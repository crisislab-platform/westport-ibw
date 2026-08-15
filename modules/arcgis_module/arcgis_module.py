"""Publish forecast impacts to ArcGIS Online.

One hosted feature layer per exposed object type -- buildings, road parcels,
properties -- each carrying one row per object per forecast timestamp, plus two
holding the water itself: the inundation extent as classified depth-band
polygons, and the inundation grid as the coarsened model grid, one row per
inundated block.

Both are published as features rather than as a rendered raster because this
org has neither raster analytics nor a named-user login: app credentials can
edit an existing feature layer but cannot create the item a tile or imagery
layer would need, and an imagery layer has no append-per-timestep path even
once created. Features also keep the water queryable and time-enabled beside
the object layers, which a picture would not.

The two are the same water at different readings. The extent answers where a
depth class reaches and stays cheap enough to draw at region scale; the grid
answers how deep it is at a place, in the ground between the modelled objects,
and carries the ensemble spread the extent has nowhere to put.

Each issue REPLACES a layer's contents: the layer is emptied and this issue's
rows written in its place, so what is published is the current forecast and
nothing else. A layer therefore holds exactly one issue's worth of rows --
``--forecast_horizon`` timesteps, one today -- and re-running a window that has
already been published overwrites it rather than duplicating it. ``issue_time``
identifies the single run every row came from; earlier issues are gone.

The replacement is not atomic. The adds are batched by the layer's
maxRecordCount, so between the truncate and the last batch the layer is empty
or partly filled, and a map open during the push sees it drain and refill --
roughly the duration of ~26k feature adds for the inundation grid. Avoiding
that would mean publishing to a second layer and swapping, which this org's app
credentials cannot do.
"""

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import logging
import time

import numpy as np
import pandas as pd
from arcgis.gis import GIS
from arcgis.features import FeatureLayer, Feature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The published schema, per object type. Trimming to an explicit list is what
# keeps the layers stable: the impact frames carry ~40 columns whose names move
# as the fuzzy inference is worked on, and a hosted layer cannot absorb a new
# field without being rebuilt. These lists are also what scripts/make_layer_
# schemas.py builds the empty layers from, so the layer you create and the rows
# you later push cannot disagree.
_SHARED_FIELDS = (
    "issue_time",
    "valid_time",
    "depth_max",
    "depth_max_std",
    "depth_mean",
    "depth_mean_std",
)

# The final risk level is the de-escalated reading -- hazard tier crossed with
# ensemble likelihood -- and every object layer carries it under the same name,
# so it means the same thing whichever layer it is read from. Beside it, the
# tier from averaging the members' depths BEFORE classifying rather than
# pooling their tiers after; it carries no likelihood of its own, so it is not
# de-escalated (see fuzzy_risk._ensemble_mean_columns). Beside that again, the
# reading from the ensemble-mean flood map -- the same collapse taken one step
# earlier, before exposure rather than after it.
_FINAL_RISK_FIELDS = (
    "ensemble_mean_hazard_level",
    "final_risk_ensemble_mean",
    "final_risk_level",
)

# Buildings also carry the crisp per-object tiers that sat BESIDE the inference
# (the FIS itself reads the fuzzified depths, not these), so the published row
# explains itself: what was measured at the footprint, its property and its
# street, and which of them drove the outcome. The _known flags matter because
# an absent link reports NaN rather than "low" -- read as a zero, a building
# with no property on record would look safe.
BUILDING_FIELDS = ("building_id",) + _SHARED_FIELDS + (
    "overall_building_hazard_level",
    "overall_building_hazard_likelihood",
    "overall_building_hazard_rule",
    "overall_building_hazard_driver",
) + _FINAL_RISK_FIELDS + (
    "building_hazard_level",
    "building_hazard_likelihood",
    "property_known",
    "property_hazard_level",
    "property_hazard_likelihood",
    "street_known",
    "street_hazard_level",
    "street_hazard_likelihood",
)

# Roads and properties are assessed on their own: no rule fires, so there is no
# rule or driver to publish, and overall_hazard_level IS the object's own crisp
# tier. Named without "building" but read the same way -- hazard tier before
# de-escalation, with final_risk_level as the de-escalated companion.
_OBJECT_HAZARD_FIELDS = (
    "overall_hazard_level",
    "overall_hazard_likelihood",
) + _FINAL_RISK_FIELDS

ROAD_FIELDS = ("id",) + _SHARED_FIELDS + _OBJECT_HAZARD_FIELDS
PROPERTY_FIELDS = ("unit_of_property_id",) + _SHARED_FIELDS + _OBJECT_HAZARD_FIELDS

# The inundation extent: one row per depth class per percentile band. Not an
# object layer, so it carries no risk tier -- the classes ARE the reading.
FLOOD_EXTENT_FIELDS = (
    "depth_class",
    "depth_min_m",
    "depth_max_m",
    "band",
    "area_ha",
    "issue_time",
    "valid_time",
)

# The inundation grid: one row per block of the coarsened model grid, covering
# the whole modelled domain rather than only the wet part, so it reads as a
# surface. Carries the ensemble MEAN depth with its spread beside it, so the
# best estimate and its uncertainty are one click apart rather than two layers
# apart. depth_mean is averaged over the block's in-domain cells with dry ones
# included, which means a block at the water's edge reads shallow because it is
# mostly dry: wet_fraction is the column that says how much of it is under, and
# the two are meant to be read together. area_ha is the WET area inside the
# block, so the column still sums to the true inundated area.
INUNDATION_GRID_FIELDS = (
    "depth_mean",
    "depth_std",
    "wet_fraction",
    "area_ha",
    "issue_time",
    "valid_time",
)

# Batches of adds in flight at once. Each is maxRecordCount features and a
# couple of megabytes, so this is bounded by what AGOL will accept before it
# throttles rather than by anything local -- the work is all waiting.
UPLOAD_WORKERS = 4

# Object ids per delete request. Sized well under an add batch because a delete
# is charged by rows touched rather than by payload, and the whole-layer delete
# this replaced was timing out server-side. Lower it if 1000 starts timing out
# too; the batches run concurrently, so smaller is not proportionally slower.
DELETE_BATCH = 1000

# Coordinates are published in degrees, where 7 decimal places is about a
# centimetre, and the impact values are depths in metres and likelihoods in
# 0..1. Full float repr spends ~18 characters saying so, and the payload is
# what the publish is actually waiting on: rounding takes about a quarter off
# it (8.2 -> 6.1 MB for a grid timestep) and changes nothing anyone can see.
COORDINATE_DECIMALS = 7
VALUE_DECIMALS = 4

# impact_forecast key -> (layer URL env var, published fields)
LAYERS = {
    "building_impact": ("ARCGIS_BUILDINGS_LAYER_URL", BUILDING_FIELDS),
    "road_impact": ("ARCGIS_ROADS_LAYER_URL", ROAD_FIELDS),
    "property_impact": ("ARCGIS_PROPERTIES_LAYER_URL", PROPERTY_FIELDS),
    "flood_extent": ("ARCGIS_FLOOD_EXTENT_LAYER_URL", FLOOD_EXTENT_FIELDS),
    "inundation_grid": ("ARCGIS_INUNDATION_GRID_LAYER_URL", INUNDATION_GRID_FIELDS),
}


class ArcGISModule():
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.arcgis_client_id = os.getenv("ARCGIS_CLIENT_ID")
        self.arcgis_client_secret = os.getenv("ARCGIS_CLIENT_SECRET")
        if not self.arcgis_client_id or not self.arcgis_client_secret:
            raise RuntimeError(
                "ARCGIS_CLIENT_ID / ARCGIS_CLIENT_SECRET are not set; add the "
                "OAuth app credentials to .env or run without --push_to_arcgis"
            )
        # One GIS for the life of the run. The publisher is called once per
        # hourly issue time, and re-authenticating on each was a round trip per
        # layer per hour for a token that had not expired.
        self.gis = GIS(
            "https://www.arcgis.com",
            client_id=self.arcgis_client_id,
            client_secret=self.arcgis_client_secret,
        )
        logger.info("Authenticated with ArcGIS Online")
        self._layer_cache = {}

    # ------------------------------------------------------------------
    # Feature layers
    # ------------------------------------------------------------------

    def get_feature_layer(self, url):
        """FeatureLayer for ``url``, cached for the life of the module.

        Accepts either a layer URL or the FeatureServer root -- the root is
        what AGOL's item page shows, and a FeatureLayer built on it fails
        later, at the first edit, with an error that says nothing about the
        URL.
        """
        url = url.strip().rstrip("/")
        if url.endswith("/FeatureServer"):
            logger.info("%s is a service root; using layer 0", url)
            url = f"{url}/0"

        if url not in self._layer_cache:
            layer = FeatureLayer(url, gis=self.gis)
            self._require_editing(layer, url)
            self._layer_cache[url] = layer
        return self._layer_cache[url]

    @staticmethod
    def _require_editing(layer, url):
        """Fail before publishing if the layer cannot be written to.

        A hosted layer created by uploading a file is query-only until editing
        is switched on, and the rejection AGOL sends back for an append is not
        obviously about permissions. Checking up front turns a confusing
        mid-run failure into one that names the fix.

        Delete is required as well as create now that publishing replaces the
        layer: a run that could add but not delete would fail partway through
        the loop, with the layers it had already reached left emptied.
        """
        capabilities = {
            capability.strip().lower()
            for capability in str(layer.properties.capabilities or "").split(",")
        }
        missing = [
            required for required in ("create", "delete")
            if required not in capabilities
        ]
        if missing:
            raise RuntimeError(
                f"{url} is missing the {' and '.join(missing)} capability "
                f"(capabilities: {layer.properties.capabilities}). Publishing "
                f"replaces the layer, so it needs both. Enable editing on the "
                f"hosted feature layer in ArcGIS Online -- Settings > Feature "
                f"layer (hosted) > Enable editing -- and allow 'Add' and 'Delete'."
            )

    def _delete_all_features(self, feature_layer, url):
        """Empty the layer before this issue's rows go in.

        Not manager.truncate(): truncate is an ADMIN operation (the manager
        rewrites the URL to /rest/admin/) that only the item's owner or an org
        administrator may call. App credentials log in as the application,
        which is never an item owner, so truncate answers 403 here however the
        layer is shared. Deleting is an ordinary layer edit, gated by the
        Delete capability _require_editing already checks for.

        Not delete_features(where="1=1") either. AGOL evaluates that as ONE
        server-side operation, and at the row counts here it exceeds the
        service's execution timeout and fails whole -- one result with
        objectId -1 and no rows removed. Asking for the object ids first and
        deleting them in explicit batches keeps each request small enough to
        finish, and the batches overlap the same way the adds do.
        """
        oids = feature_layer.query(where="1=1", return_ids_only=True).get("objectIds") or []
        if not oids:
            logger.info("%s is already empty", url)
            return

        batches = [oids[i:i + DELETE_BATCH] for i in range(0, len(oids), DELETE_BATCH)]
        with ThreadPoolExecutor(max_workers=min(UPLOAD_WORKERS, len(batches))) as pool:
            results = list(pool.map(
                lambda batch: feature_layer.delete_features(
                    deletes=",".join(str(oid) for oid in batch),
                    rollback_on_failure=True,
                ),
                batches,
            ))

        for batch, result in zip(batches, results):
            failures = [r for r in result.get("deleteResults", []) if not r.get("success")]
            if failures:
                # Stop rather than warn, as _push_adds does: adding this
                # issue's rows on top of rows that refused to go would
                # republish the duplicates this exists to remove.
                raise RuntimeError(
                    f"{len(failures)} of {len(batch)} features failed to delete "
                    f"from {url}; first error: {failures[0]}"
                )
        logger.info("Deleted %d features from %s in %d batches",
                    len(oids), url, len(batches))

    def push_impacts_to_spatial_db(self, impact_forecast, issue_time):
        """Replace the hosted feature layers with one forecast's impacts.

        Each layer is emptied and rewritten, so it ends up holding this issue's
        timesteps and nothing earlier.

        Parameters
        ----------
        impact_forecast : dict
            Output of ImpactModule.get_impact_forecast -- one
            ``{timestamp: GeoDataFrame}`` dict per object type -- optionally
            with a ``flood_extent`` entry from get_flood_extent.
        issue_time : timestamp
            When this forecast was issued, stamped on every row.

        Returns
        -------
        dict
            ``{"delete_s": ..., "publish_s": ...}``, summed over the layers.
            Split rather than reported as one number because the two measure
            different things: the delete is the replace-the-layer design
            clearing the PREVIOUS issue, while publish_s is what it costs to
            put this forecast on the map. Only the second is the framework's
            update cost.
        """
        delete_s = 0.0
        publish_s = 0.0

        for key, (url_var, fields) in LAYERS.items():
            url = os.getenv(url_var)
            if not url:
                # Publishing one layer before the others exist is a normal
                # state to be in, not an error -- say which one is missing and
                # carry on with the rest.
                logger.warning("%s is not set; skipping the %s layer", url_var, key)
                continue

            impact_series = impact_forecast.get(key) or {}
            if not impact_series:
                # Checked BEFORE the truncate: an issue that produced nothing
                # for this layer should leave the last good forecast standing
                # rather than empty the layer and leave nothing to draw.
                logger.warning("no %s frames in this forecast; nothing to publish", key)
                continue

            feature_layer = self.get_feature_layer(url)

            start = time.perf_counter()
            self._delete_all_features(feature_layer, url)
            delete_s += time.perf_counter() - start

            # Building the Esri JSON is counted as publishing: it is work this
            # forecast causes, unlike the delete above.
            start = time.perf_counter()
            for timestamp, gdf in impact_series.items():
                features = self._to_features(gdf, fields, issue_time, timestamp)
                self._push_adds(feature_layer, features)
                logger.info(
                    "Published %d %s features for %s", len(features), key, timestamp
                )
            publish_s += time.perf_counter() - start

        return {"delete_s": delete_s, "publish_s": publish_s}

    def _to_features(self, gdf, fields, issue_time, valid_time):
        """Convert one timestamp's impact frame into Esri features.

        Reprojects to WGS84 and stamps the two times, then keeps only the
        published fields.
        """
        out = gdf.to_crs(epsg=4326)
        out = out.assign(
            issue_time=pd.Timestamp(issue_time), valid_time=pd.Timestamp(valid_time)
        )

        missing = [field for field in fields if field not in out.columns]
        if missing:
            raise KeyError(
                f"impact frame is missing published fields {missing}; the layer "
                f"schema and the impact module have drifted apart"
            )

        wkid = out.crs.to_epsg()
        geometry_name = out.geometry.name
        features = []
        for _, row in out.iterrows():
            attributes = self._sanitize_attributes(
                {field: row[field] for field in fields}
            )
            esri_geom = self._geojson_geom_to_esri(row[geometry_name].__geo_interface__, wkid)
            features.append(Feature(geometry=esri_geom, attributes=attributes))
        return features

    def _push_adds(self, feature_layer, adds):
        """Add features in batches, several batches in flight at once.

        Building the Esri JSON costs under a second for the largest layer; the
        publish is all waiting on AGOL, one multi-megabyte applyEdits at a time.
        Sending the batches concurrently overlaps that waiting, which is most of
        the wall clock. Kept to UPLOAD_WORKERS rather than the full fan-out
        because the batches are megabytes each and AGOL throttles.
        """
        max_batch = feature_layer.properties.maxRecordCount or 1000
        batches = [adds[i:i + max_batch] for i in range(0, len(adds), max_batch)]
        if not batches:
            return

        with ThreadPoolExecutor(max_workers=min(UPLOAD_WORKERS, len(batches))) as pool:
            results = list(pool.map(lambda b: feature_layer.edit_features(adds=b), batches))

        for batch, result in zip(batches, results):
            failures = [r for r in result["addResults"] if not r.get("success")]
            if failures:
                # Stop rather than warn. A partly-written layer looks complete
                # in Map Viewer, so a failure that only shows up in a log is a
                # forecast with holes in it that nobody knows about.
                raise RuntimeError(
                    f"{len(failures)} of {len(batch)} features failed to publish; "
                    f"first error: {failures[0]}"
                )
        logger.info("Added %d features in %d batches", len(adds), len(batches))

    def _sanitize_attributes(self, attributes: dict) -> dict:
        clean = {}
        for k, v in attributes.items():
            if v is None or (np.isscalar(v) and pd.isna(v)):
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                # Esri date fields expect epoch milliseconds (UTC)
                clean[k] = int(v.timestamp() * 1000)
            elif isinstance(v, (np.bool_, bool)):
                # Before the float branch: numpy bools have .item() too, and a
                # rounded True is not a bool any more.
                clean[k] = bool(v)
            elif isinstance(v, (float, np.floating)):
                clean[k] = round(float(v), VALUE_DECIMALS)
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        return clean

    def _geojson_geom_to_esri(self, geom, wkid):
        geom_type = geom["type"]
        coords = geom["coordinates"]
        sr = {"wkid": wkid}

        if geom_type == "Point":
            x, y = coords
            return {
                "x": round(x, COORDINATE_DECIMALS),
                "y": round(y, COORDINATE_DECIMALS),
                "spatialReference": sr,
            }
        elif geom_type == "MultiPoint":
            return {"points": _rounded(coords), "spatialReference": sr}
        elif geom_type == "LineString":
            return {"paths": [_rounded(coords)], "spatialReference": sr}
        elif geom_type == "MultiLineString":
            return {"paths": [_rounded(path) for path in coords], "spatialReference": sr}
        elif geom_type == "Polygon":
            return {"rings": _oriented_rings(coords), "spatialReference": sr}
        elif geom_type == "MultiPolygon":
            rings = []
            for polygon in coords:
                rings.extend(_oriented_rings(polygon))
            return {"rings": rings, "spatialReference": sr}
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")


def _rounded(points):
    """One coordinate sequence at COORDINATE_DECIMALS."""
    return [
        [round(x, COORDINATE_DECIMALS), round(y, COORDINATE_DECIMALS)]
        for x, y in points
    ]


def _oriented_rings(polygon):
    """Esri ring winding for one GeoJSON polygon: outer clockwise, holes
    counter-clockwise -- the opposite of the GeoJSON convention.

    Without this a parcel with a hole in it publishes as a solid parcel, since
    Esri reads winding order rather than ring position to tell the two apart.
    Property and road parcels are multipart often enough for it to matter.
    """
    rings = []
    for index, ring in enumerate(polygon):
        is_hole = index > 0
        if _is_clockwise(ring) == is_hole:
            ring = ring[::-1]
        rings.append(_rounded(ring))
    return rings


def _is_clockwise(ring):
    """Shoelace sign for a closed ring of (x, y) pairs."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        area += (x2 - x1) * (y2 + y1)
    return area > 0
