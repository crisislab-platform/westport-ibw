import hashlib
import rasterio

from modules.lib import BUILDING_FOOTPRINTS_PATH, ROAD_NETWORK_FILE_PATH, OUTPUT_DIR, PROPERTY_BOUNDARIES_PATH, BUILDING_LINKS_PATH, BUILDING_RINGS_DIR
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import logging
import shapely
from shapely.ops import unary_union
from shapely.geometry import shape
from rasterio import features
from rasterio.transform import Affine
from modules.utils.spatial_operators.operators import  get_sample_georeference,_get_feature_labels
from modules.utils.spatial_operators.operators import load_features
# The reductions below run wherever the depth tensor lands. check_device()
# returns CUDA when it is there and CPU otherwise, so the same code path
# serves a GPU workstation and a CPU-only laptop.
from modules.utils.utils import check_device
from modules.impact_module.fuzzy_risk import (
    overall_building_hazard, object_hazard_columns,
)
import torch

logger = logging.getLogger(__name__)

# Probability at or above which a feature is treated as "likely" exposed.
LIKELY_PROBABILITY = 0.5

MIN_RING_CELLS = 4         # tune to your resolution / stats needs
MAX_BUFFER_PCT = 3        # cap growth so it doesn't blow up in open areas
GROWTH_STEP_PCT = 1.0

# Depth (m) at or above which a cell counts as wet. Cells below it are floored
# to zero before any statistic is taken.
#
# depth_mean divides by EVERY ring cell, dry ones included as zeros, so it is
# the product of two things: how deep the water is and how much of the ring it
# covers. A ring 40% covered at 0.55 m and one uniformly 0.22 m deep both score
# 0.22 m. That conflation is why wet_fraction is reported alongside it -- the
# coverage half, measured directly and separably. Note the two are correlated
# by construction (depth_mean = wet_fraction x mean-depth-where-wet), which
# matters if both are ever fed to the FIS as independent antecedents.
WET_DEPTH_MIN = 0.05

# Depth class breaks for the published inundation extent, in metres. The first
# is the wet threshold; the rest are the ARR class limits H1/H2/H3/H4 that the
# impact tiers are built on, so the extent polygons and the object tiers break
# at the same depths and a map reader can line the two up.
DEPTH_BREAKS_M = (0.05, 0.30, 0.50, 1.20, 2.00)

# Which percentile bands get vectorised: the best estimate and the reasonable
# worst case. Publishing only one would discard the spread the percentiles
# exist to carry.
FLOOD_EXTENT_BANDS = ("depth_p50", "depth_p90")

# Smallest polygon kept in the extent layer (m^2). Below roughly a tenth of a
# hectare the shapes are single-cell speckle: ~85% of the polygon count, none
# of the mapped area.
MIN_EXTENT_POLYGON_M2 = 100.0

# Cells aggregated per side for the published inundation grid, so 12 turns the
# 4.8 m model grid into 57.6 m blocks. The grid is the map of the water BETWEEN
# the modelled objects -- roads, open ground, access routes -- which the object
# layers cannot show and the dissolved extent polygons only show as a class.
#
# The org has no raster analytics licence and publishes through app credentials
# that cannot create items, so a hosted imagery layer is not available and the
# grid has to travel as features (see arcgis_module's docstring for the same
# constraint on the extent). That makes the block size a publishing budget, and
# a tighter one than it looks: the grid covers the whole domain rather than the
# wet part, and the domain is ~2.3M cells. Per timestep that is ~147k features
# at 4, ~38k at 8, ~17k at 12 and ~9.8k at 16, against 2000 features an upload
# batch. 12 buys full coverage for about what the wet-only 19.2 m grid cost,
# and still resolves which streets go under. Raise it to coarsen.
INUNDATION_BLOCK = 12

def _ensemble_mean(members, col):
    """Concatenate a column across ensemble members and return the per-row mean."""
    return pd.concat([m[col] for m in members], axis=1).mean(axis=1)


def _ensemble_std(members, col):
    """As _ensemble_mean, but the spread across members -- the forecast
    uncertainty on that depth, in the same units as the depth itself.
    """
    return pd.concat([m[col] for m in members], axis=1).std(axis=1)


def _build_exposure_columns(members, depth_cols=("depth_max", "depth_mean")):
    """Ensemble-mean depth columns and their spread, plus the fraction of
    members flagging the feature inundated.

    Each depth is reported with a ``_std`` companion: a mean over members with
    no spread beside it cannot distinguish ten members that agree on 0.4 m from
    ten that range over 0.0-0.9 m, and those are different forecasts to act on.
    This is the depth-space uncertainty; the tier-space one is measured
    separately from level agreement (see fuzzy_risk.hazard_likelihoods).
    """
    cols = {"inundated_confidence": _ensemble_mean(members, "inundated")}
    cols["inundated"] = cols["inundated_confidence"] >= 0.5
    for depth_col in depth_cols:
        cols[depth_col] = _ensemble_mean(members, depth_col)
        cols[f"{depth_col}_std"] = _ensemble_std(members, depth_col)
    return cols


def _depth_stack(members, index=None):
    """(n_members, n_features) depth_max stack for one object type.

    h_max is the whole input to the inference -- depth_mean is still carried in
    the exposure frames, but the FIS no longer reads it (see fuzzy_risk).

    ``index`` reindexes another object's rows onto the building rows (the
    property parcel or nearest road of each building); -1 means the building
    has no such link and gets zero depth, which fuzzifies to fully "low".
    """
    depth_max = np.vstack([m["depth_max"].to_numpy() for m in members])
    if index is None:
        return depth_max

    missing = index < 0
    depth_max = depth_max[:, np.where(missing, 0, index)]
    depth_max[:, missing] = 0.0
    return depth_max


def _mean_map_depths(mean_map_exposure, property_idx, road_idx):
    """The three objects' h_max from the ensemble-mean map, on building rows.

    One exposure frame per object type, each computed from the single mean map,
    so there is no member axis: _depth_stack is handed a one-member ensemble
    purely to reuse its reindexing -- the same -1-means-no-link handling the
    member stacks get, rather than a second copy of it here.
    """
    return {
        "building": _depth_stack([mean_map_exposure["building"]])[0],
        "property": _depth_stack([mean_map_exposure["property"]], property_idx)[0],
        "street": _depth_stack([mean_map_exposure["road"]], road_idx)[0],
    }


def compute_risk_matrix(building_impact, building_members, property_members,
                        road_members, property_idx, road_idx,
                        mean_map_exposure=None):
    """Attach the hazard levels and their likelihoods from the fuzzy inference
    (fuzzy_risk.overall_building_hazard).

    ``mean_map_exposure`` is the same three exposure frames computed once from
    the ensemble-mean flood map, which the inference cannot derive for itself:
    that mean is taken over the depth rasters, upstream of exposure.
    """
    return building_impact.assign(**overall_building_hazard(
        _depth_stack(building_members),
        _depth_stack(property_members, property_idx),
        _depth_stack(road_members, road_idx),
        # -1 means the building has no such object. Marked unknown rather than
        # left to fuzzify as a dry surrounding, which biased those buildings'
        # risk downwards.
        unknown={"property": property_idx < 0, "street": road_idx < 0},
        mean_map=(
            None if mean_map_exposure is None
            else _mean_map_depths(mean_map_exposure, property_idx, road_idx)
        ),
    ))


class ImpactModule():
    def __init__(self, scenario_id, domain_raster=None):
       self.scenario_id = scenario_id
       # Which buildings are modelled at all is decided by the depth raster this
       # run uses, so a validation against the surveyed raster is bounded by
       # what was surveyed rather than by the flood model's grid. None keeps the
       # model grid, which is what a forecast wants.
       self.domain_raster = domain_raster
       self.buildings = load_features(BUILDING_FOOTPRINTS_PATH, domain_raster)
       self._load_building_links()
       self.building_labels = None
       self.num_buildings = None
       # (depth.shape, raster_file) the cached building labels were
       # rasterized against. Labels are grid-specific, so a call on a
       # different grid (e.g. the surveyed raster in validate_exposure_method
       # vs the model georeference) must rebuild them rather than reuse a
       # stale, wrong-sized cache.
       self._building_label_grid = None
       # Same, per object type, for the zonal (non-ring) rasterizations --
       # roads and properties each get their own cached label raster.
       self._zonal_labels = {}
       self._zonal_label_grid = {}
       # Every footprint in the layer dissolved into one geometry, built once
       # on first use. What gets cut out of the rings -- see _footprint_union.
       self._footprint_union_cache = None
       self.use_adaptive_ring=True

    def _load_building_links(self):
        """Take the objects to model from the building link table.

        The exposure question is per building: what is the depth on ITS
        property and ITS road. So rather than measure every parcel and every
        road in the domain and index into them afterwards, only the ones some
        building actually points at are kept. That is the same computation a
        strictly per-building loop would do, minus the repetition where several
        buildings share a parcel.

        Positional indices rather than ids because the combined FIS gathers
        each building's property and road rows out of (n_members, n_objects, 3)
        activation stacks with a single take; -1 marks a building with no such
        object, which overall_building_hazard marginalises out.
        """
        links = pd.read_csv(BUILDING_LINKS_PATH)

        # The DOMAIN decides which buildings are modelled; the link table only
        # supplies the optional property and road. A building absent from it is
        # kept and reindexed to a null row, which the mapping below turns into
        # the -1 that overall_building_hazard already marginalises -- the same
        # treatment a building with no parcel gets. Dropping them instead made
        # the link
        # table a second, silent domain filter, and since it was built against
        # the model grid it removed buildings the surveyed raster covers,
        # including two placarded at the most severe damage level.
        known = self.buildings["building_id"].isin(links["building_id"])
        if not known.all():
            logger.warning(
                f"{int((~known).sum())} of {len(self.buildings)} buildings are absent "
                f"from {BUILDING_LINKS_PATH} and are scored without a property or "
                f"road; rebuild the link table to give them one"
            )
        links = links.set_index("building_id").reindex(self.buildings["building_id"])

        # Only the parcels and road parcels the buildings refer to. Read raw
        # rather than through load_features: the link table has already decided
        # these are the right objects, and domain-filtering them again would
        # drop parcels that merely straddle the edge, handing their buildings a
        # missing input for a purely geometric reason. Zonal depth over a parcel
        # that is partly outside simply uses the cells that are inside.
        used_properties = links["property_id"].dropna().unique()
        properties = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(self.buildings.crs)
        self.properties = (
            properties[properties["unit_of_property_id"].isin(used_properties)]
            .reset_index(drop=True)
        )
        used_roads = links["road_parcel_id"].dropna().unique()
        roads = gpd.read_file(ROAD_NETWORK_FILE_PATH).to_crs(self.buildings.crs)
        self.roads = roads[roads["id"].isin(used_roads)].reset_index(drop=True)

        property_row = pd.Series(
            np.arange(len(self.properties)), index=self.properties["unit_of_property_id"]
        )
        road_row = pd.Series(np.arange(len(self.roads)), index=self.roads["id"])
        self._building_property_idx = (
            links["property_id"].map(property_row).fillna(-1).to_numpy(dtype=int)
        )
        self._building_road_idx = (
            links["road_parcel_id"].map(road_row).fillna(-1).to_numpy(dtype=int)
        )
        self._building_links = links.reset_index()

        logger.info(
            f"building-centred objects: {len(self.buildings)} buildings, "
            f"{len(self.properties)} properties, {len(self.roads)} road parcels"
        )
        for label, idx in (("property", self._building_property_idx),
                           ("road parcel", self._building_road_idx)):
            missing = int((idx < 0).sum())
            if missing:
                logger.warning(
                    f"{missing} of {len(self.buildings)} buildings have no {label}; "
                    f"the combined FIS marginalises that input rather than scoring it dry"
                )

    def get_impact_forecast(self, hazard_forecast, rasterfile=None):
        # One GeoDataFrame per feature type; per-timestamp results are collected
        # as columns and added in a single assign to avoid fragmenting the frame.
        
        building_impact_series = {}
        road_impact_series = {}
        property_impact_series = {}
        for timestamp, flood_maps_ensemble in hazard_forecast.items():
            building_impact = self.buildings[[self.buildings.geometry.name, 'building_id']].copy()
            road_impact = self.roads[[self.roads.geometry.name, "id"]].copy()
            property_impact = self.properties[
                [self.properties.geometry.name, "unit_of_property_id"]
            ].copy()

            # Every member is propagated independently through the exposure
            # computation for all three object types; they are only pooled
            # after the full inference (see fuzzy_risk.overall_building_hazard).
            building_members = []
            road_members = []
            property_members = []
            for flood_map in flood_maps_ensemble:
                building_members.append(self.buffered_flood_depth(self.buildings, flood_map, raster_file=rasterfile, use_adapative_ring=self.use_adaptive_ring))
                road_members.append(self.zonal_flood_depth(self.roads, flood_map, rasterfile, cache_key="road"))
                property_members.append(self.zonal_flood_depth(self.properties, flood_map, rasterfile, cache_key="property"))

            # One more exposure pass, on the ensemble mean of the depth RASTERS
            # -- the ensemble collapsed before exposure instead of after the
            # inference. It cannot be reconstructed from the member frames
            # above: an object's h_max in the mean map is the deepest cell of
            # the average flood, not the average of the members' deepest cells,
            # and the members need not peak in the same cell. Costs one pass in
            # M + 1 rather than M. Plain mean, matching the depth_mean band in
            # _summarise_ensemble, so a cell outside the domain stays NaN.
            mean_flood_map = np.asarray(flood_maps_ensemble).mean(axis=0)
            mean_map_exposure = {
                "building": self.buffered_flood_depth(
                    self.buildings, mean_flood_map, raster_file=rasterfile,
                    use_adapative_ring=self.use_adaptive_ring),
                "road": self.zonal_flood_depth(
                    self.roads, mean_flood_map, rasterfile, cache_key="road"),
                "property": self.zonal_flood_depth(
                    self.properties, mean_flood_map, rasterfile, cache_key="property"),
            }


            building_cols = _build_exposure_columns(building_members)
            building_cols["num_ring_cells"] = pd.concat(
                [m["num_ring_cells"] for m in building_members], axis=1
            ).iloc[:, 0]
            # Ring coverage, carried through beside the depths. depth_mean is
            # now the mean over wet cells only, so how much of the ring was wet
            # is no longer recoverable from it and has to travel separately --
            # it is what distinguishes a building at the edge of the water from
            # one standing in it. Averaged over members like the depths.
            for column in ("num_wet_cells", "wet_fraction"):
                if column in building_members[0]:
                    building_cols[column] = _ensemble_mean(building_members, column)
            building_impact = building_impact.assign(**building_cols)

            building_impact = compute_risk_matrix(
                building_impact,
                building_members,
                property_members,
                road_members,
                self._building_property_idx,
                self._building_road_idx,
                mean_map_exposure,
            )

            building_impact_series[timestamp] = building_impact

            # Roads and properties reported in their own right, not only as
            # antecedents on the building rows. Their depths were computed
            # above to feed the combined FIS; assessing them here costs one
            # more pass over arrays already in hand, and gives a road parcel a
            # tier of its own rather than one only readable off the buildings
            # that happen to face it.
            #
            # _depth_stack with no index returns each object's own rows (the
            # indexed form above is what reprojects them onto buildings), and
            # the object_type must match the object so it is read against that
            # type's cutoffs (see OBJECT_FUZZIFICATION).
            for series, impact, members, object_type in (
                (road_impact_series, road_impact, road_members, "road"),
                (property_impact_series, property_impact, property_members, "property"),
            ):
                impact = impact.assign(**_build_exposure_columns(members))
                impact = impact.assign(
                    **object_hazard_columns(
                        _depth_stack(members), object_type,
                        # This layer's own rows, so no reindexing onto buildings.
                        mean_map_depth_max=_depth_stack(
                            [mean_map_exposure[object_type]]
                        )[0],
                    )
                )
                series[timestamp] = impact

        return {
            "building_impact": building_impact_series,
            "road_impact": road_impact_series,
            "property_impact": property_impact_series,
        }
        


    def save_impact_forecast(self, impact_forecast, flood_maps, issue_time, output_dir=OUTPUT_DIR):
        """
        Persist an impact forecast to disk under ``output_dir/<issue_time>/``.

        get_impact_forecast returns one GeoDataFrame per feature type PER
        TIMESTAMP (``{"building_impact": {timestamp: gdf}, "road_impact": {...}}``).
        Each feature type's per-timestamp frames are concatenated into a
        single long-format GeoDataFrame (one row per feature per timestamp,
        with added ``timestamp`` and ``issue_time`` columns) before being
        written out, so every forecast timestamp in the horizon is represented
        in the saved layers.

        Parameters:
        impact_forecast (dict): Output of :meth:`get_impact_forecast`.
        issue_time: Time the forecast was issued (used as the run folder name).
        output_dir (str): Root directory for forecast outputs.

        Returns:
        str: Path to the created run directory.
        """
        issue_str = pd.to_datetime(issue_time).strftime("%Y%m%dT%H%M%S")
        run_dir = os.path.join(output_dir, issue_str)
        os.makedirs(run_dir, exist_ok=True)

        # One long frame per feature type, all three carrying the same
        # timestamp/issue_time pair so a row is always traceable to the run
        # that produced it as well as the hour it describes.
        longs = {
            name: self._concat_timestamped(impact_forecast[f"{name}_impact"], issue_time)
            for name in ("building", "road", "property")
        }

        gpkg_path = os.path.join(run_dir, f"impact_{issue_str}.gpkg")
        for name, layer in (("building", "buildings"), ("road", "roads"),
                            ("property", "properties")):
            # One layer per feature type, every timestamp stacked as rows.
            longs[name].to_file(gpkg_path, layer=layer, driver="GPKG")

            # Also export WGS84 GeoJSON per feature type for web/GIS tools
            # (e.g. ArcGIS Online / Scene Viewer), which do not read GeoPackage,
            # plus the same frame with a string valid_time field so the layer
            # can be time-enabled in ArcGIS Online / Map Viewer.
            self._write_geojson(longs[name], os.path.join(run_dir, f"{layer}.geojson"))
            self._write_timeseries_geojson(
                longs[name], os.path.join(run_dir, f"{layer}_timeseries.geojson")
            )

        # The inundation extent is deliberately NOT written here. get_flood_extent
        # computes it per issue time to be PUBLISHED (see arcgis_module.LAYERS),
        # and the multiband GeoTIFF written below already holds the same field
        # locally at full resolution -- the polygons are a classified, simplified,
        # dissolved reading of its p50/p90 bands, so saving them too spent ~5 MB
        # a run on a lossy copy of a file already in the same directory.

        # Summarise the ensemble per timestamp into one multiband GeoTIFF,
        # georeferenced the same way as the zonal sampling so it aligns with
        # the vector layers.
        transform, raster_crs, nodata = get_sample_georeference()
        for timestamp, flood_maps_ensemble in flood_maps.items():
            # Coerce (possibly torch, possibly extra dims) to a 3D float32 stack
            # of (n_members, H, W).
            ensemble = flood_maps_ensemble
            if hasattr(ensemble, "detach"):
                ensemble = ensemble.detach().cpu().numpy()
            ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))
            if ensemble.ndim != 3:
                raise ValueError(
                    f"flood map ensemble must be (n_members, H, W), got shape {ensemble.shape}"
                )

            timestamp_str = pd.to_datetime(timestamp).strftime("%Y%m%dT%H%M%S")
            self._write_geotiff(
                os.path.join(run_dir, f"flood_map_{timestamp_str}.tif"),
                *self._summarise_ensemble(ensemble), transform, raster_crs, nodata,
            )
        return run_dir

    def get_flood_extent(self, flood_maps, bands=FLOOD_EXTENT_BANDS):
        """Inundation extent as classified depth-band polygons, per timestamp.

        The full-resolution rasters stay on disk; this is the form the extent
        travels in. Polygons publish through the same feature-layer path as the
        buildings, roads and properties, which is what makes the extent
        queryable and time-enabled beside them rather than a picture that has
        to be read off a legend.

        Both the median and the reasonable-worst-case band are emitted and
        tagged, so the map can show a best estimate and the spread around it as
        two readings of the same hour -- publishing one alone would throw away
        the reason the percentiles are computed.

        Returns ``{timestamp: GeoDataFrame}`` with one row per depth class per
        band, in the raster's CRS.
        """
        transform, raster_crs, _ = get_sample_georeference()
        extent_series = {}
        for timestamp, flood_maps_ensemble in flood_maps.items():
            ensemble = flood_maps_ensemble
            if hasattr(ensemble, "detach"):
                ensemble = ensemble.detach().cpu().numpy()
            ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))

            summary, band_names = self._summarise_ensemble(ensemble)
            frames = []
            for band in bands:
                depth = summary[band_names.index(band)]
                frame = self._depth_band_polygons(depth, transform, raster_crs)
                frame["band"] = band
                frames.append(frame)

            extent_series[timestamp] = gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True), geometry="geometry", crs=raster_crs
            )
        return extent_series

    def get_inundation_grid(self, flood_maps, block=INUNDATION_BLOCK,
                            threshold=WET_DEPTH_MIN):
        """Inundated area as a coarsened cell grid, per timestamp.

        The extent polygons say where each depth CLASS reaches; this says what
        the depth actually is, cell by cell, in the ground between the modelled
        objects. Only the ensemble mean is gridded -- the spread travels as a
        ``depth_std`` field on the same block rather than as a second geometry,
        so a reader sees the best estimate and its uncertainty in one click
        instead of comparing two layers.

        Blocks are ``block`` x ``block`` model cells (see INUNDATION_BLOCK), and
        every block holding modelled ground is emitted -- dry ones included --
        so the result is a contiguous surface over the study area rather than
        isolated patches of water. ``threshold`` no longer decides what is
        emitted, only what counts towards ``wet_fraction`` and ``area_ha``.

        Returns ``{timestamp: GeoDataFrame}`` with one row per in-domain block,
        in the raster's CRS.
        """
        transform, raster_crs, _ = get_sample_georeference()
        grid_series = {}
        for timestamp, flood_maps_ensemble in flood_maps.items():
            ensemble = flood_maps_ensemble
            if hasattr(ensemble, "detach"):
                ensemble = ensemble.detach().cpu().numpy()
            ensemble = np.squeeze(np.asarray(ensemble, dtype="float32"))
            if ensemble.ndim != 3:
                raise ValueError(
                    f"flood map ensemble must be (n_members, H, W), got shape {ensemble.shape}"
                )

            summary, band_names = self._summarise_ensemble(ensemble)
            grid_series[timestamp] = self._block_grid(
                summary[band_names.index("depth_mean")],
                summary[band_names.index("depth_std")],
                transform, raster_crs, block, threshold,
            )
        return grid_series

    @staticmethod
    def _block_grid(depth, spread, transform, crs, block, threshold):
        """Aggregate a depth/spread raster pair into inundated block polygons."""
        cell_area = abs(transform.a) * abs(transform.e)
        rows, cols = depth.shape
        # Trim any partial block at the far edge rather than pad it: a padded
        # block would carry cells that do not exist and dilute its statistics.
        rows, cols = rows // block * block, cols // block * block

        def to_blocks(array):
            """(H, W) -> (H/block, W/block, block*block), cells last."""
            return (
                array[:rows, :cols]
                .reshape(rows // block, block, cols // block, block)
                .transpose(0, 2, 1, 3)
                .reshape(rows // block, cols // block, block * block)
            )

        depth_cells, spread_cells = to_blocks(depth), to_blocks(spread)
        # NaN is outside the computational domain, so it is neither wet nor a
        # zero depth -- nan_to_num only decides the comparison, and in_domain is
        # what every statistic here is measured over.
        in_domain = np.isfinite(depth_cells)
        wet = np.nan_to_num(depth_cells) >= threshold
        n_wet, n_domain = wet.sum(axis=-1), in_domain.sum(axis=-1)

        # Floored below the wet threshold before averaging, which is what
        # WET_DEPTH_MIN is for and what _gpu_zonal_stats already does. Sub-
        # threshold output is the decoder's noise around zero on dry ground, and
        # ~60% of in-domain cells come back very slightly NEGATIVE: averaged raw
        # it publishes negative depths, and clipping only the negative half
        # would bias every dry block upwards instead. Flooring both halves
        # leaves dry ground at exactly 0.0.
        depth_cells = np.where(wet, depth_cells, 0.0)

        # Every block holding any modelled ground, not just the wet ones, so the
        # layer is a surface covering the study area rather than islands of water
        # on a blank background. The domain is irregular and fills only ~39% of
        # its own bounding box, so this is still far from a full rectangle.
        selected = n_domain > 0

        # Averaged over the IN-DOMAIN cells, dry ones entering as zero. That
        # makes this a straight downsample of the raster: the surface stays
        # continuous across the water's edge, and a block of marginal ground
        # keeps a real spread instead of being flattened away. A partly-wet
        # block therefore reads shallow BECAUSE it is mostly dry -- depth_mean
        # is wet_fraction x mean-depth-where-wet, the same conflation the
        # WET_DEPTH_MIN comment describes -- so the two columns have to be read
        # together, and wet_fraction is what makes that possible.
        def domain_mean(cells):
            return np.divide(
                np.where(in_domain, np.nan_to_num(cells), 0.0).sum(axis=-1), n_domain,
                out=np.zeros(n_domain.shape, dtype=float), where=n_domain > 0,
            )

        depth_mean, depth_std = domain_mean(depth_cells), domain_mean(spread_cells)

        block_rows, block_cols = np.nonzero(selected)
        block_transform = transform * Affine.scale(block)
        left = block_transform.c + block_cols * block_transform.a
        top = block_transform.f + block_rows * block_transform.e
        # e is negative for a north-up raster, so top + e is the lower edge.
        geometry = shapely.box(
            left, top + block_transform.e, left + block_transform.a, top
        )

        return gpd.GeoDataFrame(
            {
                # Four places, not three: dilution puts a block holding one wet
                # cell in 144 at ~0.0003 m, which rounds to a flat 0.000 at
                # three and becomes indistinguishable from genuinely dry ground.
                # depth_mean == 0 has to keep meaning dry.
                "depth_mean": np.round(depth_mean[selected], 4),
                "depth_std": np.round(depth_std[selected], 4),
                "wet_fraction": np.round(
                    np.divide(
                        n_wet[selected], n_domain[selected],
                        out=np.zeros(int(selected.sum()), dtype=float),
                        where=n_domain[selected] > 0,
                    ), 3,
                ),
                # The WET area inside the block, not the block's footprint, so
                # the column sums to the true inundated area instead of the
                # area of every block the water clipped a corner of.
                "area_ha": np.round(n_wet[selected] * cell_area / 1e4, 4),
            },
            geometry=geometry, crs=crs,
        )

    @staticmethod
    def _depth_band_polygons(depth, transform, crs,
                             breaks=DEPTH_BREAKS_M, min_area_m2=MIN_EXTENT_POLYGON_M2):
        """Vectorise one depth raster into one multipolygon per depth class."""
        # NaN marks cells outside the computational domain. digitize would send
        # them to the top bin, painting the whole dry domain as the deepest
        # class, so they are masked to 0 (nothing) before classifying rather
        # than after.
        finite = np.isfinite(depth)
        classes = np.digitize(np.where(finite, depth, -1.0), breaks).astype("int16")
        classes[~finite] = 0

        shapes = [
            (shape(geom), int(value))
            for geom, value in features.shapes(
                classes, mask=classes > 0, transform=transform
            )
        ]
        frame = gpd.GeoDataFrame(
            {"depth_class": [value for _, value in shapes]},
            geometry=[geom for geom, _ in shapes],
            crs=crs,
        )

        # Single-cell speckle is most of the polygon count and none of the
        # extent -- dropping it below a tenth of a hectare cuts the shape count
        # by ~85% without moving the mapped area.
        frame = frame[frame.area >= min_area_m2].copy()
        if frame.empty:
            return gpd.GeoDataFrame(
                {"depth_class": [], "depth_min_m": [], "depth_max_m": [], "area_ha": []},
                geometry=[], crs=crs,
            )

        # Half a cell: enough to drop the staircase edge the rasterisation
        # introduces, not enough to move a boundary by a cell.
        frame["geometry"] = frame.simplify(2.0)
        # buffer(0) repairs the self-intersections simplification can leave
        # behind; AGOL rejects an invalid ring at push time, not at write time.
        invalid = ~frame.is_valid
        if invalid.any():
            frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].buffer(0)

        dissolved = frame.dissolve(by="depth_class").reset_index()
        # The class bounds travel with the row so the layer explains its own
        # symbology instead of depending on a legend kept somewhere else.
        upper = list(breaks[1:]) + [None]
        dissolved["depth_min_m"] = [breaks[c - 1] for c in dissolved["depth_class"]]
        dissolved["depth_max_m"] = [upper[c - 1] for c in dissolved["depth_class"]]
        dissolved["area_ha"] = dissolved.area / 1e4
        return dissolved

    @staticmethod
    def _summarise_ensemble(ensemble):
        """Reduce an (n_members, H, W) depth ensemble to the reported bands.

        A mean and a spread say how uncertain a cell is but not where the
        plausible depths actually lie, and the spread is only interpretable
        relative to the depth it sits on. The percentiles put the same
        uncertainty back in metres: p10-p90 is the range to plan against, read
        directly against the ARR class limits the impact tiers use, and p90 is
        the reasonable worst case a warning should be built on.

        Returns a (5, H, W) stack and the matching band names.
        """
        # These are nanpercentile's semantics -- a cell outside the computational
        # domain is nodata, not a zero depth -- but taken over the in-domain
        # cells only rather than by calling nanpercentile on the whole grid.
        # numpy has no vectorised NaN-aware percentile, so that call falls back
        # to a Python-level pass over all 5.9M columns and costs ~50 s per
        # timestamp, even though ~60% of the grid lies outside the domain and
        # can only ever be NaN. Masking first puts the rest on the C path and
        # returns the same values bit for bit in ~0.6 s.
        finite = np.isfinite(ensemble)
        complete = finite.all(axis=0)
        percentiles = np.full((3,) + ensemble.shape[1:], np.nan, dtype="float32")
        if complete.any():
            percentiles[:, complete] = np.percentile(
                ensemble[:, complete], [10, 50, 90], axis=0
            )

        # A cell finite in some members but not all cannot take the fast path.
        # The decoder has never produced one -- every member shares the same
        # domain mask, so a cell is NaN in all ten or none -- but routing any
        # through nanpercentile keeps this an optimisation of the existing
        # reduction rather than a redefinition of it.
        partial = finite.any(axis=0) & ~complete
        if partial.any():
            percentiles[:, partial] = np.nanpercentile(
                ensemble[:, partial], [10, 50, 90], axis=0
            )

        p10, p50, p90 = percentiles
        bands = np.stack(
            [ensemble.mean(axis=0), ensemble.std(axis=0), p10, p50, p90]
        ).astype("float32")
        return bands, ("depth_mean", "depth_std", "depth_p10", "depth_p50", "depth_p90")

    @staticmethod
    def _concat_timestamped(impact_series, issue_time=None):
        """Concatenate a ``{timestamp: GeoDataFrame}`` dict (get_impact_forecast's
        per-feature-type output) into one long-format GeoDataFrame with added
        ``timestamp`` and ``issue_time`` columns (datetime64) -- one row per
        feature per timestamp.

        ``issue_time`` is stamped alongside the valid time because the two are
        not recoverable from each other: successive hourly runs forecast
        overlapping valid times, and without the issue time an appended layer
        cannot tell a re-run from a new forecast.
        """
        if not impact_series:
            raise ValueError(
                "empty impact series -- get_impact_forecast returned no frames for "
                "this feature type, so there is nothing to concatenate"
            )
        frames = []
        for timestamp, gdf in impact_series.items():
            frame = gdf.copy()
            frame["timestamp"] = pd.Timestamp(timestamp)
            if issue_time is not None:
                frame["issue_time"] = pd.Timestamp(issue_time)
            frames.append(frame)
        first = next(iter(impact_series.values()))
        return gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry=first.geometry.name, crs=first.crs
        )

    def _write_geojson(self, gdf, path):
        """Write a GeoDataFrame to WGS84 GeoJSON for web/GIS consumption.

        Reprojects to EPSG:4326 and stringifies datetime columns (e.g.
        ``timestamp``, or an onset column with NaT) so they serialise cleanly.
        """
        out = gdf.to_crs(epsg=4326).copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        out.to_file(path, driver="GeoJSON")

    def _write_timeseries_geojson(self, impact_long, path, use_centroids=False):
        """Write a long-format WGS84 GeoJSON with a string ``valid_time`` field
        (derived from the ``timestamp`` column), letting ArcGIS Online /
        Map Viewer time-enable the layer.

        Parameters:
        impact_long (GeoDataFrame): Long-format impact output (one row per
            feature per timestamp, with a ``timestamp`` column) -- see
            :meth:`_concat_timestamped`.
        path (str): Destination GeoJSON path.
        use_centroids (bool): Emit centroids (points) instead of polygons, to
            keep the (heavily duplicated across timestamps) geometry small.
        """
        out = impact_long.copy()
        if use_centroids:
            out[out.geometry.name] = out.geometry.centroid
        out = out.to_crs(epsg=4326)
        out["valid_time"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        out = out.drop(columns=["timestamp"])
        if "issue_time" in out.columns:
            out["issue_time"] = pd.to_datetime(out["issue_time"]).dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        out.to_file(path, driver="GeoJSON")

    def _write_geotiff(self, path, array, band_names, transform, crs, nodata):
        """Write a 2D array or an (n_bands, H, W) stack to a GeoTIFF at ``path``.

        ``band_names`` are set as band descriptions so a multiband file is
        self-describing -- "Band 3" in QGIS is not a useful label for a
        percentile, and the alternative is a naming convention nobody outside
        this module knows about.
        """
        array = np.asarray(array)
        if array.ndim == 2:
            array = array[np.newaxis]
        band_names = (band_names,) if isinstance(band_names, str) else tuple(band_names)
        if len(band_names) != array.shape[0]:
            raise ValueError(
                f"got {len(band_names)} band names for {array.shape[0]} bands"
            )

        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            for index, name in enumerate(band_names, start=1):
                dst.write(array[index - 1], index)
                dst.set_band_description(index, name)



    def buffered_flood_depth(
        self,
        exposed_objects,
        flood_map,
        buffer_pct: float = 2.0,
        mean_threshold: float = 0.05,
        max_threshold: float = 0.15,
        raster_file: str = None,
        use_adapative_ring:bool =False,
    ):
        """
        Compute buffered (ring) flood-depth statistics using GPU reductions.
        Rings are allowed to overlap between neighboring buildings (common in
        tightly packed blocks); the sparse rasterization path preserves each
        building's own pixels instead of losing them to whichever neighbor's
        ring rasterizes last. Labels/indices are computed once and cached.
        """

        # ---------------------------------------------------------------
        # Keep flood map on GPU if possible
        # ---------------------------------------------------------------
        if isinstance(flood_map, np.ndarray):
            depth = torch.from_numpy(flood_map).float().to(check_device())
        elif torch.is_tensor(flood_map):
            depth = flood_map.squeeze().float()
            if depth.device.type != check_device().type:
                depth = depth.to(check_device())
        else:
            raise TypeError(
                "flood_map must be either a NumPy array or a torch.Tensor"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"buffered_flood_depth expects a 2D flood map, got shape {tuple(depth.shape)}"
            )

        # ---------------------------------------------------------------
        # Georeferencing
        # ---------------------------------------------------------------
        transform, raster_crs, _ = get_sample_georeference(raster_file)

        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)

        if not np.isclose(cell_size_x, cell_size_y, rtol=1e-3):
            raise ValueError(
                f"Non-square pixels ({cell_size_x} x {cell_size_y})"
            )

        buffer_dist = cell_size_x * buffer_pct

        # ---------------------------------------------------------------
        # Build ring indices once per grid (rebuild if the depth grid /
        # raster file changes, since the cached indices are grid-specific).
        # ---------------------------------------------------------------
        grid_key = (tuple(depth.shape), raster_file)
        if self.building_labels is None or self._building_label_grid != grid_key:
            ring_objects = exposed_objects.copy()
            if use_adapative_ring:
                # Precomputed rings if there are any for this grid. The geometry
                # is identical to what the loop would produce -- the same method
                # wrote it -- so this is a pure time saving, and a mismatched or
                # absent file falls through to building them here.
                ring_key = self._ring_cache_key(
                    tuple(depth.shape), transform, raster_crs)
                cached = self._load_ring_cache(exposed_objects, ring_key)
                if cached is None:
                    ring_geom, _, exhausted = self._build_adaptive_rings(
                        exposed_objects, tuple(depth.shape), transform,
                        raster_crs, cell_size_x
                    )
                else:
                    ring_geom, exhausted, self._ring_buffer_pct = cached
                self._exhausted_buildings = exhausted
                ring_objects["geometry"] = ring_geom.values
            else:
                 all_buildings_geom = self._footprint_union()
                 ring_objects["geometry"] = (
                     exposed_objects.geometry            
                     .buffer(buffer_dist)            
                     .difference(all_buildings_geom)            
                 )
            # allow_overlap=True: rings can legitimately overlap in packed
            # blocks; every building keeps its own uncontested pixels instead of
            # losing them to whichever neighbor's ring rasterizes last.
            self._row_idx, self._col_idx, self._feat_idx, self.num_buildings = _get_feature_labels(
                ring_objects,
                tuple(depth.shape),
                transform,
                raster_crs,
                allow_overlap=True,
            )
            self.building_labels = True  # sentinel: cache has been built
            self._building_label_grid = grid_key

        # ---------------------------------------------------------------
        # GPU zonal statistics (sparse, overlap-safe)
        # ---------------------------------------------------------------
        (
            depth_max,
            _,
            depth_sum,
            depth_mean,
            counts,
            wet_counts,
        ) = self._gpu_zonal_stats_sparse(
            depth,
            self._row_idx,
            self._col_idx,
            self._feat_idx,
            self.num_buildings,
        )

        # ---------------------------------------------------------------
        # Build output
        # ---------------------------------------------------------------
        result = exposed_objects.copy()

        result["depth_max"] = np.round(depth_max.cpu().numpy(), 2)
        result["depth_mean"] = np.round(depth_mean.cpu().numpy(), 2)
        result["buffer_dist"] = buffer_dist
        result["ring_pixel_count"] = counts.cpu().numpy()
        result["num_ring_cells"] = counts.cpu().numpy()
        # Ring coverage, kept separate from depth_mean now that the mean is
        # taken over wet cells only. num_ring_cells stays the GEOMETRIC cell
        # count, so the "ring never sampled anything" filter is unaffected.
        result["num_wet_cells"] = wet_counts.cpu().numpy()
        result["wet_fraction"] = np.divide(
            wet_counts.cpu().numpy(), counts.cpu().numpy(),
            out=np.zeros(len(result), dtype=float),
            where=counts.cpu().numpy() > 0,
        )

        result["inundated"] = (
            (result["depth_max"] >= max_threshold)
            & (result["depth_mean"] >= mean_threshold)
        )
        return result

    def zonal_flood_depth(
        self,
        exposed_objects,
        flood_map,
        raster_file=None,
        cache_key="road",
    ):
        """
        Compute zonal flood statistics using GPU reductions.
        Rasterization is cached and performed only once.

        ``cache_key`` names the object type so each set of zones (roads,
        properties) keeps its own cached label raster; passing one key for two
        different geometries would reuse the wrong labels.
        """

        mean_threshold = 0.05
        max_threshold = 0.15

        # ---------------------------------------------------------------
        # Keep the flood map on the GPU if possible
        # ---------------------------------------------------------------
        if isinstance(flood_map, np.ndarray):
            depth = torch.from_numpy(flood_map).float().to(check_device())
        elif torch.is_tensor(flood_map):
            depth = flood_map.squeeze().float()
            if depth.device.type != check_device().type:
                depth = depth.to(check_device())
        else:
            raise TypeError(
                "flood_map must be either a NumPy array or a torch.Tensor"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"zonal_flood_depth expects a 2D flood map, got shape {tuple(depth.shape)}"
            )

        # ---------------------------------------------------------------
        # Georeferencing
        # ---------------------------------------------------------------
        transform, raster_crs, _ = get_sample_georeference(raster_file)

        if exposed_objects.crs != raster_crs:
            exposed_objects = exposed_objects.to_crs(raster_crs)

        # ---------------------------------------------------------------
        # Build labels once per grid (rebuild if the depth grid / raster file
        # changes, since the cached labels are grid-specific).
        # ---------------------------------------------------------------
        grid_key = (tuple(depth.shape), raster_file)
        if self._zonal_label_grid.get(cache_key) != grid_key:
            labels, n_features = _get_feature_labels(
                exposed_objects,
                tuple(depth.shape),
                transform,
                raster_crs,
            )
            self._zonal_label_grid[cache_key] = grid_key
            self._zonal_labels[cache_key] = (
                torch.from_numpy(labels).long().to(depth.device),
                n_features,
            )

        labels_gpu, n_features = self._zonal_labels[cache_key]

        # ---------------------------------------------------------------
        # Prepare flood map
        # ---------------------------------------------------------------
        depth = torch.nan_to_num(depth, nan=0.0)
        depth[depth < mean_threshold] = 0.0

        # ---------------------------------------------------------------
        # GPU zonal statistics
        # ---------------------------------------------------------------
        (
            depth_max,
            depth_min,
            depth_sum,
            depth_mean,
            counts,
            wet_counts,
        ) = self._gpu_zonal_stats(
            depth,
            labels_gpu,
            n_features,
        )

        # ---------------------------------------------------------------
        # Copy only the statistics back to CPU
        # ---------------------------------------------------------------
        result = exposed_objects.copy()

        result["depth_max"] = depth_max.cpu().numpy()
        result["depth_min"] = depth_min.cpu().numpy()
        result["depth_mean"] = depth_mean.cpu().numpy()
        result["depth_sum"] = depth_sum.cpu().numpy()
        result["num_pixels"] = counts.cpu().numpy()
        result["num_wet_pixels"] = wet_counts.cpu().numpy()

        result["inundated"] = (
            (result["depth_max"] >= max_threshold)
            & (result["depth_mean"] >= mean_threshold)
        )

        return result
    
    import torch


    def _gpu_zonal_stats(self, depth, labels, n_features):
        """
        depth  : torch.FloatTensor (H,W) on GPU
        labels : torch.LongTensor  (H,W) on GPU

        Returns
        -------
        depth_max
        depth_min
        depth_sum
        depth_mean
        counts
        """

    
        labels_flat = labels.reshape(-1)
        depth_flat = depth.reshape(-1)

        # Ignore background
        valid = (labels_flat > 0) & torch.isfinite(depth_flat)
        labels_flat = labels_flat[valid]
        depth_flat = depth_flat[valid]
        counts = torch.bincount(labels_flat, minlength=n_features + 1)

        depth_flat = torch.nan_to_num(depth_flat, nan=0.0)
        depth_flat[depth_flat < WET_DEPTH_MIN] = 0.0

        # Wet-cell tally per feature, the denominator of the ring mean. Taken
        # after the flooring so it uses exactly the cells that contribute to
        # depth_sum -- a cell in the sum but not the count would bias the mean
        # upward.
        wet_counts = torch.zeros(
            n_features + 1,
            device=depth.device,
            dtype=depth.dtype,
        )
        wet_counts.scatter_add_(0, labels_flat, (depth_flat > 0).to(depth.dtype))

        # ---------------- Sum ----------------

        depth_sum = torch.zeros(
            n_features + 1,
            device=depth.device,
            dtype=depth.dtype,
        )

        depth_sum.scatter_add_(0, labels_flat, depth_flat)

        # ---------------- Max ----------------

        depth_max = torch.full(
            (n_features + 1,),
            -torch.inf,
            device=depth.device,
        )

        depth_max.scatter_reduce_(
            0,
            labels_flat,
            depth_flat,
            reduce="amax",
            include_self=True,
        )

        # ---------------- Min ----------------

        depth_min = torch.full(
            (n_features + 1,),
            torch.inf,
            device=depth.device,
        )

        depth_min.scatter_reduce_(
            0,
            labels_flat,
            depth_flat,
            reduce="amin",
            include_self=True,
        )

        depth_max[torch.isinf(depth_max)] = 0
        depth_min[torch.isinf(depth_min)] = 0

        counts = counts[1:]
        wet_counts = wet_counts[1:]
        depth_sum = depth_sum[1:]
        depth_max = depth_max[1:]
        depth_min = depth_min[1:]

        # Mean over EVERY ring cell, dry ones entering as zeros.
        depth_mean = torch.where(
            counts > 0,
            depth_sum / counts.float(),
            torch.zeros_like(depth_sum),
        )

        return (
            depth_max,
            depth_min,
            depth_sum,
            depth_mean,
            counts,
            wet_counts,
        )
        
        
    def _gpu_zonal_stats_sparse(self, depth, row_idx, col_idx, feature_idx, n_features):
        """Same outputs as _gpu_zonal_stats, but driven by sparse (row, col,
        feature) triples so overlapping features are supported — a pixel can
        contribute to more than one feature_idx.
        """
        if row_idx.size == 0:
            z = torch.zeros(n_features, device=depth.device, dtype=depth.dtype)
            c = torch.zeros(n_features, device=depth.device, dtype=torch.long)
            return z.clone(), z.clone(), z.clone(), z.clone(), c, z.clone()

        row_t = torch.from_numpy(row_idx).long().to(depth.device)
        col_t = torch.from_numpy(col_idx).long().to(depth.device)
        feat_t = torch.from_numpy(feature_idx).long().to(depth.device)

        vals = depth[row_t, col_t]
        vals = torch.nan_to_num(vals, nan=0.0)
        vals = torch.where(vals < WET_DEPTH_MIN, torch.zeros_like(vals), vals)

        counts = torch.bincount(feat_t, minlength=n_features)

        depth_sum = torch.zeros(n_features, device=depth.device, dtype=depth.dtype)
        depth_sum.scatter_add_(0, feat_t, vals)

        # Denominator of the ring mean: wet cells only, counted after the
        # flooring so it matches exactly what went into depth_sum.
        wet_counts = torch.zeros(n_features, device=depth.device, dtype=depth.dtype)
        wet_counts.scatter_add_(0, feat_t, (vals > 0).to(depth.dtype))

        depth_max = torch.full((n_features,), -torch.inf, device=depth.device)
        depth_max.scatter_reduce_(0, feat_t, vals, reduce="amax", include_self=True)

        depth_min = torch.full((n_features,), torch.inf, device=depth.device)
        depth_min.scatter_reduce_(0, feat_t, vals, reduce="amin", include_self=True)

        depth_max[torch.isinf(depth_max)] = 0
        depth_min[torch.isinf(depth_min)] = 0

        # Mean over EVERY ring cell, dry ones entering as zeros.
        depth_mean = torch.where(
            counts > 0, depth_sum / counts.float(), torch.zeros_like(depth_sum)
        )

        return depth_max, depth_min, depth_sum, depth_mean, counts, wet_counts
        

    def _footprint_union(self):
        """Every building footprint in the layer, dissolved, in the modelled CRS.

        The geometry subtracted from the buffers so a ring is the ground around
        a building and never a roof. It has to come from the FULL layer rather
        than from self.buildings: load_features keeps only footprints sitting on
        a finite depth cell, so a building just outside the model domain is not
        in the modelled set, yet is still physically there next to one that is.
        Subtracting only the modelled set leaves those roofs inside a
        neighbour's ring, where their cells are sampled as if they were ground.

        Read raw rather than through load_features for exactly that reason --
        the domain filter is what has to be bypassed here -- and cached, since
        the union over the whole layer is the same for every call.
        """
        if self._footprint_union_cache is None:
            footprints = gpd.read_file(BUILDING_FOOTPRINTS_PATH).to_crs(
                self.buildings.crs
            )
            self._footprint_union_cache = unary_union(footprints.geometry.values)
        return self._footprint_union_cache

    def _ring_cache_key(self, out_shape, transform, raster_crs,
                        min_cells=MIN_RING_CELLS, max_buffer_pct=MAX_BUFFER_PCT,
                        step_pct=GROWTH_STEP_PCT):
        """Everything the ring geometry depends on, as one comparable string.

        Deliberately NOT the raster file: the growth loop counts cells
        geometrically and never reads a depth value, so rings are a property of
        the grid and the footprints, not of the scenario. One cached file
        therefore serves every scenario and timestep sharing a grid, which is
        the whole point of caching it. What does invalidate it -- a different
        grid, a different CRS, retuned growth parameters, a changed building
        population -- is exactly what this key carries.

        The grid matters for more than the labels: a building outside a raster's
        extent clips no cell at any width, so it grows to the cap there and not
        on a grid that covers it. The surveyed July 2021 raster is smaller than
        the model grid and 86 buildings fall outside it, which is why each grid
        gets its own file rather than sharing one.
        """
        return "|".join([
            f"{out_shape[0]}x{out_shape[1]}",
            ",".join(f"{v:.6f}" for v in (transform.a, transform.b, transform.c,
                                          transform.d, transform.e, transform.f)),
            str(raster_crs),
            f"min_cells={min_cells:g},max_buffer={max_buffer_pct:g},step={step_pct:g}",
            f"buildings={len(self.buildings)}",
        ])

    def _ring_cache_path(self, key, directory=BUILDING_RINGS_DIR):
        """Where the rings for one grid live.

        Named from the key so every grid gets its own file and neither run has
        to be told which one it wants: the grid it is working on picks the path.
        The shape is kept legible in the name and the digest carries the rest of
        the key, which is too long to read but has to be distinguished.
        """
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        return os.path.join(
            directory, f"building_rings_{key.split('|')[0]}_{digest}.geojson"
        )

    def build_ring_cache(self, out_shape, transform, raster_crs, cell_size,
                         path=None, **growth_kwargs):
        """Compute the rings once and write them where a run can load them.

        Called by scripts/build_building_rings.py rather than from a forecast:
        writing during a run would have several processes racing for the same
        file, and the rebuild is the cost the cache exists to avoid.
        """
        key = self._ring_cache_key(out_shape, transform, raster_crs,
                                   **growth_kwargs)
        path = path or self._ring_cache_path(key)

        ring_geom, _, exhausted = self._build_adaptive_rings(
            self.buildings, out_shape, transform, raster_crs, cell_size,
            **growth_kwargs,
        )

        rings = gpd.GeoDataFrame(
            {
                "building_id": self.buildings["building_id"].values,
                "ring_buffer_pct": self._ring_buffer_pct,
                "exhausted": exhausted,
                # Stamped on every feature because GeoJSON has nowhere else to
                # put it, and a cache without provenance is worse than none.
                "ring_key": key,
            },
            geometry=ring_geom.values,
            crs=self.buildings.crs,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written in lon/lat, which is what GeoJSON is defined in. Handing the
        # driver a projected frame instead writes metre coordinates with no crs
        # member, and every reader -- including this module's own loader -- then
        # takes them for degrees. The reprojection round trip is exact to about
        # a nanometre, well under the rasterization's reach.
        rings.to_crs(epsg=4326).to_file(path, driver="GeoJSON")
        logger.info(f"wrote {len(rings)} rings to {path}")
        return rings

    def _load_ring_cache(self, exposed_objects, key, path=None):
        """The cached rings if they match this grid and population, else None.

        Every rejection is logged with its reason and falls back to rebuilding,
        so a stale file costs a slow run rather than wrong geometry.
        """
        path = path or self._ring_cache_path(key)
        if not os.path.exists(path):
            logger.info(f"no ring cache at {path}; building the rings in-process")
            return None

        rings = gpd.read_file(path)
        cached_key = rings["ring_key"].iloc[0] if len(rings) else ""
        if cached_key != key:
            logger.warning(
                f"ring cache at {path} was built for a different setup and is "
                f"ignored\n  cached: {cached_key}\n  wanted: {key}\n"
                f"  rebuild it with scripts/build_building_rings.py"
            )
            return None

        # Matched on building_id rather than row order: the cache is keyed to
        # buildings, and a reordered layer would otherwise hand every building
        # its neighbour's ring.
        rings = rings.set_index("building_id").reindex(
            exposed_objects["building_id"].values
        )
        if rings.geometry.isna().any():
            logger.warning(
                f"ring cache at {path} is missing "
                f"{int(rings.geometry.isna().sum())} of the modelled buildings "
                f"and is ignored; rebuild it with scripts/build_building_rings.py"
            )
            return None

        if rings.crs != exposed_objects.crs:
            rings = rings.to_crs(exposed_objects.crs)

        logger.info(f"loaded {len(rings)} rings from {path}")
        return (
            gpd.GeoSeries(rings.geometry.values, crs=exposed_objects.crs),
            rings["exhausted"].to_numpy(dtype=bool),
            rings["ring_buffer_pct"].to_numpy(dtype=float),
        )

    def _build_adaptive_rings(self, exposed_objects, out_shape, transform,
                              raster_crs, cell_size, min_cells=MIN_RING_CELLS,max_buffer_pct=MAX_BUFFER_PCT, step_pct=GROWTH_STEP_PCT):
        """Grow each building's ring until it holds min_cells, capped.

        Takes the grid's SHAPE rather than the depth field, because since the
        counting fix nothing here reads a depth value -- the loop rasterizes
        candidate rings and counts cells. That is what makes the result cacheable
        across scenarios; see _ring_cache_key.
        """
        all_buildings_geom = self._footprint_union()
        n = len(exposed_objects)
        buffer_pct = np.full(n, 1, dtype=float)
        needs_growth = np.ones(n, dtype=bool)

        final_geom = None
        final_counts = None
        final_buffer_pct = buffer_pct.copy()

        while needs_growth.any():
            dist = buffer_pct * cell_size
            candidate_geom = exposed_objects.geometry.buffer(dist).difference(all_buildings_geom)

            # Counted through the overlap-preserving path, which is what
            # _gpu_zonal_stats_sparse uses downstream, so the loop tests the
            # sample size the model will actually get. Rings SHOULD overlap in
            # packed blocks -- a cell of water between two buildings is real
            # water for both of them, and neither has a claim to it over the
            # other. The dense path cannot express that: one pixel carries one
            # label, so a shared cell goes to whichever ring rasterizes last and
            # counts for nobody else. Under it a building hemmed in by later
            # neighbours reads as empty however wide it grows, and the loop runs
            # to the cap chasing a number set by row order in the layer.
            _, _, feat_idx, n_feat = _get_feature_labels(
                gpd.GeoDataFrame(geometry=candidate_geom, crs=exposed_objects.crs),
                out_shape, transform, raster_crs, allow_overlap=True,
            )
            counts_np = np.bincount(feat_idx, minlength=n_feat)

            if final_geom is None:
                final_geom, final_counts = candidate_geom, counts_np
            else:
                update = needs_growth
                final_geom = final_geom.where(~update, candidate_geom)
                final_counts = np.where(update, counts_np, final_counts)

            final_buffer_pct = np.where(needs_growth, buffer_pct, final_buffer_pct)

            maxed_out = buffer_pct >= max_buffer_pct
            needs_growth = (final_counts < min_cells) & (~maxed_out)
            buffer_pct = np.where(needs_growth, buffer_pct + step_pct, buffer_pct)

        # The per-building buffer the loop settled on, in cells. Kept on the
        self._ring_buffer_pct = final_buffer_pct

        # these need the block-dissolve fallback, not a bigger ring.
        exhausted = (final_counts < min_cells) & (final_buffer_pct >= max_buffer_pct)
        return final_geom, final_counts, exhausted
    
                
