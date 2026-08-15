"""Write empty GeoJSONs to create the three hosted feature layers from.

Uploading one of these to ArcGIS Online creates a hosted feature layer whose
schema matches what the forecast will later push, without waiting for a run:
the field lists come from arcgis_module, so the layer you create and the rows
you publish into it cannot disagree.

The rows carry real geometry and real ids but zeroed measures. That is
deliberate -- AGOL infers field types from the data, so an all-null column is
guessed at, and a column guessed wrong (a date read as a string, say) can only
be fixed by rebuilding the layer. Truncate each layer after creating it.

    python scripts/make_layer_schemas.py [--output-dir out/layer_schemas]
"""

import argparse
import glob
import logging
import os
import sys

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.arcgis_module.arcgis_module import (  # noqa: E402
    BUILDING_FIELDS, ROAD_FIELDS, PROPERTY_FIELDS, FLOOD_EXTENT_FIELDS,
    INUNDATION_GRID_FIELDS,
)
from modules.impact_module.impact_module import (  # noqa: E402
    ImpactModule, DEPTH_BREAKS_M, FLOOD_EXTENT_BANDS, INUNDATION_BLOCK,
    WET_DEPTH_MIN,
)
from modules.lib import OUTPUT_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("make_layer_schemas")

# Placeholder values by field. Integer tiers stay integers and probabilities
# stay doubles so AGOL types them apart; strings get a non-empty value because
# an empty one can come back as null and be typed as something else.
_INT_FIELDS = {"overall_building_hazard_level", "overall_hazard_level",
               "final_risk_level", "building_hazard_level",
               "property_hazard_level", "street_hazard_level", "depth_class",
               "ensemble_mean_hazard_level", "final_risk_ensemble_mean"}
_STR_FIELDS = {"overall_building_hazard_rule",
               "overall_building_hazard_driver",
               "band"}
_BOOL_FIELDS = {"property_known", "street_known"}
_TIME_FIELDS = {"issue_time", "valid_time"}

# Long enough for the values actually written (colours, rule labels like "R5",
# drivers like "structure") with room to spare -- AGOL sizes a string field
# from the sample and will truncate later pushes that outgrow it.
_STRING_PLACEHOLDER = "unset"
_EPOCH = pd.Timestamp("1970-01-01T00:00:00")


def _placeholder(field):
    if field in _TIME_FIELDS:
        return _EPOCH
    if field in _STR_FIELDS:
        return _STRING_PLACEHOLDER
    if field in _BOOL_FIELDS:
        return False
    if field in _INT_FIELDS:
        return 0
    return 0.0


def build_schema_frame(objects, fields, id_field):
    """One row per object: real geometry and id, every measure zeroed."""
    frame = objects[[objects.geometry.name, id_field]].copy()
    for field in fields:
        if field == id_field:
            continue
        frame[field] = _placeholder(field)
    # Column order follows the published field list, with geometry last, so the
    # created layer's field order matches the pushes.
    return frame[list(fields) + [frame.geometry.name]].to_crs(epsg=4326)


def build_flood_extent_schema(impact_module, raster_path):
    """Schema frame for the extent layer, from a real run's raster.

    The extent has no fixed object set to enumerate -- its geometry is derived
    per timestep -- so the placeholder is built by classifying an existing
    forecast raster and zeroing the measures. That keeps genuine multipolygon
    geometry and a correct layer extent, which is what AGOL infers the geometry
    type and spatial index from.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        band = src.read(list(src.descriptions).index(FLOOD_EXTENT_BANDS[-1]) + 1)
        frame = impact_module._depth_band_polygons(band, src.transform, src.crs)

    # One row per class per band, so every combination the publisher can emit
    # is represented in the sample AGOL types the layer from.
    frames = []
    for band_name in FLOOD_EXTENT_BANDS:
        rows = frame.copy()
        rows["band"] = band_name
        frames.append(rows)
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=frame.crs)

    for field in FLOOD_EXTENT_FIELDS:
        if field in ("depth_class", "band"):
            continue  # real values: they are what the layer is keyed and styled on
        out[field] = _placeholder(field)
    # The class bounds are constants, not measurements -- keep them real so the
    # created layer already documents its own symbology.
    upper = list(DEPTH_BREAKS_M[1:]) + [None]
    out["depth_min_m"] = [DEPTH_BREAKS_M[c - 1] for c in out["depth_class"]]
    out["depth_max_m"] = [upper[c - 1] for c in out["depth_class"]]
    return out[list(FLOOD_EXTENT_FIELDS) + ["geometry"]].to_crs(epsg=4326)


def build_inundation_grid_schema(impact_module, raster_path, sample=500):
    """Schema frame for the inundation grid layer, from a real run's raster.

    Unlike the object layers this keeps its measures REAL rather than zeroed.
    AGOL types a field from the sample it is given, and depth_mean, depth_std,
    wet_fraction and area_ha are all fractional -- handed a column of zeros it
    can type them as Integer, which silently truncates every depth pushed
    afterwards and can only be undone by rebuilding the layer. The times keep
    the epoch placeholder, which typed correctly on the extent layer.

    Only ``sample`` blocks are written: enough for type inference and a correct
    layer extent, without uploading a full timestep's worth of geometry to
    create an empty layer.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        depth = src.read(list(src.descriptions).index("depth_mean") + 1)
        spread = src.read(list(src.descriptions).index("depth_std") + 1)
        frame = impact_module._block_grid(
            depth, spread, src.transform, src.crs, INUNDATION_BLOCK, WET_DEPTH_MIN,
        )

    # Half wet, half dry, each strided across its own rows. The grid covers the
    # whole domain and most of it is dry on any given hour -- roughly 86% at
    # Westport -- so an evenly strided sample would be mostly zeros and hand
    # AGOL a column it can read as Integer, which is the one mistake here that
    # costs a layer rebuild. Forcing wet blocks in guarantees fractional depths
    # in the sample; keeping dry ones in keeps the extent honest.
    if len(frame) > sample:
        wet = frame[frame["wet_fraction"] > 0]
        dry = frame[frame["wet_fraction"] == 0]
        half = max(sample // 2, 1)
        parts = [
            part.iloc[:: max(len(part) // half, 1)]
            for part in (wet, dry) if len(part)
        ]
        frame = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), geometry="geometry", crs=frame.crs
        )

    for field in _TIME_FIELDS:
        frame[field] = _placeholder(field)
    return frame[list(INUNDATION_GRID_FIELDS) + ["geometry"]].to_crs(epsg=4326)


def find_latest_flood_map():
    """Newest flood_map_*.tif under OUTPUT_DIR, or None."""
    candidates = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "*", "flood_map_*.tif")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="out/layer_schemas")
    parser.add_argument(
        "--scenario_id", type=int, default=0,
        help="Only used to construct ImpactModule; does not affect the output.",
    )
    parser.add_argument(
        "--flood-map", default=None,
        help="Raster to derive the flood-extent geometry from. Defaults to the "
             "newest flood_map_*.tif under OUTPUT_DIR.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Loading the impact module is what selects the modelled objects: the
    # building link table decides which buildings, parcels and road parcels are
    # in play, and the layers must hold exactly those.
    impact_module = ImpactModule(args.scenario_id)

    for name, objects, fields, id_field in (
        ("buildings", impact_module.buildings, BUILDING_FIELDS, "building_id"),
        ("roads", impact_module.roads, ROAD_FIELDS, "id"),
        ("properties", impact_module.properties, PROPERTY_FIELDS, "unit_of_property_id"),
    ):
        frame = build_schema_frame(objects, fields, id_field)
        for column in _TIME_FIELDS:
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%dT%H:%M:%S")

        path = os.path.join(args.output_dir, f"{name}_schema.geojson")
        frame.to_file(path, driver="GeoJSON")
        logger.info("Wrote %s: %d features, %d fields", path, len(frame), len(fields))

    raster_path = args.flood_map or find_latest_flood_map()
    if raster_path:
        frame = build_flood_extent_schema(impact_module, raster_path)
        for column in _TIME_FIELDS:
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        path = os.path.join(args.output_dir, "flood_extent_schema.geojson")
        frame.to_file(path, driver="GeoJSON")
        logger.info(
            "Wrote %s: %d features, %d fields (geometry from %s)",
            path, len(frame), len(FLOOD_EXTENT_FIELDS), raster_path,
        )

        frame = build_inundation_grid_schema(impact_module, raster_path)
        for column in _TIME_FIELDS:
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        path = os.path.join(args.output_dir, "inundation_grid_schema.geojson")
        frame.to_file(path, driver="GeoJSON")
        logger.info(
            "Wrote %s: %d features, %d fields (geometry from %s)",
            path, len(frame), len(INUNDATION_GRID_FIELDS), raster_path,
        )
    else:
        logger.warning(
            "no flood_map_*.tif under %s, so the flood extent schema was skipped; "
            "run a forecast first, or pass --flood-map", OUTPUT_DIR,
        )

    logger.info(
        "Upload each file to ArcGIS Online to create the layer, then TRUNCATE it "
        "before the first real push."
    )


if __name__ == "__main__":
    main()
