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

from modules.update_module.arcgis_module import (  # noqa: E402
    BUILDING_FIELDS, FLOOD_EXTENT_FIELDS, INUNDATION_GRID_FIELDS,
    BUILDING_PEAK_FIELDS,
)
# RiskModule in modules.risk_module as of the 2026-08 refactor; this script
# still imported ImpactModule from modules.impact_module and could not run.
from modules.risk_module.exposure_module import (  # noqa: E402
    ExposureModule, DEPTH_BREAKS_M, FLOOD_EXTENT_BANDS, INUNDATION_BLOCK,
    WET_DEPTH_MIN,
)
from modules.lib import OUTPUT_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("make_layer_schemas")

# Placeholder values by field. Integer tiers stay integers and probabilities
# stay doubles so AGOL types them apart; strings get a non-empty value because
# an empty one can come back as null and be typed as something else.
# Every hazard class and likelihood band is an ordinal level; matched by suffix
# rather than listed, since the same names appear once per configuration prefix
# on the building layer (see arcgis_module.BUILDING_FIELDS).
_INT_SUFFIXES = ("hazard_level", "likelihood_level", "risk_level")
_INT_FIELDS = {"depth_class"}
_STR_SUFFIXES = ("hazard_class", "likelihood_class")
_STR_FIELDS = {"band"}
_TIME_FIELDS = {"issue_time", "valid_time", "peak_valid_time"}

# AGOL sizes a string field from the sample it is given and truncates later
# pushes that outgrow it, so the placeholder has to be at least as long as the
# longest real value: "Very Likely", 11 characters, since the likelihood bands
# were renamed to the paper's terms. "unset" (5) would have sized the field
# short enough to clip every Very Likely and Unlikely band pushed afterwards.
_STRING_PLACEHOLDER = "placeholder"
_EPOCH = pd.Timestamp("1970-01-01T00:00:00")


def _placeholder(field):
    if field in _TIME_FIELDS:
        return _EPOCH
    if field in _STR_FIELDS or field.endswith(_STR_SUFFIXES):
        return _STRING_PLACEHOLDER
    if field in _INT_FIELDS or field.endswith(_INT_SUFFIXES):
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

    # Loading the exposure module is what selects the modelled buildings, and
    # the layers must hold exactly those.
    # min_exceed_prob does not affect an all-placeholder schema, but the
    # constructor requires it; 0.1 is what the sweeps run with.
    impact_module = ExposureModule(args.scenario_id, 0.1)

    for name, objects, fields, id_field in (
        ("buildings", impact_module.buildings, BUILDING_FIELDS, "building_id"),
        # The peak layer: the same buildings and geometry, one row each,
        # carrying the window collapse instead of an hour.
        ("buildings_peak", impact_module.buildings, BUILDING_PEAK_FIELDS,
         "building_id"),
    ):
        frame = build_schema_frame(objects, fields, id_field)
        # Intersected, not iterated whole: an hourly frame has no
        # peak_valid_time and a peak frame has no valid_time.
        for column in _TIME_FIELDS.intersection(frame.columns):
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%dT%H:%M:%S")

        path = os.path.join(args.output_dir, f"{name}_schema.geojson")
        frame.to_file(path, driver="GeoJSON")
        logger.info("Wrote %s: %d features, %d fields", path, len(frame), len(fields))

    # The flood-extent and inundation-grid schemas are PAUSED as of 2026-08-27:
    # neither layer is needed, and neither can be built anyway. Both builders
    # call methods that stayed in fuzzy_risk_back.py through the refactor and
    # are not on RiskModule -- _depth_band_polygons for the extent, _block_grid
    # for the grid. Restoring those two methods, and the LAYERS entries in
    # arcgis_module, is what un-pauses this:
    #
    #     raster_path = args.flood_map or find_latest_flood_map()
    #     if raster_path:
    #         frame = build_flood_extent_schema(impact_module, raster_path)
    #         ... write flood_extent_schema.geojson ...
    #         frame = build_inundation_grid_schema(impact_module, raster_path)
    #         ... write inundation_grid_schema.geojson ...
    #
    # build_flood_extent_schema, build_inundation_grid_schema, the --flood-map
    # argument and find_latest_flood_map are left in place for that.

    logger.info(
        "Upload each file to ArcGIS Online to create the layer, then TRUNCATE it "
        "before the first real push."
    )


if __name__ == "__main__":
    main()
