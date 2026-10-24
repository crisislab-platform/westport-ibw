"""One-off precompute: nearest qualifying road parcel for each in-domain building.

Run once (and whenever the underlying exposure data changes) to refresh
BUILDING_ROAD_MAPPING_PATH; ImpactModule loads the resulting CSV at init
instead of recomputing the spatial join on every run.
"""
import logging

from modules.lib import BUILDING_ROAD_MAPPING_PATH
from modules.utils.spatial_operators.operators import build_building_road_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_access_mapping")

if __name__ == "__main__":
    mapping = build_building_road_mapping()
    mapping.to_csv(BUILDING_ROAD_MAPPING_PATH, index=False)
    logger.info(f"Wrote {len(mapping)} building->road mappings to {BUILDING_ROAD_MAPPING_PATH}")
