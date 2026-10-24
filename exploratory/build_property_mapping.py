"""One-off precompute: the property parcel each in-domain building sits on.

Run once (and whenever the underlying exposure data changes) to refresh
BUILDING_PROPERTY_MAPPING_PATH; ImpactModule loads the resulting CSV at init
instead of reading the ~5M-row national link table on every run.
"""
import logging

from modules.lib import BUILDING_PROPERTY_MAPPING_PATH
from modules.utils.spatial_operators.operators import build_building_property_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_property_mapping")

if __name__ == "__main__":
    mapping = build_building_property_mapping()
    mapping.to_csv(BUILDING_PROPERTY_MAPPING_PATH, index=False)
    logger.info(
        f"Wrote {len(mapping)} building->property mappings to {BUILDING_PROPERTY_MAPPING_PATH}"
    )
