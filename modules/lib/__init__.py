import os

# Both roots are overridable so the tree can be relocated -- the Docker image
# mounts a staged data bundle somewhere other than /data/westport, and the repo
# is checked out under /app rather than the author's home. Unset, they resolve
# to exactly the paths the workstation runs use, so nothing changes there.
DATA_ROOT = os.environ.get("WESTPORT_DATA_ROOT", "/data/westport")
REPO_ROOT = os.environ.get(
    "WESTPORT_REPO_ROOT",
    # modules/lib/__init__.py -> modules/lib -> modules -> repo root
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

HAZARD_MODEL_PATH = f"{DATA_ROOT}/out/model_checkpoints/"
LATENT_FLOW_MATCHING = "LATENT_FLOW_MATCHING_V1"
# Deep-ensemble members (member_<index>_seed_<seed>.onnx), one velocity network
# per trained member. The numeric index is the canonical order.
LATENT_FLOW_MATCHING_ENSEMBLE = os.path.join(LATENT_FLOW_MATCHING, "ensemble")
AUTO_ENCODER = "AUTO_ENCODER"
AUTO_ENCODER_DECODER = "AUTO_ENCODER.decoder"
LATENT_DATA_DIR = f"{DATA_ROOT}/data/latent_floodmaps/"
LATENT_FLOOD_MAP_STATS_DIR = os.path.join(LATENT_DATA_DIR, "latent_flood_map_stats.pt")
LATENT_DEM_PATH = os.path.join(LATENT_DATA_DIR, "DEM", "dem_latent.pt")
DEM_FILE_PATH = f"{DATA_ROOT}/data/dem/Topo.tif"
FLOOD_MAPS_DIR = f"{DATA_ROOT}/data/flood_maps"
BOUNDARY_CONDITIONS_DIR = f"{DATA_ROOT}/data/boundary_conditions"
BUILDING_FOOTPRINTS_PATH = f"{DATA_ROOT}/data/exposure/buildings.gpkg"
PLACRD_FILE_PATH = f"{DATA_ROOT}/data/Impact-based_data/obs_placard.gpkg"
ROAD_NETWORK_FILE_PATH = f"{DATA_ROOT}/data/exposure/road_network.gpkg"
ROAD_CENTERLINES_FILE_PATH = f"{DATA_ROOT}/data/exposure/road_centerlines.gpkg"
BUILDING_ROAD_MAPPING_PATH = f"{DATA_ROOT}/data/exposure/building_nearest_road_parcel.csv"
PROPERTY_BOUNDARIES_PATH = f"{DATA_ROOT}/data/exposure/property_boundaries.gpkg"
# LINZ property <-> building link table (national, ~5M rows); read only by the
# one-off precompute that writes BUILDING_PROPERTY_MAPPING_PATH.
PROPERTY_BUILDINGS_PATH = f"{DATA_ROOT}/data/exposure/property_buildings.csv"
BUILDING_PROPERTY_MAPPING_PATH = f"{DATA_ROOT}/data/exposure/building_property.csv"
# Building-centred link table: one row per in-domain building with the property
# it belongs to and the road parcel it fronts, built by
# exploratory/build_building_links.py from the LINZ property references.
BUILDING_LINKS_PATH = f"{REPO_ROOT}/out/building_links.csv"
# The grid cells each building's footprint touches, saved by ExposureModule the
# first time it meets a grid so later runs load them instead of rasterizing.
# One file per grid and building set, named by a hash of both.
FOOTPRINT_INDEX_DIR = f"{DATA_ROOT}/out/exposure/footprint_index"
JULY_2021_BC_FILE_PATH = f"{DATA_ROOT}/data/boundary_conditions/July2021.csv"
MODEL_CHECKPOINTS_DIR = f"{DATA_ROOT}/out/model_checkpoints"
OUTPUT_DIR = f"{DATA_ROOT}/out/forecasts"
SURVEYED_FLOOD_DEPTH_JUL_2021 = os.path.join(FLOOD_MAPS_DIR, "July_2021_depth_snapshots", "July_2021_surveyed_flood_depth.tif")
EXPOSURE_STUDY_RESULTS = f"{REPO_ROOT}/out/"
# Every metrics file the validation module writes (spatial-summary calibration,
# placards, hourly and per-building event depth scores, runtime).
VALIDATION_RESULTS_DIR = f"{DATA_ROOT}/out/exposure/validation"
# Forecast models main.py can score. FLOW_MATCHING_MODEL is the surrogate this
# system runs; CNN1D_MODEL is the westport 1D-CNN benchmark, run with
# onnxruntime from what scripts/export_cnn1d_onnx.py writes to CNN1D_CHECKPOINT_DIR.
FLOW_MATCHING_MODEL = "flow_matching"
CNN1D_MODEL = "cnn1d"
FORECAST_MODELS = (FLOW_MATCHING_MODEL, CNN1D_MODEL)
CNN1D_CHECKPOINT_DIR = f"{DATA_ROOT}/out/model_checkpoints/CNN1D_V1"