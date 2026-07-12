import os

HAZARD_MODEL_PATH = "/data/westport/out/model_checkpoints/"
LATENT_FLOW_MATCHING = "LATENT_FLOW_MATCHING_V1"
AUTO_ENCODER = "AUTO_ENCODER"
AUTO_ENCODER_DECODER = "AUTO_ENCODER.decoder"
LATENT_DATA_DIR = "/data/westport/data/latent_flood_maps/"
LATENT_FLOOD_MAP_STATS_DIR = os.path.join(LATENT_DATA_DIR, "latent_flood_map_stats.pt")
LATENT_DEM_PATH = os.path.join(LATENT_DATA_DIR, "dem_latent.pt")
DEM_FILE_PATH = "/data/westport/data/dem/Topo.tif"
FLOOD_MAPS_DIR = "/data/westport/data/flood_maps"
BOUNDARY_CONDITIONS_DIR = "/data/westport/data/flood-mapping-with-ml-main-DATA/DATA/boundary_conditions"
BUILDING_FOOTPRINTS_PATH = "/data/westport/data/exposure/buildings.gpkg"
PLACRD_FILE_PATH = "/data/westport/data/Impact-based_data/obs_placard.gpkg"
ROAD_NETWORK_FILE_PATH = "/data/westport/data/exposure/road_network.gpkg"
ROAD_CENTERLINES_FILE_PATH = "/data/westport/data/exposure/road_centerlines.gpkg"
JULY_2021_BC_FILE_PATH = "/data/westport/data/flood-mapping-with-ml-main-DATA/DATA/boundary_conditions/July2021.csv"
MODEL_CHECKPOINTS_DIR = "/data/westport/out/model_checkpoints"
OUTPUT_DIR = "/data/westport/out/forecasts"
SURVEYED_FLOOD_DEPTH_JUL_2021 = os.path.join(FLOOD_MAPS_DIR, "July_2021_depth_snapshots", "July_2021_surveyed_flood_depth.tif")
EXPOSURE_STUDY_RESULTS = "/data/westport/out/exposure/results"