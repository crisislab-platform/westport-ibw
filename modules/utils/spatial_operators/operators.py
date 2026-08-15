import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
import glob
from functools import lru_cache
from modules.lib import (
    FLOOD_MAPS_DIR,
    SURVEYED_FLOOD_DEPTH_JUL_2021,
    BUILDING_FOOTPRINTS_PATH,
    ROAD_NETWORK_FILE_PATH,
    ROAD_CENTERLINES_FILE_PATH,
    PROPERTY_BOUNDARIES_PATH,
    PROPERTY_BUILDINGS_PATH,
)
import geopandas as gpd
import pandas as pd
import torch
from rasterio.features import rasterize
import os

def load_features(file_path, domain_raster=None):
    """Features that the given depth raster actually says something about.

    The domain is a property of the raster being used, not a fixed grid: a
    forecast is bounded by the flood model's valid cells, while a validation run
    against the surveyed July 2021 raster is bounded by what was surveyed. The
    two disagree -- 542 buildings the survey covers are outside the model grid,
    87 the other way -- so filtering everything against one sample map either
    scores buildings the raster in hand says nothing about, or discards
    buildings it does. Callers that do not care keep the model grid.
    """
    with rasterio.open(domain_raster or find_sample_flood_map()) as src:
        depth = src.read(1).astype(float)
        transform = src.transform
    domain_mask = ~np.isnan(depth)
    features = gpd.read_file(file_path)
    features = features.to_crs(src.crs)
    inside = []

    for geom in features.geometry:
        if geom.geom_type == "Polygon":
            points = [geom.centroid] + list(geom.exterior.coords[::10])
        else:
            points = [geom.representative_point()]

        found = False
        for p in points:
            if not hasattr(p, "x"):
                from shapely.geometry import Point
                p = Point(p)

            row, col = rasterio.transform.rowcol(transform, p.x, p.y)

            if (
                0 <= row < domain_mask.shape[0]
                and
                0 <= col < domain_mask.shape[1]
                and
                domain_mask[row, col]
            ):
                found = True
                break

        inside.append(found)
            
    features_in_domain = features[np.array(inside)]
    return features_in_domain

def build_building_road_mapping(
    building_footprints_path=BUILDING_FOOTPRINTS_PATH,
    road_network_path=ROAD_NETWORK_FILE_PATH,
    road_centerlines_path=ROAD_CENTERLINES_FILE_PATH,
):
    """Nearest qualifying road parcel for each in-domain building.

    A "qualifying" parcel is one that intersects a mapped road centreline,
    excluding unformed paper roads. Buildings are restricted to the
    computational domain via load_features, matching ImpactModule.__init__.
    Road parcels/centrelines are read raw (not via load_features): candidate
    roads must not be domain-masked, since a real road just outside the mask
    can still be a building's true nearest match, and the "has a centreline"
    filter is a data-quality distinction unrelated to the flood domain mask.

    Returns:
    DataFrame: columns building_id, road_id, distance_to_road_m.
    """
    buildings_in_domain = load_features(building_footprints_path)

    road_parcels = gpd.read_file(road_network_path).to_crs(buildings_in_domain.crs)
    centerlines = gpd.read_file(road_centerlines_path).to_crs(buildings_in_domain.crs)

    parcels_with_road = road_parcels[
        road_parcels.intersects(centerlines.union_all())
    ].copy()

    nearest = gpd.sjoin_nearest(
        buildings_in_domain[["building_id", "geometry"]],
        parcels_with_road[["id", "geometry"]].rename(columns={"id": "road_id"}),
        how="left",
        distance_col="distance_to_road_m",
    )
    # sjoin_nearest emits one row per tied nearest parcel; keep a single
    # closest match per building.
    nearest = (
        nearest.sort_values("distance_to_road_m")
        .drop_duplicates(subset="building_id", keep="first")
        .sort_index()
    )
    return nearest[["building_id", "road_id", "distance_to_road_m"]].reset_index(drop=True)


def build_building_property_mapping(
    building_footprints_path=BUILDING_FOOTPRINTS_PATH,
    property_boundaries_path=PROPERTY_BOUNDARIES_PATH,
    property_buildings_path=PROPERTY_BUILDINGS_PATH,
):
    """The property parcel each in-domain building sits on.

    Buildings are linked to properties through the LINZ property<->building
    table, which is national and many-to-many: a building straddling a parcel
    boundary appears against every parcel it touches. Those are resolved
    geometrically, by keeping the parcel the footprint overlaps most, so each
    building ends up with exactly one property whose flood depth represents
    its grounds.

    Buildings are restricted to the computational domain via load_features,
    matching ImpactModule.__init__. Property polygons are read raw: the parcel
    a building sits on is a cadastral fact, not a function of the flood mask.

    Returns:
    DataFrame: columns building_id, property_id.
    """
    buildings_in_domain = load_features(building_footprints_path)

    properties = gpd.read_file(property_boundaries_path).to_crs(buildings_in_domain.crs)
    properties = properties[properties["unit_of_property_id"].notna()]

    # Only the link rows for buildings actually in the domain; the national
    # table is ~5M rows, so this filter has to happen before anything else.
    links = pd.read_csv(
        property_buildings_path, usecols=["unit_of_property_id", "building_id"]
    )
    links = links[links["building_id"].isin(set(buildings_in_domain["building_id"]))]
    links = links.rename(columns={"unit_of_property_id": "property_id"}).drop_duplicates()

    candidates = links.merge(
        buildings_in_domain[["building_id", "geometry"]], on="building_id", how="inner"
    ).merge(
        properties[["unit_of_property_id", "geometry"]].rename(
            columns={"unit_of_property_id": "property_id", "geometry": "property_geometry"}
        ),
        on="property_id",
        how="inner",
    )

    # Overlap area decides only where a building genuinely straddles parcels;
    # computing it for every candidate keeps the tie-break uniform.
    overlap = gpd.GeoSeries(candidates["geometry"], crs=buildings_in_domain.crs).intersection(
        gpd.GeoSeries(candidates["property_geometry"], crs=buildings_in_domain.crs)
    )
    candidates["overlap_area"] = overlap.area

    resolved = (
        candidates.sort_values("overlap_area", ascending=False)
        .drop_duplicates(subset="building_id", keep="first")
        .sort_values("building_id")
    )
    return resolved[["building_id", "property_id"]].reset_index(drop=True)

def find_sample_flood_map():

    test_event_name = "Base0_Buller20yrARI_v2"
    
    inundation_files = glob.glob(f"{FLOOD_MAPS_DIR}/{test_event_name}_depth_snapshots/depth_at_*.tif")
    inundation_files.sort()
    sample_file = inundation_files[0] if inundation_files else None
    if sample_file is None:
        raise ValueError(f"No inundation files found for event {test_event_name}")
    return sample_file

def get_raster_transform_and_crs(rasterfile):
    with rasterio.open(rasterfile) as src:
        transform = src.transform
        raster_crs = src.crs
        nodata = np.nan
    return transform, raster_crs, nodata


@lru_cache(maxsize=1)
def get_sample_georeference(raster_file=None):
    if raster_file is None:
        raster_file = find_sample_flood_map()
  
    return get_raster_transform_and_crs(raster_file)

from affine import Affine

def _get_feature_labels(gdf, out_shape, transform, raster_crs, allow_overlap=False):
    """
    Rasterize features against a raster grid.

    allow_overlap=False (default): dense (H, W) label array, one label per
    pixel — unchanged legacy behaviour, used by roads/centerlines.

    allow_overlap=True: sparse (row_idx, col_idx, feature_idx) triples — a
    pixel covered by multiple features appears once per feature, so
    overlapping ring geometry around tightly packed buildings is preserved
    instead of being collapsed to whichever feature rasterizes last.
    """
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    n = len(gdf)

    if not allow_overlap:
        shapes = (
            (geom, i + 1)
            for i, geom in enumerate(gdf.geometry)
        )
        labels = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.int32,
            all_touched=True,
        )
        return labels, n

    # --- sparse, overlap-preserving path ---
    row_idx_parts = []
    col_idx_parts = []
    feat_idx_parts = []

    for i, geom in enumerate(gdf.geometry.values):
        if geom is None or geom.is_empty:
            continue

        minx, miny, maxx, maxy = geom.bounds
        col_start, row_start = ~transform * (minx, maxy)
        col_stop, row_stop = ~transform * (maxx, miny)

        row_start = max(int(np.floor(row_start)) - 1, 0)
        col_start = max(int(np.floor(col_start)) - 1, 0)
        row_stop = min(int(np.ceil(row_stop)) + 1, out_shape[0])
        col_stop = min(int(np.ceil(col_stop)) + 1, out_shape[1])

        if row_stop <= row_start or col_stop <= col_start:
            continue

        window_shape = (row_stop - row_start, col_stop - col_start)
        window_transform = transform * Affine.translation(col_start, row_start)

        mask = rasterize(
            [(geom, 1)],
            out_shape=window_shape,
            transform=window_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

        local_rows, local_cols = np.nonzero(mask)
        if local_rows.size == 0:
            continue

        row_idx_parts.append(local_rows + row_start)
        col_idx_parts.append(local_cols + col_start)
        feat_idx_parts.append(np.full(local_rows.size, i, dtype=np.int64))

    if not row_idx_parts:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, n

    row_idx = np.concatenate(row_idx_parts)
    col_idx = np.concatenate(col_idx_parts)
    feat_idx = np.concatenate(feat_idx_parts)

    return row_idx, col_idx, feat_idx, n


def get_domain_mask(self):
    """Static computational-domain mask at full flood-map resolution.

    1.0 = inside the computational domain, 0.0 = outside (NaN / nodata).
    Read from the RAW sample raster because process_inundation_file() zeroes
    NaN. The mask is static across events/timesteps, so it is built once and
    cached. Returned on CPU to match ``pixel_tiles_to_image`` output.
    """
       
    with rasterio.open(self.sample_flood_map) as src:
        data = src.read(1).astype(np.float32)
        nodata_value = src.nodata
        m = ~np.isnan(data)
        if nodata_value is not None:
            m &= (data != nodata_value)
        domain_mask = torch.from_numpy(m.astype(np.float32))
    return domain_mask
    
def load_flood_map(file, event_name):
    file_path = os.path.join(FLOOD_MAPS_DIR, f"{event_name}_depth_snapshots", file)
    
    with rasterio.open(file_path) as src:
        data = src.read(1)
        nodata_value = src.nodata
        
        # Handle NoData values by replacing them with 0 (no flooding)
        # This is appropriate for flood depth data where NoData typically means
        # areas outside the study domain or dry areas
        
        # Replace negative, nan and nodata values with 0
        data = np.where(np.isnan(data), 0, data)
        data = np.where(data == nodata_value, 0, data)
        
        return data


    tensor_data = torch.from_numpy(data).float().to(self.device)
    return tensor_data

