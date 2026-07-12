import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
import glob
from functools import lru_cache
from modules.lib import FLOOD_MAPS_DIR, SURVEYED_FLOOD_DEPTH_JUL_2021
import geopandas as gpd
import pandas as pd
from arcgis.features import Feature
import torch

def load_features(file_path):
    with rasterio.open(find_sample_flood_map()) as src:
        depth = src.read(1).astype(float)
        transform = src.transform
    domain_mask = ~np.isnan(depth)
    features = gpd.read_file(file_path)
    points = features.representative_point()
    inside = []

    for pt in points:
        row, col = rasterio.transform.rowcol(transform, pt.x, pt.y)

        if (
            0 <= row < domain_mask.shape[0]
            and 0 <= col < domain_mask.shape[1]
        ):
            inside.append(domain_mask[row, col])
        else:
            inside.append(False)
            
    features_in_domain = features[np.array(inside)]
    return features_in_domain

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
def get_sample_georeference(scenario_id=None):
    """Read (transform, crs, nodata) from the sample flood map once and cache it.

    The sample map's georeferencing is constant for a run, so this avoids
    re-opening the raster on every zonal_flood_depth call.
    """
    if scenario_id == 0:
        return get_raster_transform_and_crs(SURVEYED_FLOOD_DEPTH_JUL_2021)
    return get_raster_transform_and_crs(find_sample_flood_map())


# Cache of rasterized feature labels keyed by (id(gdf), out_shape). The feature
# geometries and raster grid are constant for a run, so we only rasterize once.
_LABEL_CACHE = {}


def _get_feature_labels(exposed_objects, out_shape, transform, raster_crs):
    """Return (labels, n_features) rasterizing each feature to a unique id once.

    ``labels`` is an int32 array of ``out_shape`` where pixel value ``i + 1``
    marks feature row ``i`` (0 = background). Results are cached per feature set
    and grid so repeated calls reuse the same rasterization.
    """
    key = (id(exposed_objects), out_shape)
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        return cached

    gdf = exposed_objects.to_crs(raster_crs)
    shapes = ((geom, idx + 1) for idx, geom in enumerate(gdf.geometry))
    labels = features.rasterize(
        shapes, out_shape=out_shape, transform=transform,
        fill=0, all_touched=False, dtype="int32",
    )
    result = (labels, len(gdf))
    _LABEL_CACHE[key] = result
    return result

def buffered_flood_depth(
    exposed_objects,
    flood_map: np.ndarray,
    buffer_pct: float = 2.0,       # 200% of grid resolution (Table 3, Bertsch et al. 2022)
    mean_threshold: float = 0.05,  # Table 2's adapted classification scheme
    max_threshold: float = 0.1,
    scenario_id:int = 0,
):
    """
    Zonal flood-depth statistics computed over the buffer RING around each
    exposed object's footprint -- the footprint itself is excluded.

    This follows the paper's stated method exactly: depth is extracted
    "from cells adjacent to the building outline," not from the footprint.
    The ring is computed in vector space as buffer(d).difference(footprint),
    which stays correct for concave/non-convex footprints.
    """
    # ---- same tensor -> 2D numpy coercion as zonal_flood_depth --------
    if hasattr(flood_map, "detach"):
        flood_map = flood_map.detach().cpu().numpy()
    flood_map = np.squeeze(np.asarray(flood_map, dtype="float32"))
    if flood_map.ndim != 2:
        raise ValueError(
            f"buffered_flood_depth expects a 2D flood map, got shape {flood_map.shape}"
        )

    transform, raster_crs, _ = get_sample_georeference(scenario_id)

    # Pixel resolution drives the buffer distance. Assumes square pixels;
    # if your raster is non-square, buffer x/y separately instead.
    cell_size_x, cell_size_y = abs(transform.a), abs(transform.e)
    if not np.isclose(cell_size_x, cell_size_y, rtol=1e-3):
        raise ValueError(
            f"Non-square pixels ({cell_size_x} x {cell_size_y}); "
            "buffer_pct assumes an isotropic buffer distance."
        )
    buffer_dist = cell_size_x * buffer_pct

    # Ring-only geometry: buffer minus the footprint itself. Requires
    # exposed_objects to be in a projected (metric) CRS matching raster_crs --
    # buffering in a lat/lon CRS would apply buffer_dist in degrees, not meters.
    ring_objects = exposed_objects.copy()
    ring_objects["geometry"] = exposed_objects.geometry.buffer(buffer_dist).difference(
        exposed_objects.geometry
    )

    # A degenerate (zero-area or empty) footprint could make the ring empty
    # too; that building will simply rasterize to zero pixels and fall out
    # safely via the counts>0 guard below, rather than raising.
    labels, n = _get_feature_labels(
        ring_objects, flood_map.shape, transform, raster_crs
    )

    # ---- same vectorized reduction pattern as zonal_flood_depth -------
    valid_labels = np.where(np.isfinite(flood_map), labels, 0)
    depth = np.nan_to_num(flood_map, nan=0.0)
    depth[depth < 0] = 0.0
    index = np.arange(1, n + 1)

    depth_max = np.nan_to_num(ndimage.maximum(depth, valid_labels, index), nan=0.0)
    depth_sum = np.nan_to_num(ndimage.sum(depth, valid_labels, index), nan=0.0)

    counts = np.bincount(valid_labels.ravel(), minlength=n + 1)[1:]
    depth_mean = np.divide(
        depth_sum, counts, out=np.zeros_like(depth_sum), where=counts > 0
    )

    result = exposed_objects.copy()
    result["depth_max"] = depth_max
    result["depth_mean"] = depth_mean
    result["buffer_dist"] = buffer_dist
    result["ring_pixel_count"] = counts  # useful to sanity-check thin/degenerate rings

    # Table 2 classification: both mean and max thresholds must be met
    # within the ring for "high" / internally-flooded exposure.
    result["inundated"] =   result["inundated"] = result["inundated"] = (
        (result["depth_max"] >= max_threshold)
    )

    return result

def zonal_flood_depth(exposed_objects, flood_map: np.ndarray):
    mean_threshold = 0.10
    max_threshold = 0.3

    # The model emits torch tensors (often with leading batch/channel dims),
    # but the reduction needs a single 2D numpy array. Coerce here and fail
    # loudly if the map cannot be reduced to 2D.
    if hasattr(flood_map, "detach"):
        flood_map = flood_map.detach().cpu().numpy()
    flood_map = np.squeeze(np.asarray(flood_map, dtype="float32"))
    if flood_map.ndim != 2:
        raise ValueError(
            f"zonal_flood_depth expects a 2D flood map, got shape {flood_map.shape}"
        )

    # Georeferencing of the sample map (cached) and the rasterized feature
    # labels (cached) — neither is re-read/re-rasterized across calls.
    transform, raster_crs, _ = get_sample_georeference()
    labels, n = _get_feature_labels(exposed_objects, flood_map.shape, transform, raster_crs)

    # Exclude no-data/NaN pixels by demoting them to background (label 0) so they
    # are ignored by the grouped reductions.
    valid_labels = np.where(np.isfinite(flood_map), labels, 0)
    depth = np.nan_to_num(flood_map, nan=0.0)
    index = np.arange(1, n + 1)

    # Vectorized per-feature statistics over the whole grid in one pass each.
    depth_max = np.nan_to_num(ndimage.maximum(depth, valid_labels, index), nan=0.0)
    depth_min = np.nan_to_num(ndimage.minimum(depth, valid_labels, index), nan=0.0)
    depth_sum = np.nan_to_num(ndimage.sum(depth, valid_labels, index), nan=0.0)

    # Mean = sum / valid-pixel-count, computed with a safe divide so features
    # with no covered pixels yield 0.0 (avoids ndimage.mean's 0/0 warning).
    counts = np.bincount(valid_labels.ravel(), minlength=n + 1)[1:]
    depth_mean = np.divide(depth_sum, counts, out=np.zeros_like(depth_sum),
                           where=counts > 0)

    result = exposed_objects.copy()
    result["depth_max"] = depth_max
    result["depth_min"] = depth_min
    result["depth_mean"] = depth_mean
    result["depth_sum"] = depth_sum
    result["inundated"] = result["inundated"] = (
        (result["depth_max"] >= max_threshold) &
        (result["depth_mean"] >= mean_threshold)
    )

    return result


def gdf_to_features(gdf):
    features = []
    for _, row in gdf.iterrows():
        attributes = row.drop(labels="geometry").to_dict()
        attributes = _sanitize_attributes(attributes)

        geom = row.geometry.__geo_interface__
        esri_geom = _geojson_geom_to_esri(geom)

        features.append(Feature(geometry=esri_geom, attributes=attributes))
    return features

def _sanitize_attributes(attributes: dict) -> dict:
    clean = {}
    for k, v in attributes.items():
        # catch NaN, NaT, None, pandas.NA all at once
        if pd.isna(v):
            clean[k] = None
        elif isinstance(v, pd.Timestamp):
            # Esri date fields expect epoch milliseconds (UTC)
            clean[k] = int(v.timestamp() * 1000)
        elif hasattr(v, "item"):
            # numpy int64/float64 -> native Python int/float
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean

def _geojson_geom_to_esri(geom):
    geom_type = geom["type"]
    coords = geom["coordinates"]
    sr = {"wkid": 4326}  # adjust if your gdf isn't in WGS84

    if geom_type == "Point":
        x, y = coords
        return {"x": x, "y": y, "spatialReference": sr}
    elif geom_type == "MultiPoint":
        return {"points": coords, "spatialReference": sr}
    elif geom_type == "LineString":
        return {"paths": [coords], "spatialReference": sr}
    elif geom_type == "MultiLineString":
        return {"paths": coords, "spatialReference": sr}
    elif geom_type == "Polygon":
        return {"rings": coords, "spatialReference": sr}
    elif geom_type == "MultiPolygon":
        rings = []
        for polygon in coords:
            rings.extend(polygon)
        return {"rings": rings, "spatialReference": sr}
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")
    
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