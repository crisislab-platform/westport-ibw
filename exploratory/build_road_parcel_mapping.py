"""Map road cadastral parcels to the streets that run through them.

The three road layers are keyed differently and none of them join directly:

  road_network.gpkg   cadastral road parcels, their own ``id``      (744)
  address_roads.gpkg  one line per addressed street, ``road_id``    (130)
  Road Sections       carriageway segments, road_section_id+road_id (layer-123109)

Sections are the bridge: they carry the addressing ``road_id`` and follow the
actual carriageway, so intersecting them with the parcels says which street
runs through which parcel. Whole-street lines are too coarse for this -- one
MultiLineString can clip dozens of parcels it merely passes near.

Writes:
  road_sections.gpkg        the downloaded sections, domain bbox
  road_parcel_sections.csv  road_parcel_id, road_section_id, road_id, street, overlap_m
"""
import json
import os
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import geopandas as gpd
import pandas as pd

SIMULATION_DATA_DIR = "/data/westport/data"
ROAD_SECTIONS_GPKG = f"{SIMULATION_DATA_DIR}/exposure/road_sections.gpkg"
ROAD_PARCEL_SECTIONS = f"{SIMULATION_DATA_DIR}/exposure/road_parcel_sections.csv"
ROAD_CADASTRAL = f"{SIMULATION_DATA_DIR}/exposure/road_network.gpkg"
DEM_FILE_PATH = f"{SIMULATION_DATA_DIR}/dem/Topo.tif"

LINZ_KEY = "a06a74b6c02f4d4a81f999bd09034897"
WFS = f"https://data.linz.govt.nz/services;key={LINZ_KEY}/wfs"
ROAD_SECTIONS_LAYER = "layer-123109"      # NZ Addresses: Road Sections

WORK_CRS = 2193
MIN_OVERLAP_M = 5.0    # a section merely clipping a parcel corner is not "on" it
REFRESH = False

# ── DOWNLOAD THE SECTIONS ─────────────────────────────────────────────────────

if os.path.exists(ROAD_SECTIONS_GPKG) and not REFRESH:
    sections = gpd.read_file(ROAD_SECTIONS_GPKG).to_crs(WORK_CRS)
    print(f"[skip] {ROAD_SECTIONS_GPKG} already has {len(sections)} sections")
else:
    import rasterio
    with rasterio.open(DEM_FILE_PATH) as dem:
        b = dem.bounds
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": ROAD_SECTIONS_LAYER, "outputFormat": "application/json",
        "bbox": f"{b.left},{b.bottom},{b.right},{b.top},EPSG:2193",
    }
    url = f"{WFS}?{urllib.parse.urlencode(params)}"
    sections = gpd.read_file(url).to_crs(WORK_CRS)
    sections.to_file(ROAD_SECTIONS_GPKG, driver="GPKG")
    print(f"[ok]   fetched {len(sections)} road sections -> {ROAD_SECTIONS_GPKG}")

parcels = gpd.read_file(ROAD_CADASTRAL).to_crs(WORK_CRS)
print(f"road parcels: {len(parcels)} | sections: {len(sections)} "
      f"| distinct streets in sections: {sections['road_id'].nunique()}")

# ── PARCEL <-> SECTION <-> STREET ─────────────────────────────────────────────

pairs = gpd.sjoin(
    sections[["road_section_id", "road_id", "full_road_name", "geometry"]],
    parcels[["id", "geometry"]].rename(columns={"id": "road_parcel_id"}),
    how="inner", predicate="intersects",
).drop(columns="index_right")

# How much of the section actually runs inside the parcel, so a line grazing a
# boundary can be told from one the parcel genuinely carries.
parcel_geom = parcels.set_index("id").geometry
pairs["overlap_m"] = gpd.GeoSeries(pairs.geometry.values, crs=WORK_CRS).intersection(
    gpd.GeoSeries(pairs["road_parcel_id"].map(parcel_geom).values, crs=WORK_CRS)
).length.values

kept = pairs[pairs["overlap_m"] >= MIN_OVERLAP_M].copy()
print(f"\nsection-parcel intersections: {len(pairs)}, "
      f"{len(kept)} with >= {MIN_OVERLAP_M:.0f} m inside the parcel")

mapping = (kept[["road_parcel_id", "road_section_id", "road_id", "full_road_name",
                 "overlap_m"]]
           .sort_values(["road_parcel_id", "overlap_m"], ascending=[True, False])
           .reset_index(drop=True))
mapping["overlap_m"] = mapping["overlap_m"].round(1)
mapping.to_csv(ROAD_PARCEL_SECTIONS, index=False)
print(f"wrote {ROAD_PARCEL_SECTIONS}: {len(mapping)} rows")

print(f"\nroad parcels carrying a street: {mapping['road_parcel_id'].nunique()} "
      f"of {len(parcels)}")
per_parcel = mapping.groupby("road_parcel_id")["road_id"].nunique()
print(f"  streets per parcel: median {per_parcel.median():.0f}, max {per_parcel.max()}")
per_street = mapping.groupby("road_id")["road_parcel_id"].nunique()
print(f"  parcels per street: median {per_street.median():.0f}, max {per_street.max()}")
print(f"streets with no parcel: "
      f"{sections['road_id'].nunique() - mapping['road_id'].nunique()}")
