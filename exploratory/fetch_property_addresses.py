"""Download the LINZ Property-Address Reference for parcels inside the flood domain.

table-115638 is LINZ's authoritative property <-> address association: three
columns (id, unit_of_property_id, address_id) and no geometry. Being a table it
cannot be cut down with a WFS bbox, so the in-domain parcels are identified
locally and their ids are sent back as CQL filters in chunks.

This is the link the geometric address cascade has been approximating: it says
which addresses belong to which property outright, with no distance rule.

Writes /data/westport/data/exposure/property_addresses.csv.
"""
import json
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape
from shapely.ops import unary_union

from modules.lib import PROPERTY_BOUNDARIES_PATH
from modules.utils.spatial_operators.operators import find_sample_flood_map

LINZ_KEY = "a06a74b6c02f4d4a81f999bd09034897"
WFS = f"https://data.linz.govt.nz/services;key={LINZ_KEY}/wfs"
TYPE_NAME = "table-115638"      # NZ Properties: Property-Address Reference
OUT_CSV = "/data/westport/data/exposure/property_addresses.csv"
ADDRESS_POINTS_FILE_PATH = "/data/westport/data/exposure/address_points.gpkg"

WORK_CRS = 2193
CHUNK = 100

# ── PARCELS INSIDE THE COMPUTATIONAL DOMAIN ───────────────────────────────────

with rasterio.open(find_sample_flood_map()) as src:
    mask = (~np.isnan(src.read(1).astype("float32"))).astype("uint8")
    polys = [shape(g) for g, v in rio_features.shapes(
        mask, mask=mask.astype(bool), transform=src.transform) if v == 1]
    domain_crs = src.crs
domain = gpd.GeoSeries([unary_union(polys)], crs=domain_crs).to_crs(WORK_CRS).iloc[0].simplify(10)

parcels = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(WORK_CRS)
in_domain = parcels[parcels.intersects(domain)]
ids = sorted(in_domain["unit_of_property_id"].dropna().unique())
print(f"domain {domain.area / 1e6:.1f} km2 | parcels in domain {len(in_domain)} of {len(parcels)}")
print(f"requesting address references for {len(ids)} property ids")

# ── FETCH ─────────────────────────────────────────────────────────────────────


def fetch(chunk):
    quoted = ",".join("'" + i.replace("'", "''") + "'" for i in chunk)
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": TYPE_NAME, "outputFormat": "application/json",
        "CQL_FILTER": f"unit_of_property_id IN ({quoted})",
    }
    with urllib.request.urlopen(f"{WFS}?{urllib.parse.urlencode(params)}", timeout=300) as r:
        payload = json.load(r)
    if "features" not in payload:
        raise RuntimeError(f"unexpected WFS response: {str(payload)[:300]}")
    return [f["properties"] for f in payload["features"]]


records = []
for start in range(0, len(ids), CHUNK):
    records.extend(fetch(ids[start:start + CHUNK]))
    print(f"  {min(start + CHUNK, len(ids)):5}/{len(ids)} ids -> {len(records)} rows", flush=True)

links = pd.DataFrame.from_records(records)[["unit_of_property_id", "address_id"]]
links = links.drop_duplicates().sort_values(["unit_of_property_id", "address_id"])
links.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}: {len(links)} property-address links")

# ── WHAT IT COVERS ────────────────────────────────────────────────────────────

n_prop = links["unit_of_property_id"].nunique()
print(f"properties with >=1 address: {n_prop} of {len(ids)} ({n_prop / len(ids):.1%})")
per = links.groupby("unit_of_property_id").size()
print(f"addresses per property: median {per.median():.0f}, max {per.max()}")

addresses = gpd.read_file(ADDRESS_POINTS_FILE_PATH)
known = set(addresses["address_id"])
hit = links["address_id"].isin(known)
print(f"referenced addresses present in address_points.gpkg: "
      f"{links.loc[hit, 'address_id'].nunique()} of {links['address_id'].nunique()}")
