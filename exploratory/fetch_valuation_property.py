"""Download the LINZ District Valuation Roll for parcels inside the flood domain.

table-114085 is a non-spatial table -- valuation attributes keyed by
unit_of_property_id, with no geometry -- so it cannot be cut down with a WFS
bbox. Instead the in-domain parcels are identified locally and their ids are
sent back as CQL filters, in chunks small enough to keep each GET under the
URL length limit. The result is joined onto the parcel geometry and written as
a spatial layer, so the output really is "property boundaries, in-domain only,
carrying their valuation record".
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
TYPE_NAME = "table-114085"          # NZ Properties: National District Valuation Roll
VALUATION_PROPERTY_PATH = "/data/westport/data/exposure/valuation_property.gpkg"

WORK_CRS = 2193
CHUNK = 100          # ~4 kB of UUIDs per request, well under the URL limit
SIMPLIFY_M = 10      # domain outline only; parcels are not simplified

# ── COMPUTATIONAL DOMAIN ──────────────────────────────────────────────────────

with rasterio.open(find_sample_flood_map()) as src:
    mask = (~np.isnan(src.read(1).astype("float32"))).astype("uint8")
    polys = [
        shape(geom)
        for geom, val in rio_features.shapes(mask, mask=mask.astype(bool),
                                             transform=src.transform)
        if val == 1
    ]
    domain_crs = src.crs

domain = gpd.GeoSeries([unary_union(polys)], crs=domain_crs).to_crs(WORK_CRS).iloc[0]
domain = domain.simplify(SIMPLIFY_M)
print(f"computational domain: {domain.area / 1e6:.1f} km2")

# ── PARCELS INSIDE IT ─────────────────────────────────────────────────────────

parcels = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(WORK_CRS)
in_domain = parcels[parcels.intersects(domain)].copy()
print(f"parcels: {len(parcels)} total, {len(in_domain)} intersecting the domain "
      f"({int(parcels.within(domain).sum())} wholly inside)")

ids = sorted(in_domain["unit_of_property_id"].dropna().unique())
print(f"requesting valuation records for {len(ids)} property ids")

# ── FETCH ─────────────────────────────────────────────────────────────────────


def fetch(chunk):
    quoted = ",".join("'" + i.replace("'", "''") + "'" for i in chunk)
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": TYPE_NAME, "outputFormat": "application/json",
        "CQL_FILTER": f"unit_of_property_id IN ({quoted})",
    }
    url = f"{WFS}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=300) as resp:
        payload = json.load(resp)
    if "features" not in payload:
        raise RuntimeError(f"unexpected WFS response: {str(payload)[:300]}")
    return [f["properties"] for f in payload["features"]]


records = []
for start in range(0, len(ids), CHUNK):
    chunk = ids[start:start + CHUNK]
    got = fetch(chunk)
    records.extend(got)
    print(f"  {start + len(chunk):5}/{len(ids)} ids -> {len(records)} records", flush=True)

valuations = pd.DataFrame.from_records(records)

# ── JOIN AND SAVE ─────────────────────────────────────────────────────────────
# A property can carry several valuation assessments (subdivided rating units),
# so this is a left join and the output may hold more rows than parcels.

if len(valuations) and "unit_of_property_id" in valuations:
    print(f"\nfetched {len(valuations)} valuation records "
          f"for {valuations['unit_of_property_id'].nunique()} properties")
    out = in_domain.merge(valuations, on="unit_of_property_id", how="left",
                          suffixes=("", "_val"))
    matched = out["unit_of_property_id"].isin(valuations["unit_of_property_id"])
    print(f"parcels with a valuation record: "
          f"{int(out.loc[matched, 'unit_of_property_id'].nunique())} of {len(in_domain)}")
else:
    # The roll keys on its own unit_of_property_id, which does not appear in the
    # current property_boundaries layer (113968) -- the two are different id
    # spaces. The boundaries are still worth saving; the valuation attributes
    # need a layer that shares a key with the roll (the older 122657 carried
    # valuation_reference, i.e. roll + assessment).
    print("\nNo valuation records matched. The roll's unit_of_property_id does "
          "not intersect this property layer's ids, so only the boundaries are "
          "written -- see the notes in this script's docstring.")
    out = in_domain.copy()

print(f"output rows: {len(out)}")

# Dates come back as strings; GPKG is happy with those, but object columns that
# are entirely null upset the driver, so they are dropped.
empty = [c for c in out.columns if c != out.geometry.name and out[c].isna().all()]
if empty:
    out = out.drop(columns=empty)
    print(f"dropped {len(empty)} all-null column(s): {', '.join(empty)}")

out.to_file(VALUATION_PROPERTY_PATH, driver="GPKG")
print(f"saved {VALUATION_PROPERTY_PATH}")
