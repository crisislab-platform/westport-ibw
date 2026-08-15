"""Download the LINZ Property-Building Reference for parcels inside the flood domain.

table-119620 is LINZ's authoritative property <-> building association: three
columns (id, unit_of_property_id, building_id) and no geometry. Same shape as
the Property-Address Reference, and the same reason for the chunked CQL fetch --
a table has no geometry to bbox against, so the in-domain parcel ids are sent
back as filters instead.

This is the domain-scoped equivalent of the ~5M-row national property_buildings
extract that build_property_mapping.py reads, and the authority against which
the max-overlap geometric assignment can be checked.

Writes /data/westport/data/exposure/property_buildings_reference.csv and prints
how far it agrees with out/building_links.csv.
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
TYPE_NAME = "table-119620"      # NZ Properties: Property-Building Reference
OUT_CSV = "/data/westport/data/exposure/property_buildings_reference.csv"
LINKS_CSV = f"{PROJECT_ROOT}/out/building_links.csv"

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
print(f"requesting building references for {len(ids)} property ids")

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

links = pd.DataFrame.from_records(records)[["unit_of_property_id", "building_id"]]
links["building_id"] = links["building_id"].astype("int64")
links = links.drop_duplicates().sort_values(["unit_of_property_id", "building_id"])
links.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}: {len(links)} property-building links")

n_prop = links["unit_of_property_id"].nunique()
print(f"properties with >=1 building: {n_prop} of {len(ids)} ({n_prop / len(ids):.1%})")
per = links.groupby("unit_of_property_id").size()
print(f"buildings per property: median {per.median():.0f}, max {per.max()}")
multi = links.groupby("building_id").size()
print(f"buildings referenced against >1 property: {int((multi > 1).sum())}")

# ── AGREEMENT WITH THE GEOMETRIC ASSIGNMENT ───────────────────────────────────

wide = pd.read_csv(LINKS_CSV)
ours = wide[["building_id", "property_id", "property_source", "overlap_frac"]]
merged = ours.merge(links.rename(columns={"unit_of_property_id": "linz_property_id"}),
                    on="building_id", how="left")

in_ref = merged["linz_property_id"].notna()
print(f"\nof {len(ours)} in-domain buildings, {merged.loc[in_ref, 'building_id'].nunique()} "
      f"appear in the reference")
agree = merged["property_id"] == merged["linz_property_id"]
by_building = merged.groupby("building_id").agg(
    any_agree=("property_id", lambda s: False),  # placeholder, replaced below
)
by_building["any_agree"] = merged.assign(a=agree).groupby("building_id")["a"].any()
by_building["in_ref"] = merged.assign(r=in_ref).groupby("building_id")["r"].any()

both = by_building[by_building["in_ref"]]
print(f"  geometry matches LINZ for {int(both['any_agree'].sum())} of {len(both)} "
      f"({both['any_agree'].mean():.1%})")
disagree = both[~both["any_agree"]]
print(f"  disagreements: {len(disagree)}")
if len(disagree):
    d = ours[ours["building_id"].isin(disagree.index)]
    print(f"    their overlap_frac: median {d['overlap_frac'].median():.2f}, "
          f"min {d['overlap_frac'].min():.2f}")
    print(f"    by property_source:\n"
          + d["property_source"].value_counts().to_string())
