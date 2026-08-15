"""Reference-first exposure links: one row per building, with its property,
address and street.

LINZ's own association tables decide membership wherever they cover it --
property_buildings_reference says which buildings belong to a property,
property_addresses says which addresses do. Geometry is demoted to the two jobs
those tables cannot do: choosing WHICH building on a property takes WHICH
address point, and filling the gaps where the references are silent.

Covers every building in the footprint layer rather than only those inside a
flood model domain: the links are geography, and the run decides its own
population from the depth raster it uses.

This is the script form of the notebook's mapping cell; keep the two in step.
Writes out/building_links.csv, out/building_addresses.csv and
out/buildings_needing_address.csv.
"""
import os
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd

from modules.utils.spatial_operators.operators import load_features

SIMULATION_DATA_DIR = "/data/westport/data"
BUILDING_FILE_PATH  = f"{SIMULATION_DATA_DIR}/exposure/buildings.gpkg"
PROPERTY_BUILDING_REFERENCE_PATH = f"{SIMULATION_DATA_DIR}/exposure/property_buildings_reference.csv"
PROPERTY_FILE_PATH       = f"{SIMULATION_DATA_DIR}/exposure/property_boundaries.gpkg"
PROPERTY_ADDRESS_FILE_PATH = f"{SIMULATION_DATA_DIR}/exposure/property_addresses.csv"
ADDRESS_ROADS_FILE_PATH  = f"{SIMULATION_DATA_DIR}/exposure/address_roads.gpkg"
MANUAL_ADDRESS_POINTS    = f"{SIMULATION_DATA_DIR}/exposure/manual_address_points.csv"
ADDRESS_POINTS_FILE_PATH = f"{SIMULATION_DATA_DIR}/exposure/address_points.gpkg"
EXCLUDED_BUILDINGS       = f"{SIMULATION_DATA_DIR}/exposure/excluded_buildings.csv"
ROAD_CADASTRAL           = f"{SIMULATION_DATA_DIR}/exposure/road_network.gpkg"
ROAD_PARCEL_SECTIONS     = f"{SIMULATION_DATA_DIR}/exposure/road_parcel_sections.csv"


OUT_WIDE = f"{PROJECT_ROOT}/out/building_links.csv"
OUT_LONG = f"{PROJECT_ROOT}/out/building_addresses.csv"
OUT_TODO = f"{PROJECT_ROOT}/out/buildings_needing_address.csv"

WORK_CRS = 2193
MAX_ADDR_DIST_M = 15.0   # geometric fallback only: further than this is someone else's

# ── LOAD ──────────────────────────────────────────────────────────────────────

addresses  = gpd.read_file(ADDRESS_POINTS_FILE_PATH).to_crs(WORK_CRS)
# The WHOLE footprint layer, not the flood model's domain. Which parcel, address
# and road a building has is geography and holds whatever raster is in play, so
# domain-filtering here made the table a second population filter: ImpactModule
# takes its domain from the depth raster the run uses, and a building the survey
# covers but the model grid misses arrived with no property and no road for no
# reason but the order the two were built in.
buildings  = gpd.read_file(BUILDING_FILE_PATH).to_crs(WORK_CRS)
properties = gpd.read_file(PROPERTY_FILE_PATH).to_crs(WORK_CRS)
roads      = gpd.read_file(ADDRESS_ROADS_FILE_PATH).to_crs(WORK_CRS)
ref_buildings = pd.read_csv(PROPERTY_BUILDING_REFERENCE_PATH)
ref_addresses = pd.read_csv(PROPERTY_ADDRESS_FILE_PATH)

if os.path.exists(EXCLUDED_BUILDINGS):
    excluded = pd.read_csv(EXCLUDED_BUILDINGS)
    if len(excluded):
        gone = buildings["building_id"].isin(excluded["building_id"])
        buildings = buildings[~gone]
        print(f"excluded {int(gone.sum())} building(s): "
              + ", ".join(f"{r.building_id} ({r.reason})" for r in excluded.itertuples()))

bgeom, ageom, pgeom = buildings.geometry.name, addresses.geometry.name, properties.geometry.name

titled = properties[properties["unit_of_property_id"].notna()][
    ["unit_of_property_id", pgeom]].rename(columns={"unit_of_property_id": "property_id"})
street_names = roads[["road_id", "full_road_name"]].drop_duplicates("road_id").rename(
    columns={"full_road_name": "street_name"})

ref_buildings = ref_buildings.rename(columns={"unit_of_property_id": "property_id"})
ref_addresses = ref_addresses.rename(columns={"unit_of_property_id": "property_id"})

links = buildings[["building_id", "use", bgeom]].copy()
print(f"buildings {len(links)} | parcels {len(titled)} | addresses {len(addresses)} "
      f"| ref: {len(ref_buildings)} building links, {len(ref_addresses)} address links")

# ── MANUAL OVERRIDES ──────────────────────────────────────────────────────────
# Hand-entered rows outrank every automatic rule. Two kinds, told apart by
# whether address_id already exists in LINZ: a new point (blank/negative id,
# placed on the named building's centroid) or an assignment of an existing
# address to a building.

manual_assignments = pd.DataFrame(columns=["building_id", "address_id"])
if os.path.exists(MANUAL_ADDRESS_POINTS):
    manual = pd.read_csv(MANUAL_ADDRESS_POINTS)
    if len(manual) and "address_id" in manual.columns:
        is_existing = manual["address_id"].isin(addresses["address_id"])
        manual_assignments = manual[is_existing][["building_id", "address_id"]].copy()
        manual = manual[~is_existing]
    if len(manual):
        for col in ("address_id", "road_id", "easting", "northing", "building_id"):
            if col not in manual.columns:
                manual[col] = np.nan
        blank = manual["address_id"].isna()
        if blank.any():
            start = int(manual["address_id"].min()) - 1 if manual["address_id"].notna().any() else -1
            manual.loc[blank, "address_id"] = range(start, start - int(blank.sum()), -1)
        manual["address_id"] = manual["address_id"].astype(int)

        centroid_by_id = buildings.set_index("building_id")[bgeom].centroid
        need_xy = manual["easting"].isna() | manual["northing"].isna()
        if need_xy.any():
            pts = manual.loc[need_xy, "building_id"].map(centroid_by_id)
            if pts.isna().any():
                raise ValueError("manual rows need easting/northing or an in-domain building_id")
            manual.loc[need_xy, "easting"] = [p.x for p in pts]
            manual.loc[need_xy, "northing"] = [p.y for p in pts]

        need_road = manual["road_id"].isna()
        for i in manual.index[need_road]:
            text = str(manual.at[i, "full_address"]).upper()
            hits = [(len(r.street_name), r.road_id) for r in street_names.itertuples()
                    if str(r.street_name).upper() in text]
            if hits:
                manual.at[i, "road_id"] = max(hits)[1]

        addresses = pd.concat([addresses, gpd.GeoDataFrame(
            manual.drop(columns=["easting", "northing"]),
            geometry=gpd.points_from_xy(manual["easting"], manual["northing"]),
            crs=2193).to_crs(addresses.crs)], ignore_index=True)
        print(f"added {len(manual)} manual address point(s), "
              f"{len(manual_assignments)} manual assignment(s) to apply")

addr = addresses[["address_id", "full_address", "road_id", ageom]].copy()

# ── 1. BUILDING -> PROPERTY ───────────────────────────────────────────────────
# The reference is authoritative. It is many-to-many -- 148 buildings are listed
# against more than one property -- so where it offers several, geometry breaks
# the tie on largest overlap, which is the only way to say which parcel the
# footprint actually sits on.

overlap_area = gpd.sjoin(links[["building_id", bgeom]], titled,
                         how="inner", predicate="intersects").drop(columns="index_right")
overlap_area = overlap_area.merge(
    titled.rename(columns={pgeom: "parcel_geometry"}), on="property_id", how="left")
overlap_area["overlap_area"] = gpd.GeoSeries(overlap_area[bgeom].values, crs=WORK_CRS).intersection(
    gpd.GeoSeries(overlap_area["parcel_geometry"].values, crs=WORK_CRS)).area
overlap_area["overlap_frac"] = overlap_area["overlap_area"] / overlap_area[bgeom].area.values
overlap_lookup = overlap_area.set_index(["building_id", "property_id"])["overlap_frac"]

referenced = ref_buildings[ref_buildings["building_id"].isin(links["building_id"])].copy()
referenced["overlap_frac"] = pd.MultiIndex.from_frame(
    referenced[["building_id", "property_id"]]).map(overlap_lookup)

# Where a building is listed against several properties they are usually unit or
# cross-lease titles stacked on the same ground, so overlap is identical for all
# of them and cannot choose. Prefer the one that carries an address: it is the
# title the property is actually known by, and it keeps the property and address
# columns talking about the same parcel. Overlap only breaks what is left.
addressed_properties = set(ref_addresses["property_id"])
referenced["has_address"] = referenced["property_id"].isin(addressed_properties)

_candidates = referenced.groupby("building_id")["property_id"].size()
_multi = set(_candidates[_candidates > 1].index)
_addressed_per_building = (referenced[referenced["building_id"].isin(_multi)]
                           .groupby("building_id")["has_address"].sum())
print(f"buildings listed against >1 property: {len(_multi)} "
      f"({int((_addressed_per_building == 1).sum())} settled by a single addressed "
      f"property, {int((_addressed_per_building == 0).sum())} with none addressed, "
      f"{int((_addressed_per_building > 1).sum())} with several)")

# Then the smallest parcel, which is the most specific title where several are
# stacked, and finally property_id so the choice is at least deterministic --
# for genuinely identical stacked titles nothing in the data can separate them.
referenced["parcel_area"] = referenced["property_id"].map(
    titled.set_index("property_id")[pgeom].area)
by_reference = (referenced.sort_values(
                    ["has_address", "overlap_frac", "parcel_area", "property_id"],
                    ascending=[False, False, True, True], na_position="last")
                .drop_duplicates("building_id", keep="first")
                [["building_id", "property_id", "overlap_frac"]])
by_reference["property_source"] = "linz_reference"

# Buildings the reference does not mention fall back to largest overlap.
missing = links[~links["building_id"].isin(by_reference["building_id"])]["building_id"]
by_geometry = (overlap_area[overlap_area["building_id"].isin(missing)]
               .sort_values("overlap_area", ascending=False)
               .drop_duplicates("building_id", keep="first")
               [["building_id", "property_id", "overlap_frac"]])
by_geometry["property_source"] = "max_overlap"

building_property = pd.concat([by_reference, by_geometry], ignore_index=True)
links = links.merge(building_property, on="building_id", how="left")
links["property_source"] = links["property_source"].fillna("none")
print("\nproperty_source:\n" + links["property_source"].value_counts().to_string())

# ── 2. BUILDING -> ADDRESS ────────────────────────────────────────────────────
# The reference gives each property its addresses; geometry only decides which
# building on that property holds which point. A point inside a footprint goes
# to that building, otherwise to the nearest building on the same property.

prop_of_building = links.set_index("building_id")["property_id"]
geom_of_building = links.set_index("building_id")[bgeom]

candidates = (ref_addresses.merge(links[["building_id", "property_id"]].dropna(),
                                  on="property_id", how="inner")
              .merge(addr, on="address_id", how="inner"))
if len(candidates):
    candidates["dist"] = gpd.GeoSeries(
        candidates["building_id"].map(geom_of_building).values, crs=WORK_CRS).distance(
        gpd.GeoSeries(candidates[ageom].values, crs=WORK_CRS)).values
    candidates["inside"] = candidates["dist"] <= 1e-6
    # Closest building on the property wins the point; a point standing inside a
    # footprint always wins over one that is merely nearby.
    holder = (candidates.sort_values(["inside", "dist"], ascending=[False, True])
              .drop_duplicates("address_id", keep="first"))
    holder["address_source"] = np.where(holder["inside"], "reference_in_footprint", "reference")
    reference_pairs = holder[["building_id", "address_id", "full_address", "road_id",
                              "address_source", "dist"]].rename(columns={"dist": "address_dist_m"})
else:
    reference_pairs = pd.DataFrame(columns=["building_id", "address_id", "full_address",
                                            "road_id", "address_source", "address_dist_m"])

# Points the reference never places -- their property is not in the table, or
# they belong to a property with no building here. Geometry takes over: inside a
# footprint first, then the nearest building within MAX_ADDR_DIST_M.
unplaced = addr[~addr["address_id"].isin(reference_pairs["address_id"])]
inside = gpd.sjoin(unplaced, links[["building_id", bgeom]],
                   how="inner", predicate="within").drop(columns="index_right")
inside = inside.assign(address_source="in_footprint", address_dist_m=0.0)

leftover = unplaced[~unplaced["address_id"].isin(inside["address_id"])]
near = gpd.sjoin_nearest(leftover, links[["building_id", bgeom]], how="inner",
                         max_distance=MAX_ADDR_DIST_M, distance_col="address_dist_m",
                         ).drop(columns="index_right").assign(address_source="nearby")

building_addresses = pd.concat(
    [reference_pairs,
     pd.DataFrame(inside.drop(columns=ageom)),
     pd.DataFrame(near.drop(columns=ageom))],
    ignore_index=True,
)[["building_id", "address_id", "full_address", "road_id", "address_source", "address_dist_m"]]
building_addresses["building_id"] = building_addresses["building_id"].astype("int64")

# A manual assignment whose point no other building has claimed is recorded as a
# real pair; where several buildings share one number the point stays with its
# holder and the assignment applies to the wide table only.
if len(manual_assignments):
    # Only for buildings the references left without an address: a hand entry is
    # there to fill a gap, not to argue with LINZ.
    already = set(building_addresses["building_id"])
    sole = manual_assignments["address_id"].map(
        manual_assignments.groupby("address_id").size()).eq(1)
    unclaimed = manual_assignments[
        sole & ~manual_assignments["address_id"].isin(building_addresses["address_id"])
        & ~manual_assignments["building_id"].isin(already)]
    if len(unclaimed):
        extra = unclaimed.merge(addr, on="address_id", how="left")
        extra["address_dist_m"] = gpd.GeoSeries(
            extra["building_id"].map(geom_of_building).values, crs=WORK_CRS).distance(
            gpd.GeoSeries(extra[ageom].values, crs=WORK_CRS)).values
        extra["address_source"] = "manual"
        building_addresses = pd.concat(
            [building_addresses, pd.DataFrame(extra.drop(columns=ageom))], ignore_index=True
        )[["building_id", "address_id", "full_address", "road_id",
           "address_source", "address_dist_m"]]

# The references outrank a hand entry; manual only fills what they leave empty.
SOURCE_RANK = {"reference_in_footprint": 0, "reference": 1, "in_footprint": 2,
               "manual": 3, "nearby": 4}
primary = (building_addresses.assign(_rank=building_addresses["address_source"].map(SOURCE_RANK))
           .sort_values(["_rank", "address_dist_m", "address_id"])
           .drop_duplicates("building_id", keep="first").drop(columns="_rank"))
n_addr = building_addresses.groupby("building_id").size().rename("n_addresses")

links = links.merge(primary, on="building_id", how="left").merge(n_addr, on="building_id", how="left")
links["n_addresses"] = links["n_addresses"].fillna(0).astype(int)
links["address_source"] = links["address_source"].fillna("none")

# ── 2b. MANUAL ASSIGNMENTS ────────────────────────────────────────────────────
# Applied only where nothing authoritative reached the building, so the hand
# list fills gaps rather than overriding LINZ. They still beat the inherited
# address below, being a statement about this building rather than its parcel.

if len(manual_assignments):
    point_lookup = addresses.set_index("address_id")
    applied = superseded = 0
    for row in manual_assignments.itertuples():
        target = links["building_id"] == row.building_id
        if not target.any():
            continue
        if links.loc[target, "address_id"].notna().all():
            superseded += 1
            continue
        point = point_lookup.loc[row.address_id]
        links.loc[target, "address_id"] = row.address_id
        links.loc[target, "full_address"] = point["full_address"]
        links.loc[target, "road_id"] = point["road_id"]
        links.loc[target, "address_source"] = "manual"
        links.loc[target, "address_dist_m"] = links.loc[target, bgeom].distance(point[ageom]).values
        applied += 1
    print(f"manual assignments: {applied} applied, {superseded} already addressed "
          f"by the references")

# ── 2c. SHARE THE PROPERTY'S ADDRESS ──────────────────────────────────────────
# LINZ issues about one address per property, not per structure, so the garage
# behind the house gets nothing of its own. For exposure it genuinely is at that
# address, so it inherits the address of the largest addressed building on its
# property. n_addresses keeps counting only a building's own points.

addressed = links[links["address_id"].notna() & links["property_id"].notna()]
donor = (addressed.assign(_area=addressed[bgeom].area)
         .sort_values(["_area", "address_id"], ascending=[False, True])
         .drop_duplicates("property_id", keep="first")
         .set_index("property_id")[["address_id", "full_address", "road_id"]])
share = (links["address_id"].isna() & links["property_id"].notna()
         & links["property_id"].isin(donor.index))
for col in ("address_id", "full_address", "road_id"):
    links.loc[share, col] = links.loc[share, "property_id"].map(donor[col]).values
links.loc[share, "address_source"] = "shared_property"
print(f"shared a property address with {int(share.sum())} buildings")

# ── 3. BUILDING -> STREET ─────────────────────────────────────────────────────
# The addressed street is a lookup on the address's own road_id; only buildings
# with no address at all fall back to the nearest road, flagged and stamped with
# the distance.

links = links.merge(street_names, on="road_id", how="left")
links["street_source"] = np.where(links["street_name"].notna(), "address", "none")
links["street_dist_m"] = np.nan

need = links["street_name"].isna()
if need.any():
    fallback = gpd.sjoin_nearest(
        gpd.GeoDataFrame(links.loc[need, ["building_id"]],
                         geometry=links.loc[need, bgeom].values, crs=WORK_CRS),
        roads[["road_id", "full_road_name", roads.geometry.name]].rename(
            columns={"full_road_name": "street_name"}),
        how="left", distance_col="street_dist_m").drop(columns="index_right")
    fallback = fallback.sort_values("street_dist_m").drop_duplicates("building_id").set_index("building_id")
    idx = links.index[need]
    for col in ("road_id", "street_name", "street_dist_m"):
        links.loc[idx, col] = links.loc[idx, "building_id"].map(fallback[col]).values
    links.loc[idx, "street_source"] = "nearest_road"

# ── 4. BUILDING -> ROAD PARCEL ────────────────────────────────────────────────
# The street gives a name; the exposure model needs the cadastral parcel that
# carries it, since that is the object the flood depth is measured over. Road
# parcels and streets share no key, so they are bridged by the road sections
# (road_parcel_sections.csv): a building takes the parcel of ITS street that
# lies closest to it, rather than whatever parcel happens to be nearest.

road_parcels = gpd.read_file(ROAD_CADASTRAL).to_crs(WORK_CRS)
parcel_streets = pd.read_csv(ROAD_PARCEL_SECTIONS)[
    ["road_parcel_id", "road_id"]].drop_duplicates()
road_parcel_geom = road_parcels.set_index("id").geometry

candidates = links[["building_id", "road_id", bgeom]].merge(
    parcel_streets, on="road_id", how="inner")
if len(candidates):
    candidates["road_parcel_dist_m"] = gpd.GeoSeries(
        candidates[bgeom].values, crs=WORK_CRS).distance(
        gpd.GeoSeries(candidates["road_parcel_id"].map(road_parcel_geom).values,
                      crs=WORK_CRS)).values
    on_street = (candidates.sort_values("road_parcel_dist_m")
                 .drop_duplicates("building_id", keep="first")
                 .set_index("building_id"))
    links["road_parcel_id"] = links["building_id"].map(on_street["road_parcel_id"])
    links["road_parcel_dist_m"] = links["building_id"].map(on_street["road_parcel_dist_m"])
    links["road_parcel_source"] = np.where(links["road_parcel_id"].notna(),
                                           "street_match", "none")
else:
    links["road_parcel_id"] = np.nan
    links["road_parcel_dist_m"] = np.nan
    links["road_parcel_source"] = "none"

# A building whose street carries no parcel falls back to the nearest road
# parcel of any street, flagged so the weaker link stays visible.
need_parcel = links["road_parcel_id"].isna()
if need_parcel.any():
    nearest = gpd.sjoin_nearest(
        gpd.GeoDataFrame(links.loc[need_parcel, ["building_id"]],
                         geometry=links.loc[need_parcel, bgeom].values, crs=WORK_CRS),
        road_parcels[["id", "geometry"]].rename(columns={"id": "road_parcel_id"}),
        how="left", distance_col="road_parcel_dist_m").drop(columns="index_right")
    nearest = (nearest.sort_values("road_parcel_dist_m")
               .drop_duplicates("building_id").set_index("building_id"))
    idx = links.index[need_parcel]
    for col in ("road_parcel_id", "road_parcel_dist_m"):
        links.loc[idx, col] = links.loc[idx, "building_id"].map(nearest[col]).values
    links.loc[idx, "road_parcel_source"] = "nearest_parcel"

print("\nroad_parcel_source:\n" + links["road_parcel_source"].value_counts().to_string())
print(f"distance to the assigned road parcel: "
      f"median {links['road_parcel_dist_m'].median():.1f} m, "
      f"p90 {links['road_parcel_dist_m'].quantile(0.9):.1f} m, "
      f"max {links['road_parcel_dist_m'].max():.0f} m")

# ── RESULT ────────────────────────────────────────────────────────────────────

building_links = links[["building_id", "use",
                        "property_id", "property_source", "overlap_frac",
                        "address_id", "full_address", "address_source",
                        "address_dist_m", "n_addresses",
                        "road_id", "street_name", "street_source", "street_dist_m",
                        "road_parcel_id", "road_parcel_source", "road_parcel_dist_m"]]
assert building_links["building_id"].is_unique, "building_id must be the primary key"
assert building_addresses.groupby("address_id")["building_id"].nunique().le(1).all(), \
    "an address point must map to exactly one building"

todo = links[links["address_id"].isna()]
buildings_needing_address = todo[["building_id", "use", "property_id", "street_name"]].copy()

n = len(building_links)
print(f"\none row per building: {n} rows, {len(building_addresses)} building-address pairs")
for col, label in (("property_id", "property"), ("address_id", "address"),
                   ("street_name", "street")):
    have = int(building_links[col].notna().sum())
    print(f"  {label:9} {have:5} / {n}  ({have / n:.1%})")
print("\naddress_source:\n" + building_links["address_source"].value_counts().to_string())
print("\nstreet_source:\n" + building_links["street_source"].value_counts().to_string())
print(f"\nfrom the LINZ references: "
      f"{int(building_links['property_source'].eq('linz_reference').sum())} properties, "
      f"{int(building_links['address_source'].isin(['reference', 'reference_in_footprint']).sum())} addresses")
print(f"still needing an address: {len(buildings_needing_address)}")

building_links.to_csv(OUT_WIDE, index=False)
building_addresses.to_csv(OUT_LONG, index=False)
buildings_needing_address.to_csv(OUT_TODO, index=False)
print(f"\nwrote {OUT_WIDE}, {OUT_LONG}, {OUT_TODO}")
