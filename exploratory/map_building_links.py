"""Interactive map of the building-centred link table.

Reads out/building_links.csv (built by build_building_links.py), joins the
footprints back on and draws each building coloured by how its address was
resolved, with the property parcels, address points and addressed roads as
context layers. Writes out/building_links_map.html.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from shapely.geometry import shape
from shapely.ops import unary_union

from modules.lib import BUILDING_FOOTPRINTS_PATH, PROPERTY_BOUNDARIES_PATH
from modules.utils.spatial_operators.operators import find_sample_flood_map

ADDRESS_POINTS_FILE_PATH = "/data/westport/data/exposure/address_points.gpkg"
PROPERTY_ADDRESS_FILE_PATH = "/data/westport/data/exposure/property_addresses.csv"
ADDRESS_ROADS_FILE_PATH = "/data/westport/data/exposure/address_roads.gpkg"
LINKS_CSV = f"{PROJECT_ROOT}/out/building_links.csv"
PAIRS_CSV = f"{PROJECT_ROOT}/out/building_addresses.csv"
OUT_HTML = f"{PROJECT_ROOT}/out/building_links_map.html"

MAX_ADDR_DIST_M = 15.0   # must match build_building_links.py

WORK_CRS = 2193
SIMPLIFY_M = 0.3

# How each building got (or failed to get) an address -- the question the map
# exists to answer, so it drives the colouring.
ADDRESS_COLOURS = {
    "reference_in_footprint": ("#2ecc71", "LINZ reference, point on this building"),
    "reference":       ("#27ae60", "LINZ reference, nearest building on the property"),
    "in_footprint":    ("#3498db", "Point inside the footprint (geometric)"),
    "nearby":          ("#5dade2", "Point within 15 m (geometric)"),
    "shared_property": ("#a29bfe", "Shares its property's address"),
    "manual":          ("#34495e", "Assigned by hand"),
    "none":            ("#e74c3c", "No address"),
}

# Why an address point reached no building. Ordered most- to least-explicable,
# and applied in that order, since a point can satisfy several of them.
ORPHAN_COLOURS = {
    "building outside flood domain": ("#e67e22", "Its building was dropped by the domain filter"),
    "vacant parcel":                 ("#1abc9c", "Parcel holds no building at all"),
    "building further than 15 m":    ("#e84393", "Parcel has a building, but out of range"),
    "no titled parcel":              ("#c0392b", "Not on a titled parcel"),
}

links = pd.read_csv(LINKS_CSV)
pairs = pd.read_csv(PAIRS_CSV)
buildings = gpd.read_file(BUILDING_FOOTPRINTS_PATH).to_crs(WORK_CRS)
properties = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(WORK_CRS)
addresses = gpd.read_file(ADDRESS_POINTS_FILE_PATH).to_crs(WORK_CRS)
roads = gpd.read_file(ADDRESS_ROADS_FILE_PATH).to_crs(WORK_CRS)
ref_addresses = pd.read_csv(PROPERTY_ADDRESS_FILE_PATH)

gdf = buildings[["building_id", "geometry"]].merge(links, on="building_id", how="inner")
assert gdf["building_id"].is_unique
print(f"{len(gdf)} buildings joined to links ({len(links)} rows in the CSV)")
print(gdf["address_source"].value_counts().to_string())

# ── COMPUTATIONAL DOMAIN ──────────────────────────────────────────────────────
# The flood model's valid cells, polygonised from a sample depth raster the same
# way load_features derives it (~isnan). Computed up front because it decides
# what gets drawn at all, not just where the boundary line goes.

with rasterio.open(find_sample_flood_map()) as src:
    domain_mask = (~np.isnan(src.read(1).astype("float32"))).astype("uint8")
    domain_polys = [
        shape(geom)
        for geom, val in rio_features.shapes(domain_mask, mask=domain_mask.astype(bool),
                                             transform=src.transform)
        if val == 1
    ]
    domain_crs = src.crs

domain = gpd.GeoDataFrame(
    {"name": ["Computational domain"]},
    geometry=[unary_union(domain_polys)],
    crs=domain_crs,
).to_crs(WORK_CRS)
# The mask is pixel-edged, so its outline is a staircase; 10 m smooths that to
# a readable boundary without moving it meaningfully at map scale.
domain["geometry"] = domain.geometry.simplify(10)
domain_geom = domain.geometry.iloc[0]
print(f"domain: {domain.geometry.area.iloc[0] / 1e6:.1f} km2")

# Only addresses the model actually covers are drawn. The full building layer is
# kept in `buildings` for the orphan classifier below -- it has to look outside
# the domain to say "your building was dropped by the domain filter" -- but
# nothing out there reaches the map.
# Manually added points carry no LINZ attributes, so the tooltip would read
# "nan" for the street.
addresses["full_road_name"] = addresses["full_road_name"].fillna("(manual point)")

n_addresses_all = len(addresses)
# Exception to the domain clip: a point outside the domain that has been linked
# to an in-domain building is kept, so the building's address is visible on the
# map rather than pointing at nothing. 67 Punt Road is the case in hand.
linked_outside = addresses["address_id"].isin(pairs["address_id"]) & ~addresses.within(domain_geom)
addresses = addresses[addresses.within(domain_geom) | linked_outside].copy()
print(f"addresses inside the domain: {len(addresses)} of {n_addresses_all} "
      f"(including {int(linked_outside.sum())} linked from outside)")

# ── ORPHANED ADDRESSES ────────────────────────────────────────────────────────
# Points that reached no building. Classified by cause, because "vacant section"
# is correct behaviour while "its building was domain-filtered away" is not.

orphans = addresses[~addresses["address_id"].isin(set(pairs["address_id"]))].copy()

nearest = gpd.sjoin_nearest(
    orphans[["address_id", "geometry"]],
    buildings[["building_id", "geometry"]],
    how="left", distance_col="nearest_building_m",
).sort_values("nearest_building_m").drop_duplicates("address_id").set_index("address_id")
orphans["nearest_building_m"] = orphans["address_id"].map(nearest["nearest_building_m"])
orphans["_near_id"] = orphans["address_id"].map(nearest["building_id"])

titled_parcels = properties[properties["unit_of_property_id"].notna()][
    ["unit_of_property_id", "geometry"]].rename(columns={"unit_of_property_id": "property_id"})
orphans["_parcel"] = orphans["address_id"].map(
    gpd.sjoin(orphans[["address_id", "geometry"]], titled_parcels,
              how="left", predicate="within").drop_duplicates("address_id")
    .set_index("address_id")["property_id"]
)
occupied = set(gpd.sjoin(buildings[["building_id", "geometry"]], titled_parcels,
                         how="inner", predicate="intersects")["property_id"])
in_domain_ids = set(links["building_id"])


def orphan_cause(row):
    if row["nearest_building_m"] <= MAX_ADDR_DIST_M and row["_near_id"] not in in_domain_ids:
        return "building outside flood domain"
    if pd.isna(row["_parcel"]):
        return "no titled parcel"
    if row["_parcel"] not in occupied:
        return "vacant parcel"
    return "building further than 15 m"


orphans["orphan_cause"] = orphans.apply(orphan_cause, axis=1)
orphans["nearest_building_m"] = orphans["nearest_building_m"].round(1)
print(f"\n{len(orphans)} orphaned addresses:")
print(orphans["orphan_cause"].value_counts().to_string())

# Tooltips render NaN as "nan", which reads like a value rather than a gap.
for col in ("full_address", "street_name", "property_id", "use"):
    gdf[col] = gdf[col].fillna("—")
for col in ("address_dist_m", "street_dist_m", "overlap_frac"):
    gdf[col] = gdf[col].round(2).fillna("—")

TOOLTIP = (
    ["building_id", "use", "full_address", "address_source", "address_dist_m",
     "n_addresses", "street_name", "street_source", "street_dist_m",
     "property_id", "overlap_frac"],
    ["Building id", "Use", "Address", "Address via", "Address dist (m)",
     "Addresses", "Street", "Street via", "Street dist (m)",
     "Property", "Footprint in parcel"],
)


def to_wgs84(g, cols):
    out = g[cols + ["geometry"]].copy()
    out["geometry"] = out.geometry.simplify(SIMPLIFY_M)
    return out.to_crs(epsg=4326)


centre = gdf.to_crs(epsg=4326).union_all().centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=14,
               tiles="Esri.WorldImagery", attr="Esri")

# ── COMPUTATIONAL DOMAIN ──────────────────────────────────────────────────────
# Drawn because the domain edge explains a whole class of link failures: the
# orphan addresses and lost property parcels are ones falling outside it.

folium.GeoJson(
    domain.to_crs(epsg=4326),
    name="Computational domain",
    style_function=lambda x: {"fillColor": "#ffffff", "color": "#e74c3c",
                              "weight": 2.5, "dashArray": "6,4", "fillOpacity": 0.0},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=[""]),
).add_to(m)

# ── PARCELS IN THE DOMAIN ─────────────────────────────────────────────────────

addressed_properties = set(ref_addresses["unit_of_property_id"])
parcels_in_domain = properties[properties["unit_of_property_id"].notna()
                               & properties.intersects(domain_geom)].copy()
parcels_in_domain = parcels_in_domain.rename(columns={"unit_of_property_id": "property_id"})
buildings_per_property = links.groupby("property_id").size()
parcels_in_domain["n_buildings"] = (
    parcels_in_domain["property_id"].map(buildings_per_property).fillna(0).astype(int))

# A parcel with nothing built on it cannot affect building exposure, so empty
# ones are dropped rather than drawn -- an unaddressed bare section is not a
# gap worth chasing.
occupied_parcels = parcels_in_domain[parcels_in_domain["n_buildings"] > 0]
print(f"\nin-domain parcels: {len(parcels_in_domain)}, of which "
      f"{len(occupied_parcels)} hold buildings "
      f"({len(parcels_in_domain) - len(occupied_parcels)} empty, excluded)")

# ── CONTEXT (off by default; heavy and not the point of the map) ──────────────

folium.GeoJson(
    to_wgs84(occupied_parcels, ["property_id", "n_buildings"]),
    name="Property parcels (holding buildings)",
    show=False,
    style_function=lambda x: {"fillColor": "#2ecc71", "color": "#27ae60",
                              "weight": 0.8, "fillOpacity": 0.08},
    tooltip=folium.GeoJsonTooltip(fields=["property_id", "n_buildings"],
                                  aliases=["Property", "Buildings on it"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(properties[properties["unit_of_property_id"].isna()], []),
    name="Untitled land",
    show=False,
    style_function=lambda x: {"fillColor": "#95a5a6", "color": "#7f8c8d",
                              "weight": 0.8, "fillOpacity": 0.3},
).add_to(m)

folium.GeoJson(
    to_wgs84(roads, ["full_road_name"]),
    name="Addressed roads",
    style_function=lambda x: {"color": "#ecf0f1", "weight": 1.5, "opacity": 0.7},
    tooltip=folium.GeoJsonTooltip(fields=["full_road_name"], aliases=["Street"]),
).add_to(m)

# ── BUILDINGS, one layer per address_source ───────────────────────────────────

for source, (colour, label) in ADDRESS_COLOURS.items():
    subset = gdf[gdf["address_source"] == source]
    if subset.empty:
        continue
    folium.GeoJson(
        to_wgs84(subset, TOOLTIP[0]),
        name=f"{label} ({len(subset)})",
        style_function=lambda x, c=colour: {"fillColor": c, "color": "black",
                                            "weight": 0.5, "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=TOOLTIP[0], aliases=TOOLTIP[1]),
    ).add_to(m)

# ── PROBLEM LAYERS, for auditing rather than browsing ─────────────────────────

# Buildings on no parcel at all are drawn on top and shown by default: they are
# the hard floor on property coverage, and the reason a matching number of
# buildings can never inherit an address.
for source, colour, label, show in (
    ("none", "#ff6b00", "No property at all", True),
    ("max_overlap", "#9b59b6", "Property by geometry, not in the LINZ reference", False),
):
    subset = gdf[gdf["property_source"] == source]
    if subset.empty:
        continue
    print(f"  {label}: {len(subset)} buildings")
    folium.GeoJson(
        to_wgs84(subset, TOOLTIP[0]),
        name=f"{label} ({len(subset)})",
        show=show,
        style_function=lambda x, c=colour: {"fillColor": c, "color": "black",
                                            "weight": 2, "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=TOOLTIP[0], aliases=TOOLTIP[1]),
    ).add_to(m)

guessed = gdf[gdf["street_source"] == "nearest_centreline"]
if not guessed.empty:
    folium.GeoJson(
        to_wgs84(guessed, TOOLTIP[0]),
        name=f"Street guessed by proximity ({len(guessed)})",
        show=False,
        style_function=lambda x: {"fillColor": "#e67e22", "color": "#d35400",
                                  "weight": 1, "fillOpacity": 0.7},
        tooltip=folium.GeoJsonTooltip(fields=TOOLTIP[0], aliases=TOOLTIP[1]),
    ).add_to(m)

# ── ADDRESS POINTS ────────────────────────────────────────────────────────────

# ── PROPERTIES WITH NO ADDRESS ────────────────────────────────────────────────
# Parcels the LINZ property-address reference never names. The ones holding
# buildings are the reason buildings end up unaddressed -- there is nothing on
# the property to inherit -- while the empty ones are just bare land.

no_address = occupied_parcels[~occupied_parcels["property_id"].isin(addressed_properties)]
print(f"occupied parcels with no address reference: {len(no_address)}")

if len(no_address):
    folium.GeoJson(
        to_wgs84(no_address, ["property_id", "n_buildings"]),
        name=f"Property with no address ({len(no_address)})",
        style_function=lambda x: {"fillColor": "#ff00a0", "color": "#ff00a0",
                                  "weight": 2, "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=["property_id", "n_buildings"],
                                      aliases=["Property", "Buildings on it"]),
    ).add_to(m)

# One GeoJson layer rather than 3480 CircleMarker objects: folium emits a
# separate JS block per marker, which alone doubled the size of the page.
folium.GeoJson(
    addresses[["address_id", "full_address", "full_road_name", "geometry"]].to_crs(epsg=4326),
    name=f"Address points ({len(addresses)})",
    marker=folium.CircleMarker(radius=2.5, color="#f1c40f", fill=True,
                               fill_opacity=0.9, weight=0.5),
    tooltip=folium.GeoJsonTooltip(
        fields=["address_id", "full_address", "full_road_name"],
        aliases=["Address id", "Address", "Street"]),
).add_to(m)

for cause, (colour, _) in ORPHAN_COLOURS.items():
    subset = orphans[orphans["orphan_cause"] == cause]
    if subset.empty:
        continue
    folium.GeoJson(
        subset[["address_id", "full_address", "full_road_name", "orphan_cause",
                "nearest_building_m", "geometry"]].to_crs(epsg=4326),
        name=f"Orphan address: {cause} ({len(subset)})",
        marker=folium.CircleMarker(radius=5, color="black", weight=1,
                                   fill=True, fill_color=colour, fill_opacity=1.0),
        style_function=lambda x, c=colour: {"fillColor": c},
        tooltip=folium.GeoJsonTooltip(
            fields=["address_id", "full_address", "full_road_name", "orphan_cause",
                    "nearest_building_m"],
            aliases=["Address id", "Address", "Street", "Why orphaned",
                     "Nearest building (m)"]),
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

swatches = "".join(
    f'<i style="background:{c};width:14px;height:14px;display:inline-block;'
    f'margin-right:6px;"></i> {lab}<br>'
    for c, lab in ADDRESS_COLOURS.values()
)
orphan_swatches = "".join(
    f'<i style="background:{c};width:10px;height:10px;display:inline-block;'
    f'margin-right:10px;border-radius:50%;border:1px solid black;"></i> {lab}<br>'
    for c, lab in ORPHAN_COLOURS.values()
)
n_orphans = len(orphans)
m.get_root().html.add_child(folium.Element(f"""
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
     background-color: white; padding: 12px; border-radius: 6px;
     border: 1px solid grey; font-size: 13px;">
  <b>How each building got its address</b><br>
  {swatches}
  <hr style="margin:6px 0">
  <i style="background:#ff6b00;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Building with no property<br>
  <i style="background:#9b59b6;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Property by geometry only<br>
  <i style="background:#ff00a0;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Property with no address<br>

  <i style="background:#f1c40f;width:10px;height:10px;display:inline-block;
     margin-right:10px;border-radius:50%;"></i> Address point<br>
  <hr style="margin:6px 0">
  <span style="border-top:2.5px dashed #e74c3c;display:inline-block;width:14px;
     margin-right:6px;vertical-align:middle;"></span> Computational domain<br>
  <hr style="margin:6px 0">
  <b>Orphaned addresses ({n_orphans})</b><br>
  {orphan_swatches}
  <span style="color:#7f8c8d">Hover anything for its details.</span>
</div>
"""))

m.save(OUT_HTML)
print(f"saved {OUT_HTML}")
