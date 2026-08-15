"""Map each building together with the property and road parcel it was linked to.

The link table says building 2525143 belongs to property X and fronts road parcel
Y; this draws all three so the association can be checked by eye rather than by
reading ids. A connector line runs from every building to its assigned road
parcel, which is the link most worth seeing -- it is the one chosen through the
building's street rather than by plain proximity.

Writes out/building_context_map.html.
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
from shapely.geometry import LineString, shape
from shapely.ops import nearest_points, unary_union

from modules.utils.spatial_operators.operators import find_sample_flood_map

SIMULATION_DATA_DIR = "/data/westport/data"
BUILDING_FILE_PATH = f"{SIMULATION_DATA_DIR}/exposure/buildings.gpkg"
PROPERTY_FILE_PATH = f"{SIMULATION_DATA_DIR}/exposure/property_boundaries.gpkg"
ROAD_CADASTRAL = f"{SIMULATION_DATA_DIR}/exposure/road_network.gpkg"
ROAD_PARCEL_SECTIONS = f"{SIMULATION_DATA_DIR}/exposure/road_parcel_sections.csv"
ADDRESS_POINTS_FILE_PATH = f"{SIMULATION_DATA_DIR}/exposure/address_points.gpkg"
MANUAL_ADDRESS_POINTS = f"{SIMULATION_DATA_DIR}/exposure/manual_address_points.csv"
LINKS_CSV = f"{PROJECT_ROOT}/out/building_links.csv"
OUT_HTML = f"{PROJECT_ROOT}/out/building_context_map.html"

WORK_CRS = 2193
SIMPLIFY_M = 0.5

links = pd.read_csv(LINKS_CSV)
buildings = gpd.read_file(BUILDING_FILE_PATH).to_crs(WORK_CRS)
properties = gpd.read_file(PROPERTY_FILE_PATH).to_crs(WORK_CRS)
road_parcels = gpd.read_file(ROAD_CADASTRAL).to_crs(WORK_CRS)
parcel_streets = pd.read_csv(ROAD_PARCEL_SECTIONS)

# Manual points live outside the LINZ layer, so they are folded back in or the
# hand-addressed buildings would appear to point at nothing.
addresses = gpd.read_file(ADDRESS_POINTS_FILE_PATH).to_crs(WORK_CRS)[
    ["address_id", "full_address", "geometry"]]
import os
if os.path.exists(MANUAL_ADDRESS_POINTS):
    manual = pd.read_csv(MANUAL_ADDRESS_POINTS)
    manual = manual[manual["easting"].notna() & manual["northing"].notna()]
    if len(manual):
        addresses = pd.concat([addresses, gpd.GeoDataFrame(
            manual[["address_id", "full_address"]],
            geometry=gpd.points_from_xy(manual["easting"], manual["northing"]),
            crs=2193)], ignore_index=True)

gdf = buildings[["building_id", "geometry"]].merge(links, on="building_id", how="inner")
print(f"{len(gdf)} buildings linked")

# ── THE TWO PARCEL SETS ACTUALLY IN USE ───────────────────────────────────────
# Only parcels some building points at are drawn: the rest are context nobody
# asked for, and they are what makes these pages heavy.

used_property = set(gdf["property_id"].dropna())
used_road = set(gdf["road_parcel_id"].dropna())

prop_used = properties[properties["unit_of_property_id"].isin(used_property)].copy()
prop_used = prop_used.rename(columns={"unit_of_property_id": "property_id"})
prop_used["n_buildings"] = prop_used["property_id"].map(
    gdf.groupby("property_id").size()).fillna(0).astype(int)

road_used = road_parcels[road_parcels["id"].isin(used_road)].copy()
road_used = road_used.rename(columns={"id": "road_parcel_id"})
road_used["n_buildings"] = road_used["road_parcel_id"].map(
    gdf.groupby("road_parcel_id").size()).fillna(0).astype(int)
street_of_parcel = (parcel_streets.sort_values("overlap_m", ascending=False)
                    .drop_duplicates("road_parcel_id")
                    .set_index("road_parcel_id")["full_road_name"])
road_used["street"] = road_used["road_parcel_id"].map(street_of_parcel).fillna("—")

print(f"property parcels in use: {len(prop_used)} | road parcels in use: {len(road_used)}")

# ── CONNECTORS: BUILDING -> ITS ROAD PARCEL ───────────────────────────────────
# Drawn to the closest point on the parcel rather than its centroid, so the line
# shows the actual frontage relationship instead of pointing at a road's middle.

road_geom = road_used.set_index("road_parcel_id").geometry
rows = gdf.dropna(subset=["road_parcel_id"])
connectors = []
for building_id, geom, parcel_id, source, dist in zip(
        rows["building_id"], rows.geometry, rows["road_parcel_id"],
        rows["road_parcel_source"], rows["road_parcel_dist_m"]):
    parcel = road_geom.get(parcel_id)
    if parcel is None:
        continue
    a, b = nearest_points(geom.centroid, parcel)
    connectors.append({"building_id": building_id, "road_parcel_id": parcel_id,
                       "road_parcel_source": source,
                       "road_parcel_dist_m": round(float(dist), 1),
                       "geometry": LineString([a, b])})
connectors = gpd.GeoDataFrame(connectors, crs=WORK_CRS)
print(f"connectors: {len(connectors)}, "
      f"{int((connectors['road_parcel_source'] == 'nearest_parcel').sum())} from the fallback")

# ── ADDRESS POINTS AND THEIR CONNECTORS ───────────────────────────────────────
# Only points some building carries. A building sharing its property's address
# gets a connector too, which is what makes the shared-address cases visible:
# several buildings fanning out to one point.

address_geom = addresses.drop_duplicates("address_id").set_index("address_id").geometry
addressed = gdf[gdf["address_id"].notna()]
addr_links = []
for building_id, geom, address_id, source, full in zip(
        addressed["building_id"], addressed.geometry, addressed["address_id"],
        addressed["address_source"], addressed["full_address"]):
    point = address_geom.get(address_id)
    if point is None:
        continue
    addr_links.append({"building_id": building_id, "address_id": int(address_id),
                       "full_address": full, "address_source": source,
                       "geometry": LineString([geom.centroid, point])})
addr_links = gpd.GeoDataFrame(addr_links, crs=WORK_CRS)
used_addresses = addresses[addresses["address_id"].isin(addr_links["address_id"])].copy()
print(f"address points in use: {len(used_addresses)} | connectors: {len(addr_links)}")

# ── DOMAIN ────────────────────────────────────────────────────────────────────

with rasterio.open(find_sample_flood_map()) as src:
    mask = (~np.isnan(src.read(1).astype("float32"))).astype("uint8")
    polys = [shape(g) for g, v in rio_features.shapes(
        mask, mask=mask.astype(bool), transform=src.transform) if v == 1]
    domain_crs = src.crs
domain = gpd.GeoDataFrame({"name": ["Computational domain"]},
                          geometry=[unary_union(polys)], crs=domain_crs).to_crs(WORK_CRS)
domain["geometry"] = domain.geometry.simplify(10)

# ── MAP ───────────────────────────────────────────────────────────────────────

for col in ("full_address", "street_name", "property_id", "road_parcel_id", "use"):
    gdf[col] = gdf[col].fillna("—")
for col in ("address_dist_m", "road_parcel_dist_m", "overlap_frac"):
    gdf[col] = gdf[col].round(2).fillna("—")

BUILDING_FIELDS = ["building_id", "use", "full_address", "address_source", "street_name",
                   "property_id", "property_source", "road_parcel_id",
                   "road_parcel_source", "road_parcel_dist_m"]
BUILDING_ALIASES = ["Building", "Use", "Address", "Address via", "Street",
                    "Property", "Property via", "Road parcel",
                    "Road parcel via", "Distance to road (m)"]


def to_wgs84(g, cols):
    out = g[cols + ["geometry"]].copy()
    out["geometry"] = out.geometry.simplify(SIMPLIFY_M)
    return out.to_crs(epsg=4326)


centre = gdf.to_crs(epsg=4326).union_all().centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=15,
               tiles="Esri.WorldImagery", attr="Esri")

folium.GeoJson(
    domain.to_crs(epsg=4326), name="Computational domain",
    style_function=lambda x: {"fillOpacity": 0, "color": "#e74c3c",
                              "weight": 2.5, "dashArray": "6,4"},
).add_to(m)

folium.GeoJson(
    to_wgs84(prop_used, ["property_id", "n_buildings"]),
    name=f"Property parcels ({len(prop_used)})",
    style_function=lambda x: {"fillColor": "#2ecc71", "color": "#27ae60",
                              "weight": 1, "fillOpacity": 0.18},
    tooltip=folium.GeoJsonTooltip(fields=["property_id", "n_buildings"],
                                  aliases=["Property", "Buildings on it"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(road_used, ["road_parcel_id", "street", "n_buildings"]),
    name=f"Road parcels ({len(road_used)})",
    style_function=lambda x: {"fillColor": "#3498db", "color": "#2980b9",
                              "weight": 1, "fillOpacity": 0.3},
    tooltip=folium.GeoJsonTooltip(fields=["road_parcel_id", "street", "n_buildings"],
                                  aliases=["Road parcel", "Street", "Buildings fronting"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(connectors, ["building_id", "road_parcel_id", "road_parcel_source",
                          "road_parcel_dist_m"]),
    name=f"Building to road parcel ({len(connectors)})",
    style_function=lambda x: {
        "color": "#f39c12" if x["properties"]["road_parcel_source"] == "street_match"
        else "#e74c3c", "weight": 1, "opacity": 0.7},
    tooltip=folium.GeoJsonTooltip(
        fields=["building_id", "road_parcel_id", "road_parcel_source", "road_parcel_dist_m"],
        aliases=["Building", "Road parcel", "Matched via", "Distance (m)"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(addr_links, ["building_id", "address_id", "full_address", "address_source"]),
    name=f"Building to address ({len(addr_links)})",
    style_function=lambda x: {"color": "#9b59b6", "weight": 1, "opacity": 0.6,
                              "dashArray": "3,3"},
    tooltip=folium.GeoJsonTooltip(
        fields=["building_id", "address_id", "full_address", "address_source"],
        aliases=["Building", "Address id", "Address", "Matched via"]),
).add_to(m)

folium.GeoJson(
    used_addresses.to_crs(epsg=4326),
    name=f"Address points ({len(used_addresses)})",
    marker=folium.CircleMarker(radius=3, color="#8e44ad", fill=True,
                               fill_color="#f1c40f", fill_opacity=1.0, weight=1),
    tooltip=folium.GeoJsonTooltip(fields=["address_id", "full_address"],
                                  aliases=["Address id", "Address"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(gdf, BUILDING_FIELDS),
    name=f"Buildings ({len(gdf)})",
    style_function=lambda x: {"fillColor": "#ecf0f1", "color": "#2c3e50",
                              "weight": 0.7, "fillOpacity": 0.85},
    tooltip=folium.GeoJsonTooltip(fields=BUILDING_FIELDS, aliases=BUILDING_ALIASES),
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.get_root().html.add_child(folium.Element("""
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
     background-color: white; padding: 12px; border-radius: 6px;
     border: 1px solid grey; font-size: 13px;">
  <b>Each building with its property and road parcel</b><br>
  <i style="background:#ecf0f1;width:14px;height:14px;display:inline-block;
     margin-right:6px;border:1px solid #2c3e50;"></i> Building<br>
  <i style="background:#2ecc71;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Property parcel it belongs to<br>
  <i style="background:#3498db;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Road parcel it fronts<br>
  <span style="border-top:2px solid #f39c12;display:inline-block;width:14px;
     margin-right:6px;vertical-align:middle;"></span> Link, matched through its street<br>
  <span style="border-top:2px solid #e74c3c;display:inline-block;width:14px;
     margin-right:6px;vertical-align:middle;"></span> Link, nearest-parcel fallback<br>
  <i style="background:#f1c40f;width:10px;height:10px;display:inline-block;
     margin-right:10px;border-radius:50%;border:1px solid #8e44ad;"></i> Address point<br>
  <span style="border-top:2px dashed #9b59b6;display:inline-block;width:14px;
     margin-right:6px;vertical-align:middle;"></span> Link to its address<br>
  <span style="color:#7f8c8d">Hover a building for all of its ids.</span>
</div>
"""))

m.save(OUT_HTML)
print(f"saved {OUT_HTML}")
