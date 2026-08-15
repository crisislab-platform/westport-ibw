"""Why do some in-domain buildings end up with no property parcel?

Draws the buildings, the parcels underneath them and the address points on one
folium map, colouring each unmapped building by the reason the mapping fails.
Writes out/unmapped_buildings_map.html.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from modules.lib import (
    BUILDING_FOOTPRINTS_PATH,
    BUILDING_PROPERTY_MAPPING_PATH,
    PROPERTY_BOUNDARIES_PATH,
)
from modules.utils.spatial_operators.operators import load_features

ADDRESS_POINTS_FILE_PATH = "/data/westport/data/exposure/address_points.gpkg"
OUT_HTML = f"{PROJECT_ROOT}/out/unmapped_buildings_map.html"

# Parcels/addresses are drawn only near an unmapped building -- the whole domain
# is ~3800 parcels and 3480 points, which makes the page slow and buries the
# handful of cases the map is meant to explain. None -> draw everything.
CONTEXT_BUFFER_M = 250
SIMPLIFY_M = 0.3

# ── LOAD, exactly as ImpactModule sees them ───────────────────────────────────

buildings = load_features(BUILDING_FOOTPRINTS_PATH)
props_domain = load_features(PROPERTY_BOUNDARIES_PATH)
props_all = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(buildings.crs)
addresses = gpd.read_file(ADDRESS_POINTS_FILE_PATH).to_crs(buildings.crs)
mapping = pd.read_csv(BUILDING_PROPERTY_MAPPING_PATH)

titled = props_domain[props_domain["unit_of_property_id"].notna()].reset_index(drop=True)
untitled = props_domain[props_domain["unit_of_property_id"].isna()]
titled_outside = props_all[
    props_all["unit_of_property_id"].notna()
    & ~props_all["unit_of_property_id"].isin(titled["unit_of_property_id"])
]

# ── RESOLVE the mapping the way _load_building_property_map does ──────────────

property_row = pd.Series(np.arange(len(titled)), index=titled["unit_of_property_id"])
building_to_property = mapping.set_index("building_id")["property_id"].map(property_row)
buildings["property_idx"] = (
    buildings["building_id"].map(building_to_property).fillna(-1).to_numpy(dtype=int)
)
unmapped = buildings[buildings["property_idx"] < 0].copy()

# ── CLASSIFY the failures ─────────────────────────────────────────────────────

in_csv = set(mapping["building_id"])
tp = titled[["unit_of_property_id", "geometry"]].rename(
    columns={"unit_of_property_id": "property_id"}
)
overlaps_titled = set(
    gpd.sjoin(unmapped[["building_id", "geometry"]], tp, how="inner", predicate="intersects")[
        "building_id"
    ]
)

CAUSE_COLOURS = {
    # In the CSV with a real property_id, but that parcel fails load_features
    # and so is missing from ImpactModule's self.properties.
    "parcel outside flood domain": "#f39c12",
    # LINZ property<->building table has no row, though the footprint does sit
    # on a titled parcel -- recoverable with a geometric fallback.
    "no LINZ link row": "#3498db",
    # Genuinely on road reserve / untitled land: no titled parcel to link to.
    "untitled land only": "#e74c3c",
}


def classify(building_id):
    if building_id in in_csv:
        return "parcel outside flood domain"
    if building_id in overlaps_titled:
        return "no LINZ link row"
    return "untitled land only"


unmapped["cause"] = [classify(b) for b in unmapped["building_id"]]
print(f"{len(unmapped)} of {len(buildings)} buildings unmapped:")
print(unmapped["cause"].value_counts().to_string())

# ── TRIM THE CONTEXT LAYERS ───────────────────────────────────────────────────

if CONTEXT_BUFFER_M:
    near = unmapped.geometry.buffer(CONTEXT_BUFFER_M).union_all()
    titled = titled[titled.intersects(near)]
    untitled = untitled[untitled.intersects(near)]
    titled_outside = titled_outside[titled_outside.intersects(near)]
    addresses = addresses[addresses.within(near)]
    context_buildings = buildings[
        (buildings["property_idx"] >= 0) & buildings.intersects(near)
    ]
else:
    context_buildings = buildings[buildings["property_idx"] >= 0]

print(
    f"drawing {len(titled)} titled + {len(untitled)} untitled parcels, "
    f"{len(addresses)} addresses, {len(context_buildings)} mapped buildings"
)


def to_wgs84(gdf, cols):
    out = gdf[cols + [gdf.geometry.name]].copy()
    out["geometry"] = out.geometry.simplify(SIMPLIFY_M)
    return out.to_crs(epsg=4326)


# ── MAP ───────────────────────────────────────────────────────────────────────

centre = unmapped.to_crs(epsg=4326).union_all().centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=14,
               tiles="Esri.WorldImagery", attr="Esri")

folium.GeoJson(
    to_wgs84(titled, ["unit_of_property_id", "legal_description"]),
    name="Titled parcels (in domain)",
    style_function=lambda x: {"fillColor": "#2ecc71", "color": "#2ecc71",
                              "weight": 1, "fillOpacity": 0.12},
    tooltip=folium.GeoJsonTooltip(fields=["unit_of_property_id", "legal_description"],
                                  aliases=["Property id", "Legal description"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(titled_outside, ["unit_of_property_id", "legal_description"]),
    name="Titled parcels (outside domain)",
    style_function=lambda x: {"fillColor": "#9b59b6", "color": "#9b59b6",
                              "weight": 1, "dashArray": "4,4", "fillOpacity": 0.18},
    tooltip=folium.GeoJsonTooltip(fields=["unit_of_property_id", "legal_description"],
                                  aliases=["Property id", "Legal description"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(untitled, ["legal_description"]),
    name="Untitled land (road / reserve)",
    style_function=lambda x: {"fillColor": "#95a5a6", "color": "#7f8c8d",
                              "weight": 1, "fillOpacity": 0.35},
    tooltip=folium.GeoJsonTooltip(fields=["legal_description"], aliases=["Legal description"]),
).add_to(m)

folium.GeoJson(
    to_wgs84(context_buildings, ["building_id"]),
    name="Buildings with a property",
    style_function=lambda x: {"fillColor": "#ecf0f1", "color": "#bdc3c7",
                              "weight": 0.6, "fillOpacity": 0.5},
    tooltip=folium.GeoJsonTooltip(fields=["building_id"], aliases=["Building id"]),
).add_to(m)

for cause, colour in CAUSE_COLOURS.items():
    subset = unmapped[unmapped["cause"] == cause]
    if subset.empty:
        continue
    folium.GeoJson(
        to_wgs84(subset, ["building_id", "use", "cause"]),
        name=f"Unmapped: {cause} ({len(subset)})",
        style_function=lambda x, c=colour: {"fillColor": c, "color": "black",
                                            "weight": 1.2, "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=["building_id", "use", "cause"],
                                      aliases=["Building id", "Use", "Why unmapped"]),
    ).add_to(m)

address_layer = folium.FeatureGroup(name=f"Address points ({len(addresses)})")
for _, row in addresses.to_crs(epsg=4326).iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2.5, color="#f1c40f", fill=True, fill_opacity=0.9, weight=0.5,
        tooltip=row["full_address"],
    ).add_to(address_layer)
address_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

legend_html = """
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
     background-color: white; padding: 12px; border-radius: 6px;
     border: 1px solid grey; font-size: 13px;">
  <b>Buildings with no property parcel</b><br>
  <i style="background:#f39c12;width:14px;height:14px;display:inline-block;margin-right:6px;"></i>
    On a titled parcel that is outside the flood domain<br>
  <i style="background:#3498db;width:14px;height:14px;display:inline-block;margin-right:6px;"></i>
    On a titled parcel, but no LINZ link row<br>
  <i style="background:#e74c3c;width:14px;height:14px;display:inline-block;margin-right:6px;"></i>
    On untitled land only (road reserve)<br>
  <hr style="margin:6px 0">
  <i style="background:#2ecc71;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Titled parcel (in domain)<br>
  <i style="background:#9b59b6;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Titled parcel (outside domain)<br>
  <i style="background:#95a5a6;width:14px;height:14px;display:inline-block;margin-right:6px;"></i> Untitled land<br>
  <i style="background:#f1c40f;width:10px;height:10px;display:inline-block;
     margin-right:10px;border-radius:50%;"></i> Address point<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save(OUT_HTML)
print(f"saved {OUT_HTML}")
