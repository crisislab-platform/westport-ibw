"""Interactive map of the ring each building is sampled through.

Every depth antecedent the FIS sees is a statistic over a ring: the footprint
buffered outward with every footprint in the layer subtracted, so the ring is
the ground around the building and never a roof (ImpactModule.buffered_flood_depth).
This draws those rings as the growth loop actually settles them, coloured by the
buffer each building ended on, so the two things that are hard to see in a table
are visible at a glance: where the loop had to grow, and what a ring is actually
sampling when a building sits in a packed block.

Layers, all toggleable:
  Rings           coloured by final buffer width, 1 / 2 / 3+ cells
  Exhausted       rings still under the floor at the cap, drawn on top
  Footprints      the modelled buildings, the rings' subjects
  Unmodelled      footprints dropped by the domain filter -- not scored, but
                  subtracted from the rings, which is why they are worth seeing
  Domain          the flood model's valid cells, which is what "unmodelled" means

Usage:
  python exploratory/map_building_rings.py
  python exploratory/map_building_rings.py --min-cells 6 --max-buffer 5

Writes out/building_rings_map.html.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXPLORATORY = f"{PROJECT_ROOT}/exploratory"
if EXPLORATORY not in sys.path:
    sys.path.insert(0, EXPLORATORY)

import argparse

import folium
import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio import features as rio_features
from shapely.geometry import shape
from shapely.ops import unary_union

from modules.impact_module.risk_module import (
    GROWTH_STEP_PCT, MAX_BUFFER_PCT, MIN_RING_CELLS,
)
from modules.lib import BUILDING_FOOTPRINTS_PATH
from modules.utils.spatial_operators.operators import find_sample_flood_map
from ring_growth_tuning import context, count_sparse, grow

OUT_HTML = f"{PROJECT_ROOT}/out/building_rings_map.html"

WORK_CRS = 2193

# Ring outlines are buffered footprints, so their corners carry a lot of
# vertices that no one can see at map scale. 0.4 m is well under a tenth of a
# cell and takes the page from unusable to responsive.
SIMPLIFY_M = 0.4

# By final buffer width in cells: most buildings never grow, so the colours have
# to make the minority that did stand out rather than blend into a wash.
BUFFER_COLOURS = {
    1: ("#3498db", "Settled at 1 cell (no growth needed)"),
    2: ("#f39c12", "Grew to 2 cells"),
    3: ("#e74c3c", "Grew to the cap"),
}
EXHAUSTED_COLOUR = "#8e44ad"


def domain_layer():
    """The flood model's valid cells, polygonised the way load_features reads them.

    Drawn because it is the line that decides which footprints are modelled at
    all, and therefore why some buildings on the map have no ring.
    """
    with rasterio.open(find_sample_flood_map()) as src:
        mask = (~np.isnan(src.read(1).astype("float32"))).astype("uint8")
        polys = [
            shape(geom)
            for geom, val in rio_features.shapes(
                mask, mask=mask.astype(bool), transform=src.transform)
            if val == 1
        ]
        crs = src.crs

    domain = gpd.GeoDataFrame(
        {"name": ["Computational domain"]},
        geometry=[unary_union(polys)],
        crs=crs,
    ).to_crs(WORK_CRS)
    # Pixel-edged, so the raw outline is a staircase; 10 m reads as a boundary
    # without moving it meaningfully at this scale.
    domain["geometry"] = domain.geometry.simplify(10)
    return domain


def build_rings(min_cells, max_buffer_pct, step_pct):
    """Run the growth loop and return the rings with their per-building stats."""
    ctx = context()
    buildings, cell_size = ctx["buildings"], ctx["cell_size"]

    geom, dense, buffer_pct, exhausted, _ = grow(
        buildings, ctx["footprints"], ctx["shape"], ctx["transform"],
        ctx["raster_crs"], ctx["valid"], cell_size, min_cells, max_buffer_pct,
        step_pct, ctx["cache"],
    )
    sparse = count_sparse(geom, buildings.crs, ctx["shape"], ctx["transform"],
                          ctx["raster_crs"])

    rings = gpd.GeoDataFrame(
        {
            "building_id": buildings["building_id"].values,
            "use": buildings["use"].fillna("—").values,
            "buffer_cells": buffer_pct,
            "buffer_m": (buffer_pct * cell_size).round(1),
            "ring_cells": sparse,
            "ring_cells_dense": dense,
            "ring_area": geom.area.values.round(0),
            "fp_area": buildings.geometry.area.values.round(0),
            "exhausted": exhausted,
        },
        geometry=geom.values,
        crs=buildings.crs,
    )
    return ctx, rings


def to_wgs84(gdf):
    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(SIMPLIFY_M)
    out = out.to_crs(epsg=4326)
    # Coordinates are serialised into the page at full float precision, which is
    # several megabytes of digits nobody can see. 1e-6 degrees is ~0.1 m.
    out["geometry"] = shapely.set_precision(out.geometry.values, 1e-6)
    return out


TOOLTIP = (
    ["building_id", "use", "buffer_cells", "buffer_m", "ring_cells",
     "ring_cells_dense", "ring_area", "fp_area", "exhausted"],
    ["Building id", "Use", "Buffer (cells)", "Buffer (m)",
     "Ring cells (production)", "Ring cells (growth loop)", "Ring area (m²)",
     "Footprint area (m²)", "Exhausted"],
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-cells", type=float, default=MIN_RING_CELLS)
    parser.add_argument("--max-buffer", type=float, default=MAX_BUFFER_PCT)
    parser.add_argument("--step", type=float, default=GROWTH_STEP_PCT)
    args = parser.parse_args()

    ctx, rings = build_rings(args.min_cells, args.max_buffer, args.step)
    buildings = ctx["buildings"]
    print(f"{len(rings)} rings, {int(rings['exhausted'].sum())} exhausted")
    print(rings["buffer_cells"].value_counts().sort_index().to_string())

    centre = rings.to_crs(epsg=4326).union_all().centroid
    m = folium.Map(location=[centre.y, centre.x], zoom_start=15,
                   tiles="Esri.WorldImagery", attr="Esri")

    folium.GeoJson(
        to_wgs84(domain_layer()),
        name="Computational domain",
        style_function=lambda x: {"fillColor": "#ffffff", "color": "#ecf0f1",
                                  "weight": 2.5, "dashArray": "6,4",
                                  "fillOpacity": 0.0},
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=[""]),
    ).add_to(m)

    # Footprints under the rings. Without them a ring reads as a blob rather
    # than as the ground around something.
    folium.GeoJson(
        to_wgs84(buildings[["building_id", "geometry"]]),
        name="Building footprints (modelled)",
        style_function=lambda x: {"fillColor": "#2c3e50", "color": "#2c3e50",
                                  "weight": 0.4, "fillOpacity": 0.75},
    ).add_to(m)

    # The footprints the domain filter drops. They are subtracted from the
    # rings but never scored, so on the map they appear as bites taken out of a
    # neighbour's ring by a building that has none of its own.
    all_footprints = gpd.read_file(BUILDING_FOOTPRINTS_PATH).to_crs(WORK_CRS)
    unmodelled = all_footprints[
        ~all_footprints["building_id"].isin(buildings["building_id"])
    ][["building_id", "geometry"]]
    folium.GeoJson(
        to_wgs84(unmodelled),
        name=f"Footprints outside the model ({len(unmodelled)})",
        style_function=lambda x: {"fillColor": "#7f8c8d", "color": "#95a5a6",
                                  "weight": 0.4, "fillOpacity": 0.45},
        tooltip=folium.GeoJsonTooltip(fields=["building_id"],
                                      aliases=["Building id (not modelled)"]),
    ).add_to(m)

    # One layer per buffer width rather than one styled layer, so the widths can
    # be turned on and off independently -- the question is usually "show me the
    # ones that had to grow".
    for width, (colour, label) in BUFFER_COLOURS.items():
        if width == max(BUFFER_COLOURS):
            subset = rings[rings["buffer_cells"] >= width]
        else:
            subset = rings[rings["buffer_cells"] == width]
        if subset.empty:
            continue
        folium.GeoJson(
            to_wgs84(subset),
            name=f"{label} — {len(subset)}",
            style_function=lambda x, c=colour: {
                "fillColor": c, "color": c, "weight": 0.6, "fillOpacity": 0.45},
            tooltip=folium.GeoJsonTooltip(fields=TOOLTIP[0], aliases=TOOLTIP[1]),
        ).add_to(m)

    exhausted = rings[rings["exhausted"]]
    if not exhausted.empty:
        folium.GeoJson(
            to_wgs84(exhausted),
            name=f"Exhausted — {len(exhausted)}",
            style_function=lambda x: {"fillColor": EXHAUSTED_COLOUR,
                                      "color": EXHAUSTED_COLOUR, "weight": 2.0,
                                      "fillOpacity": 0.55},
            tooltip=folium.GeoJsonTooltip(fields=TOOLTIP[0], aliases=TOOLTIP[1]),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_HTML)
    print(f"map -> {OUT_HTML}")


if __name__ == "__main__":
    main()
