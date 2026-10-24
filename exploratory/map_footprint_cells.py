"""Interactive map of the grid cells each building's depth is read from.

Draws, on the model grid, exactly what ExposureModule uses: every cell each
footprint touches (the max / median / p90 summaries read these), the cell
holding each centroid (the "centroid" summary), and the footprints themselves,
over street and aerial basemaps. Cells touched by more than one footprint are
drawn in a layer of their own.

Usage:
  python exploratory/map_footprint_cells.py
  python exploratory/map_footprint_cells.py --bbox 1488000 5374000 1490000 5376000
"""
import argparse
import os
import sys

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.exposure_module.exposure_module import ExposureModule  # noqa: E402
from modules.lib import REPO_ROOT  # noqa: E402
from modules.utils.spatial_operators.operators import (  # noqa: E402
    find_sample_flood_map,
    get_sample_georeference,
)

OUT_HTML = os.path.join(REPO_ROOT, "out", "footprint_cells_map.html")


def cell_boxes(transform, rows, cols):
    """One square polygon per (row, col) cell, in the grid's CRS."""
    x0 = transform.c + cols * transform.a
    y0 = transform.f + rows * transform.e
    return shapely.box(x0, y0 + transform.e, x0 + transform.a, y0)


def main(bbox=None, out_html=OUT_HTML):
    sample = find_sample_flood_map()
    with rasterio.open(sample) as src:
        shape = src.shape
    transform, crs, _ = get_sample_georeference(None)

    module = ExposureModule("map_footprint_cells")
    module._build_footprint_index(shape, None)
    buildings = module.buildings.reset_index(drop=True)
    ids = module._cell_ids.to_numpy()

    rows = module._row_t.cpu().numpy()
    cols = module._col_t.cpu().numpy()
    feat = module._feat_t.cpu().numpy()
    cells = gpd.GeoDataFrame(
        {"building_id": ids[feat], "row": rows, "col": cols},
        geometry=cell_boxes(transform, rows, cols), crs=crs,
    )
    centroid_on_grid = module._centroid_on_grid
    centroid_cells = gpd.GeoDataFrame(
        {"building_id": ids[centroid_on_grid]},
        geometry=cell_boxes(transform,
                            module._centroid_row_t.cpu().numpy()[centroid_on_grid],
                            module._centroid_col_t.cpu().numpy()[centroid_on_grid]),
        crs=crs,
    )
    num_cells = pd.Series(module._counts, index=ids, name="num_cells")
    footprints = buildings[["building_id", buildings.geometry.name]].to_crs(crs)
    footprints["num_cells"] = footprints["building_id"].map(num_cells)
    centroids = footprints.assign(geometry=footprints.geometry.centroid)[["building_id", "geometry"]]

    if bbox is not None:
        window = shapely.box(*bbox)
        keep = footprints.intersects(window)
        kept = set(footprints.loc[keep, "building_id"])
        footprints, centroids = footprints[keep], centroids[keep]
        cells = cells[cells["building_id"].isin(kept)]
        centroid_cells = centroid_cells[centroid_cells["building_id"].isin(kept)]

    shared_counts = cells.groupby(["row", "col"]).size()
    shared_keys = shared_counts[shared_counts > 1].index
    shared = cells.set_index(["row", "col"]).loc[lambda f: f.index.isin(shared_keys)]
    shared = shared.groupby(level=["row", "col"]).agg(
        building_ids=("building_id", lambda v: ", ".join(map(str, sorted(v)))),
        geometry=("geometry", "first"),
    ).reset_index()
    shared = gpd.GeoDataFrame(shared, geometry="geometry", crs=crs)
    shared["num_buildings"] = shared["building_ids"].str.count(",") + 1

    # Each building's cells as one multipolygon, so a hover names the building.
    per_building = cells.groupby("building_id").geometry.apply(
        lambda g: shapely.multipolygons(list(g))).reset_index()
    per_building = gpd.GeoDataFrame(per_building, geometry="geometry", crs=crs)
    per_building["num_cells"] = per_building["building_id"].map(num_cells)

    def wgs84(frame):
        frame = frame.to_crs(epsg=4326)
        frame["geometry"] = shapely.set_precision(frame.geometry.values, 1e-7)
        return frame

    centre = wgs84(footprints).geometry.union_all().centroid
    fmap = folium.Map(location=[centre.y, centre.x], zoom_start=17, max_zoom=22, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Streets", max_zoom=22, max_native_zoom=19).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Aerial", max_zoom=22, max_native_zoom=19,
    ).add_to(fmap)

    folium.GeoJson(
        wgs84(per_building), name=f"Footprint cells ({len(cells):,})",
        style_function=lambda _: {"color": "#d95f02", "weight": 0.6, "fillColor": "#fdb863", "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(["building_id", "num_cells"], aliases=["building", "cells"]),
    ).add_to(fmap)
    folium.GeoJson(
        wgs84(shared), name=f"Cells shared by 2+ footprints ({len(shared):,})",
        style_function=lambda _: {"color": "#b2182b", "weight": 0.8, "fillColor": "#ef8a62", "fillOpacity": 0.6},
        tooltip=folium.GeoJsonTooltip(["building_ids", "num_buildings"], aliases=["buildings", "count"]),
    ).add_to(fmap)
    folium.GeoJson(
        wgs84(centroid_cells), name=f"Centroid cells ({len(centroid_cells):,})",
        style_function=lambda _: {"color": "#1b7837", "weight": 1.2, "fillColor": "#5aae61", "fillOpacity": 0.45},
        tooltip=folium.GeoJsonTooltip(["building_id"], aliases=["building"]),
    ).add_to(fmap)
    folium.GeoJson(
        wgs84(footprints), name=f"Footprints ({len(footprints):,})",
        style_function=lambda _: {"color": "#2166ac", "weight": 1.5, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(["building_id", "num_cells"], aliases=["building", "cells"]),
    ).add_to(fmap)
    folium.GeoJson(
        wgs84(centroids), name="Centroids",
        marker=folium.CircleMarker(radius=2, color="#00441b", fill=True, fill_opacity=1, weight=0),
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fmap.save(out_html)
    print(f"{len(footprints):,} footprints, {len(cells):,} footprint cells "
          f"({len(shared):,} shared), {len(centroid_cells):,} centroid cells")
    print(f"wrote {out_html} ({os.path.getsize(out_html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="Only buildings intersecting this NZTM box")
    parser.add_argument("--out", default=OUT_HTML)
    args = parser.parse_args()
    main(args.bbox, args.out)
