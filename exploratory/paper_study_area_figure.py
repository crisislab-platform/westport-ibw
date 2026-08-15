"""Study-area maps for the paper.

Writes the same two maps in both of the arrangements worth having:

  study_area_inset    ONE self-contained figure -- the modelled domain with the
                      detail carried as an inset in the top-right corner, a
                      leader from the locator box, and the colour key inside
                      the map. Drop this next to the TikZ sampling-window
                      diagram and the figure is done.
  study_area_panel_a  the domain on its own
  study_area_panel_b  the detail on its own
  study_area_legend   the colour key on its own

The separate panels are for composing the layout in TikZ instead; they are
saved full-bleed at exact square figure sizes with no title, panel letter or
frame, since TikZ owns the labelling there and anything drawn here would have
to be positioned twice. The script prints the locator box's position in panel A
axes fractions so a TikZ leader can be anchored without measuring the image.
"""
import os
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import contextily as cx
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.ops import nearest_points

from modules.lib import (
    BUILDING_FOOTPRINTS_PATH,
    BUILDING_LINKS_PATH,
    PROPERTY_BOUNDARIES_PATH,
    ROAD_NETWORK_FILE_PATH,
)

WORK_CRS = 2193      # NZTM2000, metres -- all geometry work happens here
PLOT_CRS = 3857      # Web Mercator, for the tile basemap

# How far either side of the property the road parcel is followed when framing
# panel B. True metres, measured in WORK_CRS -- Web Mercator units are
# stretched by 1/cos(latitude) and would not mean this.
FRONTAGE_CONTEXT_M = 80

OUT_DIR = os.path.join(PROJECT_ROOT, "exploratory")
COMBINED = os.path.join(OUT_DIR, "study_area_inset")
PANEL_A = os.path.join(OUT_DIR, "study_area_panel_a")
PANEL_B = os.path.join(OUT_DIR, "study_area_panel_b")
LEGEND = os.path.join(OUT_DIR, "study_area_legend")

# Where the inset sits in the combined figure, as figure fractions. Top-right:
# the town fills the middle and lower left of the window, so that corner is the
# one holding least of the map.
INSET_RECT = (0.588, 0.588, 0.395, 0.395)

# Side length of each saved map, inches. Both windows are square, so one number
# fixes the page and TikZ can scale from there.
PANEL_A_IN = 6.4
PANEL_B_IN = 3.4

# Categorical slots 1-3 of the reference palette, in fixed order. Those three
# are the documented all-pairs-safe subset (worst pair CVD dE 9.2), which is
# what a map needs -- every layer is adjacent to every other one.
C_STREET = "#2a78d6"
C_BUILDING = "#eb6834"
C_PROPERTY = "#1baf7a"

PAPER = "#ffffff"
INK = "#0b0b0b"
MUTED = "#52514e"

# Text over imagery is never on a known surface, so every label carries a halo
# rather than relying on contrast with whatever is behind it.
HALO = [pe.withStroke(linewidth=2.6, foreground=PAPER)]


def scale_bar(ax, length_m, label, loc=(0.055, 0.055), height=0.012):
    """Alternating black/white scale bar, sized in metres of the axes' CRS."""
    x0, y0 = loc
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    frac = length_m / span
    for i, (a, b) in enumerate(((0.0, 0.5), (0.5, 1.0))):
        ax.add_patch(Rectangle(
            (x0 + frac * a, y0), frac * (b - a), height,
            transform=ax.transAxes, facecolor=PAPER if i == 0 else INK,
            edgecolor=INK, linewidth=0.7, zorder=11))
    ax.text(x0 + frac / 2, y0 + height + 0.012, label, transform=ax.transAxes,
            ha="center", va="bottom", color=INK, fontsize=9,
            fontweight="bold", zorder=12, path_effects=HALO)


def north_arrow(ax, loc=(0.945, 0.90)):
    x, y = loc
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.07),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", color=INK, fontsize=11,
                fontweight="bold", zorder=12, path_effects=HALO,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                path_effects=HALO))


def square(bounds, pad_frac=0.0):
    """Square window centred on a bounds tuple, so both panels share a shape."""
    x0, y0, x1, y1 = bounds
    cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * (1 + pad_frac)
    return cx_ - half, cy_ - half, cx_ + half, cy_ + half


def full_bleed(size_in, extent):
    """A figure that is nothing but one map: no margins, frame or ticks.

    The saved page is then exactly the map, so TikZ can place it as a plain
    node without trimming.
    """
    fig = plt.figure(figsize=(size_in, size_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    return fig, ax


def basemap(ax):
    cx.add_basemap(ax, crs=f"EPSG:{PLOT_CRS}",
                   source=cx.providers.Esri.WorldImagery,
                   attribution=False, zorder=0)


def save(fig, stem, close=True):
    # No bbox_inches="tight": the page must stay exactly the figure size, or
    # the printed geometry below would not match the file TikZ receives.
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    # Kept open only for the combined figure, which main() hands back so a
    # notebook cell can display it.
    if close:
        plt.close(fig)
    print(f"wrote {os.path.basename(stem)}.pdf / .png")


def load():
    """Every layer the maps draw, plus the building the detail panel is about."""
    links = pd.read_csv(BUILDING_LINKS_PATH)

    buildings = gpd.read_file(BUILDING_FOOTPRINTS_PATH).to_crs(WORK_CRS)
    properties = gpd.read_file(PROPERTY_BOUNDARIES_PATH).to_crs(WORK_CRS)
    roads = gpd.read_file(ROAD_NETWORK_FILE_PATH).to_crs(WORK_CRS)

    # The link table keys properties by unit_of_property_id and road parcels by
    # the numeric parcel id, so both sides are aligned onto those before any
    # membership test -- matching on the layers' own "id" silently returns
    # nothing.
    properties = properties.rename(columns={"unit_of_property_id": "property_id"})
    roads["road_parcel_id"] = pd.to_numeric(roads["id"], errors="coerce")
    links["road_parcel_id"] = pd.to_numeric(links["road_parcel_id"], errors="coerce")

    # Only what the model actually runs on: the link table is the authority.
    buildings = buildings[buildings["building_id"].isin(links["building_id"])]
    linked_properties = properties[properties["property_id"].isin(links["property_id"])]
    linked_roads = roads[roads["road_parcel_id"].isin(links["road_parcel_id"].dropna())]

    # A property carrying exactly one building, with a road parcel linked, and
    # of typical residential size -- so the zoom shows the ordinary case rather
    # than an outlier.
    per_property = links.groupby("property_id").size()
    candidates = links[
        links["property_id"].isin(per_property[per_property == 1].index)
        & links["road_parcel_id"].notna()
    ]
    detail = linked_properties[
        linked_properties["property_id"].isin(candidates["property_id"])
    ].copy()
    detail["area"] = detail.area
    detail = detail[(detail["area"] > 400) & (detail["area"] < 1400)]

    # Nearest such property to the middle of the built-up area, so the zoom
    # sits in town rather than on a rural edge.
    detail["d"] = detail.distance(buildings.union_all().centroid)
    chosen_property = detail.sort_values("d").iloc[[0]]

    link = candidates[
        candidates["property_id"] == chosen_property["property_id"].iloc[0]
    ].iloc[0]
    chosen_building = buildings[buildings["building_id"] == link["building_id"]]
    chosen_road = roads[roads["road_parcel_id"] == link["road_parcel_id"]]

    # The stretch of road parcel panel B has to hold. Clipping to the property's
    # neighbourhood first bounds the window: a road parcel can run the length of
    # a street, and framing all of one would shrink the building out of
    # legibility. This building's parcel is 104 x 42 m, inside the clip, so it
    # is framed whole.
    frontage = gpd.GeoSeries(
        [chosen_road.geometry.iloc[0].intersection(
            chosen_property.geometry.iloc[0].buffer(FRONTAGE_CONTEXT_M))],
        crs=WORK_CRS,
    )

    layers = {k: v.to_crs(PLOT_CRS) for k, v in dict(
        buildings=buildings, properties=linked_properties, roads=linked_roads,
        d_property=chosen_property, d_building=chosen_building,
        d_road=chosen_road, d_frontage=frontage,
    ).items()}
    return layers, link


def extents(layers):
    """The window each panel covers, both square."""
    # The domain reaches ~10 km inland over farmland where a handful of
    # buildings sit; framing on that leaves the town an unreadable speck. The
    # window is taken from the central band of building centroids instead,
    # which is the built-up area the model is actually about.
    pts = layers["buildings"].representative_point()
    q = 0.2
    overview = square((pts.x.quantile(q), pts.y.quantile(q),
                       pts.x.quantile(1 - q), pts.y.quantile(1 - q)), pad_frac=0.04)
    inside = int(((pts.x > overview[0]) & (pts.x < overview[2])
                  & (pts.y > overview[1]) & (pts.y < overview[3])).sum())

    # Panel B holds the property, its building and the road parcel frontage in
    # full. Framing on the nearest point of the road instead left the parcel
    # running off the panel edge, which reads as though the building were
    # linked to a sliver of road rather than to the street.
    near = layers["d_frontage"].total_bounds
    built = pd.concat([layers["d_property"], layers["d_building"]]).total_bounds
    detail = square((min(near[0], built[0]), min(near[1], built[1]),
                     max(near[2], built[2]), max(near[3], built[3])), pad_frac=0.12)
    return overview, detail, inside


def paint_overview(ax, layers, overview, detail, scale_len=500, scale_txt="500 m"):
    """The domain map, and the locator box marking what the detail shows.

    Returns the box as (x0, y0, side) in data coordinates.
    """
    # No white casing at this density: at ~2 km across the casing is wider than
    # the line it protects and turns the panel white.
    layers["roads"].plot(ax=ax, facecolor=C_STREET, edgecolor=C_STREET,
                         linewidth=0.25, alpha=0.5, zorder=3)
    layers["properties"].plot(ax=ax, facecolor="none", edgecolor=C_PROPERTY,
                              linewidth=0.35, alpha=0.9, zorder=4)
    layers["buildings"].plot(ax=ax, facecolor=C_BUILDING, edgecolor=C_BUILDING,
                             linewidth=0.1, alpha=0.95, zorder=5)
    basemap(ax)

    # The detail window is ~120 m across against a ~2 km panel, so drawn to
    # scale the box would be thinner than its own outline; it is enlarged to a
    # legible minimum and marks the location rather than the exact footprint.
    span = overview[2] - overview[0]
    box = max(detail[2] - detail[0], 0.035 * span)
    lx = (detail[0] + detail[2]) / 2 - box / 2
    ly = (detail[1] + detail[3]) / 2 - box / 2
    for lw, color in ((3.2, PAPER), (1.5, INK)):
        ax.add_patch(Rectangle((lx, ly), box, box, facecolor="none",
                               edgecolor=color, lw=lw, zorder=8))

    scale_bar(ax, scale_len, scale_txt)
    return lx, ly, box


def paint_detail(ax, layers, label_size=9.5):
    """The one-property map: building, its section, and the street it fronts."""
    layers["d_road"].plot(ax=ax, facecolor=C_STREET, edgecolor=PAPER,
                          linewidth=2.6, alpha=0.42, zorder=3)
    layers["d_road"].plot(ax=ax, facecolor="none", edgecolor=C_STREET,
                          linewidth=1.6, zorder=4)
    layers["d_property"].plot(ax=ax, facecolor=C_PROPERTY, edgecolor=PAPER,
                              linewidth=2.6, alpha=0.3, zorder=5)
    layers["d_property"].plot(ax=ax, facecolor="none", edgecolor=C_PROPERTY,
                              linewidth=1.8, zorder=6)
    layers["d_building"].plot(ax=ax, facecolor=C_BUILDING, edgecolor=PAPER,
                              linewidth=1.6, alpha=0.9, zorder=7)
    basemap(ax)

    # Direct labels, so identity holds up if the map is read without the
    # legend beside it.
    b_pt = layers["d_building"].geometry.iloc[0].centroid
    p_geom = layers["d_property"].geometry.iloc[0]
    r_pt = nearest_points(b_pt, layers["d_road"].geometry.iloc[0])[1]
    ax.annotate("building", xy=(b_pt.x, b_pt.y), ha="center", va="center",
                fontsize=label_size, fontweight="bold", color=INK,
                path_effects=HALO, zorder=10)
    # Outside the parcel's south-west corner: the building fills most of the
    # section, so a label placed inside it lands on the footprint.
    ax.annotate("property", xy=(p_geom.bounds[0], p_geom.bounds[1]),
                xytext=(-4, -4), textcoords="offset points", ha="right",
                va="top", fontsize=label_size, fontweight="bold",
                color="#0f7a55", path_effects=HALO, zorder=10)
    ax.annotate("street parcel", xy=(r_pt.x, r_pt.y), xytext=(0, -22),
                textcoords="offset points", ha="center", va="top",
                fontsize=label_size, fontweight="bold", color="#1c5fa8",
                path_effects=HALO, zorder=10)


def key_handles():
    return [
        Patch(facecolor=C_BUILDING, edgecolor=C_BUILDING, label="Building footprint"),
        Patch(facecolor=C_PROPERTY, edgecolor=C_PROPERTY, alpha=0.45,
              label="Property parcel"),
        Patch(facecolor=C_STREET, edgecolor=C_STREET, alpha=0.45,
              label="Street parcel"),
    ]


def draw_combined(layers, overview, detail):
    """One figure: the domain, with the detail inset into its top-right."""
    fig, ax = full_bleed(PANEL_A_IN, overview)
    lx, ly, box = paint_overview(ax, layers, overview, detail)
    north_arrow(ax, loc=(0.055, 0.945))

    ax_in = fig.add_axes(INSET_RECT)
    ax_in.set_xlim(detail[0], detail[2])
    ax_in.set_ylim(detail[1], detail[3])
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for side in ax_in.spines.values():
        side.set_color(PAPER)
        side.set_linewidth(2.4)
    paint_detail(ax_in, layers, label_size=8.5)
    scale_bar(ax_in, 20, "20 m", loc=(0.06, 0.06), height=0.018)

    # Leaders from the locator box up to the inset's near corners, drawn on the
    # figure so they can cross the axes boundary. Solid and cased: they run over
    # imagery of unpredictable brightness, where a thin dashed line disappears.
    from matplotlib.patches import ConnectionPatch
    # Both leaders rise from the top of the box to the bottom of the inset, so
    # neither has to traverse the map sideways.
    for corner, inset_corner in (((0, 1), (0, 0)), ((1, 1), (1, 0))):
        fig.add_artist(ConnectionPatch(
            xyA=(lx + corner[0] * box, ly + corner[1] * box), coordsA=ax.transData,
            xyB=inset_corner, coordsB=ax_in.transAxes,
            color=INK, lw=1.2, zorder=7,
            path_effects=[pe.withStroke(linewidth=3.2, foreground=PAPER)]))

    leg = ax.legend(handles=key_handles(), loc="lower right", ncol=1,
                    fontsize=8.5, frameon=True, framealpha=0.9,
                    borderpad=0.6, handlelength=1.3, borderaxespad=0.6)
    leg.get_frame().set_edgecolor(MUTED)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(12)

    save(fig, COMBINED, close=False)
    return fig


def draw_panel_a(layers, overview, detail):
    fig, ax = full_bleed(PANEL_A_IN, overview)
    lx, ly, box = paint_overview(ax, layers, overview, detail)
    north_arrow(ax)
    save(fig, PANEL_A)

    # Axes fractions, so a TikZ leader can be anchored to the box corners
    # without measuring the exported image.
    span = overview[2] - overview[0]
    return ((lx - overview[0]) / span, (ly - overview[1]) / span, box / span)


def draw_panel_b(layers, detail):
    fig, ax = full_bleed(PANEL_B_IN, detail)
    paint_detail(ax, layers)
    scale_bar(ax, 20, "20 m")
    save(fig, PANEL_B)


def draw_legend():
    """The colour key on its own page, sized to its content."""
    fig = plt.figure(figsize=(7.2, 0.4))
    fig.legend(handles=key_handles(), loc="center", ncol=3, frameon=False,
               fontsize=9.5)
    # The one figure that IS sized to its content rather than to a fixed page.
    fig.savefig(f"{LEGEND}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{LEGEND}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {os.path.basename(LEGEND)}.pdf / .png")


def main():
    layers, link = load()
    overview, detail, inside = extents(layers)

    fig = draw_combined(layers, overview, detail)
    lx, ly, box = draw_panel_a(layers, overview, detail)
    draw_panel_b(layers, detail)
    draw_legend()

    print(f"domain map {PANEL_A_IN} x {PANEL_A_IN} in, holds {inside} of "
          f"{len(layers['buildings'])} modelled buildings")
    print(f"detail map {PANEL_B_IN} x {PANEL_B_IN} in, "
          f"{link['full_address']}, {link['street_name']}")
    print(f"locator box in panel A, as fractions of its width from the "
          f"bottom-left: x={lx:.4f} y={ly:.4f} side={box:.4f}")
    return fig


if __name__ == "__main__":
    main()
