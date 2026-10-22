"""How the ring sampling window relates to the footprint it surrounds.

Every depth antecedent the FIS sees -- h_max, h_mean, wet_fraction -- is a
statistic over the cells of a ring: the footprint buffered outward with every
building in the domain subtracted, so the ring is the ground around the building
and never the building itself (ImpactModule.buffered_flood_depth). num_ring_cells
is that statistic's sample size, and on a 4.8 m grid a fixed-width buffer would
hand a small house far fewer cells than a warehouse, because ring area scales
with PERIMETER. _build_adaptive_rings exists to break that dependence: it grows
each building's buffer a cell at a time until the ring holds MIN_RING_CELLS,
capped at MAX_BUFFER_PCT.

This script measures whether that works, over every modelled building, and
answers three questions the manuscript needs:

  1. Does the adaptive rule decouple sample size from footprint size, and what
     would a fixed buffer have done instead? (the counterfactual table)
  2. What does the ring actually sample -- how far does it reach on a small
     footprint, and how much of the nominal annulus do neighbours cut away?
  3. How big is the exhausted tail: buildings still short of MIN_RING_CELLS at
     MAX_BUFFER_PCT, which nothing downstream currently handles?

It is geometry only. Nothing here touches forecast skill; the errors those rings
produce are error_depth_analysis.py's subject.

Writes, all under out/:
  ring_footprint_buildings.csv      per-building record, the underlying data
  ring_footprint_summary.csv        metrics by footprint-area decile, long form
  ring_footprint_counterfactual.csv fixed buffers against adaptive, the argument
  ring_footprint_saturation.csv     true cell count against the count the growth
                                    loop consults, at matched widths

Every number quoted in the manuscript should come from these files rather than
from a transcript, so a config change regenerates the text's evidence.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from scipy import stats
from shapely.ops import unary_union

from modules.lib import SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.impact_module.risk_module import (
    GROWTH_STEP_PCT, MAX_BUFFER_PCT, MIN_RING_CELLS, ImpactModule,
)
from modules.utils.spatial_operators.operators import (
    _get_feature_labels, get_sample_georeference,
)

OUT = f"{PROJECT_ROOT}/out"

# The buffer the model is actually run with (run_scripts/validate.sh passes
# --buffer_multiplier 2), in cells. It is the adaptive rule's starting point,
# and the fixed setting the counterfactual has to beat.
BASE_BUFFER_PCT = 2.0

# Fixed buffers to score the adaptive rule against, in cells. 2 is the
# configured one; 1 and 3 bracket it; 5 is what a fixed buffer would have to be
# to match the adaptive cap, and shows the cost of getting the floor that way.
FIXED_BUFFER_PCTS = (1.0, 2.0, 3.0, 5.0)

# The grid the rings are rasterized against. The surveyed July 2021 raster is
# the one the validation runs on, so the geometry measured here is the geometry
# the reported skill came from. Only its shape and transform matter -- the
# adaptive loop counts cells, it does not read depths.
GRID_RASTER = SURVEYED_FLOOD_DEPTH_JUL_2021

PERCENTILES = [.05, .25, .5, .75, .95]

N_DECILES = 10


def population():
    """The modelled buildings, in the grid's CRS, with footprint metrics.

    Built through ImpactModule rather than by reading buildings.gpkg: __init__
    already applies both filters that define "modelled" -- load_features drops
    footprints outside the flood-model domain, _load_building_links drops
    anything absent from the link table -- and reading the layer directly would
    silently analyse a larger, different population.
    """
    impact = ImpactModule("ring_footprint")
    buildings = impact.buildings.reset_index(drop=True)

    metrics = pd.DataFrame({
        "building_id": buildings["building_id"].values,
        "fp_area": buildings.geometry.area.values,
        "fp_perim": buildings.geometry.length.values,
    })
    # Isoperimetric ratio: 1.0 is a circle, a long thin shed tends to 0. A ring
    # of fixed width around a sprawling footprint holds more cells than one
    # around a compact footprint of the same area, so shape is a confounder
    # that has to be on the record alongside area.
    metrics["fp_compactness"] = (
        4 * np.pi * metrics["fp_area"] / metrics["fp_perim"] ** 2
    )
    return impact, buildings, metrics


def grid():
    """Georeference and the real depth field the validation runs on.

    The depth VALUES do not enter any statistic here -- this script counts cells
    -- but the raster is read rather than faked for two reasons: the shape and
    transform have to be the real ones, since all_touched rasterization depends
    on where the cell edges fall, and the growth loop's counting path drops
    non-finite cells, so a synthetic all-zero field would quietly hand it more
    cells than production gives it.

    Returns the finite-cell mask alongside, so the same exclusion can be applied
    when the dense count is reproduced outside the loop.
    """
    transform, raster_crs, _ = get_sample_georeference(GRID_RASTER)
    with rio.open(GRID_RASTER) as src:
        values = src.read(1).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth = torch.from_numpy(values).to(device)
    depth_valid = np.isfinite(values)

    cell_size = abs(transform.a)
    if not np.isclose(cell_size, abs(transform.e), rtol=1e-3):
        raise ValueError(f"Non-square pixels on {GRID_RASTER}")
    return depth, depth_valid, transform, raster_crs, cell_size


def count_cells(geometry, crs, shape, transform, raster_crs):
    """Cells per ring, counted the way production counts them.

    The sparse overlap-preserving path, then a bincount over the feature index
    -- exactly what _gpu_zonal_stats_sparse does to produce num_ring_cells. Two
    properties of it matter for reading these numbers: all_touched=True, so a
    ring gets every cell it clips rather than every cell it covers, and counts
    are geometric, so cells outside the model domain are included as dry.
    """
    rings = gpd.GeoDataFrame(geometry=geometry, crs=crs)
    _, _, feat_idx, n = _get_feature_labels(
        rings, shape, transform, raster_crs, allow_overlap=True,
    )
    return np.bincount(feat_idx, minlength=n).astype(int)


def count_cells_dense(geometry, crs, shape, transform, raster_crs, depth_valid):
    """Cells per ring the way the GROWTH LOOP counts them, for comparison.

    _build_adaptive_rings consults the dense label path, where one pixel carries
    one label, so a pixel claimed by several overlapping rings is credited only
    to whichever rasterizes last; _gpu_zonal_stats then drops non-finite cells
    as well. Neither applies to the production count. Reproduced here rather
    than described, because the gap between this and count_cells is what decides
    every buffer the rule chooses.
    """
    rings = gpd.GeoDataFrame(geometry=geometry, crs=crs)
    labels, n = _get_feature_labels(rings, shape, transform, raster_crs)
    labels = np.where(depth_valid, labels, 0)
    return np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(int)


def fixed_rings(buildings, all_buildings_geom, distance):
    """A non-adaptive ring at one fixed buffer distance, in metres.

    The same two operations buffered_flood_depth performs when
    use_adaptive_ring is off, so the counterfactual is the real alternative
    rather than an idealised annulus.
    """
    return buildings.geometry.buffer(distance).difference(all_buildings_geom)


def adaptive_on_true_counts(buildings, all_buildings_geom, shape, transform,
                            raster_crs, cell_size):
    """The adaptive rule as intended: same growth loop, production cell counts.

    _build_adaptive_rings is the same algorithm driven by the dense count, which
    saturates -- widening a ring adds cells but also adds overlap, and under
    dense rasterization the two roughly cancel, so for many buildings growing
    the buffer does not move the number the loop is testing and it runs to the
    cap. Rerunning the identical loop against the sparse count separates the
    rule from that artefact, which is what lets the manuscript say whether the
    design is sound and the implementation wrong, or the design itself is wrong.
    """
    n = len(buildings)
    buffer_pct = np.full(n, BASE_BUFFER_PCT, dtype=float)
    needs_growth = np.ones(n, dtype=bool)
    final_counts = np.zeros(n, dtype=int)
    final_buffer_pct = buffer_pct.copy()
    final_geom = None

    while needs_growth.any():
        candidate = (buildings.geometry.buffer(buffer_pct * cell_size)
                     .difference(all_buildings_geom))
        counts = count_cells(candidate, buildings.crs, shape, transform, raster_crs)

        if final_geom is None:
            final_geom, final_counts = candidate, counts
        else:
            final_geom = final_geom.where(~needs_growth, candidate)
            final_counts = np.where(needs_growth, counts, final_counts)
        final_buffer_pct = np.where(needs_growth, buffer_pct, final_buffer_pct)

        maxed_out = buffer_pct >= MAX_BUFFER_PCT
        needs_growth = (final_counts < MIN_RING_CELLS) & (~maxed_out)
        buffer_pct = np.where(needs_growth, buffer_pct + GROWTH_STEP_PCT, buffer_pct)

    exhausted = (final_counts < MIN_RING_CELLS) & (final_buffer_pct >= MAX_BUFFER_PCT)
    return final_geom, final_counts, final_buffer_pct, exhausted


def measure(impact, buildings, metrics, depth, depth_valid, transform,
            raster_crs, cell_size):
    """Per-building ring geometry: adaptive, corrected, and every fixed ring."""
    shape = tuple(depth.shape)
    crs = buildings.crs
    all_buildings_geom = unary_union(buildings.geometry.values)

    ring_geom, dense_counts, exhausted = impact._build_adaptive_rings(
        buildings, depth, transform, raster_crs, cell_size,
        base_buffer_pct=BASE_BUFFER_PCT,
    )
    buffer_pct = impact._ring_buffer_pct

    df = metrics.copy()
    df["ring_area"] = ring_geom.area.values
    df["num_ring_cells"] = count_cells(ring_geom, crs, shape, transform, raster_crs)
    # What the growth loop itself saw. It counts through the DENSE label path,
    # which differs from the production sparse count two ways: overlapping rings
    # collapse to whichever rasterizes last, and cells outside the model domain
    # are dropped as non-finite. Both make it an undercount, so some rings are
    # grown wider than the floor actually required. Carried per building so the
    # size of that effect is measurable rather than assumed.
    df["num_ring_cells_dense"] = dense_counts.astype(int)
    df["ring_buffer_pct"] = buffer_pct
    df["ring_buffer_m"] = buffer_pct * cell_size
    df["exhausted"] = exhausted

    # The ring a building would have had with no neighbours: the full annulus at
    # its own final width. What it actually got is that minus the footprints cut
    # out of it, so the shortfall measures how much of the sampling window the
    # neighbours consumed.
    widened = buildings.geometry.buffer(df["ring_buffer_m"].values)
    df["annulus_area_ideal"] = (widened.area - buildings.geometry.area).values
    df["occlusion_frac"] = np.where(
        df["annulus_area_ideal"] > 0,
        1 - df["ring_area"] / df["annulus_area_ideal"],
        np.nan,
    )

    for pct in FIXED_BUFFER_PCTS:
        geometry = fixed_rings(buildings, all_buildings_geom, pct * cell_size)
        df[f"cells_fixed_{pct:g}"] = count_cells(
            geometry, crs, shape, transform, raster_crs)
        # The same fixed ring counted the growth loop's way. Comparing the two
        # across widths is what shows the dense count saturating: the sparse
        # count rises with the buffer, the dense one does not.
        df[f"cells_fixed_{pct:g}_dense"] = count_cells_dense(
            geometry, crs, shape, transform, raster_crs, depth_valid)
        df[f"ring_area_fixed_{pct:g}"] = geometry.area.values

    corrected_geom, corrected_counts, corrected_pct, corrected_exhausted = (
        adaptive_on_true_counts(buildings, all_buildings_geom, shape, transform,
                                raster_crs, cell_size)
    )
    df["cells_corrected"] = corrected_counts
    df["corrected_buffer_m"] = corrected_pct * cell_size
    df["corrected_exhausted"] = corrected_exhausted
    df["ring_area_corrected"] = corrected_geom.area.values

    return df


def counterfactual(df, cell_size):
    """Each buffer setting scored on the two things the floor exists for.

    Below-floor fraction is whether the rule delivers a usable sample at all;
    the correlation with footprint area is whether the sample size it delivers
    still depends on how big the building is. A setting has to win both -- a
    fixed 5-cell buffer clears the floor by sampling 24 m of a neighbourhood
    that may have nothing to do with the building.
    """
    rows = []
    settings = [(f"fixed_{pct:g}", f"cells_fixed_{pct:g}", pct * cell_size)
                for pct in FIXED_BUFFER_PCTS]
    settings.append(("adaptive", "num_ring_cells", np.nan))
    settings.append(("adaptive_corrected", "cells_corrected", np.nan))

    widths = {"adaptive": "ring_buffer_m", "adaptive_corrected": "corrected_buffer_m"}
    areas = {"adaptive": "ring_area", "adaptive_corrected": "ring_area_corrected"}

    for name, column, width in settings:
        cells = df[column]
        rows.append({
            "setting": name,
            "buffer_m": width if np.isfinite(width) else df[widths[name]].median(),
            "buffer_is_fixed": np.isfinite(width),
            "median_cells": cells.median(),
            "p05_cells": cells.quantile(0.05),
            "p95_cells": cells.quantile(0.95),
            "frac_below_floor": (cells < MIN_RING_CELLS).mean(),
            "n_below_floor": int((cells < MIN_RING_CELLS).sum()),
            "spearman_fp_area": stats.spearmanr(cells, df["fp_area"]).statistic,
            "pearson_fp_area": stats.pearsonr(cells, df["fp_area"]).statistic,
            "spearman_fp_perim": stats.spearmanr(cells, df["fp_perim"]).statistic,
            "median_ring_area": df[areas.get(name, f"ring_area_{name}")].median(),
        })
    return pd.DataFrame(rows)


def saturation(df, cell_size):
    """Why the growth loop cannot converge, as a table.

    One row per fixed buffer, with the true cell count beside the count the
    growth loop would have seen at that same width. A rule that grows a ring
    until it holds enough cells only terminates if the count it consults RISES
    with the buffer; if the dense median is flat across widths while the sparse
    median climbs, growth is futile and every unsatisfied building runs to the
    cap. This table is the evidence for that claim.
    """
    rows = []
    for pct in FIXED_BUFFER_PCTS:
        sparse, dense = df[f"cells_fixed_{pct:g}"], df[f"cells_fixed_{pct:g}_dense"]
        rows.append({
            "buffer_pct": pct,
            "buffer_m": pct * cell_size,
            "median_cells_true": sparse.median(),
            "median_cells_growth_loop": dense.median(),
            "frac_below_floor_true": (sparse < MIN_RING_CELLS).mean(),
            "frac_below_floor_growth_loop": (dense < MIN_RING_CELLS).mean(),
            "median_cells_lost_to_overlap": (sparse - dense).median(),
        })
    return pd.DataFrame(rows)


def summary(df):
    """Ring metrics by footprint-area decile, long form, ready to be a table.

    Deciles rather than a fitted line: the relationship is not expected to be
    monotonic -- the adaptive rule should flatten it -- and a slope near zero
    from a regression cannot distinguish "flat" from "U-shaped".
    """
    binned = df.copy()
    binned["fp_area_decile"] = pd.qcut(binned["fp_area"], N_DECILES, labels=False) + 1

    columns = ["num_ring_cells", "cells_fixed_2", "cells_corrected",
               "ring_buffer_m", "corrected_buffer_m", "occlusion_frac",
               "fp_area", "fp_compactness"]
    rows = []
    for column in columns:
        described = (binned.groupby("fp_area_decile")[column]
                     .describe(percentiles=PERCENTILES))
        described.insert(0, "statistic", column)
        rows.append(described.reset_index())

    exhausted_rate = (binned.groupby("fp_area_decile")["exhausted"]
                      .mean().rename("50%").to_frame())
    exhausted_rate.insert(0, "statistic", "exhausted_rate")
    rows.append(exhausted_rate.reset_index())

    return pd.concat(rows, ignore_index=True)


def main():
    impact, buildings, metrics = population()
    depth, depth_valid, transform, raster_crs, cell_size = grid()
    print(f"{len(buildings)} modelled buildings on a "
          f"{tuple(depth.shape)} grid at {cell_size:.1f} m")

    df = measure(impact, buildings, metrics, depth, depth_valid, transform,
                 raster_crs, cell_size)

    table = counterfactual(df, cell_size)
    print("\nsample size by buffer rule "
          f"(floor = {MIN_RING_CELLS} cells, cap = {MAX_BUFFER_PCT:g} cells, "
          f"step = {GROWTH_STEP_PCT:g}):")
    print(table[["setting", "buffer_m", "median_cells", "frac_below_floor",
                 "spearman_fp_area"]].round(3).to_string(index=False))

    adaptive = table[table["setting"] == "adaptive"].iloc[0]
    fixed_base = table[table["setting"] == f"fixed_{BASE_BUFFER_PCT:g}"].iloc[0]
    print(f"\nthe result the ring design rests on: a fixed buffer makes sample "
          f"size a function of building size "
          f"(rho={fixed_base['spearman_fp_area']:.2f} at the configured "
          f"{BASE_BUFFER_PCT:g} cells / {BASE_BUFFER_PCT * cell_size:.1f} m), "
          f"the adaptive rule removes that dependence "
          f"(rho={adaptive['spearman_fp_area']:.2f})")
    print(f"  the floor itself is not the binding constraint: a fixed "
          f"{BASE_BUFFER_PCT:g}-cell ring already leaves only "
          f"{fixed_base['frac_below_floor']:.1%} of buildings below "
          f"{MIN_RING_CELLS} cells")

    sat = saturation(df, cell_size)
    print("\nwhy the growth loop cannot converge -- true count against the "
          "count the loop consults, at the same widths:")
    print(sat.round(3).to_string(index=False))

    grown = df["ring_buffer_pct"] > BASE_BUFFER_PCT
    print(f"\n{grown.mean():.1%} of rings were grown past the base buffer; "
          f"median final width {df['ring_buffer_m'].median():.1f} m, "
          f"max {df['ring_buffer_m'].max():.1f} m")
    print(f"  median occlusion by neighbouring footprints "
          f"{df['occlusion_frac'].median():.1%} of the nominal annulus, "
          f"{(df['occlusion_frac'] > 0.5).mean():.1%} of buildings above 50%")

    exhausted = df[df["exhausted"]]
    print(f"\nflagged exhausted (short of {MIN_RING_CELLS} cells at the "
          f"{MAX_BUFFER_PCT:g}-cell cap): {len(exhausted)} buildings "
          f"({len(exhausted) / len(df):.2%})")
    if len(exhausted):
        print(f"  median footprint {exhausted['fp_area'].median():.0f} m2 "
              f"vs {df['fp_area'].median():.0f} m2 overall; "
              f"median occlusion {exhausted['occlusion_frac'].median():.1%} "
              f"vs {df['occlusion_frac'].median():.1%}")
        print(f"  but their rings are NOT short: median "
              f"{exhausted['num_ring_cells'].median():.0f} cells in production "
              f"(the growth loop saw {exhausted['num_ring_cells_dense'].median():.0f}); "
              f"{int((exhausted['num_ring_cells'] < MIN_RING_CELLS).sum())} of them "
              f"are genuinely below the floor")

    undercounted = df["num_ring_cells_dense"] < df["num_ring_cells"]
    shortfall = (df["num_ring_cells"] - df["num_ring_cells_dense"])[undercounted]
    print(f"\ngrowth loop undercounted {undercounted.mean():.1%} of rings "
          f"against the production count "
          f"(median shortfall {shortfall.median():.0f} cells, "
          f"max {shortfall.max():.0f})")

    corrected = table[table["setting"] == "adaptive_corrected"].iloc[0]
    print(f"\nsame rule driven by the true count: median "
          f"{corrected['median_cells']:.0f} cells, "
          f"{corrected['frac_below_floor']:.1%} below floor, "
          f"rho={corrected['spearman_fp_area']:.2f}, median width "
          f"{df['corrected_buffer_m'].median():.1f} m "
          f"(vs {df['ring_buffer_m'].median():.1f} m as implemented); "
          f"{int(df['corrected_exhausted'].sum())} exhausted "
          f"(vs {int(df['exhausted'].sum())})")
    print(f"  rings that would stay at the base buffer: "
          f"{(df['corrected_buffer_m'] <= BASE_BUFFER_PCT * cell_size).mean():.1%} "
          f"(vs {(~grown).mean():.1%} as implemented)")

    below = df["num_ring_cells"] < MIN_RING_CELLS
    print(f"\ninvariant check: {int((below & ~df['exhausted']).sum())} "
          f"non-exhausted buildings below the floor (expected 0)")

    summarised = summary(df)
    table.to_csv(f"{OUT}/ring_footprint_counterfactual.csv", index=False)
    sat.to_csv(f"{OUT}/ring_footprint_saturation.csv", index=False)
    summarised.to_csv(f"{OUT}/ring_footprint_summary.csv", index=False)
    df.to_csv(f"{OUT}/ring_footprint_buildings.csv", index=False)

    print(f"\nwrote {OUT}/ring_footprint_buildings.csv ({len(df)} rows)")
    print(f"wrote {OUT}/ring_footprint_summary.csv")
    print(f"wrote {OUT}/ring_footprint_counterfactual.csv")
    print(f"wrote {OUT}/ring_footprint_saturation.csv")


if __name__ == "__main__":
    main()
