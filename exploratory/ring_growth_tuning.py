"""What the adaptive ring growth loop settles on, per building, for tuning.

_build_adaptive_rings grows each building's buffer one step at a time until its
ring holds MIN_RING_CELLS, capped at MAX_BUFFER_PCT. This runs that loop outside
ImpactModule so (min_cells, max_buffer_pct, step_pct) can be swept without
touching the module, and reports the two numbers the tuning turns on: how many
buildings come out EXHAUSTED (still short of the floor at the cap, which nothing
downstream handles) and the spread of ring cell counts.

Two cell counts are reported per building, and they are not the same number:

  ring_cells_dense  what the growth loop consults -- the dense label path, where
                    a pixel claimed by several overlapping rings is credited to
                    whichever rasterizes last, minus non-finite cells.
  ring_cells        what production actually samples -- the sparse
                    overlap-preserving path _gpu_zonal_stats_sparse uses, where
                    a shared pixel counts for every ring covering it.

The loop decides on the first and the model sees the second, so a building can
be flagged exhausted while holding plenty of cells downstream. Both are written
per building so the gap is measurable rather than assumed.

Usage, from the shell:
  python exploratory/ring_growth_tuning.py
  python exploratory/ring_growth_tuning.py --min-cells 3,4,6 --max-buffer 3,5

or from a notebook cell, where the grid and building layer stay loaded between
calls so a sweep can be repeated interactively:
  from ring_growth_tuning import sweep
  frames, summary = sweep(min_cells=[3, 4, 6], max_buffer_pct=[3, 5])
  frames[(4.0, 3.0)].query("exhausted")

Writes, under out/:
  ring_growth_buildings_m<min>_b<max>_s<step>.csv   per-building record
  ring_growth_summary.csv                           one row per swept config
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from modules.impact_module.risk_module import (
    GROWTH_STEP_PCT, MAX_BUFFER_PCT, MIN_RING_CELLS, ImpactModule,
)
from modules.utils.spatial_operators.operators import (
    _get_feature_labels, get_sample_georeference,
)

OUT = f"{PROJECT_ROOT}/out"

# The grid the rings are rasterized against. The notebook's sample map, so these
# numbers line up with what ring_stats.ipynb shows. Only the shape, transform
# and finite-cell mask matter -- the loop counts cells, it does not read depths,
# but the dense count drops non-finite cells so the nodata pattern does enter.
DEFAULT_RASTER = (
    "/data/westport/data/flood_maps/Base0_Buller20yrARI_v2_depth_snapshots/"
    "depth_at_2000-01-02_00-00-00.tif"
)

# Buffer the loop starts every building at, in cells.
BASE_BUFFER_PCT = 1.0

PERCENTILES = (5, 25, 50, 75, 95)


def population():
    """The modelled buildings, and the geometry cut out of their rings.

    Loaded through ImpactModule rather than by reading buildings.gpkg directly:
    __init__ applies both filters that define "modelled" -- load_features drops
    footprints outside the flood-model domain, _load_building_links drops
    anything absent from the link table. Reading the layer raw would add
    buildings that sit on nothing but nodata, which count zero cells at every
    buffer and would pad the exhausted tally with a population artefact.

    The exclusion geometry comes from the module too (_footprint_union), which
    is the FULL layer including those unmodelled footprints -- a ring must not
    hold a roof just because the building it belongs to is not being scored.
    """
    impact = ImpactModule("ring_growth_tuning")
    return impact.buildings.reset_index(drop=True), impact._footprint_union()


def grid(raster):
    """Georeference plus the finite-cell mask the dense count is subject to."""
    transform, raster_crs, _ = get_sample_georeference(raster)
    with rio.open(raster) as src:
        values = src.read(1).astype(np.float32)

    cell_size = abs(transform.a)
    if not np.isclose(cell_size, abs(transform.e), rtol=1e-3):
        raise ValueError(f"Non-square pixels on {raster}")

    return np.isfinite(values), values.shape, transform, raster_crs, cell_size


def count_dense(geometry, crs, shape, transform, raster_crs, valid):
    """Cells per ring the way the GROWTH LOOP counts them.

    _build_adaptive_rings rasterizes through the dense label path and then hands
    the labels to _gpu_zonal_stats, which drops non-finite cells. Reproduced
    with numpy rather than by calling the module so it runs without a GPU; the
    arithmetic is a bincount either way.
    """
    rings = gpd.GeoDataFrame(geometry=geometry, crs=crs)
    labels, n = _get_feature_labels(rings, shape, transform, raster_crs)
    labels = np.where(valid, labels, 0)
    return np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(int)


def count_sparse(geometry, crs, shape, transform, raster_crs):
    """Cells per ring the way PRODUCTION counts them.

    The overlap-preserving path, then a bincount over the feature index --
    exactly what _gpu_zonal_stats_sparse does to produce num_ring_cells.
    """
    rings = gpd.GeoDataFrame(geometry=geometry, crs=crs)
    _, _, feat_idx, n = _get_feature_labels(
        rings, shape, transform, raster_crs, allow_overlap=True,
    )
    return np.bincount(feat_idx, minlength=n).astype(int)


def grow(buildings, all_buildings_geom, shape, transform, raster_crs, valid,
         cell_size, min_cells, max_buffer_pct, step_pct, cache):
    """_build_adaptive_rings, instrumented, driven by the dense count it uses.

    Driven by the sparse count, as the module now is: rings are allowed to
    overlap, so a cell of water between two buildings counts for both.

    Faithful to the module including the mixed-buffer rasterization: at each
    pass the candidate set holds every building at ITS current buffer, so the
    ring geometry a building is measured on depends on how wide its neighbours
    have grown. That coupling is why the pass cannot be replaced by a per-width
    lookup.

    `cache` is keyed on the exact buffer state, which is what makes a sweep
    affordable -- configs sharing a base and step share their opening passes.
    """
    n = len(buildings)
    buffer_pct = np.full(n, BASE_BUFFER_PCT, dtype=float)
    needs_growth = np.ones(n, dtype=bool)

    final_geom = None
    final_counts = None
    final_buffer_pct = buffer_pct.copy()
    history = []

    while needs_growth.any():
        key = buffer_pct.tobytes()
        if key not in cache:
            candidate = (buildings.geometry.buffer(buffer_pct * cell_size)
                         .difference(all_buildings_geom))
            cache[key] = (
                candidate,
                count_sparse(candidate, buildings.crs, shape, transform,
                             raster_crs),
            )
        candidate, counts = cache[key]

        if final_geom is None:
            final_geom, final_counts = candidate, counts
        else:
            final_geom = final_geom.where(~needs_growth, candidate)
            final_counts = np.where(needs_growth, counts, final_counts)
        final_buffer_pct = np.where(needs_growth, buffer_pct, final_buffer_pct)

        history.append({
            "buffer_pct": float(buffer_pct[needs_growth].max()),
            "n_growing": int(needs_growth.sum()),
            "median_cells": float(np.median(final_counts[needs_growth])),
        })

        maxed_out = buffer_pct >= max_buffer_pct
        needs_growth = (final_counts < min_cells) & (~maxed_out)
        buffer_pct = np.where(needs_growth, buffer_pct + step_pct, buffer_pct)

    exhausted = (final_counts < min_cells) & (final_buffer_pct >= max_buffer_pct)
    return final_geom, final_counts, final_buffer_pct, exhausted, history


def describe(values, prefix):
    """mean/min/max plus the percentile spread, as flat summary fields."""
    out = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": int(np.min(values)),
        f"{prefix}_max": int(np.max(values)),
    }
    for p in PERCENTILES:
        out[f"{prefix}_p{p:02d}"] = float(np.percentile(values, p))
    return out


def run(buildings, all_buildings_geom, shape, transform, raster_crs, valid,
        cell_size, min_cells, max_buffer_pct, step_pct, cache, write=True,
        verbose=True):
    """One configuration: grow, measure, write the per-building record."""
    geom, sparse, buffer_pct, exhausted, history = grow(
        buildings, all_buildings_geom, shape, transform, raster_crs, valid,
        cell_size, min_cells, max_buffer_pct, step_pct, cache,
    )
    # The count the loop no longer uses, kept for one turn longer because it is
    # the number the old behaviour was decided on and the gap is the fix.
    dense = count_dense(geom, buildings.crs, shape, transform, raster_crs, valid)

    df = pd.DataFrame({
        "building_id": buildings["building_id"].values,
        "fp_area": buildings.geometry.area.values,
        "ring_buffer_pct": buffer_pct,
        "ring_buffer_m": buffer_pct * cell_size,
        "ring_area": geom.area.values,
        "ring_cells_dense": dense,
        "ring_cells": sparse,
        "exhausted": exhausted,
    })

    tag = f"m{min_cells:g}_b{max_buffer_pct:g}_s{step_pct:g}"
    path = f"{OUT}/ring_growth_buildings_{tag}.csv"
    if write:
        df.to_csv(path, index=False)

    summary = {
        "min_cells": min_cells,
        "max_buffer_pct": max_buffer_pct,
        "step_pct": step_pct,
        "n_buildings": len(df),
        "n_exhausted": int(exhausted.sum()),
        "pct_exhausted": 100.0 * exhausted.mean(),
        # What the loop USED to be decided on, for comparison: rings the dense
        # path credits with nothing, which under the old counting ran to the cap.
        "n_zero_cells_dense": int((dense == 0).sum()),
        "n_below_floor_production": int((sparse < min_cells).sum()),
        "n_at_cap": int((buffer_pct >= max_buffer_pct).sum()),
        "passes": len(history),
    }
    summary.update(describe(dense, "dense"))
    summary.update(describe(sparse, "sparse"))

    if not verbose:
        return df, summary

    print(f"\n=== min_cells={min_cells:g}  max_buffer_pct={max_buffer_pct:g}  "
          f"step_pct={step_pct:g} ===")
    print(f"buildings              {summary['n_buildings']}")
    print(f"exhausted              {summary['n_exhausted']} "
          f"({summary['pct_exhausted']:.1f}%)")
    print(f"  of which 0 cells     {summary['n_zero_cells_dense']}")
    print(f"grew to the cap        {summary['n_at_cap']}")
    print("ring cells, loop's view (dense): "
          f"mean {summary['dense_mean']:.1f}  min {summary['dense_min']}  "
          f"max {summary['dense_max']}  median {summary['dense_p50']:.0f}")
    print("ring cells, production (sparse): "
          f"mean {summary['sparse_mean']:.1f}  min {summary['sparse_min']}  "
          f"max {summary['sparse_max']}  median {summary['sparse_p50']:.0f}")
    print(f"  still under the floor downstream: "
          f"{summary['n_below_floor_production']}")

    print("buffer the loop settled on (cells):")
    for value, count in df["ring_buffer_pct"].value_counts().sort_index().items():
        print(f"  {value:>4.1f}  {count:>6d}  ({100.0 * count / len(df):.1f}%)")

    print("growth passes:")
    for i, h in enumerate(history, 1):
        print(f"  pass {i}: {h['n_growing']:>6d} growing at buffer "
              f"{h['buffer_pct']:.1f}, median cells {h['median_cells']:.0f}")
    if write:
        print(f"per-building record -> {path}")

    return df, summary


def parse_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


# Loading the grid and the building layer costs a few seconds and neither
# depends on the parameters being tuned, so a notebook re-running sweep() with
# different floors pays for them once. Keyed on the raster, since that is the
# only thing that would invalidate them. The geometry cache rides along: a
# buffer state already rasterized under one configuration is the same rings
# under the next.
_CONTEXT = {}


def context(raster=DEFAULT_RASTER, verbose=True):
    """Grid, buildings and the shared rasterization cache for one raster."""
    if raster not in _CONTEXT:
        valid, shape, transform, raster_crs, cell_size = grid(raster)
        buildings, footprints = population()
        # Reprojected before `buildings` is, since the union comes out in the
        # module's CRS and that is what it has to be tagged with.
        footprints = (gpd.GeoSeries([footprints], crs=buildings.crs)
                      .to_crs(raster_crs).iloc[0])
        buildings = buildings.to_crs(raster_crs)
        if verbose:
            print(f"grid {shape} at {cell_size:g} m, {valid.sum()} finite cells")
            print(f"{len(buildings)} modelled buildings")
        _CONTEXT[raster] = dict(
            buildings=buildings, footprints=footprints, shape=shape,
            transform=transform, raster_crs=raster_crs, valid=valid,
            cell_size=cell_size, cache={},
        )
    return _CONTEXT[raster]


def sweep(min_cells=MIN_RING_CELLS, max_buffer_pct=MAX_BUFFER_PCT,
          step_pct=GROWTH_STEP_PCT, raster=DEFAULT_RASTER, write=True,
          verbose=True):
    """Run the growth loop over every (min_cells, max_buffer_pct) combination.

    The notebook entry point. Scalars or sequences for either parameter.

    Returns (frames, summary): `frames` maps (min_cells, max_buffer_pct) to the
    per-building record, `summary` is one row per configuration.
    """
    ctx = context(raster, verbose=verbose)
    min_cells = np.atleast_1d(min_cells).astype(float)
    max_buffer_pct = np.atleast_1d(max_buffer_pct).astype(float)

    frames = {}
    summaries = []
    for floor in min_cells:
        for cap in max_buffer_pct:
            df, summary = run(
                ctx["buildings"], ctx["footprints"], ctx["shape"],
                ctx["transform"], ctx["raster_crs"], ctx["valid"],
                ctx["cell_size"], floor, cap, step_pct, ctx["cache"],
                write=write, verbose=verbose,
            )
            frames[(floor, cap)] = df
            summaries.append(summary)

    summary = pd.DataFrame(summaries)
    if write:
        path = f"{OUT}/ring_growth_summary.csv"
        summary.to_csv(path, index=False)
        if verbose:
            print(f"\nsummary -> {path}")
    return frames, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=DEFAULT_RASTER)
    parser.add_argument("--min-cells", default=str(MIN_RING_CELLS),
                        help="comma-separated floors to sweep")
    parser.add_argument("--max-buffer", default=str(MAX_BUFFER_PCT),
                        help="comma-separated caps to sweep, in cells")
    parser.add_argument("--step", type=float, default=GROWTH_STEP_PCT)
    args = parser.parse_args()

    sweep(
        min_cells=parse_list(args.min_cells),
        max_buffer_pct=parse_list(args.max_buffer),
        step_pct=args.step,
        raster=args.raster,
    )


if __name__ == "__main__":
    main()
