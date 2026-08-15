"""Precompute the ring each building is sampled through, once, to out/.

ImpactModule.buffered_flood_depth needs a ring per building: the footprint
buffered outward with every footprint in the layer subtracted, widened where a
building would otherwise hold fewer than MIN_RING_CELLS. Building that costs
tens of seconds and it is the same geometry on every run, so this writes it to
BUILDING_RINGS_PATH and the forecast loads it.

The rings are grid-specific but NOT scenario-specific -- the growth loop counts
cells geometrically and never reads a depth -- so each raster below is used for
its shape and transform only, and one file serves every scenario and timestep on
that grid. The grid, CRS, growth parameters and building count are stamped into
each file; ImpactModule names the file after the grid and ignores one whose stamp
does not match what the run needs, rebuilding in-process instead, so a stale
cache costs time rather than correctness.

Both grids the system runs on are built by default: the model grid the forecasts
use, and the surveyed July 2021 raster the placard validation uses. They are not
interchangeable -- the surveyed raster is the smaller footprint and 86 modelled
buildings fall outside it, which no ring width can fix, so their rings differ
between the two.

Rerun after: retuning MIN_RING_CELLS / MAX_BUFFER_PCT / GROWTH_STEP_PCT, editing
the building layer or the link table, or adding a grid.

Usage:
  python scripts/build_building_rings.py
  python scripts/build_building_rings.py --raster /path/to/depth.tif
"""
import argparse
import logging
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rasterio

from modules.impact_module.impact_module import (
    GROWTH_STEP_PCT, MAX_BUFFER_PCT, MIN_RING_CELLS, ImpactModule,
)
from modules.lib import SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.utils.spatial_operators.operators import (
    find_sample_flood_map, get_sample_georeference,
)


def build(raster, args):
    """One grid: its own population, its own rings, its own file.

    A module per raster rather than one shared: the domain is decided by the
    raster, so the surveyed grid and the model grid do not even model the same
    buildings, let alone give them the same rings.
    """
    impact = ImpactModule("build_building_rings", domain_raster=raster)
    transform, raster_crs, _ = get_sample_georeference(raster)
    with rasterio.open(raster) as src:
        out_shape = src.shape

    cell_size = abs(transform.a)
    if not abs(cell_size - abs(transform.e)) < 1e-3 * cell_size:
        raise ValueError(f"Non-square pixels on {raster}")

    print(f"\ngrid {out_shape} at {cell_size:g} m from {raster}")
    print(f"  {len(impact.buildings)} buildings in this raster's domain")

    rings = impact.build_ring_cache(
        out_shape, transform, raster_crs, cell_size,
        min_cells=args.min_cells, max_buffer_pct=args.max_buffer,
        step_pct=args.step,
    )

    print(f"  {len(rings)} rings, {int(rings['exhausted'].sum())} exhausted")
    print("  buffers: " + ", ".join(
        f"{width:g} cells x{count}"
        for width, count in rings["ring_buffer_pct"].value_counts().sort_index().items()
    ))
    return rings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", nargs="+", default=None,
                        help="grids to build against; defaults to the model "
                             "grid and the surveyed July 2021 raster")
    parser.add_argument("--min-cells", type=float, default=MIN_RING_CELLS)
    parser.add_argument("--max-buffer", type=float, default=MAX_BUFFER_PCT)
    parser.add_argument("--step", type=float, default=GROWTH_STEP_PCT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rasters = args.raster or [find_sample_flood_map(), SURVEYED_FLOOD_DEPTH_JUL_2021]

    for raster in rasters:
        build(raster, args)


if __name__ == "__main__":
    main()
