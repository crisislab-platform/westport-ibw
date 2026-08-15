"""Where the hazard classification goes wrong, spatially, at chosen timesteps.

For each (event, timestep) the surrogate and the simulated map are pushed
through the SAME exposure and inference chain, and every building is coloured by
its SIGNED tier error, predicted minus reference. The sign matters more than the
magnitude: over-prediction is the safe failure and under-prediction is not, so
they are never collapsed into one "wrong" colour.

Three timesteps are chosen to sit at different points of an event, since the
per-timestep verification shows the failure mode is phase-dependent rather than
uniform: early in a coastal event, mid-way through a tidal one, and after the
recession of a fluvial one.

    python exploratory/map_error_hotspots.py

Writes out/error_hotspots.png and out/error_hotspots.csv.
"""
import os
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio as rio
import torch

from modules.bc_forecast_module.bc_module import BC_Module
from modules.impact_module.impact_module import ImpactModule, WET_DEPTH_MIN
from modules.hazard_forecast_module.model_client import FlowMatchingModelClient
from modules.utils.spatial_operators.operators import load_flood_map

logging.basicConfig(level=logging.WARNING)

OUT_PNG = f"{PROJECT_ROOT}/out/error_hotspots.png"
OUT_CSV = f"{PROJECT_ROOT}/out/error_hotspots.csv"

TIER_COLUMN = "overall_building_hazard_level"
TERM_LABEL = {1: "H1", 2: "H2-H3", 3: "H4+"}

# Signed tier error -> colour and label. Diverging about "correct", so the two
# failure directions read as opposites rather than as degrees of one thing.
ERROR_STYLE = {
    -2: ("#67000d", "Under by 2"),
    -1: ("#fb6a4a", "Under by 1"),
     0: ("#9e9e9e", "Correct"),
     1: ("#6baed6", "Over by 1"),
     2: ("#08306b", "Over by 2"),
}

# (scenario, position through the event, label). Position is a fraction so the
# choice is describable rather than an index that means nothing to a reader.
PANELS = [
    (17, 0.20, "Event 17 (coastal)\nearly"),
    (57, 0.50, "Event 57 (tidal)\nmid-event"),
    (83, 0.85, "Event 83 (fluvial)\nrecession"),
]

M, K = 10, 10


def building_tiers(hazard_module, impact_module, bc_module, timestamp):
    """Predicted and reference tier per building at one timestep."""
    bc = {timestamp: bc_module.bc_dict[timestamp]}
    _, hazard = next(iter(hazard_module.iter_hazard_forecast(bc)))
    pred = next(iter(impact_module.get_impact_forecast({timestamp: hazard})
                     ["building_impact"].values()))

    gt_map = load_flood_map(bc_module.flood_map_dict[timestamp], bc_module.event_name)
    gt = next(iter(impact_module.get_impact_forecast(
        {str(timestamp): [gt_map]})["building_impact"].values()))

    # The ensemble's own confidence travels with the tier, so the saved frame
    # can answer whether a wrong call was a confident one without a second
    # model run. The per-tier probabilities come too, for the same reason --
    # they are free here and expensive to recover later.
    carried = ([TIER_COLUMN, "depth_max", "overall_building_hazard_likelihood"]
               + [c for c in pred.columns
                  if c.startswith("overall_building_hazard_prob_")])
    joined = gt[["building_id", "geometry", TIER_COLUMN, "depth_max"]].merge(
        pred[["building_id"] + carried],
        on="building_id", suffixes=("_gt", "_pred"),
    )
    joined = joined[joined["num_ring_cells"].gt(0)] if "num_ring_cells" in joined else joined
    joined["tier_error"] = (joined[f"{TIER_COLUMN}_pred"]
                            - joined[f"{TIER_COLUMN}_gt"]).astype(int)
    return joined, gt_map


def main():
    hazard_module = FlowMatchingModelClient(num_members=M, samples_per_member=K)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.2 * len(PANELS), 7.2))
    rows = []

    for ax, (scenario, position, title) in zip(np.atleast_1d(axes), PANELS):
        bc_module = BC_Module(scenario)
        impact_module = ImpactModule(scenario)
        steps = sorted(bc_module.bc_dict)
        timestamp = steps[int(position * (len(steps) - 1))]

        joined, gt_map = building_tiers(hazard_module, impact_module, bc_module, timestamp)
        joined["scenario_id"] = scenario
        joined["timestamp"] = timestamp
        rows.append(joined.drop(columns="geometry"))

        # The reference flood field behind the buildings, in GREY. Water was
        # drawn in Blues first, which collided with the cool half of the
        # diverging error scale -- an over-predicted building and deep water
        # were the same colour. Context has no business competing with the
        # variable the figure is about.
        ax.imshow(np.where(gt_map > WET_DEPTH_MIN, gt_map, np.nan),
                  cmap="Greys", vmin=0, vmax=2.5, alpha=0.30,
                  extent=_extent(impact_module), origin="upper", zorder=0)

        # Drawn as CENTROIDS, not footprints. The exposed buildings span 99%
        # of the domain width, so there is nothing to zoom into, and a footprint
        # at that scale is well under a pixel -- polygons rendered as invisible
        # specks whatever their colour. A dot map trades exact shape, which is
        # not what this figure is about, for a mark that can actually be seen
        # and sized by importance.
        points = joined.set_geometry(joined.geometry.centroid)

        # Correct first and small, errors on top and larger, so a hotspot is not
        # buried under the 4,000-odd buildings that were right.
        for error, size in ((0, 1.5), (1, 7), (-1, 7), (2, 14), (-2, 14)):
            subset = points[points.tier_error == error]
            if subset.empty:
                continue
            colour, _ = ERROR_STYLE[error]
            subset.plot(ax=ax, color=colour, markersize=size, linewidth=0,
                        alpha=0.55 if error == 0 else 0.9,
                        zorder=1 if error == 0 else 2 + abs(error))

        wrong = int((joined.tier_error != 0).sum())
        under = int((joined.tier_error < 0).sum())
        ax.set_title(f"{title}\n{timestamp:%d %b %H:%M}", fontsize=15, fontweight="bold")
        ax.set_xlabel(f"{wrong:,} of {len(joined):,} misclassified "
                      f"({under:,} under-predicted)", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    handles = [Line2D([], [], marker="o", ls="", markersize=8, color=c, label=l)
               for e, (c, l) in sorted(ERROR_STYLE.items())]
    axes[0].legend(handles=handles, loc="lower left", fontsize=11,
                   title="Tier error\n(predicted - reference)", title_fontsize=11,
                   framealpha=0.92)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    pd.concat(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


def _extent(impact_module):
    """Map extent of the grid the buildings were filtered against.

    domain_raster is None on a forecast run -- load_features then falls back to
    find_sample_flood_map(), the model grid -- so resolve it the same way rather
    than reading the attribute, which would be None here.
    """
    from modules.utils.spatial_operators.operators import find_sample_flood_map
    with rio.open(impact_module.domain_raster or find_sample_flood_map()) as src:
        bounds = src.bounds
    return (bounds.left, bounds.right, bounds.bottom, bounds.top)


if __name__ == "__main__":
    main()
