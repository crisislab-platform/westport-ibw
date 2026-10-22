"""Regenerate out/binary_validation_metrics.csv under the current FIS code.

The published CSV (2026-08-13) no longer reproduces: its depth-thresholded rows
still match a rerun, but both FIS-tier rows do not, because the fuzzy engine moved
after it was written. Rather than hand-editing it, this reruns the same scoring path
that produced it -- Integrator.test_against_placard_data -- so the file keeps its
provenance and column set.

Two things make that path awkward to call directly, both handled here:

* Integrator.__init__ constructs the boundary-condition module and the
  flow-matching client. Neither is used by the placard scoring, and loading them
  costs a model init, so the runner is built with __new__ and given only the two
  attributes write_results actually reads.
* self.buffer_m is read by write_results but assigned nowhere in modules/, so a
  real run today would raise AttributeError. It is set here from
  ImpactModule.buffered_flood_depth's buffer_pct default, which is the value the
  published rows recorded as buffer_multiplier, rather than restated as a literal.

write_results APPENDS, so the existing CSV is archived first -- rerunning without
that would stack a second set of rows under the same header.
"""
import inspect
import os
import shutil
import sys
import time

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import rasterio as rio

from modules.lib import EXPOSURE_STUDY_RESULTS, SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.impact_module.risk_module import ImpactModule
from modules.forecast_runner.forecast_runner import Integrator

METRICS_FILE = os.path.join(EXPOSURE_STUDY_RESULTS, "binary_validation_metrics.csv")
SCENARIO_ID = 0


def buffer_multiplier():
    """The buffer the scoring ran at, read off the signature rather than restated."""
    return inspect.signature(
        ImpactModule.buffered_flood_depth).parameters["buffer_pct"].default


def archive_existing():
    if not os.path.exists(METRICS_FILE):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archived = f"{os.path.splitext(METRICS_FILE)[0]}_{stamp}.csv.bak"
    shutil.move(METRICS_FILE, archived)
    print(f"archived previous metrics to {archived}")
    return archived


def main():
    archived = archive_existing()

    with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
        surveyed_depth = src.read(1).astype(float)

    # domain_raster is what Integrator.validate_exposure_method passes, and it
    # DEFINES THE SCORED POPULATION: the surveyed raster covers different ground
    # from the flood model's grid, so omitting it here scored 2244 buildings
    # where the pipeline scores 2247. Same misses and false alarms either way,
    # but different hits, POD and CSI -- so the two disagreed on the published
    # numbers while both looked self-consistent.
    impact_forecast = ImpactModule(
        "binary_validation_regen",
        domain_raster=SURVEYED_FLOOD_DEPTH_JUL_2021).get_impact_forecast(
            {"17-07-2021 00:00:00": [surveyed_depth]},
            rasterfile=SURVEYED_FLOOD_DEPTH_JUL_2021,
        )

    # Only the attributes the placard scoring path reads; see the module docstring.
    runner = Integrator.__new__(Integrator)
    runner.scenario_id = SCENARIO_ID
    runner.buffer_m = buffer_multiplier()

    runner.test_against_placard_data(impact_forecast, model="surveyed")

    fresh = pd.read_csv(METRICS_FILE)
    columns = ["indicator", "num_buildings_evaluated", "hits", "misses",
               "false_alarms", "correct_negatives", "pod", "csi"]
    print("\nregenerated:")
    print(fresh[columns].to_string(index=False))

    if archived:
        # Keep the LAST row per indicator. write_results appends, so a file
        # that a pipeline run has also written to holds one set of rows per
        # run; without this, .loc returns a Series per indicator and the
        # int() below raises. Last wins because it is the most recent scoring.
        previous = pd.read_csv(archived).drop_duplicates(
            subset="indicator", keep="last").set_index("indicator")
        rows = []
        for _, row in fresh.iterrows():
            indicator = row["indicator"]
            if indicator not in previous.index:
                rows.append({"indicator": indicator, "status": "new"})
                continue
            before = previous.loc[indicator]
            rows.append({
                "indicator": indicator,
                "misses": f"{int(before['misses'])} -> {int(row['misses'])}",
                "false_alarms": (f"{int(before['false_alarms'])} -> "
                                 f"{int(row['false_alarms'])}"),
                "n": (f"{int(before['num_buildings_evaluated'])} -> "
                      f"{int(row['num_buildings_evaluated'])}"),
                "csi": f"{before['csi']:.3f} -> {row['csi']:.3f}",
            })
        print("\nchange against the archived file:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
