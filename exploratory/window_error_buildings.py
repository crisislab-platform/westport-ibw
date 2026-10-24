"""Per-building outcomes for chosen forecast windows, which the sweep discards.

``horizon_window_metrics_*.csv`` keeps one row per issue: the counts n_hit,
n_miss and n_false. The frame those counts are computed from -- one row per
building, the class the simulated maps reach over the window against the class
the forecast reaches -- is built by ``_window_peak`` and thrown away, and a
reduction cannot be inverted. This re-runs the chosen windows and writes that
frame, so the spatial pattern behind a POD can be looked at.

The window is reproduced by driving the real Integrator for a single issue
rather than by reimplementing the horizon path, and the scored frame is taken
from ``_window_peak`` itself, so the max-over-window collapse and the
hit/miss/false-alarm rule are the runner's own. What is suppressed is only the
side effects: the per-step exceedance rows, the runtime metrics and the horizon
metrics row would otherwise append to results files that are already written,
and a reproduction has no business adding rows to them.

The ensemble is unseeded (Integrator does not pass a seed to
FlowMatchingModelClient), so the reproduced counts will not match the swept row
exactly. Both are written to the summary, side by side, so the gap is visible
rather than assumed away.

    python exploratory/window_error_buildings.py

Writes out/horizon/window_error_buildings.csv and
out/horizon/window_error_summary.csv.
"""
import os
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging

import pandas as pd

import modules.validation_module.horizon_validation as validation
from modules.forecast_runner.forecast_runner import Integrator
from modules.validation_module.validation_module import ValidationModule

logging.basicConfig(level=logging.WARNING)

HORIZON_DIR = os.path.join(PROJECT_ROOT, "out", "horizon")
OUT_CSV = os.path.join(HORIZON_DIR, "window_error_buildings.csv")
OUT_SUMMARY = os.path.join(HORIZON_DIR, "window_error_summary.csv")

# (event, horizon, issue time) -- the LOWEST-POD window of that event at that
# horizon, read off the swept metrics. The 1 h and 12 h columns of the anatomy
# figure. Event 57's two horizons happen to bottom out at the same issue.
CASES = [
    (57, 1, "2018-02-02 11:00:00"),
    (57, 12, "2018-02-02 11:00:00"),
    (83, 1, "1988-05-20 23:00:00"),
    (83, 12, "1988-05-20 19:00:00"),
]

# The sweep's ensemble: M=10 deep members x K=10 samples.
#
# MIN_EXCEED_PROB is RiskLevelFinder.min_class_probability -- the share of the
# ensemble that has to reach a class before a building is placed in it -- and it
# is a probability, not a percentage. It is recovered from the swept output
# rather than from main.py, whose --exceedence_prob defaults to 10: at 10 no
# class is ever reachable, so every building on both sides falls to H1, every
# window scores POD 1.0, and the degenerate run looks like a perfect one. In
# out/hazard_exceedance_buildprop_max.csv the lowest P at an elected class is
# 0.11 while a level below never exceeds 0.10, which places the sweep's
# threshold at 0.1.
NUM_DEEP_ENSEMBLE, SAMPLES_PER_MEMBER, MIN_EXCEED_PROB = 10, 10, 0.1


def run_window(scenario_id, forecast_horizon, issue_time):
    """The per-building scored frame for one issue, with geometry.

    Returns (frame, reproduced_metrics). The frame is what _window_peak
    produced -- observed class, forecast class, likelihood, per building --
    plus the outcome each building contributes to the counts, its tier
    shortfall, and its centroid.
    """
    issue = pd.Timestamp(issue_time)
    integrator = Integrator(
        scenario_id, forecast_horizon, False,
        samples_per_member=SAMPLES_PER_MEMBER,
        num_deep_ensemble=NUM_DEEP_ENSEMBLE,
        min_exceed_prob=MIN_EXCEED_PROB,
    )

    # Side effects off: the horizon rows append to a file the sweep already
    # wrote. The validation loop records no timings, so there is nothing else.
    validator = ValidationModule(integrator)
    captured_rows = []
    validator.write_horizon_results = captured_rows.extend

    captured_window = {}
    original_window_peak = validation._window_peak

    def spy(observed, forecast, level_column, likelihood_column):
        window = original_window_peak(observed, forecast, level_column,
                                      likelihood_column)
        captured_window["frame"] = window
        return window

    validation._window_peak = spy
    try:
        # start == end runs exactly one issue: the loop advances an hour at a
        # time while current_time <= end_time.
        validator.validate_forecast(start_time=issue, end_time=issue)
    finally:
        validation._window_peak = original_window_peak

    if "frame" not in captured_window:
        raise RuntimeError(f"no scored window for event {scenario_id} at {issue}")

    window = captured_window["frame"].copy()
    window["observed"] = window["observed"].astype(int)
    window["forecast"] = window["forecast"].astype(int)

    # The same three mutually exclusive outcomes compute_horizon_metrics counts:
    # exact agreement is a hit, an under-forecast a miss, an over-forecast a
    # false alarm. shortfall is signed, so a miss by two tiers is separable
    # from a miss by one.
    window["shortfall"] = window["forecast"] - window["observed"]
    window["outcome"] = pd.Series("hit", index=window.index)
    window.loc[window["shortfall"] < 0, "outcome"] = "miss"
    window.loc[window["shortfall"] > 0, "outcome"] = "false_alarm"

    # Geometry off the simulated-map exposure frame for the issue step, which
    # the run has just cached; every step of a window carries the same building
    # set and the same footprints.
    ground_truth = validator._ground_truth_buildings(issue)
    centroids = ground_truth.set_index("building_id").geometry.centroid
    window["centroid_x"] = centroids.x.reindex(window.index)
    window["centroid_y"] = centroids.y.reindex(window.index)

    window = window.reset_index()
    window.insert(0, "issue_time", issue.strftime("%Y-%m-%d %H:%M:%S"))
    window.insert(0, "forecast_horizon", forecast_horizon)
    window.insert(0, "scenario_id", scenario_id)

    metrics = captured_rows[0] if captured_rows else {}
    return window, metrics


def swept_row(scenario_id, forecast_horizon, issue_time):
    """The row the sweep wrote for this window, for the side-by-side check."""
    keep = lambda name: name not in {
        "horizon_window_metrics_backup.csv", "horizon_window_metrics_p10.csv",
        "horizon_window_metrics.csv"}
    frames = [pd.read_csv(os.path.join(HORIZON_DIR, name))
              for name in sorted(os.listdir(HORIZON_DIR))
              if name.startswith("horizon_window_metrics") and name.endswith(".csv")
              and keep(name)]
    swept = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["forecast_horizon", "scenario_id", "issue_time"], keep="first")
    match = swept[(swept.scenario_id == scenario_id)
                  & (swept.forecast_horizon == forecast_horizon)
                  & (swept.issue_time == issue_time)]
    return match.iloc[0] if len(match) else None


def main():
    frames, summary = [], []
    for scenario_id, forecast_horizon, issue_time in CASES:
        print(f"event {scenario_id}, {forecast_horizon} h, issued {issue_time}",
              flush=True)
        window, metrics = run_window(scenario_id, forecast_horizon, issue_time)
        frames.append(window)

        counts = window["outcome"].value_counts()
        swept = swept_row(scenario_id, forecast_horizon, issue_time)
        row = {
            "scenario_id": scenario_id,
            "forecast_horizon": forecast_horizon,
            "issue_time": issue_time,
            "n_buildings": len(window),
            "n_hit": int(counts.get("hit", 0)),
            "n_miss": int(counts.get("miss", 0)),
            "n_false": int(counts.get("false_alarm", 0)),
            "pod": metrics.get("pod"),
            "far": metrics.get("far"),
            "csi": metrics.get("csi"),
            "swept_pod": None if swept is None else swept.pod,
            "swept_n_miss": None if swept is None else int(swept.n_miss),
            "swept_n_false": None if swept is None else int(swept.n_false),
        }
        summary.append(row)
        print(f"  reproduced POD {row['pod']:.4f} (miss {row['n_miss']}, "
              f"false {row['n_false']}) vs swept POD {row['swept_pod']:.4f} "
              f"(miss {row['swept_n_miss']}, false {row['swept_n_false']})",
              flush=True)

    os.makedirs(HORIZON_DIR, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT_CSV, index=False)
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)
    print(f"wrote {OUT_CSV}\nwrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
