import logging
import os

import pandas as pd

from modules.lib import VALIDATION_RESULTS_DIR
from modules.forecast_runner.forecast_runner_helper import _align_appended_csv

logger = logging.getLogger(__name__)

# Files the validation module appends to. The calibration files and the July
# 2021 summary go straight under VALIDATION_RESULTS_DIR; everything else a
# forecast run produces goes in that run configuration's own folder
# (ValidationModule.results_dir).
SPATIAL_SUMMARY_CALIBRATION_FILE = "spatial_summary_calibration.csv"
SPATIAL_SUMMARY_DAMAGE_FILE = "spatial_summary_depth_by_damage.csv"
# July 2021: each building's forecast and surveyed depth, in the run's folder.
JULY_2021_BUILDING_FILE = "july_2021_building_depth.csv"
# July 2021: the model's peak flood map (largest flooded area), in the run's folder.
JULY_2021_PEAK_MAP_FILE = "july_2021_peak_flood_map.tif"
# The July 2021 scores of every model configuration, one row each.
JULY_2021_METRICS_FILE = "july_2021_metrics.csv"
BUILDING_HOURLY_FILE = "building_hourly_depth.csv"
# The two summary levels, as in the westport runs: every event's every
# timestep, and one row per event.
PER_TIMESTEP_METRICS_FILE = "per_timestep_metrics.csv"
BUILDING_EVENT_FILE = "building_event_metrics.csv"
PER_EVENT_METRICS_FILE = "per_event_metrics.csv"
# Ensemble runs only: per event, detection and peak depth when warning on each
# member percentile, and how often the simulated depth falls inside the
# ensemble's ranges (DepthValidation.score_percentiles).
PERCENTILE_EVENT_METRICS_FILE = "percentile_event_metrics.csv"
ENSEMBLE_SPREAD_METRICS_FILE = "ensemble_spread_metrics.csv"


def append_results(frame, filename, directory=None):
    """Append ``frame`` to ``filename`` in ``directory`` (VALIDATION_RESULTS_DIR
    when None), writing the header only when the file is new. See
    _align_appended_csv for what happens when the columns differ from the
    file's."""
    if frame is None or frame.empty:
        return None
    directory = directory or VALIDATION_RESULTS_DIR
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, filename)
    _align_appended_csv(output_file, frame.columns)
    frame.to_csv(output_file, mode="a",
                 header=not os.path.exists(output_file), index=False)
    logger.info("%d rows -> %s", len(frame), output_file)
    return output_file


def replace_results(frame, filename, key, directory=None):
    """As append_results, but one row per ``key``: the file's rows whose
    ``key`` value matches a row of ``frame`` are replaced by it, so re-running
    a configuration updates its row instead of adding another."""
    if frame is None or frame.empty:
        return None
    directory = directory or VALIDATION_RESULTS_DIR
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, filename)
    _align_appended_csv(output_file, frame.columns)
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        frame = pd.concat([existing[~existing[key].isin(frame[key])], frame], ignore_index=True)
    frame.to_csv(output_file, index=False)
    logger.info("%d rows -> %s", len(frame), output_file)
    return output_file
