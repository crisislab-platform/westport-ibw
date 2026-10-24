from modules.utils.utils import append_metrics
import pandas as pd
import numpy as np
import logging
import os
import time
from modules.forecast_runner.forecast_runner_helper import (
    TIMING_COLUMNS,
    RUNTIME_METRICS_FILE,
)

logger = logging.getLogger(__name__)


class RuntimeValidation():
    """Times the forecast end to end, one row per issue in runtime_metrics.csv.

    Mixed into ValidationModule, which sets the attributes these methods read.
    """

    def validate_runtime(self, start_time=None, end_time=None):
        """Run every issue in [start_time, end_time] the way the operational
        run does -- publishing included when the Integrator publishes -- and
        record what each stage cost.

        The stage timers sit inside the Integrator (they have to wrap the
        stages themselves); this only asks for them and writes them down.
        """
        for issue_time in self.integrator.issue_times(start_time, end_time):
            timings = dict.fromkeys(TIMING_COLUMNS, np.nan)
            issue_start = time.perf_counter()

            if self.integrator.issue_forecast(issue_time, timings) is None:
                continue

            timings["forecast_horizon"] = self.forecast_horizon
            timings["num_deep_ensemble"] = self.num_deep_ensemble
            timings["samples_per_member"] = self.samples_per_member
            timings["ensemble_size"] = self.num_deep_ensemble * self.samples_per_member
            timings["total_s"] = time.perf_counter() - issue_start
            self.record_timings(issue_time, timings)

    def record_timings(self, issue_time, timings):
        """Append one issue's component timings to out/runtime_metrics.csv.

        Written per issue rather than summarised at the end so a sweep that is
        interrupted still leaves usable rows, and so the cost of an individual
        forecast can be read rather than only the mean.
        """
        output_file = os.path.join(self.results_dir(), RUNTIME_METRICS_FILE)

        expected = ["timestamp", "scenario_id"] + list(TIMING_COLUMNS)
        if os.path.exists(output_file):
            existing_cols = pd.read_csv(output_file, nrows=0).columns.tolist()
            if existing_cols != expected:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                archived = f"{os.path.splitext(output_file)[0]}_{stamp}.csv.bak"
                os.rename(output_file, archived)
                logger.warning(
                    f"{RUNTIME_METRICS_FILE} had a different column set; archived "
                    f"it to {os.path.basename(archived)} and started a new file."
                )

        # Reused rather than reimplemented: same timestamp + scenario_id + row
        # shape as the other metrics writers.
        append_metrics(timings, issue_time, self.scenario_id, output_file)

        logger.info(
            "%s h+%d (%d steps) MxK=%dx%d: inundation %.2fs exposure %.2fs "
            "publish %.2fs (+%.2fs clearing the last issue) total %.2fs "
            "| per timestep %.2fs",
            issue_time, self.forecast_horizon, timings["num_timesteps"],
            timings["num_deep_ensemble"], timings["samples_per_member"],
            timings["inundation_s"], timings["exposure_s"], timings["publish_s"],
            timings["delete_s"], timings["total_s"],
            timings["total_s"] / timings["num_timesteps"],
        )
