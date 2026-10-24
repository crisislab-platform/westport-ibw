import logging
import time

import numpy as np
import pandas as pd

from modules.exposure_module.exposure_module import is_exposed, summarise_members
from modules.forecast_runner.forecast_runner_helper import _stamp
from modules.utils.utils import GpuMemoryMonitor, RamMonitor
from modules.validation_module.metrics import (
    contingency,
    crps_ensemble,
    has_spread,
    hours_between,
    mean_std,
    onset_times,
    peaks,
    stratified_rmse,
)
from modules.validation_module.results import (
    BUILDING_EVENT_FILE,
    BUILDING_HOURLY_FILE,
    ENSEMBLE_SPREAD_METRICS_FILE,
    PERCENTILE_EVENT_METRICS_FILE,
    PER_EVENT_METRICS_FILE,
    PER_TIMESTEP_METRICS_FILE,
    append_results,
)

logger = logging.getLogger(__name__)

# per_timestep_metrics.csv scores carried into per_event_metrics.csv as mean +- SD.
# Member percentiles scored by score_percentiles, and the hours dropped from the
# start of each event there: the warm-up, while the simulation fills from its
# dry initial state (the figures drop the same hours; the other files keep them).
ENSEMBLE_PERCENTILES = {"min": 0, "p25": 25, "median": 50, "p75": 75, "max": 100}
PERCENTILE_WARM_UP_H = 4

HOURLY_EVENT_AVERAGES = ("rmse_all", "rmse_wet", "rmse_dry", "mae", "mae_wet", "bias", "bias_wet", "crps", "crps_wet",
                         "brier", "hit_rate", "csi")


class DepthValidation():
    """Scores one run over a whole event against its simulated maps: depth and
    exposure hour by hour, then each building's peak and timing.

    Mixed into ValidationModule, which sets the attributes these methods read.
    """

    def validate_event(self, start_time=None, end_time=None):
        """ONE forecast run over the event's boundary-condition series, scored
        hour by hour as it streams, then per building over the whole series.

        Both sides go through the same spatial summary: the forecast per member,
        the simulated map as a single member.
        """
        times, observed, forecast, ensembles, hourly = [], [], [], [], []
        # What producing the forecast costs, kept apart from scoring it. Every
        # stage runs per batch of self.hours_per_batch hours (1 by default):
        #   simulation_s - the surrogate sampling AND decoding the batch's
        #                  flood-map ensembles together (filled in by the stream);
        #   exposure_s   - building depths for every hour and member of the
        #                  batch in one pass.
        # Reading the simulated maps and computing the scores happen outside
        # both. Both end with results on the CPU, so the GPU work is finished
        # when the clock is read. A batch's times are shared equally among its
        # hours on the per-timestep rows; the event row has the exact totals.
        timings = {"bc_s": 0.0, "inundation_s": 0.0}
        exposure_total, simulated_before = 0.0, 0.0
        # GPU memory is sampled every 10 ms in the background and its peak
        # taken per stage, over the same windows as the timers: what each stage
        # needs while it runs, not what is left once it is done. The process's
        # RAM (RSS) is sampled the same way: the memory figure on machines
        # without an NVIDIA GPU (Apple silicon shares it between CPU and GPU).
        monitor = GpuMemoryMonitor().start()
        ram_monitor = RamMonitor().start()
        monitor.reset()
        ram_monitor.reset()
        for batch_index, (valid_times, maps) in enumerate(self.integrator.iter_inundation_batches(
                *self.event_span(start_time, end_time), timings=timings,
                hours_per_batch=self.hours_per_batch)):
            simulation_s = timings["inundation_s"] - simulated_before
            simulated_before = timings["inundation_s"]
            gpu_peak_simulation = monitor.peak()
            ram_peak_simulation = ram_monitor.peak(settle_s=0)
            n_hours, n_members = maps.shape[0], maps.shape[1]

            monitor.reset()
            ram_monitor.reset()
            start = time.perf_counter()
            depths, num_cells = self.exposure_module.building_depths(
                maps.reshape(n_hours * n_members, *maps.shape[2:]))
            depths = depths.reshape(n_hours, n_members, -1)
            summaries = [summarise_members(hour) for hour in depths]
            exposure_s = time.perf_counter() - start
            exposure_total += exposure_s
            gpu_peak_exposure = monitor.peak()
            ram_peak_exposure = ram_monitor.peak(settle_s=0)
            del maps
            gpu_memory = {"gpu_peak_simulation_mib": gpu_peak_simulation,
                          "gpu_peak_exposure_mib": gpu_peak_exposure,
                          "gpu_peak_mib": float(np.nanmax([gpu_peak_simulation, gpu_peak_exposure]))
                          if np.isfinite([gpu_peak_simulation, gpu_peak_exposure]).any() else np.nan,
                          "ram_peak_simulation_mib": ram_peak_simulation,
                          "ram_peak_exposure_mib": ram_peak_exposure,
                          "ram_peak_mib": float(np.nanmax([ram_peak_simulation, ram_peak_exposure]))
                          if np.isfinite([ram_peak_simulation, ram_peak_exposure]).any() else np.nan}
            scored = num_cells > 0

            for valid_time, members, summary in zip(valid_times, depths, summaries):
                observed_depth = self._ground_truth_depths(valid_time)
                if observed_depth is None:
                    continue
                hourly.append(self.score_hour(valid_time, members[:, scored], observed_depth[scored],
                                {k: v[scored] for k, v in summary.items()},
                                self.exposure_module.buildings["building_id"].to_numpy()[scored],
                                time_step=len(hourly),
                                timing={"batch_index": batch_index,
                                        "simulation_s": simulation_s / n_hours,
                                        "exposure_s": exposure_s / n_hours,
                                        **gpu_memory}))

                times.append(pd.Timestamp(valid_time))
                observed.append(np.where(scored, observed_depth, np.nan))
                forecast.append(np.where(scored, summary["depth_median"], np.nan))
                # Every member's building depth this hour, (members, buildings),
                # for the probabilistic score of the peak; small next to the maps.
                ensembles.append(members.astype(np.float32))

            # The next simulation window starts here, after this batch's scoring.
            monitor.reset()
            ram_monitor.reset()
        monitor.stop()
        ram_monitor.stop()

        if not times:
            logger.warning("scenario %s: no step had a simulated map; nothing scored",
                           self.scenario_id)
            return
        self.score_event(times, np.vstack(observed), np.vstack(forecast), np.stack(ensembles),
                         pd.DataFrame(hourly),
                         timing={"bc_s": timings["bc_s"],
                                 "simulation_s": timings["inundation_s"],
                                 "exposure_s": exposure_total,
                                 "forecast_s": timings["bc_s"] + timings["inundation_s"] + exposure_total,
                                 # The event's peak: the highest over its batches.
                                 **{k: (float(np.nanmax([h[k] for h in hourly]))
                                        if any(np.isfinite(h[k]) for h in hourly) else np.nan)
                                    for k in ("gpu_peak_mib", "gpu_peak_simulation_mib", "gpu_peak_exposure_mib",
                                              "ram_peak_mib", "ram_peak_simulation_mib", "ram_peak_exposure_mib")}})

    def _run_keys(self):
        """What a summary row was produced from: the event and the ensemble,
        so rows from different events and ensemble sizes stay separable in
        the shared files."""
        return {
            "scenario_id": self.scenario_id,
            "event_name": self.hydromet_module.event_name,
            "spatial_summary": self.exposure_module.spatial_summary,
            "num_deep_ensemble": self.num_deep_ensemble,
            "samples_per_member": self.samples_per_member,
            "ensemble_size": self.num_deep_ensemble * self.samples_per_member,
            "hours_per_batch": self.hours_per_batch,
            "decode_batch": getattr(self.integrator, "decode_batch", None),
            "low_memory": bool(getattr(self.integrator, "low_memory", False)),
        }

    def score_hour(self, valid_time, members, observed_depth, summary, building_ids, time_step=None,
                   timing=None):
        """Depth (RMSE and MAE of the median, CRPS) and exposure (Brier) for one hour."""
        # The median's absolute error: CRPS reduces to it for a one-member
        # ensemble, so MAE vs CRPS shows what the ensemble spread adds. CRPS
        # and Brier are NaN for a deterministic forecast (has_spread).
        error = summary["depth_median"] - observed_depth
        absolute_error = np.abs(error)
        observed_exposed = is_exposed(observed_depth)
        if has_spread(members):
            crps = crps_ensemble(members, observed_depth)
            brier = (summary["prob_exposed"] - observed_exposed) ** 2
        else:
            crps = brier = np.full(len(observed_depth), np.nan)

        append_results(pd.DataFrame({
            "scenario_id": self.scenario_id,
            "valid_time": _stamp(valid_time),
            "building_id": building_ids,
            "observed_depth": observed_depth,
            "depth_median": summary["depth_median"],
            "depth_p10": summary["depth_p10"],
            "depth_p90": summary["depth_p90"],
            # Minimum, quartiles and maximum of the members, for warning on a
            # chosen member percentile (validation only; the forecast layer keeps
            # summarise_members' columns).
            "depth_min": members.min(axis=0),
            "depth_p25": np.quantile(members, 0.25, axis=0),
            "depth_p75": np.quantile(members, 0.75, axis=0),
            "depth_max": members.max(axis=0),
            "crps": crps,
            # The building's Brier score this hour: (P(exposed) - exposed)^2.
            "brier": brier,
        }), BUILDING_HOURLY_FILE, directory=self.results_dir())

        row = {
            **self._run_keys(),
            "time_step": time_step,
            # Seconds to produce this hour's forecast (see validate_event).
            **(timing or {}),
            "valid_time": _stamp(valid_time),
            "n_buildings": len(observed_depth),
            "n_observed_exposed": int(observed_exposed.sum()),
            "n_forecast_exposed": int(is_exposed(summary["depth_median"]).sum()),
            **stratified_rmse(summary["depth_median"], observed_depth),
            # Averages over this hour's buildings, each with its SD across them.
            **mean_std(absolute_error, "mae"),
            # Mean signed error of the median (forecast - observed): > 0 too deep.
            **mean_std(error, "bias"),
            **mean_std(crps, "crps"),
            **mean_std(brier, "brier"),
            # Over the observed-exposed buildings only, as crps_wet is over the
            # observed-wet cells in the westport flood-map scores.
            **mean_std(absolute_error[observed_exposed], "mae_wet"),
            **mean_std(error[observed_exposed], "bias_wet"),
            **mean_std(crps[observed_exposed], "crps_wet"),
        }
        # Exposure of the median depth across this hour's buildings.
        exposure = contingency(observed_exposed, is_exposed(summary["depth_median"]))
        row["hit_rate"], row["csi"] = exposure["hit_rate"], exposure["csi"]
        append_results(pd.DataFrame([row]), PER_TIMESTEP_METRICS_FILE, directory=self.results_dir())
        logger.info("%s: RMSE %.3f m (wet %.3f, dry %.3f) | MAE %.3f m | bias %+.3f m | CRPS %.3f m | Brier %.3f | "
                    "hit rate %.3f | CSI %.3f",
                    row["valid_time"], row["rmse_all"], row["rmse_wet"], row["rmse_dry"],
                    row["mae"], row["bias"], row["crps"], row["brier"], row["hit_rate"], row["csi"])
        return row

    def score_event(self, times, observed, forecast, ensembles, hourly, timing=None):
        """Each building's peak depth and peak timing from the median series,
        its exposure onset (first hour it is exposed, is_exposed), and how
        well the median series called its exposure hour by hour (hit rate, CSI).

        ``observed`` and ``forecast`` are (times, buildings) and ``ensembles``
        (times, members, buildings); NaN marks a building whose footprint
        touched no cell, which is dropped here. ``hourly`` holds score_hour's
        rows, averaged here over the event's hours with their SD (hourly_*).

        The peak's CRPS scores the ensemble AT THE FORECAST PEAK HOUR against the
        observed peak depth. Not each member's maximum over time: every hour is
        sampled with fresh noise, so a member index is not a trajectory, and the
        maximum over independent draws would inflate every peak.
        """
        keep = ~np.isnan(observed).any(axis=0)
        observed, forecast, ensembles = observed[:, keep], forecast[:, keep], ensembles[:, :, keep]
        building_ids = self.exposure_module.buildings["building_id"].to_numpy()[keep]

        observed_peak, observed_peak_time = peaks(observed, times)
        forecast_peak, forecast_peak_time = peaks(forecast, times)
        observed_onset = onset_times(observed, times)
        forecast_onset = onset_times(forecast, times)

        forecast_peak_index = forecast.argmax(axis=0)
        peak_ensemble = ensembles[forecast_peak_index, :, np.arange(forecast.shape[1])].T
        peak_crps = (crps_ensemble(peak_ensemble, observed_peak) if has_spread(peak_ensemble)
                     else np.full(len(observed_peak), np.nan))

        observed_exposed = is_exposed(observed_peak)
        forecast_exposed = is_exposed(forecast_peak)
        both = observed_exposed & forecast_exposed
        # Timing is only defined where both sides were exposed; a peak time on
        # a building that never got wet is the argmax of zeros.
        peak_time_error = np.where(
            both, hours_between(forecast_peak_time, observed_peak_time), np.nan)
        onset_error = np.where(
            both, hours_between(forecast_onset, observed_onset), np.nan)
        outcome = np.select(
            [both, observed_exposed, forecast_exposed],
            ["hit", "miss", "false_alarm"], default="correct_negative")

        # Hourly exposure of the median depth against the observed depth, counted
        # per building over the event's hours. Hit rate is NaN for a building
        # never observed exposed, CSI for one neither side ever called exposed.
        hourly_observed = is_exposed(observed)
        hourly_forecast = is_exposed(forecast)
        hits = (hourly_observed & hourly_forecast).sum(axis=0)
        misses = (hourly_observed & ~hourly_forecast).sum(axis=0)
        false_alarms = (~hourly_observed & hourly_forecast).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            hit_rate = np.where(hits + misses > 0, hits / (hits + misses), np.nan)
            csi = np.where(hits + misses + false_alarms > 0,
                           hits / (hits + misses + false_alarms), np.nan)

        append_results(pd.DataFrame({
            "scenario_id": self.scenario_id,
            "building_id": building_ids,
            "observed_peak_depth": observed_peak,
            "forecast_peak_depth": forecast_peak,
            "peak_depth_error": forecast_peak - observed_peak,
            "peak_depth_crps": peak_crps,
            "observed_peak_time": [_stamp(t) if observed_exposed[i] else None
                                   for i, t in enumerate(observed_peak_time)],
            "forecast_peak_time": [_stamp(t) if forecast_exposed[i] else None
                                   for i, t in enumerate(forecast_peak_time)],
            "peak_time_error_h": peak_time_error,
            "observed_onset_time": [None if pd.isna(t) else _stamp(t) for t in observed_onset],
            "forecast_onset_time": [None if pd.isna(t) else _stamp(t) for t in forecast_onset],
            "onset_error_h": onset_error,
            "exposure_outcome": outcome,
            "exposed_hours_observed": hits + misses,
            "exposed_hours_forecast": hits + false_alarms,
            "hit_rate": hit_rate,
            "csi": csi,
        }), BUILDING_EVENT_FILE, directory=self.results_dir())

        row = {
            **self._run_keys(),
            "event_start": _stamp(times[0]),
            "event_end": _stamp(times[-1]),
            "num_steps": len(times),
            # Seconds to produce the whole event's forecast: boundary conditions,
            # the surrogate over every hour, and building depths (see
            # validate_event). Scoring is not included.
            **(timing or {}),
            "n_buildings": int(keep.sum()),
            # Peak depth over the buildings that actually flooded (observed peak
            # exposed); dry buildings are left out of these averages.
            "n_flooded": int(observed_exposed.sum()),
            "peak_depth_rmse": (float(np.sqrt(np.mean((forecast_peak - observed_peak)[observed_exposed] ** 2)))
                                if observed_exposed.any() else np.nan),
            **mean_std(np.abs(forecast_peak - observed_peak)[observed_exposed], "peak_depth_mae"),
            **mean_std(peak_crps[observed_exposed], "peak_depth_crps"),
            # Mean and SD of the signed peak error; with the RMSE above,
            # RMSE^2 = bias^2 + SD^2.
            **{k.replace("peak_depth_error", "peak_depth_bias"): v for k, v in
               mean_std((forecast_peak - observed_peak)[observed_exposed], "peak_depth_error").items()},
            # Timing over flooded buildings the forecast also flooded: a peak
            # or onset time does not exist for a building that stays dry.
            "n_timed": int(both.sum()),
            **mean_std(np.abs(peak_time_error), "peak_time_mae_h"),
            **mean_std(peak_time_error, "peak_time_bias_h"),
            **mean_std(np.abs(onset_error), "onset_mae_h"),
            **mean_std(onset_error, "onset_bias_h"),
            **contingency(observed_exposed, forecast_exposed),
            # The hourly scores averaged over the event's hours, each with its SD
            # across hours -- as the westport per-event summaries average their
            # per-timestep scores. Hours where a score is undefined (no wet
            # building, say) are left out of its average.
            "n_hours": len(hourly),
            **{k: v for metric in HOURLY_EVENT_AVERAGES
               for k, v in mean_std(hourly[metric], f"hourly_{metric}").items()},
        }
        append_results(pd.DataFrame([row]), PER_EVENT_METRICS_FILE, directory=self.results_dir())
        if has_spread(ensembles[0]):
            self.score_percentiles(observed, ensembles)
        logger.info(
            "scenario %s event, %d flooded buildings: peak RMSE %.3f m | peak MAE %.3f m | peak CRPS %.3f m | "
            "peak timing MAE %.1f h | onset MAE %.1f h | %d hits, %d misses, %d false alarms",
            self.scenario_id, row["n_flooded"], row["peak_depth_rmse"], row["peak_depth_mae"], row["peak_depth_crps"],
            row["peak_time_mae_h"],
            row["onset_mae_h"], row["hits"], row["misses"], row["false_alarms"])

    def score_percentiles(self, observed, ensembles):
        """Per event, the ensemble read at each member percentile in
        ENSEMBLE_PERCENTILES, after the first PERCENTILE_WARM_UP_H hours.

        ``observed`` is (times, buildings) and ``ensembles`` (times, members,
        buildings). For each percentile, its series over the event is treated
        as a single forecast: a building is exposed if its peak exceeds the
        exposure depth (is_exposed), scored against the simulated peak (hits,
        misses, false alarms, hit rate, CSI), and its peak depth error over the
        buildings the simulation floods (MAE and bias, mean and SD). Warning on
        the max ~ warning if any member floods the building; on the min, only if
        all do. One row per percentile in PERCENTILE_EVENT_METRICS_FILE.

        The spread: of the building-hours the simulation floods, the share whose
        simulated depth falls inside the 25-75%, 10-90% and min-max member
        ranges (about 50%, 80% and (M-1)/(M+1) if the spread is right), and
        below the minimum or above the maximum. One row in
        ENSEMBLE_SPREAD_METRICS_FILE.
        """
        observed, ensembles = observed[PERCENTILE_WARM_UP_H:], ensembles[PERCENTILE_WARM_UP_H:]
        if not len(observed):
            return
        observed_peak = observed.max(axis=0)
        observed_exposed = is_exposed(observed_peak)
        keys = {**self._run_keys(), "warm_up_h": PERCENTILE_WARM_UP_H}

        rows = []
        for name, q in ENSEMBLE_PERCENTILES.items():
            peak = np.percentile(ensembles, q, axis=1).max(axis=0)
            error = (peak - observed_peak)[observed_exposed]
            rows.append({**keys, "percentile": name, "n_flooded": int(observed_exposed.sum()),
                         **contingency(observed_exposed, is_exposed(peak)),
                         **mean_std(np.abs(error), "peak_depth_mae"),
                         **mean_std(error, "peak_depth_bias")})
        append_results(pd.DataFrame(rows), PERCENTILE_EVENT_METRICS_FILE, directory=self.results_dir())

        wet = is_exposed(observed)
        depth = observed[wet]
        members = np.moveaxis(ensembles, 1, -1)[wet]          # (wet building-hours, members)
        p10, p25, p75, p90 = np.percentile(members, [10, 25, 75, 90], axis=1)
        low, high = members.min(axis=1), members.max(axis=1)
        share = (lambda inside: float(inside.mean())) if len(depth) else (lambda inside: np.nan)
        append_results(pd.DataFrame([{
            **keys, "n_members": ensembles.shape[1], "n_wet_building_hours": int(len(depth)),
            "inside_p25_p75": share((depth >= p25) & (depth <= p75)),
            "inside_p10_p90": share((depth >= p10) & (depth <= p90)),
            "inside_min_max": share((depth >= low) & (depth <= high)),
            "below_min": share(depth < low),
            "above_max": share(depth > high),
        }]), ENSEMBLE_SPREAD_METRICS_FILE, directory=self.results_dir())
