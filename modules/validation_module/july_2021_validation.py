from modules.exposure_module.exposure_module import (
    EXPOSURE_DEPTH_M,
    is_exposed,
    SPATIAL_SUMMARIES,
    ExposureModule,
    summarise_members,
)
from modules.lib import (
    PLACRD_FILE_PATH,
    SURVEYED_FLOOD_DEPTH_JUL_2021,
)
from modules.forecast_runner.forecast_runner_helper import _stamp
from modules.utils.spatial_operators.operators import find_sample_flood_map
from modules.validation_module.metrics import (
    _damage_to_affected,
    brier_score,
    contingency,
    crps_ensemble,
    has_spread,
    pearson_test,
    stratified_rmse,
)
from modules.validation_module.results import (
    JULY_2021_BUILDING_FILE,
    JULY_2021_METRICS_FILE,
    JULY_2021_PEAK_MAP_FILE,
    SPATIAL_SUMMARY_CALIBRATION_FILE,
    SPATIAL_SUMMARY_DAMAGE_FILE,
    append_results,
    replace_results,
)
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import torch
import logging
import os
from rasterio.transform import rowcol

logger = logging.getLogger(__name__)


class July2021Validation():
    """The July 2021 event (scenario 0), against the surveyed depths and the
    damage placards: calibrating the spatial summary (calibrate_spatial_summarisation_method)
    and scoring each model's peak flood map, the hour of largest flooded area,
    against the surveyed flood map (validate_july_2021_event).

    Mixed into ValidationModule, which sets the attributes these methods read.
    """

    def validate_july_2021_event(self, start_time=None, end_time=None):
        """Score the model's peak flood map against the surveyed flood map,
        through the spatial summary the exposure module is set to. Calibrating
        that statistic is separate: calibrate_spatial_summarisation_method.

        The peak flood map is the hour of the event with the largest flooded
        area in the model's own forecast: cells deeper than EXPOSURE_DEPTH_M,
        averaged over the ensemble members. Extent rather than volume, because
        the survey records how far the water reached, and volume is dominated by
        the river channel and estuary, which peak with the tide. Each model is
        scored at its own peak hour, recorded as valid_time.
        """
        # Buildings in the model domain (the set the simulated events score),
        # kept before the module is swapped for the survey's: only buildings in
        # both are scored, as the forecast says nothing outside the model domain.
        self.model_domain_buildings = set(self.exposure_module.buildings["building_id"])
        self.exposure_module = ExposureModule(
            self.scenario_id, domain_raster=SURVEYED_FLOOD_DEPTH_JUL_2021,
            spatial_summary=self.exposure_module.spatial_summary,
        )
        peak_extent, peak_time, peak_map = -np.inf, None, None
        forecast_times = []
        for valid_times, maps in self.integrator.iter_inundation_batches(
                *self.event_span(start_time, end_time), hours_per_batch=self.hours_per_batch):
            forecast_times.extend(valid_times)
            # (B, members, H, W) -> each hour's expected flooded area: cells
            # deeper than EXPOSURE_DEPTH_M, counted per member and averaged over
            # the members. The surrogate's maps are NaN outside the computational
            # domain: dry, as building_depths reads them.
            wet = torch.nan_to_num(maps.float(), nan=0.0) > EXPOSURE_DEPTH_M
            extents = wet.float().sum(dim=(2, 3)).mean(dim=1)
            hour = int(extents.argmax())
            if float(extents[hour]) > peak_extent:
                # A copy, so keeping the peak hour does not keep its whole batch.
                peak_extent, peak_time, peak_map = float(extents[hour]), valid_times[hour], maps[hour].clone()
        if peak_map is None:
            logger.warning("scenario %s: no forecast step; nothing scored", self.scenario_id)
            return
        logger.info("scenario %s: forecast %d hours, %s to %s; largest flooded area at %s",
                    self.scenario_id, len(forecast_times), forecast_times[0], forecast_times[-1], peak_time)
        self.save_peak_flood_map(peak_map, peak_time)
        self.score_against_survey(peak_map, peak_time)

    def save_peak_flood_map(self, peak_map, valid_time):
        """Write the peak flood map to JULY_2021_PEAK_MAP_FILE in the run's
        folder, on the simulated maps' grid: the ensemble-median depth per cell
        (the point forecast scored against the survey; for a one-member model,
        its map). Replaced on every run."""
        depth = peak_map.float().median(dim=0).values.cpu().numpy()
        with rio.open(find_sample_flood_map()) as src:
            # Grid and projection only, not the sample's block layout.
            profile = {k: v for k, v in src.profile.items() if k not in ("blockxsize", "blockysize", "tiled")}
        if depth.shape != (profile["height"], profile["width"]):
            raise ValueError(f"peak map {depth.shape} is not on the simulated maps' grid "
                             f"{(profile['height'], profile['width'])}")
        # NaN outside the computational domain (the surrogate's maps) is no data.
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
        path = os.path.join(self.results_dir(), JULY_2021_PEAK_MAP_FILE)
        with rio.open(path, "w", **profile) as dst:
            dst.write(depth.astype(np.float32), 1)
            dst.update_tags(model=self.run_config()["model"], valid_time=_stamp(valid_time),
                            statistic="ensemble median depth (m)", members=str(peak_map.shape[0]))
        logger.info("peak flood map (%s) -> %s", _stamp(valid_time), path)

    def _surveyed_depth(self):
        with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
            return src.read(1).astype(float)

    def placard_building_depths(self):
        """One row per placarded building: its damage class, whether it counts
        as affected, and its depth off the surveyed raster under every spatial
        summary (one column per name in SPATIAL_SUMMARIES).

        Only buildings with a depth under every summary are kept, so each
        summary is scored on the same buildings.
        """
        surveyed_module = ExposureModule(
            self.scenario_id, domain_raster=SURVEYED_FLOOD_DEPTH_JUL_2021)
        surveyed = self._surveyed_depth()
        placards = self.get_obs_placard_filtered()[["building_id", "DamageNb"]]

        scored = placards.copy()
        for statistic in SPATIAL_SUMMARIES:
            depths, num_cells = surveyed_module.building_depths(
                [surveyed], SURVEYED_FLOOD_DEPTH_JUL_2021, statistic=statistic)
            building = pd.DataFrame({
                "building_id": surveyed_module.buildings["building_id"].to_numpy(),
                statistic: depths[0],
            })[num_cells > 0]
            scored = scored.merge(building, on="building_id")
        scored["observed_affected"] = scored["DamageNb"].apply(_damage_to_affected)
        scored = scored.dropna(subset=["observed_affected"]).reset_index(drop=True)
        return scored

    def calibrate_spatial_summarisation_method(self):
        """Which spatial summary -- max, mean, median or 90th percentile of the
        footprint cells, or the centroid cell -- best reflects the damage the
        July 2021 placards recorded, each building's depth read off the
        surveyed raster.

        Every summary is scored on the SAME buildings by the Pearson
        correlation between building depth and damage class (DamageNb 0-3).
        The result informs SPATIAL_SUMMARY; see the note there for the choice.

        One row per summary in spatial_summary_calibration.csv, and each
        summary's depth by damage class in spatial_summary_depth_by_damage.csv.
        """
        scored = self.placard_building_depths()
        damage = scored["DamageNb"].to_numpy()
        depth = {statistic: scored[statistic].to_numpy() for statistic in SPATIAL_SUMMARIES}

        rows = []
        for statistic in SPATIAL_SUMMARIES:
            r, p_value = pearson_test(depth[statistic], damage)
            rows.append({
                "scenario_id": self.scenario_id,
                "spatial_summary": statistic,
                "n_buildings": len(scored),
                "pearson_correlation": r,
                "pearson_p_value": p_value,
            })
            logger.info("%s: Pearson %.3f (p = %.2g)", statistic, r, p_value)
        append_results(pd.DataFrame(rows), SPATIAL_SUMMARY_CALIBRATION_FILE)

        # Each summary's depth by damage class: the relation the correlation
        # scores, as a table a reader can see.
        by_damage = [
            {"scenario_id": self.scenario_id, "spatial_summary": statistic,
             "damage_class": int(level), "n_buildings": len(values),
             "depth_median": float(np.median(values)),
             "depth_q25": float(np.quantile(values, 0.25)),
             "depth_q75": float(np.quantile(values, 0.75)),
             "depth_mean": float(np.mean(values))}
            for statistic in SPATIAL_SUMMARIES
            for level in sorted(np.unique(damage))
            for values in [depth[statistic][damage == level]]
        ]
        append_results(pd.DataFrame(by_damage), SPATIAL_SUMMARY_DAMAGE_FILE)
        return rows

    def score_against_survey(self, inundation, valid_time):
        """The forecast at ``valid_time`` against the surveyed flood map, per
        building: depth (RMSE of the median, CRPS) and exposure (Brier, and hits,
        misses and false alarms of the median), a building being exposed when
        its surveyed depth is (is_exposed), as on the simulated maps.
        """
        members, forecast_cells = self.exposure_module.building_depths(inundation)
        observed, observed_cells = self.exposure_module.building_depths(
            [self._surveyed_depth()], SURVEYED_FLOOD_DEPTH_JUL_2021)
        observed = observed[0]
        building_ids = self.exposure_module.buildings["building_id"]
        scored = ((forecast_cells > 0) & (observed_cells > 0)
                  & building_ids.isin(self.model_domain_buildings).to_numpy())
        members, observed = members[:, scored], observed[scored]
        summary = summarise_members(members)
        # CRPS and Brier are NaN for a deterministic forecast (has_spread).
        probabilistic = has_spread(members)
        crps = crps_ensemble(members, observed) if probabilistic else np.full(len(observed), np.nan)

        buildings = pd.DataFrame({
            "scenario_id": self.scenario_id,
            "valid_time": _stamp(valid_time),
            "building_id": self.exposure_module.buildings["building_id"].to_numpy()[scored],
            "observed_depth": observed,
            **summary,
            "observed_exposed": is_exposed(observed),
            "crps": crps,
        })
        # One row per building: re-running the configuration replaces its rows.
        replace_results(buildings, JULY_2021_BUILDING_FILE, key="building_id", directory=self.results_dir())

        observed_exposed = is_exposed(observed)
        # One row per model configuration in the shared July 2021 file, keyed
        # by the run's results folder, which holds the per-building rows.
        row = {
            "run": os.path.basename(self.results_dir()),
            **self.run_config(),
            "valid_time": _stamp(valid_time),
            "n_buildings": len(buildings),
            **stratified_rmse(summary["depth_median"], observed),
            "crps": float(crps.mean()),
            "brier": brier_score(summary["prob_exposed"], observed_exposed) if probabilistic else np.nan,
            **contingency(observed_exposed, is_exposed(summary["depth_median"])),
        }
        replace_results(pd.DataFrame([row]), JULY_2021_METRICS_FILE, key="run")
        logger.info("%s: RMSE %.3f m | CRPS %.3f m | Brier %.3f | hit rate %.2f | CSI %.2f",
                    row["valid_time"], row["rmse_all"], row["crps"], row["brier"],
                    row["hit_rate"], row["csi"])

    def get_obs_placard_filtered(self):
        obs_placard = gpd.read_file(PLACRD_FILE_PATH)
        obs_placard = obs_placard[obs_placard['DamageNb'].isin([0.0, 1.0, 2.0, 3.0])]
        n_before = len(obs_placard)
        with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
            obs_depth = src.read(1).astype(float)
            transform = src.transform

        window = 2  # 2 pixels each side -> 5x5 window

        keep = []

        for geom in obs_placard.geometry.centroid:
            row, col = rowcol(transform, geom.x, geom.y)

            r0 = max(0, row - window)
            r1 = min(obs_depth.shape[0], row + window + 1)
            c0 = max(0, col - window)
            c1 = min(obs_depth.shape[1], col + window + 1)

            patch = obs_depth[r0:r1, c0:c1]

            # Keep if at least one surveyed depth exists nearby
            keep.append(np.any(~np.isnan(patch)))

        obs_placard = obs_placard.loc[keep].copy()
        n_after = len(obs_placard)
        n_filtered = n_before - n_after

        logger.info(f"Buildings before filtering : {n_before}")
        logger.info(f"Buildings after filtering  : {n_after}")
        logger.info(f"Buildings filtered out     : {n_filtered} ({100*n_filtered/n_before:.1f}%)")
        return obs_placard
