from modules.bc_forecast_module.bc_module import BC_Module
from modules.hazard_forecast_module.model_client import FlowMatchingModelClient
from modules.impact_module.impact_module import ImpactModule, WET_DEPTH_MIN
from modules.impact_module.fuzzy_risk import TERMS, MIDDLE
from modules.arcgis_module.arcgis_module import ArcGISModule
from modules.utils.utils import append_metrics, check_device
from modules.utils.spatial_operators.operators import load_flood_map
from modules.lib import OUTPUT_DIR, PLACRD_FILE_PATH, SURVEYED_FLOOD_DEPTH_JUL_2021, EXPOSURE_STUDY_RESULTS
from contextlib import contextmanager
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, root_mean_squared_error
import os
import time
from rasterio.transform import rowcol

logger = logging.getLogger(__name__)

# One row per issue in runtime_metrics.csv, timing the framework end to end.
# Named for the components rather than the modules that implement them, since
# this is what the efficiency results are reported as:
#
#   bc_s         BC_Module.get_bc_forecast -- reading the boundary conditions
#   inundation_s FlowMatchingModelClient.get_hazard_forecast -- the surrogate
#   risk_s       ImpactModule.get_impact_forecast -- exposure + the FIS
#   extent_s     ImpactModule.get_flood_extent -- the depth-class polygons
#   delete_s     clearing the PREVIOUS issue off the hosted layers
#   publish_s    writing THIS issue to them -- the framework's update cost
#   total_s      the whole issue, including anything not broken out above
#
# delete_s is kept out of publish_s deliberately. Publishing replaces a layer's
# contents, so every issue first removes the last one, but that teardown is a
# property of how the layers are kept current rather than the cost of putting a
# forecast on the map. total_s is still true wall clock and includes it, so the
# update cost of the framework is total_s - delete_s.
#
# The tuple is fixed and complete so the header stays stable across horizons
# and across run/test mode: a stage that did not run reports NaN rather than
# dropping its column and misaligning every later row in an appended CSV.
#
# One issue covers num_timesteps forecast steps (the horizon, or fewer once the
# boundary conditions run out near the end of the window), so per-timestep cost
# is stage_s / num_timesteps -- which is what makes horizons comparable.
TIMING_COLUMNS = (
    "forecast_horizon",
    "num_timesteps",
    # The ensemble the timings were produced at. M and K are recorded
    # separately as well as pooled because they cost differently: M loads and
    # runs that many distinct models, K draws that many samples from each, and
    # ensemble_size = M*K is what the decode and the downstream exposure
    # computation are linear in.
    "num_deep_ensemble",
    "samples_per_member",
    "ensemble_size",
    "bc_s",
    "inundation_s",
    "risk_s",
    "extent_s",
    "delete_s",
    "publish_s",
    "total_s",
)

RUNTIME_METRICS_FILE = "runtime_metrics.csv"


@contextmanager
def _timed(timings, stage, accumulate=False):
    """Record the wall-clock seconds spent in the block under ``stage``.

    perf_counter is enough for the inundation surrogate despite it running on
    the GPU: a step ends at torch.from_numpy, so the device work has already
    been synced to host by the time the block exits.

    ``accumulate`` adds to whatever is already recorded instead of replacing
    it, which is what the stepped horizon needs: inundation and risk now
    interleave per step rather than running as two contiguous blocks, so each
    stage is the sum over the steps. Requires the stage to start at 0.0.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[stage] = timings[stage] + elapsed if accumulate else elapsed


def _timed_steps(timings, stage, steps):
    """Yield from ``steps``, charging only the time spent PRODUCING each item.

    The hazard forecast is a generator now, so its work happens inside
    ``next()`` rather than in one call the caller can wrap. Timing the
    iteration this way keeps inundation_s measuring the surrogate alone, with
    the consumer's own work charged to whatever stage it opens.
    """
    timings[stage] = 0.0
    iterator = iter(steps)
    while True:
        start = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            timings[stage] += time.perf_counter() - start
            return
        timings[stage] += time.perf_counter() - start
        yield item


def _merge_impact(accumulated, step_impact):
    """Fold one step's impact frames into the run's accumulator.

    get_impact_forecast returns ``{object_type: {timestamp: gdf}}``; stepping
    the horizon means merging the inner dicts rather than replacing them. What
    accumulates here is only the impact frames -- a few thousand rows per
    object type per step -- while the depth maps they were derived from are
    released as soon as the step is done with.
    """
    for key, series in step_impact.items():
        accumulated.setdefault(key, {}).update(series)
    return accumulated


class Integrator():
    def __init__(self, scenario_id, forecast_horizon, push_to_arcgis,samples_per_member, num_deep_ensemble):
        self.scenario_id = scenario_id
        self.forecast_horizon = forecast_horizon
        self.push_to_arcgis = push_to_arcgis
        # Kept as attributes as well as handed to the model: the runtime
        # metrics report the ensemble the timings were produced at, and the
        # inundation cost is linear in M*K.
        self.samples_per_member = samples_per_member
        self.num_deep_ensemble = num_deep_ensemble
        self.bc_module = BC_Module(scenario_id)
        self.hazard_module = FlowMatchingModelClient(samples_per_member=samples_per_member, num_members=num_deep_ensemble)
        self.impact_module = ImpactModule(scenario_id)
        if self.push_to_arcgis:
            self.arcgis_module = ArcGISModule()
        self.device = check_device()
        self.max_inundation_map = None
        self.max_extent_timestamp = None
        self.max_area = -(np.inf)

    def run_hourly_forecast(self,  start_time=None, end_time=None, test_mode=False):
        # Immidiate the hourly forecast by running the forecast hourly starting from the start_time
        if start_time == None or start_time=='':
            current_time = min(self.bc_module.bc_dict.keys())
        else:
            current_time = pd.to_datetime(start_time)
        if end_time == None or end_time=='':
            end_time = max(self.bc_module.bc_dict.keys())
        else:
            end_time = pd.to_datetime(end_time)
        while current_time <= end_time:
            # Timed per issue, not per run: the horizon sweep compares the cost
            # of one forecast, and a skipped hour must not be averaged in.
            timings = dict.fromkeys(TIMING_COLUMNS, np.nan)
            issue_start = time.perf_counter()

            with _timed(timings, "bc_s"):
                bc_forecast = self.bc_module.get_bc_forecast(current_time, self.forecast_horizon)
            #bc_forecast is a dictionary with keys as timestamps and values as boundary condition data
            #check if bc_forecast is empty
            if not bc_forecast:
                print(f"No boundary condition data available for the forecast period starting at {current_time}. Skipping this hour.")
                current_time += pd.Timedelta(hours=1)
                continue

            # One forecast step at a time: decode it, score it, then let the
            # depth maps go before the next is decoded. Holding the whole
            # horizon is what put a 12-step, 100-member forecast at ~28 GB and
            # had the kernel kill the run; scored per step the peak is one
            # step's worth, ~2.4 GB, whatever the horizon.
            timings["risk_s"] = 0.0
            impact_forecast = {}
            for timestamp, hazard in _timed_steps(
                timings, "inundation_s",
                self.hazard_module.iter_hazard_forecast(bc_forecast),
            ):
                if test_mode and self.scenario_id == 0:
                    self.update_max_extent_map({timestamp: hazard})
                    continue

                #Get the impact forecast for this step using the hazard forecast
                with _timed(timings, "risk_s", accumulate=True):
                    _merge_impact(
                        impact_forecast,
                        self.impact_module.get_impact_forecast({timestamp: hazard}),
                    )
                # The inundation extent rides along in the same dict so saving
                # and publishing both see it without either growing an argument.
                # Both take a hazard dict, so re-enabling them means passing
                # this step's {timestamp: hazard} and merging, as above.
                # if not test_mode:
                #     with _timed(timings, "extent_s", accumulate=True):
                #         _merge_impact(impact_forecast, {"flood_extent":
                #             self.impact_module.get_flood_extent({timestamp: hazard})})
                #     # The same water at cell resolution: the extent says where
                #     # each depth class reaches, the grid what the depth is
                #     # between the modelled objects, with its ensemble spread.
                #     #
                #     # PAUSED. At INUNDATION_BLOCK = 12 the grid is ~17k rows per
                #     # timestep -- more than every object layer combined -- and
                #     # since publishing became replace-the-layer it dominates the
                #     # issue: roughly 54 of ~94 uploads and 46 of ~100 MB. Left
                #     # computed nowhere rather than computed and dropped, so the
                #     # cost goes too. Re-enable with the layer URL in .env.
                #     # _merge_impact(impact_forecast, {"inundation_grid":
                #     #     self.impact_module.get_inundation_grid({timestamp: hazard})})

            #Save as layers in a spatial format (e.g., GeoPackage) plus a metrics CSV.
            # if not test_mode:
            #     self.impact_module.save_impact_forecast(impact_forecast, hazard_forecast, current_time)

            if test_mode and self.scenario_id !=0 :
                self.validate_impact_forecast(impact_forecast, current_time)

            #Update feature layers on spatial db. current_time is the issue
            #time of THIS forecast; the loop bounds say nothing about the rows
            #being written here.
            if self.push_to_arcgis:
                # Timed inside the publisher rather than around the call, so
                # clearing the previous issue lands in delete_s and only the
                # write lands in publish_s.
                timings.update(
                    self.arcgis_module.push_impacts_to_spatial_db(
                        impact_forecast, current_time
                    )
                )

            timings["forecast_horizon"] = self.forecast_horizon
            timings["num_timesteps"] = len(bc_forecast)
            timings["num_deep_ensemble"] = self.num_deep_ensemble
            timings["samples_per_member"] = self.samples_per_member
            timings["ensemble_size"] = self.num_deep_ensemble * self.samples_per_member
            timings["total_s"] = time.perf_counter() - issue_start
            self.record_timings(current_time, timings)

            logger.info(f"Processed timestamp: {current_time}")
            current_time += pd.Timedelta(hours=1)

        # Placard validation only: max_inundation_map is populated by
        # update_max_extent_map, which runs under test_mode alone, so reaching
        # this outside test_mode means scoring against None.
        if test_mode and self.scenario_id == 0:
            impact_forecast = self.impact_module.get_impact_forecast(self.max_inundation_map)
            logger.info(f"Max extent timestamp for scenario {self.scenario_id} {self.max_extent_timestamp}")
            self.test_against_placard_data(impact_forecast, model="flow_matching")
    
    def record_timings(self, issue_time, timings):
        """Append one issue's component timings to out/runtime_metrics.csv.

        Written per issue rather than summarised at the end so a sweep that is
        interrupted still leaves usable rows, and so the cost of an individual
        forecast can be read rather than only the mean.
        """
        os.makedirs(EXPOSURE_STUDY_RESULTS, exist_ok=True)
        output_file = os.path.join(EXPOSURE_STUDY_RESULTS, RUNTIME_METRICS_FILE)

        # Same guard as the two validation writers: the file is appended to
        # with the header written once, so a run whose columns differ from what
        # is already there would push its values under the wrong names. Adding
        # a column to TIMING_COLUMNS -- the ensemble fields, say -- is exactly
        # that case. Archive the stale file rather than corrupt a sweep.
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
            "%s h+%d (%d steps) MxK=%dx%d: inundation %.2fs risk %.2fs "
            "publish %.2fs (+%.2fs clearing the last issue) total %.2fs "
            "| per timestep %.2fs",
            issue_time, self.forecast_horizon, timings["num_timesteps"],
            timings["num_deep_ensemble"], timings["samples_per_member"],
            timings["inundation_s"], timings["risk_s"], timings["publish_s"],
            timings["delete_s"], timings["total_s"],
            timings["total_s"] / timings["num_timesteps"],
        )

    def update_max_extent_map(self, hazard_forecast):
        """Keep the single step with the largest inundated area.

        Called with one step at a time now that the horizon is streamed, so
        max_inundation_map holds ``{timestamp: ensemble}`` for the winning step
        alone. It previously held every step of whichever issue contained the
        winner, which meant test_against_placard_data -- which takes the FIRST
        entry -- scored a step that merely shared an issue with the maximum
        rather than the maximum itself.
        """
        for timestamp, ensemble in hazard_forecast.items():
            mean_depth = ensemble.mean(axis=0)
            inundated_area = np.count_nonzero(mean_depth > 0)
            if inundated_area > self.max_area:
                self.max_area = inundated_area
                self.max_extent_timestamp = timestamp
                self.max_inundation_map = hazard_forecast

    def validate_forecast(self, start_time=None, end_time=None):
        if self.scenario_id == 0:
            self.validate_exposure_method() 
        # logger.info(f"Validating forecast for scenario {self.scenario_id} from {start_time} to {end_time}")
        self.run_hourly_forecast(start_time=start_time, end_time=end_time, test_mode=True)
        
    def validate_exposure_method(self):
        # Scored against the surveyed raster, so the surveyed raster defines the
        # population: the buildings it has depths for. The module built in
        # __init__ is bounded by the flood model's grid, which covers different
        # ground -- it misses 542 buildings the survey reaches, two of them
        # placarded at the most severe damage level, and includes 87 the survey
        # never saw.
        self.impact_module = ImpactModule(
            self.scenario_id, domain_raster=SURVEYED_FLOOD_DEPTH_JUL_2021
        )
        with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
            obs_depth = src.read(1).astype(float)
            transform = src.transform

        hazard_forecast = {
            "17-07-2021 00:00:00": [obs_depth]
        }
        impact_forecast = self.impact_module.get_impact_forecast(hazard_forecast, SURVEYED_FLOOD_DEPTH_JUL_2021)
        self.test_against_placard_data(impact_forecast,)

            
    from sklearn.metrics import cohen_kappa_score, mean_squared_error

    def validate_impact_forecast(self, impact_forecast, current_time):
        """
        Validate the sensitivity of the exposure computation to the inundation model.

        The predicted inundation map is compared against the ground-truth inundation
        map after both have been processed using the same exposure computation
        method.
        """

        # Every forecast step is scored against the simulated map AT ITS OWN
        # VALID TIME. This used to take next(iter(...)) for the prediction and
        # flood_map_dict[current_time] for the truth, which compared a forecast
        # valid at t+1h against the map at t -- an hour out of phase, every
        # step. On a receding, tide-driven event that reads as a depth bias:
        # the forecast leads the truth, so it looks too deep while the water is
        # rising and too shallow through the whole recession, which is the sign
        # flip actually observed in the S48 metrics.
        #
        # Looping rather than taking the first step also means a horizon > 1 is
        # scored at every lead time instead of only the shortest, with lead_h
        # recorded so the leads can be separated afterwards.
        for valid_time, building_pred in impact_forecast["building_impact"].items():
            gt_file = self.bc_module.flood_map_dict.get(valid_time)
            if gt_file is None:
                # bc_dict is filtered to timestamps that have a map, so this is
                # not expected; say which step was skipped rather than score it
                # against the wrong hour.
                logger.warning(
                    "no simulated map at %s; that forecast step is not scored",
                    valid_time,
                )
                continue

            gt_hazard_map = load_flood_map(gt_file, self.bc_module.event_name)
            gt_impact = self.impact_module.get_impact_forecast(
                {str(valid_time): [gt_hazard_map]}
            )
            building_gt = next(iter(gt_impact["building_impact"].values())).copy()

            # ----------------------------------------------------------------
            # Remove buildings without sampled cells
            # ----------------------------------------------------------------
            building_pred = building_pred[building_pred["num_ring_cells"] > 0].copy()
            building_gt = building_gt[building_gt["num_ring_cells"] > 0].copy()

            # Everything is compared on the building frame. Since the impact
            # module became building-centred, each row already carries its
            # property's and its street's hazard tier, so the separate road and
            # property frames are redundant -- and they were never populated,
            # which is why this path used to raise StopIteration before it
            # compared anything.
            building_metrics = self.compute_metrics(building_gt, building_pred)

            self.write_results_simulated(
                current_time,
                building_metrics,
                model="flow_matching",
                valid_time=valid_time,
            )

    def write_results_simulated(
        self,
        timestamp,
        building_metrics,
        road_metrics=None,
        model="flow_matching",
        valid_time=None,
    ):
        # ``timestamp`` is the ISSUE time and ``valid_time`` the step being
        # scored; lead_h is the gap. Recorded because a horizon > 1 now writes
        # one row per lead rather than only the shortest, and a row that does
        # not say its own lead cannot be separated from the others afterwards.
        results = {
            "scenario_id":self.scenario_id,
            "timestamp":timestamp,
            "valid_time": valid_time,
            "lead_h": (
                None if valid_time is None
                else (pd.Timestamp(valid_time) - pd.Timestamp(timestamp))
                     .total_seconds() / 3600.0
            ),
            "model": model,

            # The ensemble the scores were produced at, so an M*K sweep is
            # separable afterwards. Without these every run of the sweep writes
            # indistinguishable rows into the same file, and the error-versus-
            # ensemble-size curve cannot be recovered from it.
            "num_deep_ensemble": self.num_deep_ensemble,
            "samples_per_member": self.samples_per_member,
            "ensemble_size": self.num_deep_ensemble * self.samples_per_member,

            # Building metrics
            **{f"building_{k}": v for k, v in building_metrics.items()},

            # Road metrics, when a separate road frame was compared. The
            # building-centred path folds the street tier into building_metrics
            # instead, as street_* keys.
            **{f"road_{k}": v for k, v in (road_metrics or {}).items()},
        }

        df = pd.DataFrame([results])

        output_file = os.path.join("simulated_validation_metrics.csv")

        # Same guard as the binary metrics writer below: the header is written
        # once, so a run whose columns differ from what is already there would
        # push its values under the wrong names. Renaming a tier column is
        # exactly that case. Archive the stale file rather than corrupt it.
        if os.path.exists(output_file):
            existing_cols = pd.read_csv(output_file, nrows=0).columns.tolist()
            if existing_cols != list(df.columns):
                stamp = time.strftime("%Y%m%d-%H%M%S")
                archived = f"{os.path.splitext(output_file)[0]}_{stamp}.csv.bak"
                os.rename(output_file, archived)
                logger.warning(
                    f"{os.path.basename(output_file)} had a different column set; "
                    f"archived it to {os.path.basename(archived)} and started a "
                    f"new file."
                )

        # Append if file exists
        df.to_csv(
            output_file,
            mode="a",
            header=not os.path.exists(output_file),
            index=False,
        )

        logger.info(f"Results written to {output_file}")
    

    def compute_metrics(self, gt, pred, object_type="building"):
        """Agreement between two runs of the SAME exposure method on different
        flood maps -- the simulated map as ground truth against the flow-matching
        surrogate's prediction.

        Compared per building, on the depths and on all five tiers the
        FIS produces: the building's own, its property's, its street's, and the
        combined risk. Those last three come straight off the building row now
        that the impact module is building-centred, so no separate road or
        property frame is needed.
        """
        id_col = "building_id" if object_type == "building" else "id"
        results = {}
        tier_columns = {
            # "ensemble_mean_hazard_level": "ensemble_mean_",
            "overall_building_hazard_level":"overall_building_hazard_level"
            # "final_risk_ensemble_mean": "mean_map_",
        }
        for column, prefix in tier_columns.items():
            if column not in gt.columns or column not in pred.columns:
                continue
            comparison = gt.merge(
                pred[[id_col, column]], on=id_col, suffixes=("_gt", "_pred"),
            )
            results.update(_fuzzy_indicator_metrics(comparison, column, prefix=prefix))

        # The ensemble's per-tier probabilities, already on the frame from the
        # FIS -- averaged over buildings only so they survive into the metrics
        # CSV, which is all a run persists.
        for term in TERMS + (f"at_least_{TERMS[MIDDLE]}",):
            column = f"overall_building_hazard_prob_{term}"
            if column in pred.columns:
                results[f"prob_{term}"] = float(pred[column].mean())

        return results

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
    
    def test_against_placard_data(self, impact_forecast, model="surveyed"):
        building_impact = next(iter(impact_forecast['building_impact'].values()))
        obs_placard_surveyed = self.get_obs_placard_filtered()  # Load placard data

        wanted = [
                    "building_id",
                    "depth_max",
                    "depth_mean",
                    "num_ring_cells",
                    "inundated",
                    "building_hazard_level",
                    "building_hazard_likelihood",
                    "property_known",
                    "property_hazard_level",
                    "property_hazard_likelihood",
                    "street_known",
                    "street_hazard_level",
                    "street_hazard_likelihood",
                    "overall_building_hazard_level",
                    "overall_building_hazard_likelihood",
                    "overall_building_hazard_prob_H1",
                    "overall_building_hazard_prob_H2_H3",
                    "overall_building_hazard_prob_H4_above",
                    "overall_building_hazard_prob_at_least_H2_H3",
                    "overall_building_hazard_rule",
                    "overall_building_hazard_driver",
                    "overall_building_hazard_bp_level",
                    "overall_building_hazard_bp_likelihood",
                    "overall_building_hazard_bp_rule",
                    "ensemble_mean_hazard_level",
                    "ensemble_mean_hazard_rule",
                    "final_risk_ensemble_mean",
                    "final_risk_ensemble_mean_rule",
                    "final_risk_level",
        ]
        missing = [column for column in wanted if column not in building_impact.columns]
        if missing:
            logger.info("not in the impact frame, so not scored: %s", missing)

        survyed_buildings = obs_placard_surveyed.merge(
            building_impact[[c for c in wanted if c in building_impact.columns]],
            on="building_id",
            how="left",
        )

        survyed_buildings.rename(
            columns={
                "depth_max": "max_depth",
                "depth_mean": "mean_depth",
            },
            inplace=True,
        )

        survyed_buildings["observed_affected"] = survyed_buildings["DamageNb"].apply(
            _damage_to_affected
        )
        
        risk_eval = survyed_buildings.dropna(
            subset=["building_hazard_level", "observed_affected",
                    "overall_building_hazard_bp_level", "max_depth", "mean_depth"]
        ).copy()
        
        y_true = risk_eval["observed_affected"]
        num_buildings = len(risk_eval)
        
        
        y_pred_building = _hazards_to_affected(risk_eval[["building_hazard_level"]])


        logger.info(
            f"scored {num_buildings} buildings; inputs available: "
            f"property {int(risk_eval['property_known'].sum())}, "
            f"street {int(risk_eval['street_known'].sum())} (street not scored)"
        )
        # Observed class for the same rows, so each indicator's misses and false
        # alarms can be attributed to the damage level they came from.
        damage_levels = risk_eval["DamageNb"]

        self.write_results(num_buildings, y_true, y_pred_building,
                           "building_hazard_level", model, "July2021",
                           damage_levels)

        #Also compute based on h_max, h_mean and the two combined-FIS tiers
        y_pred_h_max = self.depth_to_affected(risk_eval[["max_depth"]], threshold=0.30)
        self.write_results(num_buildings, y_true, y_pred_h_max,
                            "max_depth", model, "July2021", damage_levels)

        y_pred_h_mean = self.depth_to_affected(risk_eval[["mean_depth"]], threshold=0.30)
        self.write_results(num_buildings, y_true, y_pred_h_mean,
                            "mean_depth", model, "July2021", damage_levels)

        y_pred_overall = _hazards_to_affected(
            risk_eval[["overall_building_hazard_level"]])
        self.write_results(num_buildings, y_true, y_pred_overall,
                            "overall_building_hazard_level", model, "July2021",
                            damage_levels)

        y_pred_overall_bp = _hazards_to_affected(
            risk_eval[["overall_building_hazard_bp_level"]])
        self.write_results(num_buildings, y_true, y_pred_overall_bp,
                    "overall_building_hazard_bp_level", model, "July2021",
                    damage_levels)

        for column in ("ensemble_mean_hazard_level", "final_risk_ensemble_mean",
                       "final_risk_level"):
            if column not in risk_eval.columns:
                logger.info("%s not in the impact frame; not scored", column)
                continue
            self.write_results(
                num_buildings, y_true, _hazards_to_affected(risk_eval[[column]]),
                column, model, "July2021", damage_levels,
            )


    def depth_to_affected(self, depth, threshold=0.1):
        depth = pd.DataFrame(depth)
        affected = (depth >= threshold).any(axis=1)
        predicted = pd.Series(
            np.where(affected, AFFECTED, NOT_AFFECTED), index=depth.index, dtype=object
        )
        return predicted.mask(depth.isna().all(axis=1))
    
    def write_results(self, num_buildings, y_true, y_pred, indicator="building",
                      model="surveyed", timestamp=None, damage_levels=None):
        """Score the binary affected/not-affected comparison and append the
        contingency metrics to the validation CSV.

        Reported as the standard warning-verification set rather than an
        agreement statistic: POD/FAR/CSI/bias describe the decision a warning
        actually makes, while PSS and kappa credit correct negatives -- which
        only carry information now that the undamaged buildings are retained
        (see get_obs_placard_filtered).

        ``damage_levels`` is the observed DamageNb column for the same rows as
        ``y_true``; passing it adds the per-class miss / false-alarm split
        described in _errors_by_damage_level alongside the overall scores.
        """
        labels = [NOT_AFFECTED, AFFECTED]

        (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred, labels=labels)
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=[f"True {label}" for label in labels],
            columns=[f"Pred {label}" for label in labels],
        )

        output_dir = EXPOSURE_STUDY_RESULTS
        os.makedirs(output_dir, exist_ok=True)

        # --- Contingency scores ---
        pod = tp / (tp + fn) if (tp + fn) else 0.0        # hit rate / recall
        csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        

        by_damage = _errors_by_damage_level(y_true, y_pred, damage_levels)

        logger.info(f"Binary verification ({indicator}), affected = DamageNb >= "
                    f"{AFFECTED_DAMAGE_MIN}:")
        logger.info(f"  hits={tp} misses={fn} false_alarms={fp} correct_negatives={tn}")
        if damage_levels is not None:
            for level in DAMAGE_LEVELS:
                logger.info(
                    f"  DamageNb {level}: n={by_damage[f'n_damage_{level}']} "
                    f"misses={by_damage[f'misses_damage_{level}']} "
                    f"false_alarms={by_damage[f'false_alarms_damage_{level}']}"
                )

        metrics = {
            "scenario_id": self.scenario_id,
            "timestamp": timestamp,
            "num_buildings_evaluated": num_buildings,
            "damage_threshold": AFFECTED_DAMAGE_MIN,
            "number_affected": tp + fn,
            "number_not_affected": tn + fp,
            "hits": tp,
            "misses": fn,
            "false_alarms": fp,
            "correct_negatives": tn,
            "pod": pod,
            "csi": csi,
            "buffer_multiplier": self.buffer_m,
            "model": model,
            "indicator": indicator,
            **by_damage,
        }

        metrics_df = pd.DataFrame([metrics])
        # Separate file: these columns do not line up with the three-tier rows
        # already appended to risk_matrix_validation_metrics.csv.
        output_file = os.path.join(output_dir, "binary_validation_metrics.csv")

        # The file is appended to with the header written once, so a run whose
        # columns differ from what is already there would push its values under
        # the wrong names. Move the stale file aside instead of corrupting it.
        if os.path.exists(output_file):
            existing_cols = pd.read_csv(output_file, nrows=0).columns.tolist()
            if existing_cols != list(metrics_df.columns):
                stamp = time.strftime("%Y%m%d-%H%M%S")
                archived = f"{os.path.splitext(output_file)[0]}_{stamp}.csv.bak"
                os.rename(output_file, archived)
                logger.warning(
                    f"{os.path.basename(output_file)} had a different column set "
                    f"(metrics added or removed); archived it to "
                    f"{os.path.basename(archived)} and started a new file."
                )

        metrics_df.to_csv(
            output_file,
            mode="a",                                  # Append
            header=not os.path.exists(output_file),    # Write header only once
            index=False,
        )
        logger.info(f"Binary validation metrics saved to {output_file}")

        return metrics, cm_df



    
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Binary outcome labels for the placard validation. The FIS still reports three
# tiers; these are only the collapsed labels the scoring works in.
AFFECTED = "affected"
NOT_AFFECTED = "not_affected"

# Lowest MBIE damage class counted as "affected". 1 (Slight damage) makes the
# outcome "water got into the building", which is the inundation threshold a
# depth-driven hazard model actually predicts; raising it to 2 or 3 moves the
# test towards structural failure, which is force-driven and needs the velocity
# this model does not carry. Exposed as a constant so the cut can be swept.
AFFECTED_DAMAGE_MIN = 1

# The MBIE damage classes the survey uses (0 = no damage ... 3 = heavy). Fixed
# rather than read off the data because the per-level columns are appended to a
# CSV whose header is written once: a level set that shrinks when a scenario
# happens to contain no class-3 building would silently misalign later rows.
DAMAGE_LEVELS = (0, 1, 2, 3)


def _errors_by_damage_level(y_true, y_pred, damage_levels):
    """Split the misses and false alarms across the observed damage classes.

    The binary score collapses DamageNb to affected/not-affected, which hides
    whether the misses are slight-damage buildings the depth threshold sits on
    top of or heavily damaged ones the model had no business missing. This
    keeps the same contingency table and only asks which observed class each
    error came from.

    Returns flat ``*_damage_<k>`` columns so the breakdown rides along in the
    existing one-row-per-indicator CSV. ``n_damage_<k>`` is the number of
    scored buildings in that class, so hits (k >= AFFECTED_DAMAGE_MIN) and
    correct negatives (k = 0) are the complement of the error column and are
    not stored twice. Note the two error columns are mutually exclusive by
    construction: only class 0 can produce a false alarm and only the affected
    classes can produce a miss -- the zeros on the other side are kept so the
    schema does not depend on where AFFECTED_DAMAGE_MIN is cut.
    """
    if damage_levels is None:
        return {
            f"{stat}_damage_{level}": pd.NA
            for level in DAMAGE_LEVELS
            for stat in ("n", "misses", "false_alarms")
        }

    if not isinstance(damage_levels, pd.Series):
        damage_levels = pd.Series(damage_levels, index=y_true.index)
    # Align on the scored rows: the caller hands in the column off the same
    # frame y_true came from, so this only guards against a stray extra row.
    levels = damage_levels.reindex(y_true.index)

    missed = (y_true == AFFECTED) & (y_pred == NOT_AFFECTED)
    false_alarms = (y_true == NOT_AFFECTED) & (y_pred == AFFECTED)

    breakdown = {}
    for level in DAMAGE_LEVELS:
        at_level = levels == level
        breakdown[f"n_damage_{level}"] = int(at_level.sum())
        breakdown[f"misses_damage_{level}"] = int((at_level & missed).sum())
        breakdown[f"false_alarms_damage_{level}"] = int(
            (at_level & false_alarms).sum()
        )
    return breakdown


def _damage_to_affected(damage_nb):
    """Collapse an observed MBIE damage class to affected / not_affected.

    DamageNb 0 (No damage / Unaffected) is the negative case; anything at or
    above AFFECTED_DAMAGE_MIN is positive. Damaged classes below the threshold
    return NaN and drop out of the scoring rather than being counted as
    negatives, so raising the cut removes the ambiguous middle instead of
    relabelling buildings that were demonstrably affected as unaffected.
    """
    if int(damage_nb) == 0:
        return NOT_AFFECTED
    if int(damage_nb) in (1, 2, 3):
        return AFFECTED
    raise Exception

def _hazards_to_affected(levels):
    """Collapse one or more predicted FIS tier columns to affected/not_affected.

    ``levels`` is a frame of tier columns (1/2/3, NaN where the object had no
    such link). Medium (2) and high (3) are the tiers that carry a warning, so
    a row is predicted affected if ANY column is above low; low (1) in every
    column is the negative. The placard scoring passes a single already-
    combined tier, so the OR only bites if several columns are handed in.

    NaN tiers contribute nothing rather than voting "not affected": an object
    that was never assessed is not evidence of a dry surrounding, the same
    reasoning that marginalises it out of the combined FIS
    (see overall_building_hazard).
    A row whose columns are all NaN returns NaN and drops out of the scoring.
    """
    levels = pd.DataFrame(levels)
    # NaN > 1 is False, so unknown tiers fall out of the OR on their own.
    affected = (levels > 1).any(axis=1)
    predicted = pd.Series(
        np.where(affected, AFFECTED, NOT_AFFECTED), index=levels.index, dtype=object
    )
    return predicted.mask(levels.isna().all(axis=1))


def _hit_rate_and_csi(y_true, y_pred, classes, labels):
    """Compute one-vs-rest hit rate (recall) and Critical Success Index (CSI)
    for each category.

    hit_rate = tp / (tp + fn)
    csi      = tp / (tp + fp + fn)
    """
    results = {}
    for c in classes:
        is_true_c = (y_true == c)
        is_pred_c = (y_pred == c)

        tp = int((is_true_c & is_pred_c).sum())
        fp = int((~is_true_c & is_pred_c).sum())
        fn = int((is_true_c & ~is_pred_c).sum())

        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        results[labels[c]] = {"hit_rate": hit_rate, "csi": csi}

    return results


def _indicator_metrics(comparison, depth_mean_col, depth_max_col, risk_mean_col, risk_max_col, prefix=""):
    """RMSE/QWK/hit-rate/CSI for one gt-vs-pred indicator (structural or
    access risk), given an already-merged (``_gt``/``_pred`` suffixed)
    comparison frame. Column-name-parameterised so the identical structural-
    risk computation in Integrator.compute_metrics can be reused for access
    risk without duplicating the metric logic.
    """
    comparison = comparison.dropna(
        subset=[
            f"{depth_mean_col}_gt", f"{depth_mean_col}_pred",
            f"{depth_max_col}_gt", f"{depth_max_col}_pred",
            f"{risk_mean_col}_gt", f"{risk_mean_col}_pred",
            f"{risk_max_col}_gt", f"{risk_max_col}_pred",
        ]
    ).copy()

    rmse_mean = root_mean_squared_error(comparison[f"{depth_mean_col}_gt"], comparison[f"{depth_mean_col}_pred"])
    rmse_max = root_mean_squared_error(comparison[f"{depth_max_col}_gt"], comparison[f"{depth_max_col}_pred"])

    y_true_mean = comparison[f"{risk_mean_col}_gt"].astype(int)
    y_pred_mean = comparison[f"{risk_mean_col}_pred"].astype(int)
    y_true_max = comparison[f"{risk_max_col}_gt"].astype(int)
    y_pred_max = comparison[f"{risk_max_col}_pred"].astype(int)

    qwk_mean = cohen_kappa_score(y_true_mean, y_pred_mean, weights="quadratic")
    qwk_max = cohen_kappa_score(y_true_max, y_pred_max, weights="quadratic")

    classes = [1, 2, 3]
    class_labels = {1: "Yellow", 2: "Orange", 3: "Red"}

    hit_csi_mean = _hit_rate_and_csi(y_true_mean, y_pred_mean, classes, class_labels)
    hit_csi_max = _hit_rate_and_csi(y_true_max, y_pred_max, classes, class_labels)

    results = {
        f"{prefix}qwk_mean": qwk_mean,
        f"{prefix}qwk_max": qwk_max,
        f"{prefix}rmse_mean": rmse_mean,
        f"{prefix}rmse_max": rmse_max,
    }
    for label in class_labels.values():
        results[f"{prefix}hit_rate_{label}_mean"] = hit_csi_mean[label]["hit_rate"]
        results[f"{prefix}csi_{label}_mean"] = hit_csi_mean[label]["csi"]
        results[f"{prefix}hit_rate_{label}_max"] = hit_csi_max[label]["hit_rate"]
        results[f"{prefix}csi_{label}_max"] = hit_csi_max[label]["csi"]

    return results


def _fuzzy_indicator_metrics(comparison, risk_col, prefix="fuzzy_"):
    """QWK/hit-rate/CSI for a single-valued risk indicator (the fuzzy risk
    tier), given an already-merged (``_gt``/``_pred`` suffixed) comparison
    frame. Counterpart to _indicator_metrics for the fuzzy classifier, which
    -- unlike the crisp max/mean design -- produces one joint risk tier per
    building rather than a mean/max pair, and no depth RMSE of its own.
    """
    comparison = comparison.dropna(
        subset=[f"{risk_col}_gt", f"{risk_col}_pred"]
    ).copy()

    y_true = comparison[f"{risk_col}_gt"].astype(int)
    y_pred = comparison[f"{risk_col}_pred"].astype(int)

    classes = [1, 2, 3]
    class_labels = {1: "Yellow", 2: "Orange", 3: "Red"}

    hit_csi = _hit_rate_and_csi(y_true, y_pred, classes, class_labels)
    results = {
        f"{prefix}qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    for label in class_labels.values():
        results[f"{prefix}hit_rate_{label}"] = hit_csi[label]["hit_rate"]
        results[f"{prefix}csi_{label}"] = hit_csi[label]["csi"]

    return results

