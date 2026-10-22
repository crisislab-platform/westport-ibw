from modules.data_module.bc_module import BC_Module
from modules.hazard_module.model_client import FlowMatchingModelClient
from modules.risk_module.risk_module import (
    RiskModule, WET_DEPTH_MIN, BUILDPROP_MAX_PREFIX, REPORTED_PREFIX, window_peak,
)
from modules.risk_module.risk_module_helper import (
    CLASS_COLUMN_TERMS, CONFIGURATIONS, HAZARD_LEVELS, MIDDLE_CLASS,
    class_lower_bounds, configuration_prefix,
)
from modules.update_module.arcgis_module import ArcGISModule
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
import shutil
import time
from rasterio.transform import rowcol
from datetime import timedelta

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

# The BuildProp-Max depth-exceedance file: ONE ROW PER BUILDING PER FORECAST
# STEP, which is what separates it from every other CSV the runner writes --
# those summarise a run to a handful of rows, this one keeps the building axis
# so a map or a time series can be drawn from it.
#
# The three probability columns are P(h >= the depth that class starts at),
# taken over the ensemble on the BuildProp-Max context indicator
# max(h_max_building, h_max_property) -- Equation (8) on Equation (7). The
# hazard class those probabilities elect travels beside them as
# bp_hazard_level, with its likelihood, so a row carries both axes of the risk
# matrix and the distribution they were read from.
EXCEEDANCE_FILE = "hazard_exceedance_buildprop_max.csv"

# The per-building forecast/outcome PAIRS, written only when a simulated map is
# available to verify against. Separate from EXCEEDANCE_FILE because that one is
# a forecast product -- written in a live run, where no outcome exists yet --
# while this is a verification artefact.
#
# It exists because reliability cannot be recovered from an aggregate. The
# metrics CSV holds one row per timestep, and a reliability diagram needs the
# individual (forecast probability, observed event) pairs so they can be
# re-binned, stratified by lead time and block-bootstrapped afterwards without
# rerunning the pipeline.
VERIFICATION_PAIRS_FILE = "verification_pairs_buildprop_max.csv"

# The horizon window scored as ONE class per building: one row per issue per
# configuration, not per forecast step. Separate from simulated_validation_metrics
# because it answers a different question -- that file scores each step against
# the simulated map of the same hour, which is what the surrogate is asked to do;
# this one scores what the WARNING issued at that hour promises over its whole
# window, which is the worst class the building reaches anywhere in it.
HORIZON_METRICS_FILE = "horizon_window_metrics.csv"

# Frame column -> published name, forecast side then outcome. Same role as
# EXCEEDANCE_COLUMNS: this tuple IS the schema of the file.
VERIFICATION_PAIR_COLUMNS = tuple(
    (f"{BUILDPROP_MAX_PREFIX}_prob_at_least_{term}", f"prob_at_least_{term}")
    for term in CLASS_COLUMN_TERMS
)

# Frame column -> published name. Ordered, and the file is appended to with the
# header written once, so this tuple IS the schema (see the drift guard in
# write_depth_exceedance).
EXCEEDANCE_COLUMNS = (
    ("building_id", "building_id"),
    ("num_ring_cells", "num_ring_cells"),
    # Whether the building had a parcel at all. Where it did not, the depth is
    # its footprint's alone (buildprop_max_depth), and the row should not be
    # read as a building whose parcel happened to be dry.
    ("property_known", "property_known"),
    (f"{BUILDPROP_MAX_PREFIX}_hazard_level", "bp_hazard_level"),
    (f"{BUILDPROP_MAX_PREFIX}_likelihood", "bp_hazard_likelihood"),
    # Ensemble-mean h_max of the footprint alone, and of the footprint-or-parcel
    # depth the probabilities were taken on. Both are means of the maxima.
    ("depth_max", "depth_max_building"),
    (f"{BUILDPROP_MAX_PREFIX}_indicator_mean", "depth_max_buildprop"),
) + tuple(
    (f"{BUILDPROP_MAX_PREFIX}_prob_at_least_{term}", f"prob_at_least_{term}")
    for term in CLASS_COLUMN_TERMS
)

# What the run is, ahead of the building axis, so rows from different issues,
# leads, scenarios and ensemble sizes can be told apart in one appended file.
EXCEEDANCE_KEYS = (
    "scenario_id", "issue_time", "valid_time", "lead_h", "ensemble_size",
)


def _stamp(value):
    """One timestamp spelling for every appended results file.

    Formatted rather than left as a Timestamp because a chunk whose stamps all
    fall on midnight writes as a bare date, so an appended file ends up with two
    time formats in one column and anything reading it back has to parse both.
    """
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


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


def _align_appended_csv(output_file, columns):
    """Make an appended results file's header match ``columns`` before writing.

    Every results file here is written with ``mode="a"`` and the header written
    ONCE, so the columns of a later run have to line up with the header already
    on disk -- positionally, not just by name -- or the run's values land under
    the wrong names. Two cases, and they are NOT the same failure:

    * The new run only ADDS columns (a statistic added to the ablation, say).
      Nothing already in the file changes meaning, so the file is REWRITTEN with
      the new columns in place and empty on the existing rows, and the run
      appends to it. The accumulated runs survive a widened schema, and an empty
      cell says exactly what happened: that run did not compute it.
    * Anything else -- a column renamed, dropped, or one whose meaning moved --
      can silently re-point a column at a different quantity, which is what this
      guard exists to stop. The file is archived and a new one started.

    Both cases leave a timestamped ``.csv.bak`` copy, so the rewrite is
    recoverable too. Returns the path of that copy, or None if the header
    already matched.
    """
    if not os.path.exists(output_file):
        return None

    existing = pd.read_csv(output_file, nrows=0).columns.tolist()
    columns = list(columns)
    if existing == columns:
        return None

    name = os.path.basename(output_file)
    stem = os.path.splitext(output_file)[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Second-resolution stamps collide if one file is migrated twice inside the
    # same second, and os.rename would overwrite the earlier copy without a
    # word -- the one thing a backup must not do.
    archived = f"{stem}_{stamp}.csv.bak"
    for attempt in range(2, 100):
        if not os.path.exists(archived):
            break
        archived = f"{stem}_{stamp}-{attempt}.csv.bak"

    # Subset, not strict subset: a pure REORDER is also safe to migrate, since
    # the rewrite below places every existing value by name.
    if set(existing) <= set(columns):
        shutil.copy2(output_file, archived)
        added = [column for column in columns if column not in existing]
        pd.read_csv(output_file).reindex(columns=columns).to_csv(
            output_file, index=False)
        change = (
            f"gained {len(added)} column(s) ({', '.join(added[:3])}"
            f"{', ...' if len(added) > 3 else ''})" if added
            else "kept its columns in a different order")
        logger.warning(
            "%s %s; rewrote it to match, existing rows carry the new columns "
            "empty, copy at %s", name, change, os.path.basename(archived),
        )
        return archived

    os.rename(output_file, archived)
    logger.warning(
        "%s had an incompatible column set (renamed or removed, not just "
        "added); archived it to %s and started a new file.",
        name, os.path.basename(archived),
    )
    return archived


class Integrator():
    def __init__(self, scenario_id, forecast_horizon, push_to_arcgis,samples_per_member, num_deep_ensemble, min_exceed_prob):
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
        self.impact_module = RiskModule(scenario_id, min_exceed_prob)
        if self.push_to_arcgis:
            self.arcgis_module = ArcGISModule()
        self.device = check_device()
        # Simulated-map exposure frames, keyed by valid time and bounded to one
        # horizon (see _ground_truth_buildings). Hourly issues with a 12 h
        # horizon ask for the same step in twelve consecutive windows, and both
        # validation paths ask for it independently; without this the exposure
        # pass over the simulated map runs ~24 times per step instead of once.
        self._gt_buildings = {}
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
            end_time = end_time - timedelta(hours=self.forecast_horizon)
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
                    step_impact = self.impact_module.get_impact_forecast(
                        {timestamp: hazard}
                    )
                    _merge_impact(impact_forecast, step_impact)
                # Per step, off this step's frames, before they are folded into
                # the accumulator -- and outside the timer, since writing the
                # file is not part of the risk computation it measures.
                self.write_depth_exceedance(step_impact, current_time)
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
                # self.validate_impact_forecast(impact_forecast, current_time)
                # The same steps scored again as ONE class per building over the
                # whole window -- what the warning issued at current_time
                # promises, rather than what each hour of it does.
                self.validate_impact_forecast_over_horizon(
                    impact_forecast, current_time)

            #Update feature layers on spatial db. current_time is the issue
            #time of THIS forecast; the loop bounds say nothing about the rows
            #being written here.
            if self.push_to_arcgis:
                # The peak layers ride along in the same dict, so the publisher
                # needs no special case: they are just three more entries, each
                # a one-timestamp series keyed by the issue.
                impact_forecast.update(
                    _window_peak_frames(impact_forecast, current_time))
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

    def write_depth_exceedance(self, impact_forecast, issue_time):
        """Append this step's BuildProp-Max exceedance probabilities to
        out/hazard_exceedance_buildprop_max.csv, one row per building.

        Called per forecast step and per building rather than summarised,
        because the question the file answers -- how likely is this building to
        be at or past each hazard class boundary, and when -- has both axes in
        it. 
        
        The columns are EXCEEDANCE_KEYS + EXCEEDANCE_COLUMNS; the
        probabilities themselves are computed in the impact module, on the
        member depth stacks, since that is the only place they still exist.

        Written OUTSIDE the risk_s timer by its caller: this is reporting, not
        the risk computation the horizon sweep measures.
        """
        frames = []
        for valid_time, building_impact in impact_forecast["building_impact"].items():
            missing = [column for column, _ in EXCEEDANCE_COLUMNS
                       if column not in building_impact.columns]
            if missing:
                logger.warning(
                    "building frame at %s is missing %s; no exceedance row written",
                    valid_time, missing,
                )
                continue
            rows = pd.DataFrame({
                published: building_impact[column].to_numpy()
                for column, published in EXCEEDANCE_COLUMNS
            })
            issue, valid = pd.Timestamp(issue_time), pd.Timestamp(valid_time)
            keys = {
                "scenario_id": self.scenario_id,
                # Formatted rather than left as datetimes: a chunk whose stamps
                # all fall on midnight writes as a bare date, so an appended
                # file would carry two time formats in one column.
                "issue_time": issue.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_time": valid.strftime("%Y-%m-%d %H:%M:%S"),
                "lead_h": (valid - issue).total_seconds() / 3600.0,
                "ensemble_size": self.num_deep_ensemble * self.samples_per_member,
            }
            # Prepended through EXCEEDANCE_KEYS rather than in written order, so
            # the tuple is what fixes the schema and a key added to one without
            # the other raises here instead of silently reordering the file.
            for name in reversed(EXCEEDANCE_KEYS):
                rows.insert(0, name, keys[name])
            frames.append(rows)

        if not frames:
            return
        rows = pd.concat(frames, ignore_index=True)

        os.makedirs(EXPOSURE_STUDY_RESULTS, exist_ok=True)
        output_file = os.path.join(EXPOSURE_STUDY_RESULTS, EXCEEDANCE_FILE)

        # Same guard as the other appending writers: the header is written
        # once, so a run whose columns differ from what is already there would
        # push its values under the wrong names. Renaming a term or adding a
        # class is exactly that case. Archive the stale file rather than
        # corrupt it.
        _align_appended_csv(output_file, rows.columns)

        rows.to_csv(output_file, mode="a",
                    header=not os.path.exists(output_file), index=False)

        bounds = class_lower_bounds()
        logger.info(
            "%s: %d exceedance rows (%d buildings x %d steps) -> %s; "
            "class lower bounds %s",
            issue_time, len(rows), len(frames[0]), len(frames), EXCEEDANCE_FILE,
            ", ".join(f"{term} >= {bound:.2f} m" for term, bound in bounds.items()),
        )

    def write_verification_pairs(self, building_gt, building_pred,
                                 issue_time, valid_time,
                                 observed_column=f"{BUILDPROP_MAX_PREFIX}_hazard_level"):
        """Append this step's per-building (forecast, outcome) pairs.

        One row per building: the three exceedance probabilities the forecast
        gave it, and the hazard class the SAME configuration assigned the
        simulated map. That pairing is the raw material of every calibration
        statistic -- the reliability diagram, the Brier decomposition, and any
        re-binning or lead-time stratification of them.

        Written beside the metrics rather than instead of them: compute_metrics
        reduces these same rows to one line, and a reduction cannot be inverted.

        The two frames are merged on building_id rather than assumed aligned;
        they are filtered independently for ring coverage upstream, so their
        row sets are not guaranteed to match.
        """
        forecast_columns = [column for column, _ in VERIFICATION_PAIR_COLUMNS]
        missing = ([observed_column] if observed_column not in building_gt.columns
                   else []) + [c for c in forecast_columns
                               if c not in building_pred.columns]
        if missing:
            logger.info("no verification pairs at %s; frames lack %s",
                        valid_time, missing)
            return

        merged = building_gt[["building_id", observed_column]].merge(
            building_pred[["building_id"] + forecast_columns], on="building_id",
        ).dropna(subset=[observed_column] + forecast_columns)
        if merged.empty:
            return

        rows = pd.DataFrame({
            "building_id": merged["building_id"].to_numpy(),
            "observed_level": merged[observed_column].to_numpy(dtype=int),
            **{published: merged[column].to_numpy(dtype=float)
               for column, published in VERIFICATION_PAIR_COLUMNS},
        })
        issue, valid = pd.Timestamp(issue_time), pd.Timestamp(valid_time)
        keys = {
            "scenario_id": self.scenario_id,
            "issue_time": issue.strftime("%Y-%m-%d %H:%M:%S"),
            "valid_time": valid.strftime("%Y-%m-%d %H:%M:%S"),
            "lead_h": (valid - issue).total_seconds() / 3600.0,
            "ensemble_size": self.num_deep_ensemble * self.samples_per_member,
        }
        for name in reversed(EXCEEDANCE_KEYS):
            rows.insert(0, name, keys[name])

        os.makedirs(EXPOSURE_STUDY_RESULTS, exist_ok=True)
        output_file = os.path.join(EXPOSURE_STUDY_RESULTS, VERIFICATION_PAIRS_FILE)
        _align_appended_csv(output_file, rows.columns)
        rows.to_csv(output_file, mode="a",
                    header=not os.path.exists(output_file), index=False)
        logger.info("%s -> %d verification pairs in %s",
                    valid_time, len(rows), VERIFICATION_PAIRS_FILE)

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
        logger.info(
            f"Validating forecast for scenario {self.scenario_id} "
            f"from {start_time} to {end_time}"
        )
        self.run_hourly_forecast(start_time=start_time, end_time=end_time,
                                 test_mode=True)
        
    def validate_exposure_method(self):
        # Scored against the surveyed raster, so the surveyed raster defines the
        # population: the buildings it has depths for. The module built in
        # __init__ is bounded by the flood model's grid, which covers different
        # ground -- it misses 542 buildings the survey reaches, two of them
        # placarded at the most severe damage level, and includes 87 the survey
        # never saw.
        self.impact_module = RiskModule(
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

        for valid_time, building_pred in impact_forecast["building_impact"].items():
            building_gt = self._ground_truth_buildings(valid_time)
            if building_gt is None:
                continue

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

            # The pairs behind those scores, per building, so the calibration
            # can be looked at rather than only summarised.
            self.write_verification_pairs(
                building_gt, building_pred, current_time, valid_time)

            self.write_results_simulated(
                current_time,
                building_metrics,
                model="flow_matching",
                valid_time=valid_time,
            )
  
            
    def _ground_truth_buildings(self, valid_time):
        """The simulated map at ``valid_time``, through the same exposure pass.

        The building frame the forecast is scored against: one "member", so the
        exceedance probabilities it carries are 0 or 1 and its hazard class is
        the deterministic class of that map's context indicator. Returns None
        when no simulated map exists for the step, having said which step that
        was -- bc_dict is filtered to timestamps that have a map, so it is not
        expected, and a step scored against the wrong hour is worse than a step
        not scored at all.

        Cached because the same valid time is asked for repeatedly: hourly
        issues with a horizon of H put each step in H consecutive windows, and
        both validation paths ask for it independently. The cache is bounded to
        one horizon and evicted oldest-first, which is enough -- the windows
        advance an hour at a time, so nothing older than the current one is ever
        asked for again -- and keeps the memory to one window's worth of frames
        rather than the whole event's.

        The frame is handed back UNCOPIED; callers filter it, which already
        makes a new frame. Anything that mutates in place must copy first.
        """
        valid_time = pd.Timestamp(valid_time)
        if valid_time in self._gt_buildings:
            return self._gt_buildings[valid_time]

        gt_file = self.bc_module.flood_map_dict.get(valid_time)
        if gt_file is None:
            logger.warning(
                "no simulated map at %s; that forecast step is not scored",
                valid_time,
            )
            return None

        gt_hazard_map = load_flood_map(gt_file, self.bc_module.event_name)
        gt_impact = self.impact_module.get_impact_forecast(
            {str(valid_time): [gt_hazard_map]}
        )
        frame = next(iter(gt_impact["building_impact"].values()))

        self._gt_buildings[valid_time] = frame
        while len(self._gt_buildings) > max(self.forecast_horizon, 1) + 1:
            # dicts keep insertion order, and the windows advance forwards, so
            # the oldest entry is the one furthest behind the current window.
            self._gt_buildings.pop(next(iter(self._gt_buildings)))
        return frame

    def validate_impact_forecast_over_horizon(self, impact_forecast, issue_time,
                                              prefixes=None):
        """Score the whole forecast window as ONE hazard class per building.

        ``validate_impact_forecast`` scores every step against the simulated map
        of the same hour, which is the right question to ask of the surrogate and
        the wrong one to ask of the warning: a warning issued at ``issue_time``
        covers the whole horizon, so what it promises a building is the WORST
        class it reaches anywhere in the window, not the class it holds at any
        one hour. Both sides are collapsed the same way -- the max over the
        window's steps -- and the pair is scored once per issue, which is the
        row this writes.

        Collapsing with a MAX rather than reading the class at the peak-depth
        step keeps the two sides symmetric and is the rule the risk matrix
        already places a building by (the highest class it reaches, Equation 8),
        so a forecast is not punished for putting the peak an hour early. What
        it does hide is timing, and deliberately: the window is the unit of the
        warning, and when inside it the water arrives is a separate question.

        The likelihood travels with the class it belongs to. For each building
        the step kept is one that actually reached its window class, tie-broken
        by the HIGHEST likelihood among the steps that reached it -- the
        strongest statement the forecast made at the class it is scored on.

        One row per issue per configuration in out/horizon_window_metrics.csv,
        defaulting to BuildProp-Max, the configuration the rest of the
        probabilistic verification is written for. More prefixes add ROWS, not
        columns, so the ablation can be swept without migrating the file.
        """
        prefixes = (BUILDPROP_MAX_PREFIX,) if prefixes is None else tuple(prefixes)
        levels = {prefix: f"{prefix}_hazard_level" for prefix in prefixes}
        likelihoods = {prefix: f"{prefix}_likelihood" for prefix in prefixes}

        # Sliced to the scored columns before they are concatenated: the frames
        # carry geometry and every configuration's columns, and the window holds
        # all of its steps at once.
        observed_steps, forecast_steps, scored_times = [], [], []
        for valid_time, building_pred in impact_forecast.get(
                "building_impact", {}).items():
            building_gt = self._ground_truth_buildings(valid_time)
            if building_gt is None:
                continue
            # The same ring-coverage filter the per-step scoring applies: a
            # building with no sampled cell has no indicator to collapse.
            observed_steps.append(_take_columns(
                building_gt[building_gt["num_ring_cells"] > 0],
                ["building_id"] + list(levels.values())))
            forecast_steps.append(_take_columns(
                building_pred[building_pred["num_ring_cells"] > 0],
                ["building_id"] + list(levels.values())
                + list(likelihoods.values())))
            scored_times.append(pd.Timestamp(valid_time))

        if not scored_times:
            logger.info("%s: no simulated map anywhere in the window; no horizon "
                        "row written", issue_time)
            return

        observed = pd.concat(observed_steps, ignore_index=True)
        forecast = pd.concat(forecast_steps, ignore_index=True)

        rows = []
        for prefix in prefixes:
            level, likelihood = levels[prefix], likelihoods[prefix]
            # dict.fromkeys, not a plain concatenation: the class column can
            # be absent on BOTH sides, and naming it twice reads as two faults.
            missing = list(dict.fromkeys(
                ([level] if level not in observed.columns else [])
                + [column for column in (level, likelihood)
                   if column not in forecast.columns]))
            if missing:
                logger.info("no horizon metrics for %s; the frames lack %s",
                            prefix, missing)
                continue

            window = _window_peak(observed, forecast, level, likelihood)
            if window.empty:
                logger.info("no horizon metrics for %s; no building carries a "
                            "class on both sides of the window", prefix)
                continue

            configuration, indicators_used = _configuration_name(prefix)
            rows.append({
                "scenario_id": self.scenario_id,
                # The issue time IS the start of the window (get_bc_forecast
                # runs from it inclusive); window_start/_end are the SCORED
                # span, which is shorter wherever a step had no simulated map.
                "issue_time": _stamp(issue_time),
                "window_start": _stamp(min(scored_times)),
                "window_end": _stamp(max(scored_times)),
                "forecast_horizon": self.forecast_horizon,
                "num_steps": len(scored_times),
                "num_deep_ensemble": self.num_deep_ensemble,
                "samples_per_member": self.samples_per_member,
                "ensemble_size": self.num_deep_ensemble * self.samples_per_member,
                "configuration": configuration,
                "indicators_used": indicators_used,
                **self.compute_horizon_metrics(window),
            })

        if not rows:
            return
        self.write_horizon_results(rows)

    def compute_horizon_metrics(self, window):
        """Compute pooled POD, FAR, CSI and signed risk-level bias.

        ``window`` is one row per building: the class the simulated maps reach
        over the window (``observed``), the class the forecast reaches
        (``forecast``), and the likelihood the forecast attached to its OWN
        class (``likelihood``).

        The ordinal comparison uses mutually exclusive outcomes: exact class
        agreement is a hit, an underestimate is a miss, and an overestimate is
        a false alert. This avoids counting an overestimate as both a hit and a
        false alert. ``risk_level_bias`` is forecast minus observed, averaged
        over buildings.
        """
        observed = window["observed"].astype(int)
        forecast = window["forecast"].astype(int)

        n_hit = int((forecast == observed).sum())
        n_miss = int((forecast < observed).sum())
        n_false = int((forecast > observed).sum())

        pod_denominator = n_hit + n_miss
        far_denominator = n_hit + n_false
        csi_denominator = n_hit + n_miss + n_false
        return {
            "n_buildings": int(len(window)),
            "n_hit": n_hit,
            "n_miss": n_miss,
            "n_false": n_false,
            "pod": n_hit / pod_denominator if pod_denominator else np.nan,
            "far": n_false / far_denominator if far_denominator else np.nan,
            "csi": n_hit / csi_denominator if csi_denominator else np.nan,
            "risk_level_bias": float((forecast - observed).mean()),
        }

    def write_horizon_results(self, rows):
        """Append this issue's window rows to out/horizon_window_metrics.csv.

        Appended per issue rather than summarised at the end, like every other
        results file here, so an interrupted sweep still leaves usable rows.
        """
        frame = pd.DataFrame(rows)

        os.makedirs(EXPOSURE_STUDY_RESULTS, exist_ok=True)
        output_file = os.path.join(EXPOSURE_STUDY_RESULTS, "horizon", HORIZON_METRICS_FILE)
        # The header is written once, so the columns of this run have to line up
        # with the file's; see _align_appended_csv for the two ways they can
        # differ and what each one costs.
        _align_appended_csv(output_file, frame.columns)
        frame.to_csv(output_file, mode="a",
                     header=not os.path.exists(output_file), index=False)

        for row in rows:
            logger.info(
                "%s +%dh window (%d steps, %d buildings) %s: "
                "POD %.2f | FAR %.2f | CSI %.2f | bias %+.2f -> %s",
                row["issue_time"], row["forecast_horizon"], row["num_steps"],
                row["n_buildings"], row["configuration"],
                row["pod"], row["far"], row["csi"],
                row["risk_level_bias"],
                HORIZON_METRICS_FILE,
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
            "timestamp":_stamp(timestamp),
            "valid_time": _stamp(valid_time),
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

        # The header is written once, so a run whose columns differ from what
        # is already there would push its values under the wrong names. Adding
        # a configuration to the ablation only widens the schema and the file is
        # migrated in place; renaming a tier column is the other case, and the
        # stale file is archived rather than corrupted. See _align_appended_csv.
        _align_appended_csv(output_file, df.columns)

        # Append if file exists
        df.to_csv(
            output_file,
            mode="a",
            header=not os.path.exists(output_file),
            index=False,
        )

        logger.info(f"Results written to {output_file}")
    

    def compute_metrics(self, gt, pred, object_type="building"):
        id_col = "building_id" if object_type == "building" else "id"
        results = {}
        for prefix in CONFIGURATION_PREFIXES:
            # The ensemble collapsed BEFORE the class: the mean of the members'
            # aggregated indicators, classified once. Not the pooled vote of the
            # members' classes, and not the class of the ensemble-mean flood map
            # -- both of those are on the frame too (see the finder's columns).
            column = f"{prefix}_ensemble_mean_hazard_level"
            if column in gt.columns and column in pred.columns:
                comparison = gt.merge(
                    pred[[id_col, column]], on=id_col, suffixes=("_gt", "_pred"),
                )
                # Only the ordered-agreement number is published. The helper also
                # returns per-class hit rate and CSI; they are dropped here to
                # keep the row to the reported set, and restoring them is
                # `results.update(metrics)` instead of the single key below.
                metrics = _class_indicator_metrics(comparison, column,
                                                   prefix=f"{prefix}_")
                results[f"{prefix}_qwk"] = metrics[f"{prefix}_qwk"]

            results.update(_exceedance_metrics(gt, pred, id_col, prefix=prefix))

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

        # One column set per scored configuration, plus the per-object
        # diagnostics that explain a row. Built from the registry rather than
        # listed, so a configuration cannot be scored without its columns being
        # pulled across (see SCORED_CONFIGURATIONS).
        wanted = [
                    "building_id",
                    "depth_max",
                    "depth_mean",
                    "num_ring_cells",
                    "property_known",
                    "street_known",
        ] + [
            f"{prefix}_{suffix}"
            for prefix in CONFIGURATION_PREFIXES
            for suffix in ("indicator_mean", "hazard_level", "hazard_class",
                           "likelihood", "likelihood_class",
                           "prob_at_least_H2", "prob_at_least_H3_above",
                           "ensemble_mean_hazard_level",
                           "mean_map_hazard_level")
        ] + [
            f"{object_type}_max_{suffix}"
            for object_type in ("property", "street")
            for suffix in ("indicator_mean", "hazard_level", "likelihood")
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
        
        # THE SCORED POPULATION. Only the h_max columns are in the subset,
        # deliberately: contingency tables can only be compared across one
        # identical population, and every configuration has to be scored on the
        # rows the h_max pass defines rather than on its own.
        risk_eval = survyed_buildings.dropna(
            subset=["observed_affected", "max_depth", "mean_depth",
                    f"{configuration_prefix('BuildingOnly', 'Max')}_hazard_level",
                    f"{BUILDPROP_MAX_PREFIX}_hazard_level"]
        ).copy()
        
        y_true = risk_eval["observed_affected"]
        num_buildings = len(risk_eval)
        
        
        logger.info(
            f"scored {num_buildings} buildings; inputs available: "
            f"property {int(risk_eval['property_known'].sum())}, "
            f"street {int(risk_eval['street_known'].sum())}"
        )
        # Observed class for the same rows, so each indicator's misses and false
        # alarms can be attributed to the damage level they came from.
        damage_levels = risk_eval["DamageNb"]

        # THE PAPER CONFIGURATIONS, in SCORED_CONFIGURATIONS order, so the
        # CSV rows come out in the order the results table reads.
        #
        # ONE RULE FOR ALL OF THEM now that no inference stands between the depths
        # and the class: a building is predicted affected when the ensemble puts
        # its context indicator at or above the H1/H2 boundary with probability
        # at least AFFECTED_MIN_PROBABILITY -- P(h >= 0.30 m) >= 0.5 by default.
        # That is Equation (8) read at the class the warning starts at, and it
        # is the same question asked of every configuration, which is what makes
        # the ablation a comparison of spatial contexts rather than of scoring
        # rules (see AFFECTED_MIN_PROBABILITY for what moving it does).
        for column in SCORED_CONFIGURATIONS:
            if column not in risk_eval.columns:
                logger.info("%s not in the impact frame; not scored", column)
                continue
            self.write_results(
                num_buildings, y_true,
                _exceedance_to_affected(risk_eval[column]),
                column, model, "July2021", damage_levels,
            )

        # The diagnostics: the same scoring applied to readings that are not
        # configurations -- the ensemble collapsed before the class instead of
        # after it. Separable from the table by having no paper name.
        for column in DIAGNOSTIC_COLUMNS:
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

        ``indicator`` is passed as the impact-frame COLUMN that was
        thresholded, and published as the configuration's PAPER NAME where it
        is one of the paper's (SCORED_CONFIGURATIONS). A column with no paper name
        is a diagnostic and is published under its own name, with an empty
        ``indicators_used`` -- so the table is exactly the rows that have one.
        """
        labels = [NOT_AFFECTED, AFFECTED]
        configuration, indicators_used = SCORED_CONFIGURATIONS.get(
            indicator, (indicator, ""))

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

        logger.info(f"Binary verification ({configuration}), affected = DamageNb >= "
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
            "model": model,
            "indicator": configuration,
            "indicators_used": indicators_used,
            **by_damage,
        }

        metrics_df = pd.DataFrame([metrics])
        # Separate file: these columns do not line up with the three-tier rows
        # already appended to risk_matrix_validation_metrics.csv.
        output_file = os.path.join(output_dir, "binary_validation_metrics.csv")

        # The header is written once, so the columns of this run have to line
        # up with the file's; see _align_appended_csv for the two ways they can
        # differ and what each one costs.
        _align_appended_csv(output_file, metrics_df.columns)

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

# The statistics in ABLATION ORDER, which is not the order they are declared in
# (hazard_indicator_finder.STATISTICS is keyed by column, not ranked). Every
# consumer that lays configurations out in rows reads this, so the results
# table, the prefixes and the CSV cannot come out in three different orders.
STATISTIC_ORDER = ("Mean", "WetMean", "Max")

# The paper's name for each scored configuration, and the indicators behind it,
# keyed by the impact-frame column the scoring thresholds -- P(h >= 0.30 m) for
# that configuration, which is what test_against_placard_data has in hand at the
# point it calls write_results. Derived from the finder's own CONFIGURATIONS so
# a context added there cannot be missing here.
#
# write_results publishes the paper name in the `indicator` column and the
# notation in `indicators_used`, so a results table is built from the CSV
# directly and there is no second mapping to keep in step with this one.
#
# Ordered by expanding spatial context (footprint, + property parcel, + adjacent
# street), and within each context by STATISTIC_ORDER, so the CSV rows come out
# in the order the table reads. The names are hyphenated to stay unique per row:
# the paper splits "BuildingOnly" and its indicator across two columns, but a
# CSV row needs one key.
#
# ONLY THE PAPER CONFIGURATIONS ARE HERE. The other scored rows are
# diagnostics, not configurations, and keep their raw column names so they are
# trivially separable from the table (see DIAGNOSTIC_COLUMNS).
SCORED_CONFIGURATIONS = {
    f"{configuration_prefix(configuration, statistic)}_prob_at_least_H2": (
        f"{configuration}-{statistic}",
        ", ".join(f"h_{statistic.lower()}_{object_type[0]}"
                  for object_type in CONFIGURATIONS[configuration]),
    )
    # Statistic inside, and in the order h_mean <= h_wet_mean <= h_max holds
    # (see hazard_indicator_finder.STATISTICS), so the rows within a context
    # come out least to most severe -- the order the results table reads.
    for configuration in CONFIGURATIONS
    for statistic in STATISTIC_ORDER
}

# The probability at which the exceedance of the H1/H2 boundary is called a
# warning. 0.5 is a majority of the ensemble: the class the members agree on
# more often than not.
#
# THIS IS THE BINARY SCORING RULE OF THE ABLATION TABLE, and it is a choice, not
# a reading of the guideline -- the risk matrix itself places a building at the
# highest class it reaches and reports the likelihood separately rather than
# collapsing the two into one call. Lowering it towards 0 scores the matrix's
# own placement (any member reaching H2 counts), which raises POD and FAR
# together; the likelihood bands of hazard_indicator_finder are the operational
# reading of the same axis and are unaffected either way.
AFFECTED_MIN_PROBABILITY = 0.5

# Column prefixes of every configuration, in the order the table reads them.
CONFIGURATION_PREFIXES = tuple(
    configuration_prefix(configuration, statistic)
    for configuration in CONFIGURATIONS
    for statistic in STATISTIC_ORDER
)

# Scored beside them but not part of the table: the same BuildProp-Max and
# BuildPropStreet-Max contexts with the ensemble collapsed BEFORE the class
# instead of after it -- the mean of the members' aggregated indicators, and the
# indicator read off the ensemble-mean flood map. They answer "does reading the
# ensemble probabilistically buy anything", which is a different question from
# which spatial context to use.
DIAGNOSTIC_COLUMNS = (
    f"{BUILDPROP_MAX_PREFIX}_ensemble_mean_hazard_level",
    f"{BUILDPROP_MAX_PREFIX}_mean_map_hazard_level",
    f"{REPORTED_PREFIX}_ensemble_mean_hazard_level",
    f"{REPORTED_PREFIX}_mean_map_hazard_level",
)


def _configuration_name(prefix):
    """``(paper name, indicators)`` for a configuration's column prefix.

    Read out of SCORED_CONFIGURATIONS rather than rebuilt, so a results row
    names its configuration exactly the way the binary table does and there is
    no second mapping to keep in step. A prefix with no paper name -- a
    diagnostic -- publishes under its own name with no indicator notation, the
    same convention write_results follows.
    """
    return SCORED_CONFIGURATIONS.get(
        f"{prefix}_prob_at_least_{CLASS_COLUMN_TERMS[MIDDLE_CLASS]}", (prefix, ""))


def _take_columns(frame, columns):
    """``frame`` narrowed to the columns it actually has, in the given order.

    Whichever of them are missing is reported by the caller against the
    concatenated frame, so a configuration that is not on the frames is skipped
    with its name said once rather than raising here per step.
    """
    seen, wanted = set(), []
    for column in columns:
        if column in frame.columns and column not in seen:
            seen.add(column)
            wanted.append(column)
    return frame[wanted]


# Which object series each peak layer collapses, and the id it is keyed on.
PEAK_LAYERS = {
    "building_impact": ("building_peak", "building_id"),
    "road_impact": ("road_peak", "id"),
    "property_impact": ("property_peak", "unit_of_property_id"),
}


def _window_peak_frames(impact_forecast, issue_time):
    """The peak layers' frames for one issue, ready to publish.

    Each is a ONE-entry series keyed by the issue time rather than by a valid
    time, because the peak is a property of the window and not of an hour --
    which is also what lets the publisher treat these like any other layer. The
    hour the peak is reached travels as the peak_valid_time COLUMN.

    Built only when the forecast is being published: nothing else reads them,
    and the collapse is a pass over every step of the window.
    """
    peaks = {}
    for source, (key, id_column) in PEAK_LAYERS.items():
        frame = window_peak(impact_forecast.get(source) or {}, id_column)
        if frame is None:
            continue
        peaks[key] = {pd.Timestamp(issue_time): frame}
        logger.info("%s: %d objects collapsed to their window peak",
                    key, len(frame))
    return peaks


def _window_peak(observed, forecast, level_column, likelihood_column):
    """Collapse a window's per-step rows to one class per building, per side.

    Both sides take the MAX class over the window's steps. On the forecast side
    the likelihood has to come from a step that actually REACHED that class --
    it is P of the class assigned, so the likelihood of a lesser step is a
    different quantity -- which is why the row is selected rather than the two
    columns aggregated apart. Sorting by (class, likelihood) and keeping each
    building's last row gives the highest class, and among the steps that
    reached it the highest likelihood: the strongest statement the forecast made
    at the class it is scored on.

    NaNs sort FIRST, so a step whose likelihood could not be evaluated cannot
    win a tie against one that could; a building whose likelihoods are all NaN
    still reports NaN.

    Returns a frame indexed by building_id with ``observed``, ``forecast`` and
    ``likelihood``, on the buildings that carry a class on both sides.
    """
    peak_observed = observed.groupby("building_id")[level_column].max()
    peak_forecast = (
        forecast
        .sort_values([level_column, likelihood_column], na_position="first")
        .groupby("building_id", sort=False)
        .tail(1)
        .set_index("building_id")
    )
    window = pd.DataFrame({
        "observed": peak_observed,
        "forecast": peak_forecast[level_column],
        "likelihood": peak_forecast[likelihood_column],
    })
    return window.dropna(subset=["observed", "forecast"])


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

def _exceedance_to_affected(probability, threshold=None):
    """Collapse a configuration's P(h >= 0.30 m) to affected / not_affected.

    The binary counterpart of the risk matrix: the matrix reports a class and
    its likelihood separately, and a contingency table needs the two crossed
    into one call. Affected when at least ``threshold`` of the ensemble puts the
    context indicator at or above the H1/H2 boundary
    (see AFFECTED_MIN_PROBABILITY).

    A NaN probability -- a building the configuration could not be evaluated for
    -- returns NaN and drops out of the scoring rather than counting as a
    negative.
    """
    threshold = AFFECTED_MIN_PROBABILITY if threshold is None else threshold
    probability = pd.Series(probability)
    predicted = pd.Series(
        np.where(probability >= threshold, AFFECTED, NOT_AFFECTED),
        index=probability.index, dtype=object,
    )
    return predicted.mask(probability.isna())


def _hazards_to_affected(levels):
    """Collapse one or more predicted hazard-class columns to affected/not_affected.

    ``levels`` is a frame of hazard-class columns (1/2/3, NaN where the class
    could not be evaluated). H2 and H3+ are the classes that carry a warning, so
    a row is predicted affected if ANY column is above H1; H1 in every column is
    the negative. The scoring passes a single already-aggregated class, so the
    OR only bites if several columns are handed in.

    NaN classes contribute nothing rather than voting "not affected": an object
    that was never assessed is not evidence of a dry surrounding. A row whose
    columns are all NaN returns NaN and drops out of the scoring.
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


# The K-1 thresholds an ordered K-class forecast needs, for one configuration.
# P(>= H1) carries no information (every building is at least H1), so the
# distribution is fully described by the exceedance of the two INTERIOR
# boundaries.
def exceedance_prob_columns(prefix):
    return tuple(f"{prefix}_prob_at_least_{term}"
                 for term in CLASS_COLUMN_TERMS[MIDDLE_CLASS:])


# All three, including P(>= H1). That one is not part of the score -- every
# building is at least H1, so it adds nothing to the distribution -- but it is
# the share of the ensemble putting ANY water in the building's spatial context,
# which is worth reporting in its own right.
def all_exceedance_columns(prefix):
    return tuple(f"{prefix}_prob_at_least_{term}" for term in CLASS_COLUMN_TERMS)


def _rps_from_cumulative(cumulative, observed):
    """Ranked probability score per row, for an ordered categorical forecast.

        RPS = sum_k (F_k - O_k)^2,    k = 1 .. K-1

    with F_k the forecast probability of class <= k and O_k the observed
    indicator. Summing the SQUARED CUMULATIVE error rather than the per-class
    error is what makes the score respect the ordering: putting the mass on H1
    when H3+ verified is penalised twice as hard as putting it on H2.

    It is a proper score, so it is minimised by reporting the forecast actually
    believed -- a hedged distribution cannot beat an honest one. And it reduces
    to |c - y| for a deterministic forecast (all mass on one class), which is
    what lets a point forecast and an ensemble be scored in the same units and
    put in the same table.

    ``cumulative`` is (n, K-1) and ``observed`` the (n,) integer class.
    """
    observed_cumulative = np.stack(
        [observed <= level for level in HAZARD_LEVELS[:-1]], axis=1
    ).astype(float)
    return ((np.asarray(cumulative, dtype=float) - observed_cumulative) ** 2).sum(axis=1)


def _skill(score, reference):
    """1 - score/reference, NaN where the reference is perfect.

    A reference RPS of 0 means the reference forecast was already exactly right
    on every row -- a timestep where every building is dry, which is most of
    them at the start and end of an event. There is no skill to express over a
    perfect reference, so the row reports NaN rather than an infinity that
    would poison any later average.
    """
    return np.nan if reference <= 0 else float(1.0 - score / reference)


def _exceedance_metrics(gt, pred, id_col, prefix=None):
    """Probabilistic verification of one configuration's exceedance probabilities.

    The forecast is a DISTRIBUTION over the three hazard classes, published as
    the exceedance of the two interior boundaries; the truth is the class the
    same configuration assigns to the simulated map. Scoring a probability
    against a class is what RPS is for -- the deterministic QWK beside it can
    only see the collapsed class and is blind to the ensemble's spread entirely.

    ONE reference: the same configuration's ``ensemble_mean_hazard_level``, its
    mean-of-maxima class, scored as a degenerate distribution (all mass on one
    class, at which point the RPS reduces to |c - y|). RPS is an absolute score
    whose magnitude depends on the class base rates and means nothing on its
    own, so it needs a reference -- and this is the one the method's claim is
    actually about: does reading the ensemble probabilistically beat collapsing
    it before classifying. It is also EXACTLY the forecast the QWK column
    scores, so the two figures are two readings of one pair of forecasts rather
    than of two different ones.

    NO CLIMATOLOGY REFERENCE. A climatology would have to be built from a
    historical archive of outcomes, which this pipeline does not have; taking
    the base rates of the very rows being scored instead is an in-sample
    construct that is not a climatology in the sense the verification
    literature uses the word, and reporting it under that name invites exactly
    the wrong comparison.

    The RAW scores are recorded alongside the skill, because RPSS does not
    average: pooling timesteps means summing the numerators and denominators
    separately, which needs rps, its reference and n, not the ratio.
    """
    prefix = BUILDPROP_MAX_PREFIX if prefix is None else prefix
    observed_column = f"{prefix}_hazard_level"
    reference_column = f"{prefix}_ensemble_mean_hazard_level"

    wanted = list(all_exceedance_columns(prefix)) + [reference_column]
    missing = ([observed_column] if observed_column not in gt.columns else []) + [
        column for column in wanted if column not in pred.columns]
    if missing:
        logger.info("no exceedance metrics for %s; the frames lack %s",
                    prefix, missing)
        return {}

    merged = gt[[id_col, observed_column]].merge(
        pred[[id_col] + wanted], on=id_col).dropna(
        subset=[observed_column] + wanted)
    if merged.empty:
        return {}

    observed = merged[observed_column].to_numpy(dtype=int)
    # P(>=H2), P(>=H3+)
    exceedance = merged[list(exceedance_prob_columns(prefix))].to_numpy(dtype=float)
    # P(class <= k) is the complement of P(class >= k+1); the columns are
    # already ordered by boundary, so this is the cumulative the score wants.
    rps = _rps_from_cumulative(1.0 - exceedance, observed).mean()

    point = merged[reference_column].to_numpy(dtype=int)
    ensemble_mean = _rps_from_cumulative(
        np.stack([point <= level for level in HAZARD_LEVELS[:-1]], axis=1),
        observed,
    ).mean()

    results = {
        f"{prefix}_n": int(len(merged)),
        f"{prefix}_rps": float(rps),
        f"{prefix}_rps_ensemble_mean": float(ensemble_mean),
        f"{prefix}_rpss_ensemble_mean": _skill(rps, ensemble_mean),
    }
    # The exceedance probabilities themselves, averaged over buildings -- the
    # only form in which a per-building quantity survives into a metrics CSV.
    # Taken on the MERGED rows, so every number in the row describes the same
    # population as the scores beside it.
    for term in CLASS_COLUMN_TERMS:
        results[f"{prefix}_prob_at_least_{term}"] = float(
            merged[f"{prefix}_prob_at_least_{term}"].mean())
    return results


def _class_indicator_metrics(comparison, risk_col, prefix="class_"):
    """QWK/hit-rate/CSI for a single-valued hazard class, given an
    already-merged (``_gt``/``_pred`` suffixed) comparison frame.

    Counterpart to _indicator_metrics, which scores a mean/max PAIR of depth
    columns and their classes. A configuration produces ONE class per building
    -- its objects have already been aggregated by Equation (7) -- so there is
    one column to score and no depth RMSE of its own.
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
