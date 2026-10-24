from modules.hydromet_module.bc_module import HydroMetModule
from modules.inundation_module.model_client import InundationModule
from modules.inundation_module.cnn1d_client import Cnn1dInundation
from modules.lib import CNN1D_MODEL, FLOW_MATCHING_MODEL, FORECAST_MODELS
from modules.update_module.arcgis_module import ArcGISModule
from modules.utils.utils import check_device

from modules.exposure_module.exposure_module import ExposureModule
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from modules.forecast_runner.forecast_runner_helper import (
    TIMING_COLUMNS,
    _timed,
    _timed_steps,
    _merge_impact,
)

logger = logging.getLogger(__name__)

# Seed for the surrogate's sampling noise, so a run reproduces exactly.
SAMPLING_SEED = 41


class Integrator():
    def __init__(self, scenario_id, forecast_horizon, push_to_arcgis, samples_per_member, num_deep_ensemble,
                 decode_batch=4, low_memory=False, model=FLOW_MATCHING_MODEL):
        """``model``: which forecast to run, one of FORECAST_MODELS. CNN1D_MODEL
        is the westport 1D-CNN benchmark; it is deterministic, so M = K = 1."""
        if model not in FORECAST_MODELS:
            raise ValueError(f"model must be one of {FORECAST_MODELS}, not {model!r}")
        if model == CNN1D_MODEL:
            samples_per_member, num_deep_ensemble = 1, 1
        self.model = model
        self.scenario_id = scenario_id
        self.forecast_horizon = forecast_horizon
        self.push_to_arcgis = push_to_arcgis
        self.samples_per_member = samples_per_member
        self.num_deep_ensemble = num_deep_ensemble
        self.hydromet_module = HydroMetModule(scenario_id)
        if model == CNN1D_MODEL:
            self.inundation_module = Cnn1dInundation(self.hydromet_module)
        else:
            self.inundation_module = InundationModule(samples_per_member=samples_per_member, num_members=num_deep_ensemble,
                                                      seed=SAMPLING_SEED, decode_batch=decode_batch,
                                                      hold_members=not low_memory)
        self.decode_batch = decode_batch
        self.low_memory = low_memory
        self.exposure_module = ExposureModule(scenario_id)
        if self.push_to_arcgis:
            self.arcgis_module = ArcGISModule()
        self.device = check_device()


    def issue_times(self, start_time=None, end_time=None):
        """The hourly issue times from start_time to end_time, inclusive.

        Blank bounds default to the whole event: the first boundary-condition
        time, and the last one less the horizon, so every issue has a full
        window of boundary conditions behind it.
        """
        if start_time == None or start_time=='':
            current_time = min(self.hydromet_module.bc_dict.keys())
        else:
            current_time = pd.to_datetime(start_time)
        if end_time == None or end_time=='':
            end_time = max(self.hydromet_module.bc_dict.keys()) 
            end_time = end_time - timedelta(hours=self.forecast_horizon)
        else:
            end_time = pd.to_datetime(end_time)

        while current_time <= end_time:
            yield current_time
            current_time += pd.Timedelta(hours=1)

    def iter_inundation(self, start_time, forecast_horizon=None, timings=None,
                        hours_per_batch=None):
        """Stream ``(valid_time, ensemble)`` over the boundary conditions from
        start_time for forecast_horizon hours (self.forecast_horizon if None).
        ``hours_per_batch``: see InundationModule.iter_hazard_forecast.

        Yields nothing when there are no boundary conditions for the span.
        ``timings``, when given, gets bc_s, inundation_s and num_timesteps.
        """
        if timings is None:
            timings = dict.fromkeys(TIMING_COLUMNS, np.nan)

        with _timed(timings, "bc_s"):
            bc_forecast = self.hydromet_module.get_bc_forecast(
                start_time, forecast_horizon or self.forecast_horizon)

        if not bc_forecast:
            print(f"No boundary condition data available for the forecast period starting at {start_time}. Skipping this hour.")
            return
        timings["num_timesteps"] = len(bc_forecast)

        yield from _timed_steps(
            timings, "inundation_s",
            self.inundation_module.iter_hazard_forecast(bc_forecast, hours_per_batch),
        )

    def iter_inundation_batches(self, start_time, forecast_horizon=None, timings=None,
                                hours_per_batch=1):
        """As iter_inundation, but yielding ``(valid_times, (B, M*K, H, W))``
        per batch of hours, each batch sampled and decoded together
        (InundationModule.iter_hazard_forecast_batches). ``timings`` gets bc_s
        and inundation_s (summed over batches) and num_timesteps.
        """
        if timings is None:
            timings = dict.fromkeys(TIMING_COLUMNS, np.nan)

        with _timed(timings, "bc_s"):
            bc_forecast = self.hydromet_module.get_bc_forecast(
                start_time, forecast_horizon or self.forecast_horizon)

        if not bc_forecast:
            print(f"No boundary condition data available for the forecast period starting at {start_time}.")
            return
        timings["num_timesteps"] = len(bc_forecast)

        yield from _timed_steps(
            timings, "inundation_s",
            self.inundation_module.iter_hazard_forecast_batches(bc_forecast, hours_per_batch),
        )

    def forecast_issue(self, current_time, timings=None):
        """One issue: boundary conditions -> inundation -> exposure, per step.

        Returns the issue's exposure forecast, or None when there are no
        boundary conditions for it. ``timings``, when given, gets bc_s,
        inundation_s and exposure_s filled in.
        """
        if timings is None:
            timings = dict.fromkeys(TIMING_COLUMNS, np.nan)

        timings["exposure_s"] = 0.0
        exposure_forecast = {}
        for timestamp, inundation in self.iter_inundation(current_time, timings=timings):
            with _timed(timings, "exposure_s", accumulate=True):
                exposure_at_timestep = self.exposure_module.get_exposure_assessment(
                    {timestamp: inundation}
                )
                _merge_impact(exposure_forecast, exposure_at_timestep)
        return exposure_forecast or None

    def issue_forecast(self, current_time, timings=None):
        """Forecast one issue and publish it, if publishing is on.

        Returns the exposure forecast, or None when the hour has no boundary
        conditions. ``timings`` is filled in as forecast_issue describes, plus
        delete_s and publish_s from the publisher; the runtime test passes it,
        the operational run does not.
        """
        exposure_forecast = self.forecast_issue(current_time, timings)
        if exposure_forecast is None:
            return None

        if self.push_to_arcgis:
            published = self.arcgis_module.push_impacts_to_spatial_db(
                exposure_forecast, current_time
            )
            if timings is not None:
                timings.update(published)
        return exposure_forecast

    def run_hourly_forecast(self,  start_time=None, end_time=None):
        for current_time in self.issue_times(start_time, end_time):
            if self.issue_forecast(current_time) is not None:
                logger.info(f"Processed timestamp: {current_time}")
