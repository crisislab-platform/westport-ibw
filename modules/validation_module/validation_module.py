from modules.utils.spatial_operators.operators import load_flood_map
import json
import os
import modules.validation_module.results as results
from modules.lib import FLOW_MATCHING_MODEL
import pandas as pd
import logging
from modules.validation_module.july_2021_validation import July2021Validation
from modules.validation_module.depth_validation import DepthValidation
from modules.validation_module.runtime_validation import RuntimeValidation

logger = logging.getLogger(__name__)


class ValidationModule(July2021Validation, DepthValidation, RuntimeValidation):
    """Scores building flood-depth forecasts against the simulated maps and,
    for the July 2021 event, the surveyed depths and damage placards. Results
    are appended to the CSVs in VALIDATION_RESULTS_DIR. Entry points:
    main.py validate / runtime / calibrate.

    One mixin per test: July2021Validation (july_2021_validation.py),
    DepthValidation (depth_validation.py) and RuntimeValidation
    (runtime_validation.py).
    """

    def __init__(self, integrator, hours_per_batch=1):
        """Validate the forecast ``integrator`` produces.

        ``hours_per_batch``: how many hours the surrogate samples together (see
        InundationModule.iter_hazard_forecast); recorded on every summary row,
        so runs at different settings can be compared for time and memory.

        The Integrator is only borrowed for its modules and its forecast
        stream: it knows nothing about validation, and the loops that score it
        live here.
        """
        self.integrator = integrator
        self.hours_per_batch = hours_per_batch
        self.scenario_id = integrator.scenario_id
        self.forecast_horizon = integrator.forecast_horizon
        self.samples_per_member = integrator.samples_per_member
        self.num_deep_ensemble = integrator.num_deep_ensemble
        self.hydromet_module = integrator.hydromet_module
        self.exposure_module = integrator.exposure_module

    def run_config(self):
        """The settings a forecast run is defined by, recorded in its folder's
        run_config.json (as westport records model_config.json per run)."""
        from modules.forecast_runner.forecast_runner import SAMPLING_SEED
        model = getattr(self.integrator, "model", FLOW_MATCHING_MODEL)
        surrogate = model == FLOW_MATCHING_MODEL
        return {
            "model": model,
            "spatial_summary": self.exposure_module.spatial_summary,
            "num_deep_ensemble": self.num_deep_ensemble,
            "samples_per_member": self.samples_per_member,
            "ensemble_size": self.num_deep_ensemble * self.samples_per_member,
            "hours_per_batch": self.hours_per_batch,
            # Settings of the surrogate's sampler and decoder; a deterministic
            # benchmark has none.
            "decode_batch": getattr(self.integrator, "decode_batch", None) if surrogate else None,
            "low_memory": bool(getattr(self.integrator, "low_memory", False)) if surrogate else None,
            "sampling_seed": SAMPLING_SEED if surrogate else None,
        }

    def results_dir(self):
        """This run configuration's own folder under VALIDATION_RESULTS_DIR,
        named after its settings, so every event run with the same settings
        lands in the same folder and different settings never share a file.
        Created, with run_config.json, the first time something is written."""
        config = self.run_config()
        if config["model"] == FLOW_MATCHING_MODEL:
            name = (f"{config['spatial_summary']}_M{config['num_deep_ensemble']}"
                    f"_K{config['samples_per_member']}_hours{config['hours_per_batch']}"
                    f"_decode{config['decode_batch']}{'_lowmem' if config['low_memory'] else ''}"
                    f"_seed{config['sampling_seed']}")
        else:
            # A deterministic benchmark: sampling and decoder settings do not apply.
            name = f"{config['spatial_summary']}_{config['model']}_hours{config['hours_per_batch']}"
        directory = os.path.join(results.VALIDATION_RESULTS_DIR, name)
        config_file = os.path.join(directory, "run_config.json")
        if not os.path.exists(config_file):
            os.makedirs(directory, exist_ok=True)
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        return directory

    def validate_forecast(self, start_time=None, end_time=None):
        """Forecast the event as one run and score it.

        Scenario 0 is the July 2021 event, scored against the surveyed depths
        and damage placards. Every other scenario is scored hour by hour, and
        per building over the event, against its simulated maps.
        """
        logger.info(
            f"Validating forecast for scenario {self.scenario_id} "
            f"from {start_time or 'the start'} to {end_time or 'the end'} of the event"
        )
        if self.scenario_id == 0:
            self.validate_july_2021_event(start_time, end_time)
        else:
            self.validate_event(start_time, end_time)

    def event_span(self, start_time=None, end_time=None):
        """``(start, hours)`` covering the boundary-condition series from
        start_time to end_time -- the whole event when they are blank."""
        times = self.hydromet_module.bc_dict.keys()
        start = pd.to_datetime(start_time) if start_time else min(times)
        end = pd.to_datetime(end_time) if end_time else max(times)
        return start, int((end - start) / pd.Timedelta(hours=1)) + 1

    def _ground_truth_depths(self, valid_time):
        """The simulated map at ``valid_time``, through the same spatial summary:
        one depth per building in exposure_module.buildings order.

        Returns None when no simulated map exists for the step, having said
        which step that was -- bc_dict is filtered to timestamps that have a
        map, so it is not expected, and a step scored against the wrong hour is
        worse than a step not scored at all.
        """
        valid_time = pd.Timestamp(valid_time)
        gt_file = self.hydromet_module.flood_map_dict.get(valid_time)
        if gt_file is None:
            logger.warning(
                "no simulated map at %s; that forecast step is not scored",
                valid_time,
            )
            return None

        gt_hazard_map = load_flood_map(gt_file, self.hydromet_module.event_name)
        depths, _ = self.exposure_module.building_depths([gt_hazard_map])
        return depths[0]
