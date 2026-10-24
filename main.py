import logging
import argparse
import multiprocessing as mp
from modules.forecast_runner.forecast_runner import Integrator
from modules.validation_module.validation_module import ValidationModule
from modules.lib import FLOW_MATCHING_MODEL, FORECAST_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

RUN_COMMAND = "run"
# Score the forecast against the simulated maps, or the July 2021 placards
# for scenario 0.
VALIDATE_COMMAND = "validate"
# Time each issue into runtime_metrics.csv.
RUNTIME_COMMAND = "runtime"
# Choose the spatial summary (max / median / p90 of footprint cells) against the July 2021
# surveyed depths and placards.
CALIBRATE_COMMAND = "calibrate"
COMMANDS = [RUN_COMMAND, VALIDATE_COMMAND, RUNTIME_COMMAND, CALIBRATE_COMMAND]

def parse_args():
    parser = argparse.ArgumentParser(description="Westport IBWS Prototype System")

    parser.add_argument('command', type=str, default="run",
                        help=f'Command to execute: {", ".join(COMMANDS)}')

    # ── General ──────────────────────────────────────────────────────────────
    parser.add_argument('--start_time', type=str, default="", help="Start time for the forecast")
    parser.add_argument('--forecast_horizon', type=int, default=0,help="Forecast horizon in hours (run, runtime); validate forecasts the whole event as one run")
    parser.add_argument('--end_time', type=str, default="", help="End time for the forecast")
    parser.add_argument("--scenario_id", type=int, default=0, help="Id of the flood scenario; 0 is the July 2021 placard test")
    parser.add_argument("--push_to_arcgis", action='store_true', help="Whether to push to arcgis or not")
    # Counts, so int: num_deep_ensemble ends up slicing the member list in
    # LatentFlowEnsemble._discover_members, and a float raises there.
    parser.add_argument("--samples_per_member", type=int, default=10,
                        help="K -- samples drawn per ensemble member")
    parser.add_argument("--num_deep_ensemble", type=int, default=10,
                        help="M -- members of the deep ensemble to load")
    parser.add_argument("--decode_batch", type=int, default=4,
                        help="flood maps the decoder takes per ONNX call (members x hours are split into chunks of this)")
    parser.add_argument("--low_memory", action="store_true",
                        help="load ensemble members one at a time and release the decoder's GPU memory after "
                             "each call: about one member's GPU memory instead of all of them, same forecasts")
    parser.add_argument("--hours_per_batch", type=int, default=1,
                        help="validate: hours the surrogate samples together (1 = one hour at a time)")
    parser.add_argument("--model", type=str, default=FLOW_MATCHING_MODEL, choices=FORECAST_MODELS,
                        help="forecast model: the flow-matching surrogate, or the westport 1D-CNN benchmark "
                             "(exported by scripts/export_cnn1d_onnx.py)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    mp.set_start_method('spawn', force=True)  
    
    logger.info(f"Starting script with command: {args.command}")
    logger.info("Arguments: " + str(args))
        
    if args.command not in COMMANDS:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)

    # Validation is never published: it scores the forecast, it does not issue
    # one. The runtime test publishes when asked, since that is part of what an
    # issue costs.
    push_to_arcgis = args.push_to_arcgis and args.command in (RUN_COMMAND, RUNTIME_COMMAND)
    integrator = Integrator(args.scenario_id, args.forecast_horizon, push_to_arcgis, args.samples_per_member, args.num_deep_ensemble,
                            decode_batch=args.decode_batch, low_memory=args.low_memory,
                            model=args.model)

    if args.command == RUN_COMMAND:
        integrator.run_hourly_forecast(start_time=args.start_time, end_time=args.end_time)
    elif args.command == VALIDATE_COMMAND:
        ValidationModule(integrator, hours_per_batch=args.hours_per_batch).validate_forecast(
            args.start_time, args.end_time)
    elif args.command == RUNTIME_COMMAND:
        ValidationModule(integrator).validate_runtime(args.start_time, args.end_time)
    elif args.command == CALIBRATE_COMMAND:
        ValidationModule(integrator).calibrate_spatial_summarisation_method()
    logger.info("Commands executed successfully")

