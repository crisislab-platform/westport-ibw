import logging
import argparse
import multiprocessing as mp
from modules.forecast_runner.forecast_runner import Integrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

RUN_COMMAND = "run"
TEST_COMMAND = "test"

def parse_args():
    parser = argparse.ArgumentParser(description="Westport IBWS Prototype System")

    parser.add_argument('command', type=str, default="run",
                        help=f'Command to execute: {RUN_COMMAND}')

    # ── General ──────────────────────────────────────────────────────────────
    parser.add_argument('--start_time', type=str, default="", help="Start time for the forecast")
    parser.add_argument('--forecast_horizon', type=int, default=0,help="Forecast horizon in hours")
    parser.add_argument('--end_time', type=str, default="", help="End time for the forecast")
    parser.add_argument("--scenario_id", type=int, help="Id of the flood scenario")
    parser.add_argument("--push_to_arcgis", action='store_true', help="Whether to push to arcgis or not")
    parser.add_argument("--buffer_multiplier", type=float, default=0.0)
    parser.add_argument("--mean_threshold", type=float, default=0.05)
    parser.add_argument("--max_threshold", type=float, default=0.15)
    # Counts, so int: num_deep_ensemble ends up slicing the member list in
    # LatentFlowEnsemble._discover_members, and a float raises there.
    parser.add_argument("--samples_per_member", type=int, default=10,
                        help="K -- samples drawn per ensemble member")
    parser.add_argument("--num_deep_ensemble", type=int, default=10,
                        help="M -- members of the deep ensemble to load")
    parser.add_argument("--exceedence_prob", type=float, default=10, help="Min exeedence probability of the lower bound of hazard claass")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    valid_commands = [RUN_COMMAND, TEST_COMMAND]
    mp.set_start_method('spawn', force=True)  
    
    logger.info(f"Starting script with command: {args.command}")
    logger.info("Arguments: " + str(args))
        
    if args.command not in valid_commands:
        logger.error(f"Invalid command '{args.command}'")
        exit(1)

    if args.command == RUN_COMMAND:
        integrator = Integrator(args.scenario_id, args.forecast_horizon, args.push_to_arcgis, args.samples_per_member, args.num_deep_ensemble, args.exceedence_prob)
        integrator.run_hourly_forecast(start_time=args.start_time, end_time=args.end_time)
    elif args.command == TEST_COMMAND:
        integrator = Integrator(args.scenario_id, args.forecast_horizon, False,args.samples_per_member, args.num_deep_ensemble, min_exceed_prob=args.exceedence_prob)
        integrator.validate_forecast(args.start_time, args.end_time)
    logger.info("Commands executed successfully")

