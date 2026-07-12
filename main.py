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
        integrator = Integrator(args.scenario_id, args.forecast_horizon, args.push_to_arcgis)
        integrator.run_hourly_forecast(start_time=args.start_time, end_time=args.end_time)
    elif args.command == TEST_COMMAND:
        integrator = Integrator(args.scenario_id, args.forecast_horizon, False)
        integrator.validate_forecast()
    logger.info("Commands executed successfully")

