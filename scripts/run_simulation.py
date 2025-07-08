#!/usr/bin/env python3
"""
Run Premier League Season Simulation

This script demonstrates the complete season simulation engine, simulating
matches week by week with realistic timing, odds generation, predictions,
and automated retraining.

Usage:
    python scripts/run_simulation.py [--weeks WEEKS] [--start-week START] [--mode MODE]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import SeasonSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main simulation runner."""
    parser = argparse.ArgumentParser(description="Run Premier League Season Simulation")
    parser.add_argument(
        "--weeks",
        type=int,
        help="Number of weeks to simulate (default: all remaining weeks)"
    )
    parser.add_argument(
        "--start-week",
        type=int,
        default=1,
        help="Week to start simulation from (default: 1)"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "interactive"],
        default="batch",
        help="Simulation mode: single (one week), batch (multiple weeks), interactive (step-by-step)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/simulation_output",
        help="Output directory for simulation results"
    )

    args = parser.parse_args()

    # Paths (adjust if needed)
    base_dir = Path(__file__).parent.parent
    simulation_data_path = base_dir / "data/simulation/simulation_data_2023_24.parquet"
    match_calendar_path = base_dir / "data/simulation/match_calendar.parquet"
    model_path = base_dir / "models/model.pkl"

    # Check if required files exist
    required_files = [simulation_data_path, match_calendar_path, model_path]
    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error("Please run 'python scripts/prepare_simulation_data.py' first")
        return 1

    try:
        # Initialize simulator
        logger.info("Initializing season simulator...")
        simulator = SeasonSimulator(
            simulation_data_path=str(simulation_data_path),
            match_calendar_path=str(match_calendar_path),
            model_path=str(model_path),
            output_dir=args.output_dir,
            retraining_threshold=0.05,
            retraining_frequency=5
        )

        # Get initial state
        initial_state = simulator.get_simulation_state()
        logger.info(f"Simulation initialized: {initial_state}")

        # Run simulation based on mode
        if args.mode == "single":
            return run_single_week(simulator, args.start_week)
        elif args.mode == "batch":
            return run_batch_simulation(simulator, args.start_week, args.weeks)
        elif args.mode == "interactive":
            return run_interactive_simulation(simulator, args.start_week, args.weeks)

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return 1


def run_single_week(simulator: SeasonSimulator, week: int) -> int:
    """Run simulation for a single week."""
    logger.info(f"Running single week simulation for week {week}")

    try:
        week_data = simulator.simulate_week(week)

        # Print summary
        print(f"\n=== Week {week} Simulation Results ===")
        print(f"Matches: {len(week_data['matches'])}")
        print(f"Predictions: {len(week_data['predictions'])}")
        print(f"Accuracy: {week_data['performance']['accuracy']:.2%}")
        print(f"Retraining triggered: {week_data['retraining_triggered']}")

        # Show some match details
        for pred, result in zip(week_data['predictions'][:3], week_data['results'][:3]):
            print(f"  {pred['home_team']} vs {pred['away_team']}: "
                  f"Predicted {pred['predicted_result']}, "
                  f"Actual {result.get('actual_result', 'N/A')}")

        return 0

    except Exception as e:
        logger.error(f"Single week simulation failed: {str(e)}")
        return 1


def run_batch_simulation(simulator: SeasonSimulator, start_week: int, weeks: int = None) -> int:
    """Run batch simulation for multiple weeks."""
    max_week = simulator.max_week
    end_week = min(start_week + weeks - 1, max_week) if weeks else max_week

    logger.info(f"Running batch simulation from week {start_week} to {end_week}")

    try:
        season_summary = simulator.simulate_season(start_week, end_week)

        # Print summary
        print(f"\n=== Season Simulation Summary ===")
        print(f"Period: {season_summary['simulation_period']}")
        print(f"Weeks simulated: {season_summary['weeks_simulated']}")
        print(f"Duration: {season_summary['simulation_duration_seconds']:.1f} seconds")
        print(f"Overall accuracy: {season_summary['overall_performance']['overall_accuracy']:.2%}")
        print(f"Total predictions: {season_summary['total_predictions']}")
        print(f"Retraining events: {season_summary['retraining_events']}")

        # Export data
        export_paths = simulator.export_simulation_data()
        print(f"\nSimulation data exported to:")
        for data_type, path in export_paths.items():
            print(f"  {data_type}: {path}")

        return 0

    except Exception as e:
        logger.error(f"Batch simulation failed: {str(e)}")
        return 1


def run_interactive_simulation(simulator: SeasonSimulator, start_week: int, weeks: int = None) -> int:
    """Run interactive simulation with user control."""
    max_week = simulator.max_week
    end_week = min(start_week + weeks - 1, max_week) if weeks else max_week

    logger.info(f"Starting interactive simulation from week {start_week}")
    print(f"\n=== Interactive Simulation Mode ===")
    print(f"Simulating weeks {start_week} to {end_week}")
    print("Commands: [Enter] = next week, 'q' = quit, 's' = skip to end, 'r' = force retraining")

    current_week = start_week

    try:
        while current_week <= end_week:
            print(f"\n--- Week {current_week} ---")

            # Get state
            state = simulator.get_simulation_state()
            print(f"Upcoming matches: {state['upcoming_matches']}")
            print(f"Total predictions so far: {state['total_predictions']}")
            print(f"Retraining count: {state['retraining_count']}")

            # Get user input
            user_input = input(f"Simulate week {current_week}? [Enter/q/s/r]: ").strip().lower()

            if user_input == 'q':
                print("Simulation stopped by user")
                break
            elif user_input == 's':
                print(f"Skipping to batch simulation of remaining weeks...")
                return run_batch_simulation(simulator, current_week, end_week - current_week + 1)
            elif user_input == 'r':
                # Force retraining
                forced = simulator.retraining_orchestrator.force_retraining(
                    current_week, "user_requested"
                )
                print(f"Forced retraining: {'Success' if forced else 'Failed'}")
                continue

            # Simulate the week
            week_data = simulator.simulate_week(current_week)

            # Show results
            print(f"  Matches simulated: {len(week_data['matches'])}")
            print(f"  Accuracy: {week_data['performance']['accuracy']:.2%}")
            print(f"  Retraining triggered: {week_data['retraining_triggered']}")

            current_week += 1

        # Final summary
        overall_performance = simulator._calculate_overall_performance()
        print(f"\n=== Final Summary ===")
        print(f"Weeks simulated: {current_week - start_week}")
        print(f"Overall accuracy: {overall_performance['overall_accuracy']:.2%}")
        print(f"Total retraining events: {simulator.retraining_orchestrator.get_retraining_count()}")

        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Interactive simulation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    exit_code = main()
    sys.exit(exit_code)
