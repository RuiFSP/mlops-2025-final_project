#!/usr/bin/env python3
"""
Demo Script for Premier League Season Simulation

A simple demonstration of the simulation engine capabilities.
This script runs a few weeks of simulation to showcase the functionality.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import SeasonSimulator, MatchScheduler, OddsGenerator, RetrainingOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a simple simulation demo."""
    print("🏈 Premier League Season Simulation Demo")
    print("=" * 50)

    # Paths
    base_dir = Path(__file__).parent.parent
    simulation_data_path = base_dir / "data/simulation/simulation_data_2023_24.parquet"
    match_calendar_path = base_dir / "data/simulation/match_calendar.parquet"
    model_path = base_dir / "models/model.pkl"

    # Check if files exist
    if not all(f.exists() for f in [simulation_data_path, match_calendar_path, model_path]):
        print("❌ Missing required files. Please run:")
        print("   python scripts/prepare_simulation_data.py")
        return 1

    try:
        print("🔧 Initializing simulation engine...")

        # Initialize simulator
        simulator = SeasonSimulator(
            simulation_data_path=str(simulation_data_path),
            match_calendar_path=str(match_calendar_path),
            model_path=str(model_path),
            output_dir="data/simulation_demo",
            retraining_threshold=0.08,  # More sensitive for demo
            retraining_frequency=3      # Retrain every 3 weeks for demo
        )

        print("✅ Simulation engine initialized!")

        # Show initial state
        state = simulator.get_simulation_state()
        print(f"\n📊 Initial State:")
        print(f"   Total weeks: {state['max_week']}")
        print(f"   Current week: {state['current_week']}")
        print(f"   Weeks remaining: {state['weeks_remaining']}")

        # Demo component functionality
        print(f"\n🗓️  Testing MatchScheduler...")
        scheduler = MatchScheduler(str(match_calendar_path))
        week_1_matches = scheduler.get_matches_for_week(1)
        print(f"   Week 1 matches: {len(week_1_matches) if week_1_matches is not None else 0}")

        print(f"\n🎲 Testing OddsGenerator...")
        odds_gen = OddsGenerator(str(simulation_data_path))
        if len(week_1_matches) > 0:
            match = week_1_matches.iloc[0]
            home_odds, draw_odds, away_odds = odds_gen.generate_odds(
                match['HomeTeam'], match['AwayTeam']
            )
            print(f"   Sample odds for {match['HomeTeam']} vs {match['AwayTeam']}:")
            print(f"     Home: {home_odds:.2f}, Draw: {draw_odds:.2f}, Away: {away_odds:.2f}")

        # Run a few weeks of simulation
        print(f"\n🏃 Running 3-week simulation demo...")

        for week in range(1, 4):
            print(f"\n--- Week {week} ---")
            week_data = simulator.simulate_week(week)

            matches = len(week_data['matches'])
            accuracy = week_data['performance']['accuracy']
            retraining = week_data['retraining_triggered']

            print(f"   Matches: {matches}")
            print(f"   Accuracy: {accuracy:.1%}")
            print(f"   Retraining: {'Yes' if retraining else 'No'}")

            # Show a sample prediction
            if week_data['predictions']:
                pred = week_data['predictions'][0]
                result = week_data['results'][0]
                print(f"   Sample: {pred['home_team']} vs {pred['away_team']}")
                print(f"     Predicted: {pred['predicted_result']} ({pred['confidence']:.1%})")
                print(f"     Actual: {result.get('actual_result', 'N/A')}")

        # Show retraining status
        retraining_count = simulator.retraining_orchestrator.get_retraining_count()
        print(f"\n🔄 Retraining Events: {retraining_count}")

        if retraining_count > 0:
            history = simulator.retraining_orchestrator.get_retraining_history()
            for event in history:
                print(f"   Week {event['week']}: {', '.join(event['trigger_reasons'])}")

        # Show final performance
        overall_perf = simulator._calculate_overall_performance()
        print(f"\n📈 Overall Performance:")
        print(f"   Accuracy: {overall_perf['overall_accuracy']:.1%}")
        print(f"   Total matches: {overall_perf['total_matches']}")
        print(f"   Weeks analyzed: {overall_perf['weeks_analyzed']}")

        print(f"\n✅ Demo completed successfully!")
        print(f"   Output saved to: data/simulation_demo/")

        return 0

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.exception("Demo error details:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
