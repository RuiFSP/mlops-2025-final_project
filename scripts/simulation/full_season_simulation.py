#!/usr/bin/env python3
"""
Full Season Simulation with Prefect Visualization

This script runs a complete Premier League season simulation showing:
1. Week-by-week match predictions
2. Performance monitoring
3. Automated retraining triggers
4. Prefect flow execution visualization
5. Complete MLOps automation in action
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Direct imports to avoid circular dependencies
from src.model_training.trainer import ModelTrainer
from src.data_preprocessing.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.automation.prefect_client import PrefectClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedSeasonSimulator:
    """
    Simplified season simulator that focuses on Prefect automation visualization.

    This version avoids circular imports and focuses on demonstrating the
    automated retraining capabilities via Prefect flows.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.current_week = 1
        self.max_week = 38  # Standard Premier League season
        self.predictions_history = []
        self.performance_history = []
        self.retraining_history = []

        # Paths
        self.simulation_data_path = base_dir / "data/simulation/simulation_data_2023_24.parquet"
        self.training_data_path = base_dir / "data/simulation/training_data_2016_2023.parquet"
        self.model_path = base_dir / "models/model.pkl"
        self.output_dir = base_dir / "data/simulation_output"

        # Simulation parameters
        self.retraining_threshold = 0.05  # 5% accuracy drop triggers retraining
        self.retraining_frequency = 5     # Check every 5 weeks minimum

        # Initialize Prefect client
        self.prefect_client = PrefectClient()

        # Load simulation data
        self._load_simulation_data()

    def _load_simulation_data(self):
        """Load the 2023-24 season data for simulation."""
        logger.info("Loading simulation data...")

        if not self.simulation_data_path.exists():
            raise FileNotFoundError(f"Simulation data not found: {self.simulation_data_path}")

        self.simulation_data = pd.read_parquet(self.simulation_data_path)
        logger.info(f"Loaded {len(self.simulation_data)} matches for simulation")

        # Add week numbers to matches (simplified)
        self.simulation_data['Week'] = ((pd.to_datetime(self.simulation_data['Date']).dt.dayofyear - 230) // 7 + 1).clip(1, 38)

    def get_matches_for_week(self, week: int) -> pd.DataFrame:
        """Get matches for a specific week."""
        week_matches = self.simulation_data[self.simulation_data['Week'] == week]
        return week_matches.head(10)  # Limit for demo purposes

    def simulate_week(self, week: int) -> dict:
        """Simulate a single week with predictions and results."""
        logger.info(f"🏟️  Simulating Week {week}")

        # Get matches for this week
        week_matches = self.get_matches_for_week(week)

        if len(week_matches) == 0:
            logger.warning(f"No matches found for week {week}")
            return {"week": week, "matches": 0, "accuracy": 0.0, "retraining_triggered": False}

        # Simulate predictions (simplified)
        predictions = []
        correct_predictions = 0

        for _, match in week_matches.iterrows():
            # Simple prediction logic based on home advantage
            home_prob = 0.45
            draw_prob = 0.30
            away_prob = 0.25

            prediction = {
                "home_team": match['HomeTeam'],
                "away_team": match['AwayTeam'],
                "predicted_result": "Home Win",
                "confidence": home_prob,
                "home_win_probability": home_prob,
                "draw_probability": draw_prob,
                "away_win_probability": away_prob
            }
            predictions.append(prediction)

            # Check if prediction was correct (simplified)
            actual_result = self._determine_actual_result(match)
            if actual_result == "Home Win":
                correct_predictions += 1

        # Calculate week accuracy
        week_accuracy = correct_predictions / len(week_matches) if len(week_matches) > 0 else 0.0
        self.performance_history.append(week_accuracy)

        # Check if retraining should be triggered
        retraining_triggered = self._check_retraining_triggers(week, week_accuracy)

        week_data = {
            "week": week,
            "matches": len(week_matches),
            "accuracy": week_accuracy,
            "predictions": predictions,
            "retraining_triggered": retraining_triggered
        }

        # Save week data
        self._save_week_data(week_data)

        return week_data

    def _determine_actual_result(self, match) -> str:
        """Determine actual match result (simplified)."""
        home_goals = match.get('FTHG', 0)
        away_goals = match.get('FTAG', 0)

        if home_goals > away_goals:
            return "Home Win"
        elif away_goals > home_goals:
            return "Away Win"
        else:
            return "Draw"

    def _check_retraining_triggers(self, week: int, current_accuracy: float) -> bool:
        """Check if automated retraining should be triggered."""

        # Need at least 3 weeks of history to detect trends
        if len(self.performance_history) < 3:
            return False

        # Calculate recent performance trend
        recent_performance = self.performance_history[-3:]
        avg_recent = sum(recent_performance) / len(recent_performance)

        # Get baseline performance (first 5 weeks)
        if len(self.performance_history) >= 5:
            baseline_performance = sum(self.performance_history[:5]) / 5
        else:
            baseline_performance = 0.55  # Default expected accuracy

        # Check triggers
        performance_drop = baseline_performance - avg_recent
        weeks_since_retraining = week - (self.retraining_history[-1]["week"] if self.retraining_history else 0)

        should_retrain = False
        reasons = []

        # Performance degradation trigger
        if performance_drop > self.retraining_threshold:
            should_retrain = True
            reasons.append(f"performance_drop_{performance_drop:.3f}")

        # Time-based trigger
        if weeks_since_retraining >= self.retraining_frequency:
            should_retrain = True
            reasons.append("time_based_trigger")

        # Random trigger for demonstration (every 8-12 weeks)
        if week % 10 == 7:  # Trigger at weeks 7, 17, 27, 37
            should_retrain = True
            reasons.append("demo_scheduled_trigger")

        if should_retrain:
            logger.warning(f"🔄 Retraining triggered at week {week}: {', '.join(reasons)}")
            self._trigger_automated_retraining(week, reasons)

        return should_retrain

    def _trigger_automated_retraining(self, week: int, reasons: list):
        """Trigger automated retraining via Prefect."""
        logger.info(f"🚀 Triggering Prefect retraining flow for week {week}")

        try:
            # Trigger Prefect deployment
            flow_run_id = self.prefect_client.trigger_retraining_deployment(
                reason=f"week_{week}_{'_'.join(reasons)}",
                force=True,
                additional_params={
                    "simulation_week": week,
                    "trigger_reasons": reasons,
                    "performance_history": self.performance_history[-5:],
                    "simulation_mode": True
                }
            )

            logger.info(f"✅ Prefect flow triggered: {flow_run_id}")

            # Record retraining event
            self.retraining_history.append({
                "week": week,
                "reasons": reasons,
                "flow_run_id": flow_run_id,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"❌ Failed to trigger Prefect retraining: {e}")

    def _save_week_data(self, week_data: dict):
        """Save week simulation data."""
        output_dir = self.output_dir / f"week_{week_data['week']:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        if week_data['predictions']:
            pred_df = pd.DataFrame(week_data['predictions'])
            pred_df.to_parquet(output_dir / "predictions.parquet")

        # Save performance
        perf_data = {
            "week": week_data['week'],
            "accuracy": week_data['accuracy'],
            "matches": week_data['matches'],
            "retraining_triggered": week_data['retraining_triggered']
        }

        with open(output_dir / "performance.json", 'w') as f:
            import json
            json.dump(perf_data, f, indent=2)

    def simulate_full_season(self, start_week: int = 1, end_week: int = None, delay: float = 2.0):
        """
        Simulate the complete season with Prefect visualization.

        Args:
            start_week: Week to start simulation
            end_week: Week to end simulation (None = full season)
            delay: Delay between weeks in seconds (for visualization)
        """
        if end_week is None:
            end_week = self.max_week

        logger.info(f"🏁 Starting full season simulation (Weeks {start_week}-{end_week})")
        logger.info(f"📊 Monitor Prefect flows at: http://localhost:4200")
        logger.info(f"⏱️  Delay between weeks: {delay}s")

        print("\n" + "="*80)
        print("🏈 PREMIER LEAGUE SEASON SIMULATION WITH AUTOMATED RETRAINING")
        print("="*80)
        print(f"📅 Simulating weeks {start_week} to {end_week}")
        print(f"🎯 Retraining threshold: {self.retraining_threshold:.1%} accuracy drop")
        print(f"⏰ Check Prefect UI: http://localhost:4200")
        print("="*80)

        total_matches = 0
        total_retraining_events = 0

        for week in range(start_week, end_week + 1):
            print(f"\n📍 WEEK {week:2d} {'='*20}")

            # Simulate the week
            week_data = self.simulate_week(week)

            total_matches += week_data['matches']
            if week_data['retraining_triggered']:
                total_retraining_events += 1

            # Display week results
            print(f"   Matches: {week_data['matches']:2d}")
            print(f"   Accuracy: {week_data['accuracy']:6.1%}")
            print(f"   Retraining: {'🔄 YES' if week_data['retraining_triggered'] else '✅ No'}")

            # Show recent trend
            if len(self.performance_history) >= 3:
                recent_avg = sum(self.performance_history[-3:]) / 3
                print(f"   Recent Avg: {recent_avg:6.1%}")

            # Show retraining events
            if week_data['retraining_triggered'] and self.retraining_history:
                latest_event = self.retraining_history[-1]
                print(f"   🚀 Flow ID: {latest_event['flow_run_id'][:8]}...")

            # Wait before next week (for visualization)
            if week < end_week:
                time.sleep(delay)

        # Final summary
        print("\n" + "="*80)
        print("🎉 SEASON SIMULATION COMPLETE")
        print("="*80)
        print(f"📊 Total matches simulated: {total_matches}")
        print(f"🔄 Total retraining events: {total_retraining_events}")
        print(f"📈 Final accuracy: {self.performance_history[-1]:.1%}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"🎯 Prefect flows: http://localhost:4200")

        if self.retraining_history:
            print(f"\n🔄 Retraining Events:")
            for event in self.retraining_history:
                print(f"   Week {event['week']:2d}: {', '.join(event['reasons'])}")

        return {
            "total_weeks": end_week - start_week + 1,
            "total_matches": total_matches,
            "retraining_events": total_retraining_events,
            "final_accuracy": self.performance_history[-1] if self.performance_history else 0.0,
            "retraining_history": self.retraining_history
        }


def main():
    """Run the full season simulation."""
    print("🏈 Premier League Season Simulation with Prefect Automation")
    print("=" * 70)

    try:
        # Initialize simulator
        base_dir = Path(__file__).parent.parent.parent
        simulator = SimplifiedSeasonSimulator(base_dir)

        # Check Prefect connection
        print("🔧 Checking Prefect connection...")
        if not simulator.prefect_client.is_connected():
            print("❌ Prefect server not accessible. Please ensure it's running:")
            print("   make prefect-server")
            return 1

        print("✅ Prefect server connected!")
        print(f"📊 Monitor automation at: http://localhost:4200")

        # Ask user for simulation preferences
        print("\n🎯 Simulation Options:")
        print("1. Quick demo (5 weeks)")
        print("2. Medium run (15 weeks)")
        print("3. Full season (38 weeks)")

        try:
            choice = input("\nSelect option (1-3): ").strip()
            delay = float(input("Delay between weeks (seconds, 1-5): ") or "2")
        except (ValueError, KeyboardInterrupt):
            choice = "1"
            delay = 2.0

        # Set simulation parameters
        if choice == "3":
            end_week = 38
            delay = max(delay, 1.0)  # Minimum 1s for full season
        elif choice == "2":
            end_week = 15
        else:
            end_week = 5

        print(f"\n🚀 Starting simulation: {end_week} weeks with {delay}s delay")
        print("⚠️  Keep Prefect UI open: http://localhost:4200")
        input("Press Enter to start...")

        # Run simulation
        results = simulator.simulate_full_season(
            start_week=1,
            end_week=end_week,
            delay=delay
        )

        print(f"\n✅ Simulation completed successfully!")
        print(f"🎯 Check Prefect UI for flow executions: http://localhost:4200")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user")
        return 0
    except Exception as e:
        logger.exception("Simulation failed")
        print(f"❌ Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
