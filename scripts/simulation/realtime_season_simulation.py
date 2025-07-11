#!/usr/bin/env python3
"""
Real-Time Season Simulation with Prefect Automation

This script demonstrates the complete MLOps automation workflow:
1. Each week represents a new batch of matches (processed every X seconds)
2. Make predictions for the week's matches
3. Reveal actual results and calculate performance
4. Check for retraining triggers (performance degradation)
5. Trigger automated retraining via Prefect if needed
6. Visualize the entire process in Prefect UI

This simulates the real-world MLOps workflow where:
- API receives weekly match data
- Model makes predictions
- Results are compared against actual outcomes
- Performance monitoring triggers retraining when needed
- New model is deployed automatically
"""

import logging
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import json

# Set Prefect API URL to connect to the main server
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.simulation import SeasonSimulator, MatchScheduler, OddsGenerator, RetrainingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RealtimeSeasonSimulation:
    """Real-time season simulation with automated retraining."""
    
    def __init__(self, week_interval_seconds: int = 10):
        """
        Initialize the real-time simulation.
        
        Args:
            week_interval_seconds: Time between each week simulation (default: 10 seconds)
        """
        self.week_interval = week_interval_seconds
        self.simulation_start_time = datetime.now()
        
        # Paths
        base_dir = Path(__file__).parent.parent.parent
        self.simulation_data_path = base_dir / "data/simulation/simulation_data_2023_24.parquet"
        self.match_calendar_path = base_dir / "data/simulation/match_calendar.parquet"
        self.model_path = base_dir / "models/model.pkl"
        
        # Check if files exist
        if not all(f.exists() for f in [self.simulation_data_path, self.match_calendar_path, self.model_path]):
            raise FileNotFoundError("Missing required files. Please ensure simulation data exists.")
        
        # Initialize simulator with realistic settings
        self.simulator = SeasonSimulator(
            simulation_data_path=str(self.simulation_data_path),
            match_calendar_path=str(self.match_calendar_path),
            model_path=str(self.model_path),
            output_dir="data/realtime_simulation",
            retraining_threshold=0.05,  # 5% performance drop triggers retraining
            retraining_frequency=4,     # Check every 4 weeks
            use_prefect=True,          # Use Prefect for retraining automation
            prefect_deployment_name="simulation-triggered-retraining"
        )
        
        logger.info("✅ Real-time simulation initialized")
        logger.info(f"📊 Total weeks to simulate: {self.simulator.max_week}")
        logger.info(f"⏱️  Week interval: {self.week_interval} seconds")
        logger.info(f"🎯 Retraining threshold: {self.simulator.retraining_threshold*100}%")
        logger.info(f"🔄 Retraining frequency: {self.simulator.retraining_frequency} weeks")
    
    def display_simulation_header(self):
        """Display simulation header with key information."""
        print("\n" + "="*80)
        print("🏟️  REAL-TIME PREMIER LEAGUE SEASON SIMULATION")
        print("="*80)
        print(f"📅 Simulation Period: 2023-24 Premier League Season")
        print(f"⏱️  Week Processing: Every {self.week_interval} seconds")
        print(f"🎯 Retraining Trigger: {self.simulator.retraining_threshold*100}% performance drop")
        print(f"🔄 Retraining Frequency: Every {self.simulator.retraining_frequency} weeks")
        print(f"🚀 Prefect Integration: {'✅ Enabled' if self.simulator.use_prefect else '❌ Disabled'}")
        print(f"📊 Total Weeks: {self.simulator.max_week}")
        print("="*80)
    
    def display_week_summary(self, week: int, week_data: dict):
        """Display summary for a completed week."""
        matches = len(week_data.get('matches', []))
        
        # Handle case where performance might be missing
        performance = week_data.get('performance', {})
        accuracy = performance.get('accuracy', 0.0)
        retraining = week_data.get('retraining_triggered', False)
        
        print(f"\n📋 WEEK {week} SUMMARY")
        print(f"   ⚽ Matches: {matches}")
        print(f"   🎯 Accuracy: {accuracy:.1%}")
        print(f"   🔄 Retraining: {'🚨 TRIGGERED' if retraining else '✅ No Action'}")
        
        # Show sample prediction if available
        predictions = week_data.get('predictions', [])
        results = week_data.get('results', [])
        
        if predictions and results:
            pred = predictions[0]
            result = results[0]
            print(f"   📊 Sample: {pred['home_team']} vs {pred['away_team']}")
            print(f"      Predicted: {pred['predicted_result']} ({pred['confidence']:.1%})")
            print(f"      Actual: {result.get('actual_result', 'N/A')}")
        elif matches == 0:
            print(f"   📊 No matches scheduled for this week")
        
        # Show retraining details if triggered
        if retraining:
            print(f"   🚨 RETRAINING TRIGGERED!")
            print(f"      📈 Performance monitoring detected degradation")
            print(f"      🔄 Prefect flow initiated for model retraining")
            print(f"      📊 Check Prefect UI for retraining progress")
    
    def display_simulation_progress(self, current_week: int, total_weeks: int):
        """Display overall simulation progress."""
        progress = (current_week / total_weeks) * 100
        elapsed = datetime.now() - self.simulation_start_time
        
        print(f"\n📊 SIMULATION PROGRESS")
        print(f"   Progress: {progress:.1f}% ({current_week}/{total_weeks} weeks)")
        print(f"   Elapsed: {elapsed.total_seconds():.0f} seconds")
        print(f"   Retraining Events: {self.simulator.retraining_orchestrator.get_retraining_count()}")
    
    def display_retraining_summary(self):
        """Display summary of all retraining events."""
        retraining_count = self.simulator.retraining_orchestrator.get_retraining_count()
        
        if retraining_count > 0:
            print(f"\n🔄 RETRAINING SUMMARY")
            print(f"   Total Events: {retraining_count}")
            
            history = self.simulator.retraining_orchestrator.get_retraining_history()
            for i, event in enumerate(history, 1):
                print(f"   Event {i}: Week {event['week']}")
                print(f"      Triggers: {', '.join(event['trigger_reasons'])}")
                print(f"      Status: {event['status']}")
                if 'performance_improvement' in event:
                    improvement = event['performance_improvement']
                    print(f"      Improvement: {improvement:+.1%}")
        else:
            print(f"\n🔄 No retraining events triggered during simulation")
    
    def display_final_summary(self):
        """Display final simulation summary."""
        overall_perf = self.simulator._calculate_overall_performance()
        elapsed = datetime.now() - self.simulation_start_time
        
        print(f"\n" + "="*80)
        print("📈 FINAL SIMULATION SUMMARY")
        print("="*80)
        print(f"📊 Overall Performance:")
        print(f"   Accuracy: {overall_perf['overall_accuracy']:.1%}")
        print(f"   Total Matches: {overall_perf['total_matches']}")
        print(f"   Weeks Analyzed: {overall_perf['weeks_analyzed']}")
        print(f"⏱️  Simulation Duration: {elapsed.total_seconds():.0f} seconds")
        print(f"🔄 Retraining Events: {self.simulator.retraining_orchestrator.get_retraining_count()}")
        print(f"📁 Output Directory: data/realtime_simulation/")
        print("="*80)
    
    def run_simulation(self, max_weeks: int = None, start_week: int = 1):
        """
        Run the real-time simulation.
        
        Args:
            max_weeks: Maximum number of weeks to simulate (default: all weeks)
            start_week: Week to start from (default: 1)
        """
        self.display_simulation_header()
        
        # Determine simulation range
        end_week = min(max_weeks or self.simulator.max_week, self.simulator.max_week)
        
        print(f"\n🚀 Starting real-time simulation...")
        print(f"   Simulating weeks {start_week} to {end_week}")
        print(f"   Each week will be processed every {self.week_interval} seconds")
        print(f"   💡 Open Prefect UI to monitor retraining flows: http://localhost:4200")
        
        processed_weeks = 0
        target_weeks = max_weeks if max_weeks else (end_week - start_week + 1)
        
        try:
            week = start_week
            while week <= end_week and processed_weeks < target_weeks:
                print(f"\n⏳ Processing Week {week}...")
                print(f"   ⏰ {datetime.now().strftime('%H:%M:%S')} - Starting week {week}")
                
                # Process the week
                week_data = self.simulator.simulate_week(week)
                
                # Check if week has matches
                if not week_data.get("matches") or len(week_data["matches"]) == 0:
                    print(f"   ⚠️  No matches in week {week}, skipping to next week...")
                    week += 1
                    continue
                
                # Display results for weeks with matches
                processed_weeks += 1
                self.display_week_summary(week, week_data)
                self.display_simulation_progress(processed_weeks, target_weeks)
                
                # Wait before next week (except for the last processed week)
                if processed_weeks < target_weeks:
                    print(f"   ⏱️  Waiting {self.week_interval} seconds before next week...")
                    time.sleep(self.week_interval)
                
                week += 1
            
            # Display final summary
            self.display_retraining_summary()
            self.display_final_summary()
            
            print(f"\n✅ Real-time simulation completed successfully!")
            print(f"   📊 Check Prefect UI for complete automation history")
            print(f"   📁 Simulation data saved to: data/realtime_simulation/")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Simulation interrupted by user")
            print(f"   Processed {processed_weeks} weeks before interruption")
            raise
        except Exception as e:
            print(f"\n❌ Simulation failed: {e}")
            logger.exception("Simulation error details:")
            raise
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Simulation interrupted by user")
            print(f"   📊 Partial results saved to: data/realtime_simulation/")
            self.display_retraining_summary()
        except Exception as e:
            print(f"\n❌ Simulation failed: {str(e)}")
            logger.exception("Simulation error details:")
            raise


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Premier League Season Simulation")
    parser.add_argument("--weeks", type=int, default=10, help="Number of weeks to simulate (default: 10)")
    parser.add_argument("--start-week", type=int, default=1, help="Week to start from (default: 1)")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between weeks (default: 10)")
    parser.add_argument("--demo", action="store_true", help="Run quick demo (5 weeks, 5 second intervals)")
    
    args = parser.parse_args()
    
    # Demo mode settings
    if args.demo:
        weeks = 10
        interval = 5
        start_week = 1
        print("🎮 Demo Mode: 10 weeks, 5 second intervals")
    else:
        weeks = args.weeks
        interval = args.interval
        start_week = args.start_week
    
    try:
        # Initialize and run simulation
        simulation = RealtimeSeasonSimulation(week_interval_seconds=interval)
        simulation.run_simulation(max_weeks=weeks, start_week=start_week)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {str(e)}")
        print("   Please ensure simulation data exists in data/simulation/")
        return 1
    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")
        logger.exception("Simulation error details:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
