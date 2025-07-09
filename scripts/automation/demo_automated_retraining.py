#!/usr/bin/env python3
"""
Automated Retraining Demo Script

This script demonstrates the automated retraining capabilities by:
1. Setting up the retraining scheduler
2. Simulating various trigger conditions
3. Showing how to monitor and manage retraining
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.automation.retraining_scheduler import AutomatedRetrainingScheduler, RetrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_notification_callback():
    """Setup a simple notification callback for demo purposes."""
    def notification_callback(event_type: str, data: dict):
        logger.info(f"🔔 NOTIFICATION: {event_type}")
        if event_type == "retraining_initiated":
            logger.info(f"   Triggers: {data.get('triggers', [])}")
            logger.info(f"   Predictions since last retraining: {data.get('prediction_count', 0)}")
        elif event_type == "retraining_completed":
            logger.info(f"   Success: {data.get('success', False)}")
            logger.info(f"   Deployed: {data.get('deployed', False)}")
        elif event_type == "retraining_failed":
            logger.info(f"   Error: {data.get('error', 'Unknown error')}")

    return notification_callback


def demo_basic_scheduler():
    """Demonstrate basic scheduler functionality."""
    logger.info("=== Demo: Basic Scheduler Setup ===")

    # Create configuration with demo settings
    config = RetrainingConfig(
        performance_threshold=0.03,  # More sensitive for demo
        max_days_without_retraining=1,  # Short for demo
        min_days_between_retraining=0,  # Allow immediate retraining for demo
        check_interval_minutes=1,  # Check every minute for demo
        min_new_predictions=5,  # Low threshold for demo
        notification_callbacks=[setup_notification_callback()],
    )

    # Initialize scheduler
    scheduler = AutomatedRetrainingScheduler(config=config)

    logger.info("✅ Scheduler initialized")
    logger.info(f"📊 Status: {scheduler.get_status()}")

    return scheduler


def demo_manual_triggers(scheduler: AutomatedRetrainingScheduler):
    """Demonstrate manual retraining triggers."""
    logger.info("\n=== Demo: Manual Retraining Triggers ===")

    # Trigger manual retraining
    logger.info("🚀 Triggering manual retraining...")
    success = scheduler.force_retraining("demo_manual_trigger")

    if success:
        logger.info("✅ Manual retraining triggered successfully")

        # Wait for retraining to complete
        while scheduler.retraining_in_progress:
            logger.info("⏳ Waiting for retraining to complete...")
            time.sleep(2)

        logger.info("✅ Retraining completed")
    else:
        logger.warning("⚠️ Manual retraining could not be triggered")

    # Show status after retraining
    status = scheduler.get_status()
    logger.info(f"📊 Post-retraining status: {status}")


def demo_prediction_tracking(scheduler: AutomatedRetrainingScheduler):
    """Demonstrate prediction tracking for volume-based triggers."""
    logger.info("\n=== Demo: Prediction Volume Tracking ===")

    # Simulate predictions
    logger.info("📊 Simulating predictions...")
    for i in range(10):
        prediction_data = {
            "home_team": f"Team{i % 4}",
            "away_team": f"Team{(i + 1) % 4}",
            "prediction": ["H", "D", "A"][i % 3],
        }

        scheduler.record_prediction(prediction_data)
        logger.info(f"   Recorded prediction {i + 1}: {prediction_data}")
        time.sleep(0.1)  # Small delay

    status = scheduler.get_status()
    logger.info(f"📊 Predictions recorded: {status['prediction_count_since_retraining']}")


def demo_scheduler_lifecycle(scheduler: AutomatedRetrainingScheduler):
    """Demonstrate starting and stopping the scheduler."""
    logger.info("\n=== Demo: Scheduler Lifecycle ===")

    # Start scheduler
    logger.info("🚀 Starting automated scheduler...")
    scheduler.start_scheduler()

    logger.info("✅ Scheduler is running")
    time.sleep(3)  # Let it run for a few seconds

    # Check status while running
    status = scheduler.get_status()
    logger.info(f"📊 Running status: {status}")

    # Stop scheduler
    logger.info("🛑 Stopping scheduler...")
    scheduler.stop_scheduler()

    logger.info("✅ Scheduler stopped")


def demo_history_and_reporting(scheduler: AutomatedRetrainingScheduler):
    """Demonstrate history tracking and reporting."""
    logger.info("\n=== Demo: History and Reporting ===")

    # Get trigger history
    trigger_history = scheduler.get_trigger_history()
    logger.info(f"📈 Trigger events: {len(trigger_history)}")

    for i, event in enumerate(trigger_history):
        logger.info(f"   Event {i + 1}: {event['triggers']} at {event['timestamp']}")

    # Get retraining history
    retraining_history = scheduler.retraining_orchestrator.get_retraining_history()
    logger.info(f"🔄 Retraining events: {len(retraining_history)}")

    for i, event in enumerate(retraining_history):
        logger.info(f"   Retraining {i + 1}: {event.get('status', 'unknown')} - {event.get('trigger_reasons', [])}")

    # Export status report
    report_path = "evaluation_reports/demo_retraining_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"📄 Exporting status report to {report_path}")
    scheduler.export_status_report(report_path)

    logger.info(f"✅ Report exported to {report_path}")


def demo_configuration_management():
    """Demonstrate configuration management."""
    logger.info("\n=== Demo: Configuration Management ===")

    # Create config from file
    config_path = "config/retraining_config.yaml"
    if Path(config_path).exists():
        logger.info(f"📁 Loading config from {config_path}")
        config = RetrainingConfig.load_from_file(config_path)
        logger.info(f"✅ Config loaded: {config.performance_threshold} threshold")
    else:
        logger.info("📁 Creating default config")
        config = RetrainingConfig()

    # Save config to demo location
    demo_config_path = "config/demo_retraining_config.yaml"
    Path(demo_config_path).parent.mkdir(parents=True, exist_ok=True)

    config.save_to_file(demo_config_path)
    logger.info(f"💾 Config saved to {demo_config_path}")

    return config


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Automated Retraining Demo")
    parser.add_argument(
        "--demo",
        choices=["all", "basic", "manual", "tracking", "lifecycle", "history", "config"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🎬 Starting Automated Retraining Demo")
    logger.info(f"📋 Running demo: {args.demo}")

    try:
        if args.demo in ["all", "config"]:
            demo_configuration_management()

        if args.demo in ["all", "basic"]:
            scheduler = demo_basic_scheduler()
        else:
            # Create basic scheduler for other demos
            config = RetrainingConfig(
                check_interval_minutes=1,
                min_days_between_retraining=0,
                notification_callbacks=[setup_notification_callback()],
            )
            scheduler = AutomatedRetrainingScheduler(config=config)

        if args.demo in ["all", "manual"]:
            demo_manual_triggers(scheduler)

        if args.demo in ["all", "tracking"]:
            demo_prediction_tracking(scheduler)

        if args.demo in ["all", "lifecycle"]:
            demo_scheduler_lifecycle(scheduler)

        if args.demo in ["all", "history"]:
            demo_history_and_reporting(scheduler)

        logger.info("🎉 Demo completed successfully!")

        # Final status
        final_status = scheduler.get_status()
        logger.info(f"📊 Final status:")
        logger.info(f"   Running: {final_status['is_running']}")
        logger.info(f"   Retraining in progress: {final_status['retraining_in_progress']}")
        logger.info(f"   Total trigger events: {final_status['total_trigger_events']}")

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
