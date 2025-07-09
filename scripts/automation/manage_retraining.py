#!/usr/bin/env python3
"""
Automated Retraining Management Script

Production management script for automated retraining system.
Provides commands for starting, stopping, monitoring, and configuring
the automated retraining scheduler.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.automation.retraining_scheduler import AutomatedRetrainingScheduler, RetrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_scheduler(config_path: str = "config/retraining_config.yaml") -> None:
    """Start the automated retraining scheduler."""
    logger.info(f"Starting automated retraining scheduler with config: {config_path}")

    try:
        # Load configuration
        if Path(config_path).exists():
            scheduler = AutomatedRetrainingScheduler(config_path=config_path)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            scheduler = AutomatedRetrainingScheduler()

        # Start scheduler
        scheduler.start_scheduler()

        logger.info("✅ Automated retraining scheduler started successfully")
        logger.info("📊 Scheduler configuration:")
        config = scheduler.get_status()["config"]
        for key, value in config.items():
            logger.info(f"   {key}: {value}")

        # Keep running until interrupted
        try:
            logger.info("🔄 Scheduler is running. Press Ctrl+C to stop.")
            while scheduler.is_running:
                time.sleep(10)
                status = scheduler.get_status()
                if status["last_check_time"]:
                    logger.debug(f"Last check: {status['last_check_time']}")
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping scheduler...")
            scheduler.stop_scheduler()
            logger.info("✅ Scheduler stopped")

    except Exception as e:
        logger.error(f"❌ Failed to start scheduler: {str(e)}")
        sys.exit(1)


def status_check(config_path: str = "config/retraining_config.yaml") -> None:
    """Check the status of retraining system."""
    logger.info("Checking retraining system status...")

    try:
        # Try to create scheduler instance to check configuration
        if Path(config_path).exists():
            scheduler = AutomatedRetrainingScheduler(config_path=config_path)
        else:
            scheduler = AutomatedRetrainingScheduler()

        status = scheduler.get_status()

        logger.info("📊 Retraining System Status:")
        logger.info(f"   Running: {status['is_running']}")
        logger.info(f"   Retraining in progress: {status['retraining_in_progress']}")
        logger.info(f"   Last check: {status['last_check_time'] or 'Never'}")
        logger.info(f"   Last retraining: {status['last_retraining_time'] or 'Never'}")
        logger.info(f"   Predictions since retraining: {status['prediction_count_since_retraining']}")
        logger.info(f"   Days since last retraining: {status['days_since_last_retraining'] or 'N/A'}")
        logger.info(f"   Total trigger events: {status['total_trigger_events']}")

        logger.info("⚙️ Configuration:")
        config = status["config"]
        for key, value in config.items():
            logger.info(f"   {key}: {value}")

        # Show recent history
        trigger_history = scheduler.get_trigger_history()
        if trigger_history:
            logger.info(f"📈 Recent trigger events (last 5):")
            for event in trigger_history[-5:]:
                logger.info(f"   {event['timestamp']}: {event['triggers']}")
        else:
            logger.info("📈 No trigger events recorded")

        retraining_history = scheduler.retraining_orchestrator.get_retraining_history()
        if retraining_history:
            logger.info(f"🔄 Recent retraining events (last 3):")
            for event in retraining_history[-3:]:
                logger.info(f"   {event.get('timestamp', 'Unknown')}: {event.get('status', 'unknown')} - {event.get('trigger_reasons', [])}")
        else:
            logger.info("🔄 No retraining events recorded")

    except Exception as e:
        logger.error(f"❌ Failed to check status: {str(e)}")
        sys.exit(1)


def trigger_retraining(reason: str, force: bool = False) -> None:
    """Manually trigger retraining."""
    logger.info(f"Triggering manual retraining. Reason: {reason}, Force: {force}")

    try:
        scheduler = AutomatedRetrainingScheduler()

        if force:
            success = scheduler.force_retraining(reason)
        else:
            # For manual triggers, we'll use force_retraining but with user confirmation
            confirmation = input(f"Are you sure you want to trigger retraining? [y/N]: ")
            if confirmation.lower() in ['y', 'yes']:
                success = scheduler.force_retraining(reason)
            else:
                logger.info("Retraining cancelled by user")
                return

        if success:
            logger.info("✅ Retraining triggered successfully")

            # Monitor progress
            logger.info("⏳ Monitoring retraining progress...")
            while scheduler.retraining_in_progress:
                logger.info("   Retraining in progress...")
                time.sleep(5)

            logger.info("✅ Retraining completed")

            # Show final status
            retraining_history = scheduler.retraining_orchestrator.get_retraining_history()
            if retraining_history:
                latest = retraining_history[-1]
                logger.info(f"📊 Final status: {latest.get('status', 'unknown')}")
                if latest.get('deployment_decision'):
                    logger.info(f"🚀 Deployment decision: {latest['deployment_decision']}")
        else:
            logger.warning("⚠️ Retraining could not be triggered (may already be in progress)")

    except Exception as e:
        logger.error(f"❌ Failed to trigger retraining: {str(e)}")
        sys.exit(1)


def export_report(output_path: str = "evaluation_reports/retraining_management_report.json") -> None:
    """Export detailed retraining report."""
    logger.info(f"Exporting retraining report to: {output_path}")

    try:
        scheduler = AutomatedRetrainingScheduler()

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export detailed report
        scheduler.export_status_report(output_path)

        logger.info(f"✅ Report exported successfully to {output_path}")

        # Also print summary to console
        with open(output_path) as f:
            report = json.load(f)

        logger.info("📄 Report Summary:")
        scheduler_status = report.get("scheduler_status", {})
        logger.info(f"   Total trigger events: {scheduler_status.get('total_trigger_events', 0)}")

        retraining_history = report.get("retraining_history", [])
        logger.info(f"   Total retraining events: {len(retraining_history)}")

        if retraining_history:
            successful_retrainings = sum(1 for r in retraining_history if r.get('status') == 'success')
            logger.info(f"   Successful retrainings: {successful_retrainings}")

            deployed_models = sum(1 for r in retraining_history
                                if r.get('deployment_decision') == 'deploy')
            logger.info(f"   Deployed models: {deployed_models}")

    except Exception as e:
        logger.error(f"❌ Failed to export report: {str(e)}")
        sys.exit(1)


def validate_config(config_path: str = "config/retraining_config.yaml") -> None:
    """Validate retraining configuration."""
    logger.info(f"Validating configuration: {config_path}")

    try:
        if not Path(config_path).exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)

        # Try to load and validate configuration
        config = RetrainingConfig.load_from_file(config_path)

        logger.info("✅ Configuration loaded successfully")
        logger.info("📋 Configuration validation:")

        # Check thresholds
        if config.performance_threshold <= 0 or config.performance_threshold > 0.5:
            logger.warning(f"⚠️ Performance threshold seems unusual: {config.performance_threshold}")
        else:
            logger.info(f"✅ Performance threshold: {config.performance_threshold}")

        if config.drift_threshold <= 0 or config.drift_threshold > 1.0:
            logger.warning(f"⚠️ Drift threshold seems unusual: {config.drift_threshold}")
        else:
            logger.info(f"✅ Drift threshold: {config.drift_threshold}")

        # Check time settings
        if config.max_days_without_retraining <= config.min_days_between_retraining:
            logger.warning("⚠️ max_days_without_retraining should be > min_days_between_retraining")
        else:
            logger.info(f"✅ Time settings: {config.min_days_between_retraining} - {config.max_days_without_retraining} days")

        # Check paths
        if not Path(config.model_path).parent.exists():
            logger.warning(f"⚠️ Model directory doesn't exist: {Path(config.model_path).parent}")
        else:
            logger.info(f"✅ Model path: {config.model_path}")

        if not Path(config.training_data_path).exists():
            logger.warning(f"⚠️ Training data not found: {config.training_data_path}")
        else:
            logger.info(f"✅ Training data: {config.training_data_path}")

        logger.info("✅ Configuration validation completed")

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {str(e)}")
        sys.exit(1)


def create_default_config(output_path: str = "config/retraining_config.yaml") -> None:
    """Create a default configuration file."""
    logger.info(f"Creating default configuration: {output_path}")

    try:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create default configuration
        config = RetrainingConfig()
        config.save_to_file(output_path)

        logger.info(f"✅ Default configuration created: {output_path}")
        logger.info("📝 You can now edit this file to customize settings")

    except Exception as e:
        logger.error(f"❌ Failed to create default config: {str(e)}")
        sys.exit(1)


def main():
    """Main management function."""
    parser = argparse.ArgumentParser(
        description="Automated Retraining Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start scheduler with default config
  %(prog)s status                   # Check current status
  %(prog)s trigger "performance"    # Manually trigger retraining
  %(prog)s export report.json       # Export detailed report
  %(prog)s validate                 # Validate configuration
  %(prog)s create-config            # Create default configuration
        """
    )

    parser.add_argument("command",
                       choices=["start", "status", "trigger", "export", "validate", "create-config"],
                       help="Management command to execute")

    parser.add_argument("--config",
                       default="config/retraining_config.yaml",
                       help="Path to configuration file (default: config/retraining_config.yaml)")

    parser.add_argument("--reason",
                       default="manual_management",
                       help="Reason for manual trigger (default: manual_management)")

    parser.add_argument("--force",
                       action="store_true",
                       help="Force retraining without confirmation")

    parser.add_argument("--output",
                       help="Output path for export or create-config commands")

    parser.add_argument("--verbose",
                       action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"🔧 Executing management command: {args.command}")

    try:
        if args.command == "start":
            start_scheduler(args.config)

        elif args.command == "status":
            status_check(args.config)

        elif args.command == "trigger":
            trigger_retraining(args.reason, args.force)

        elif args.command == "export":
            output_path = args.output or "evaluation_reports/retraining_management_report.json"
            export_report(output_path)

        elif args.command == "validate":
            validate_config(args.config)

        elif args.command == "create-config":
            output_path = args.output or "config/retraining_config.yaml"
            create_default_config(output_path)

        logger.info("✅ Management command completed successfully")

    except KeyboardInterrupt:
        logger.info("⏹️ Command interrupted by user")
    except Exception as e:
        logger.error(f"❌ Management command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
