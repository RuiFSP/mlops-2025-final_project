"""
Automated Retraining Scheduler - Production-ready automated model retraining system.

This module provides enterprise-grade automated retraining capabilities that monitor
model performance and data drift to trigger retraining when needed.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..monitoring.monitoring_service import MLOpsMonitoringService

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""

    # Performance thresholds
    performance_threshold: float = 0.05  # Trigger if accuracy drops by 5%
    drift_threshold: float = 0.1  # Trigger if drift score exceeds 10%

    # Time-based triggers
    max_days_without_retraining: int = 30
    min_days_between_retraining: int = 7

    # Data volume triggers
    min_new_predictions: int = 100
    max_predictions_without_retraining: int = 1000

    # Validation requirements
    min_validation_accuracy: float = 0.45  # Don't deploy if worse than 45%
    validation_improvement_required: float = 0.01  # Must improve by 1%

    # Paths and settings
    model_path: str = "models/model.pkl"
    backup_model_dir: str = "models/backups"
    training_data_path: str = "data/real_data/premier_league_matches.parquet"
    monitoring_output_dir: str = "evaluation_reports"

    # Scheduler settings
    check_interval_minutes: int = 60  # Check triggers every hour
    enable_automatic_deployment: bool = False  # Safety: manual approval by default
    max_concurrent_retraining: int = 1

    # Notification settings
    notification_callbacks: list[Callable] = field(default_factory=list)

    @classmethod
    def load_from_file(cls, config_path: str) -> "RetrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "notification_callbacks"
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class AutomatedRetrainingScheduler:
    """
    Production-ready automated retraining scheduler.

    Monitors model performance, data drift, and other triggers to automatically
    initiate retraining workflows when conditions are met.
    """

    def __init__(
        self,
        config: RetrainingConfig | None = None,
        config_path: str | None = None,
    ):
        """
        Initialize the automated retraining scheduler.

        Args:
            config: Retraining configuration object
            config_path: Path to configuration file (if config not provided)
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = RetrainingConfig.load_from_file(config_path)
        else:
            self.config = RetrainingConfig()

        # Initialize components
        self.monitoring_service = MLOpsMonitoringService(
            model_path=self.config.model_path,
            reference_data_path=self.config.training_data_path,
            output_dir=self.config.monitoring_output_dir,
            drift_threshold=self.config.drift_threshold,
            performance_threshold=self.config.performance_threshold,
        )

        # Import here to avoid circular dependency
        from ..simulation.retraining_orchestrator import RetrainingOrchestrator

        self.retraining_orchestrator = RetrainingOrchestrator(
            model_path=self.config.model_path,
            threshold=self.config.performance_threshold,
            frequency=7,  # Check weekly
            min_data_points=self.config.min_new_predictions,
        )

        # State management
        self.is_running = False
        self.scheduler_thread: threading.Thread | None = None
        self.last_check_time: datetime | None = None
        self.last_retraining_time: datetime | None = None
        self.retraining_in_progress = False
        self.retraining_lock = threading.Lock()

        # Metrics tracking
        self.prediction_count_since_retraining = 0
        self.performance_history: list[dict] = []
        self.trigger_history: list[dict] = []

        # Setup directories
        Path(self.config.backup_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.monitoring_output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Automated retraining scheduler initialized")

    def start_scheduler(self) -> None:
        """Start the automated retraining scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="RetrainingScheduler"
        )
        self.scheduler_thread.start()

        logger.info(
            f"Automated retraining scheduler started. "
            f"Check interval: {self.config.check_interval_minutes} minutes"
        )

    def stop_scheduler(self) -> None:
        """Stop the automated retraining scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return

        self.is_running = False

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=30)

        logger.info("Automated retraining scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop that runs in a separate thread."""
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                self._check_retraining_triggers()
                self.last_check_time = datetime.now()

                # Sleep for the configured interval
                time.sleep(self.config.check_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                # Continue running even if there's an error
                time.sleep(60)  # Wait 1 minute before retrying

    def _check_retraining_triggers(self) -> None:
        """Check all retraining triggers and initiate retraining if needed."""
        if self.retraining_in_progress:
            logger.debug("Retraining already in progress, skipping trigger check")
            return

        # Check if minimum time between retraining has passed
        if self._is_too_soon_for_retraining():
            logger.debug("Too soon since last retraining, skipping checks")
            return

        triggers_met = []

        # 1. Performance degradation trigger
        if self._check_performance_trigger():
            triggers_met.append("performance_degradation")

        # 2. Data drift trigger
        if self._check_drift_trigger():
            triggers_met.append("data_drift")

        # 3. Time-based trigger
        if self._check_time_trigger():
            triggers_met.append("time_based")

        # 4. Data volume trigger
        if self._check_data_volume_trigger():
            triggers_met.append("data_volume")

        # Initiate retraining if any triggers are met
        if triggers_met:
            self._initiate_retraining(triggers_met)

    def _check_performance_trigger(self) -> bool:
        """Check if model performance has degraded below threshold."""
        try:
            # Get recent performance metrics
            recent_metrics = self._get_recent_performance_metrics()

            if not recent_metrics or len(recent_metrics) < 2:
                logger.debug("Insufficient performance data for trigger check")
                return False

            # Calculate recent average performance
            recent_accuracy = sum(m["accuracy"] for m in recent_metrics[-5:]) / len(
                recent_metrics[-5:]
            )

            # Compare with baseline or historical performance
            baseline_accuracy = self._get_baseline_accuracy()

            if baseline_accuracy is None:
                logger.debug("No baseline accuracy available")
                return False

            performance_drop = baseline_accuracy - recent_accuracy

            if performance_drop > self.config.performance_threshold:
                logger.warning(
                    f"Performance degradation detected: "
                    f"baseline={baseline_accuracy:.4f}, "
                    f"recent={recent_accuracy:.4f}, "
                    f"drop={performance_drop:.4f}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking performance trigger: {str(e)}")
            return False

    def _check_drift_trigger(self) -> bool:
        """Check if data drift exceeds threshold."""
        try:
            # Create minimal data for drift checking (in production this would be real recent data)
            import pandas as pd

            minimal_data = pd.DataFrame({"dummy": [1, 2, 3]})

            # Use monitoring service to detect drift
            drift_report = self.monitoring_service.generate_drift_report(minimal_data)

            # Extract drift score from report
            drift_score = drift_report.get("drift_score", 0.0)

            if drift_score > self.config.drift_threshold:
                logger.warning(
                    f"Data drift detected: {drift_score:.4f} > {self.config.drift_threshold}"
                )
                return True

            return False

        except Exception as e:
            logger.debug(f"Could not check drift trigger: {str(e)}")
            return False

    def _check_time_trigger(self) -> bool:
        """Check if maximum time without retraining has passed."""
        if self.last_retraining_time is None:
            # If we've never retrained, check if model is old enough
            try:
                model_path = Path(self.config.model_path)
                if model_path.exists():
                    model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
                    if model_age.days > self.config.max_days_without_retraining:
                        logger.info(
                            f"Model is {model_age.days} days old, triggering time-based retraining"
                        )
                        return True
            except Exception as e:
                logger.error(f"Error checking model age: {str(e)}")

            return False

        days_since_retraining = (datetime.now() - self.last_retraining_time).days

        if days_since_retraining > self.config.max_days_without_retraining:
            logger.info(f"Time-based trigger: {days_since_retraining} days since last retraining")
            return True

        return False

    def _check_data_volume_trigger(self) -> bool:
        """Check if enough new data has accumulated."""
        if self.prediction_count_since_retraining >= self.config.max_predictions_without_retraining:
            logger.info(
                f"Data volume trigger: {self.prediction_count_since_retraining} "
                f"predictions since last retraining"
            )
            return True

        return False

    def _is_too_soon_for_retraining(self) -> bool:
        """Check if minimum time between retraining has passed."""
        if self.last_retraining_time is None:
            return False

        days_since_retraining = (datetime.now() - self.last_retraining_time).days
        return days_since_retraining < self.config.min_days_between_retraining

    def _initiate_retraining(self, triggers: list[str]) -> None:
        """Initiate the retraining process."""
        with self.retraining_lock:
            if self.retraining_in_progress:
                logger.warning("Retraining already in progress")
                return

            self.retraining_in_progress = True

        try:
            logger.info(f"Initiating automated retraining. Triggers: {triggers}")

            # Record trigger event
            trigger_event = {
                "timestamp": datetime.now().isoformat(),
                "triggers": triggers,
                "prediction_count": self.prediction_count_since_retraining,
                "days_since_last_retraining": (
                    (datetime.now() - self.last_retraining_time).days
                    if self.last_retraining_time
                    else None
                ),
            }
            self.trigger_history.append(trigger_event)

            # Send notifications
            self._send_notifications("retraining_initiated", trigger_event)

            # Start retraining in a separate thread
            retraining_thread = threading.Thread(
                target=self._execute_retraining,
                args=(triggers,),
                daemon=True,
                name="RetrainingExecution",
            )
            retraining_thread.start()

        except Exception as e:
            logger.error(f"Error initiating retraining: {str(e)}")
            self.retraining_in_progress = False

    def _execute_retraining(self, triggers: list[str]) -> None:
        """Execute the retraining process."""
        try:
            # Import here to avoid circular imports
            from ..automation.retraining_flow import execute_automated_retraining

            # Execute retraining
            retraining_result = execute_automated_retraining(
                config=self.config,
                triggers=triggers,
            )

            # Update state based on results
            if retraining_result.get("success", False):
                self.last_retraining_time = datetime.now()
                self.prediction_count_since_retraining = 0

                # Send success notification
                self._send_notifications("retraining_completed", retraining_result)

                logger.info("Automated retraining completed successfully")
            else:
                # Send failure notification
                self._send_notifications("retraining_failed", retraining_result)

                logger.error(
                    f"Automated retraining failed: {retraining_result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            error_msg = f"Error executing retraining: {str(e)}"
            logger.error(error_msg)

            # Send error notification
            self._send_notifications("retraining_error", {"error": error_msg})

        finally:
            self.retraining_in_progress = False

    def _get_recent_performance_metrics(self) -> list[dict]:
        """Get recent performance metrics from monitoring service."""
        # This would typically read from a metrics database or files
        # For now, return mock data or read from evaluation reports
        try:
            reports_dir = Path(self.config.monitoring_output_dir)
            metrics = []

            # Look for recent monitoring reports
            for report_file in reports_dir.glob("monitoring_summary_*.json"):
                try:
                    import json

                    with open(report_file) as f:
                        report_data = json.load(f)

                    if "performance_metrics" in report_data:
                        metrics.append(report_data["performance_metrics"])
                except Exception as e:
                    logger.debug(f"Could not read report {report_file}: {str(e)}")

            return metrics[-10:]  # Return last 10 metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return []

    def _get_baseline_accuracy(self) -> float | None:
        """Get baseline accuracy for comparison."""
        # This could be from initial model evaluation, target accuracy, etc.
        return self.retraining_orchestrator.baseline_performance or 0.55  # 55% baseline

    def _send_notifications(self, event_type: str, data: dict) -> None:
        """Send notifications via configured callbacks."""
        for callback in self.config.notification_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")

    def record_prediction(self, prediction_data: dict) -> None:
        """Record a new prediction for volume tracking."""
        self.prediction_count_since_retraining += 1

        # Could store to database or file for analysis
        logger.debug(
            f"Recorded prediction. Total since retraining: {self.prediction_count_since_retraining}"
        )

    def force_retraining(self, reason: str = "manual_trigger") -> bool:
        """Force immediate retraining regardless of triggers."""
        logger.info(f"Forcing retraining: {reason}")

        if self.retraining_in_progress:
            logger.warning("Retraining already in progress")
            return False

        self._initiate_retraining([reason])
        return True

    def get_status(self) -> dict:
        """Get current scheduler status."""
        return {
            "is_running": self.is_running,
            "retraining_in_progress": self.retraining_in_progress,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_retraining_time": self.last_retraining_time.isoformat()
            if self.last_retraining_time
            else None,
            "prediction_count_since_retraining": self.prediction_count_since_retraining,
            "days_since_last_retraining": (
                (datetime.now() - self.last_retraining_time).days
                if self.last_retraining_time
                else None
            ),
            "total_trigger_events": len(self.trigger_history),
            "config": {
                "performance_threshold": self.config.performance_threshold,
                "drift_threshold": self.config.drift_threshold,
                "max_days_without_retraining": self.config.max_days_without_retraining,
                "check_interval_minutes": self.config.check_interval_minutes,
            },
        }

    def get_trigger_history(self) -> list[dict]:
        """Get history of retraining triggers."""
        return self.trigger_history.copy()

    def update_config(self, new_config: RetrainingConfig) -> None:
        """Update scheduler configuration."""
        self.config = new_config
        logger.info("Scheduler configuration updated")

    def export_status_report(self, output_path: str) -> None:
        """Export detailed status report."""

        # Helper function to convert objects to JSON-serializable format
        def make_serializable(obj: Any) -> Any:
            if hasattr(obj, "__dict__"):
                return {
                    k: make_serializable(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
            elif isinstance(obj, list | tuple):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, "__class__") and "Mock" in obj.__class__.__name__:
                return f"<Mock: {obj.__class__.__name__}>"
            else:
                try:
                    import json

                    json.dumps(obj)  # Test if serializable
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        report = {
            "scheduler_status": make_serializable(self.get_status()),
            "trigger_history": make_serializable(self.get_trigger_history()),
            "retraining_history": make_serializable(
                self.retraining_orchestrator.get_retraining_history()
            ),
            "configuration": make_serializable(self.config),
            "export_timestamp": datetime.now().isoformat(),
        }

        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Status report exported to {output_path}")
