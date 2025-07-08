"""
Retraining Orchestrator - Manages automated model retraining during simulation.

Monitors model performance and triggers retraining when performance degrades
or at regular intervals, simulating realistic MLOps retraining workflows.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class RetrainingOrchestrator:
    """
    Orchestrates automated model retraining during season simulation.

    Monitors performance metrics and triggers retraining based on:
    - Performance degradation below threshold
    - Regular time-based intervals
    - Accumulated new data volume
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.05,
        frequency: int = 5,  # weeks
        min_data_points: int = 20,
        output_dir: str = "data/simulation_output/retraining",
    ):
        """
        Initialize the retraining orchestrator.

        Args:
            model_path: Path to the current model
            threshold: Performance drop threshold for triggering retraining
            frequency: Number of weeks between automatic retraining checks
            min_data_points: Minimum number of new predictions before retraining
            output_dir: Directory to save retraining artifacts
        """
        self.model_path = model_path
        self.threshold = threshold
        self.frequency = frequency
        self.min_data_points = min_data_points
        self.output_dir = Path(output_dir)

        # Initialize state
        self.baseline_performance: Optional[float] = None
        self.retraining_history: list[dict] = []
        self.performance_buffer: list[dict] = []
        self.last_retraining_week: int = 0
        self.retraining_count: int = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load current model for inspection
        self._load_current_model()

        logger.info(f"Retraining orchestrator initialized with threshold: {threshold}")

    def _load_current_model(self):
        """Load and inspect the current model."""
        try:
            if os.path.exists(self.model_path):
                self.current_model = joblib.load(self.model_path)
                logger.info(f"Loaded current model from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                self.current_model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.current_model = None

    def set_baseline_performance(self, performance: float):
        """Set the baseline performance for comparison."""
        self.baseline_performance = performance
        logger.info(f"Baseline performance set to: {performance:.4f}")

    def check_retraining_trigger(self, week: int, performance_data: dict) -> bool:
        """
        Check if retraining should be triggered based on current performance.

        Args:
            week: Current simulation week
            performance_data: Performance metrics for the week

        Returns:
            True if retraining should be triggered
        """
        # Add performance to buffer
        performance_data["week"] = week
        self.performance_buffer.append(performance_data)

        # Set baseline if not set
        if self.baseline_performance is None and performance_data.get("accuracy"):
            self.set_baseline_performance(performance_data["accuracy"])

        # Check triggers
        trigger_reasons = []

        # 1. Performance degradation trigger
        if self._check_performance_degradation(performance_data):
            trigger_reasons.append("performance_degradation")

        # 2. Time-based trigger
        if self._check_time_based_trigger(week):
            trigger_reasons.append("time_based")

        # 3. Data volume trigger
        if self._check_data_volume_trigger():
            trigger_reasons.append("data_volume")

        # Trigger retraining if any condition is met
        if trigger_reasons:
            self._trigger_retraining(week, trigger_reasons)
            return True

        return False

    def _check_performance_degradation(self, performance_data: dict) -> bool:
        """Check if performance has degraded below threshold."""
        if self.baseline_performance is None or not performance_data.get("accuracy"):
            return False

        current_accuracy = performance_data["accuracy"]
        performance_drop = self.baseline_performance - current_accuracy

        if performance_drop > self.threshold:
            logger.warning(
                f"Performance degradation detected: "
                f"baseline={self.baseline_performance:.4f}, "
                f"current={current_accuracy:.4f}, "
                f"drop={performance_drop:.4f} > threshold={self.threshold}"
            )
            return True

        return False

    def _check_time_based_trigger(self, week: int) -> bool:
        """Check if enough time has passed since last retraining."""
        weeks_since_retraining = week - self.last_retraining_week

        if weeks_since_retraining >= self.frequency:
            logger.info(
                f"Time-based retraining trigger: {weeks_since_retraining} weeks "
                f"since last retraining (frequency: {self.frequency})"
            )
            return True

        return False

    def _check_data_volume_trigger(self) -> bool:
        """Check if enough new data has accumulated."""
        new_data_points = len(self.performance_buffer)

        if new_data_points >= self.min_data_points:
            logger.info(
                f"Data volume retraining trigger: {new_data_points} new data points "
                f"(minimum: {self.min_data_points})"
            )
            return True

        return False

    def _trigger_retraining(self, week: int, reasons: list[str]):
        """Execute the retraining process."""
        logger.info(f"Triggering retraining at week {week}, reasons: {reasons}")

        retraining_start = datetime.now()

        # Create retraining record
        retraining_record = {
            "retraining_id": f"retrain_{week}_{self.retraining_count + 1}",
            "week": week,
            "timestamp": retraining_start.isoformat(),
            "trigger_reasons": reasons,
            "performance_before": self._calculate_recent_performance(),
            "data_points_used": len(self.performance_buffer),
            "baseline_performance": self.baseline_performance,
        }

        try:
            # Simulate retraining process
            retraining_results = self._simulate_retraining(week)
            retraining_record.update(retraining_results)

            # Update state
            self.last_retraining_week = week
            self.retraining_count += 1
            self.performance_buffer = []  # Clear buffer after retraining

            # Update baseline if retraining improved performance
            if retraining_results.get("performance_after"):
                self.baseline_performance = retraining_results["performance_after"]

            retraining_record["status"] = "success"

        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            retraining_record["status"] = "failed"
            retraining_record["error"] = str(e)

        retraining_end = datetime.now()
        retraining_record["duration_seconds"] = (retraining_end - retraining_start).total_seconds()

        # Save retraining record
        self.retraining_history.append(retraining_record)
        self._save_retraining_record(retraining_record)

        logger.info(f"Retraining completed: {retraining_record['status']}")

    def _simulate_retraining(self, week: int) -> dict:
        """
        Simulate the retraining process.

        In a real implementation, this would:
        1. Prepare new training data
        2. Retrain the model
        3. Validate the new model
        4. Deploy if performance improves

        For simulation, we'll mock this process.
        """
        import random
        import time

        # Simulate retraining time (2-10 seconds)
        retraining_time = random.uniform(2, 10)
        time.sleep(retraining_time / 10)  # Scaled down for demo

        # Calculate performance improvement (simulate realistic outcomes)
        current_performance = self._calculate_recent_performance()

        # 70% chance of improvement, 30% chance of no improvement or slight degradation
        if random.random() < 0.7:
            # Improvement case
            improvement = random.uniform(0.01, 0.08)
            new_performance = min(1.0, current_performance + improvement)
            deployment_decision = "deploy"
        else:
            # No improvement or slight degradation
            change = random.uniform(-0.02, 0.01)
            new_performance = max(0.0, current_performance + change)
            deployment_decision = "keep_current" if change < 0 else "deploy"

        # Generate training metrics
        training_metrics = {
            "training_accuracy": random.uniform(0.75, 0.95),
            "validation_accuracy": new_performance,
            "training_loss": random.uniform(0.3, 0.8),
            "validation_loss": random.uniform(0.4, 0.9),
            "epochs": random.randint(10, 50),
            "training_samples": random.randint(1000, 3000),
        }

        return {
            "performance_after": new_performance,
            "performance_improvement": new_performance - current_performance,
            "deployment_decision": deployment_decision,
            "training_metrics": training_metrics,
            "model_version": f"v{self.retraining_count + 1}_week{week}",
            "retraining_method": "automated_incremental",
        }

    def _calculate_recent_performance(self) -> float:
        """Calculate average performance from recent buffer."""
        if not self.performance_buffer:
            return self.baseline_performance or 0.5

        recent_accuracies = [
            p["accuracy"]
            for p in self.performance_buffer[-5:]  # Last 5 weeks
            if p.get("accuracy") is not None
        ]

        if recent_accuracies:
            return sum(recent_accuracies) / len(recent_accuracies)
        else:
            return self.baseline_performance or 0.5

    def _save_retraining_record(self, record: dict):
        """Save retraining record to file."""
        import json

        record_filename = f"{record['retraining_id']}.json"
        record_path = self.output_dir / record_filename

        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

        # Also save to master retraining log
        log_path = self.output_dir / "retraining_log.json"

        try:
            if log_path.exists():
                with open(log_path) as f:
                    log_data = json.load(f)
            else:
                log_data = {"retraining_events": []}

            log_data["retraining_events"].append(record)

            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving to retraining log: {str(e)}")

    def get_retraining_count(self) -> int:
        """Get the total number of retraining events."""
        return self.retraining_count

    def get_retraining_history(self) -> list[dict]:
        """Get the complete retraining history."""
        return self.retraining_history.copy()

    def get_recent_performance_trend(self, weeks: int = 5) -> dict:
        """Get recent performance trend analysis."""
        if len(self.performance_buffer) < 2:
            return {
                "trend": "insufficient_data",
                "recent_performance": None,
                "trend_direction": None,
            }

        recent_data = self.performance_buffer[-weeks:]
        accuracies = [p["accuracy"] for p in recent_data if p.get("accuracy")]

        if len(accuracies) < 2:
            return {
                "trend": "insufficient_data",
                "recent_performance": None,
                "trend_direction": None,
            }

        # Simple trend analysis
        first_half = accuracies[: len(accuracies) // 2]
        second_half = accuracies[len(accuracies) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        trend_change = second_avg - first_avg

        if abs(trend_change) < 0.01:
            trend_direction = "stable"
        elif trend_change > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"

        return {
            "trend": "calculated",
            "recent_performance": second_avg,
            "trend_direction": trend_direction,
            "trend_change": trend_change,
            "data_points": len(accuracies),
        }

    def force_retraining(self, week: int, reason: str = "manual_trigger") -> bool:
        """Force a retraining event regardless of triggers."""
        logger.info(f"Forcing retraining at week {week}, reason: {reason}")

        try:
            self._trigger_retraining(week, [reason])
            return True
        except Exception as e:
            logger.error(f"Forced retraining failed: {str(e)}")
            return False

    def reset_orchestrator(self):
        """Reset the orchestrator state (useful for testing)."""
        self.baseline_performance = None
        self.retraining_history = []
        self.performance_buffer = []
        self.last_retraining_week = 0
        self.retraining_count = 0

        logger.info("Retraining orchestrator state reset")

    def export_retraining_data(self) -> dict[str, str]:
        """Export retraining data for analysis."""
        export_paths = {}

        if self.retraining_history:
            # Export retraining history
            import json

            history_path = self.output_dir / "retraining_history_export.json"
            with open(history_path, "w") as f:
                json.dump(
                    {
                        "retraining_events": self.retraining_history,
                        "total_retrainings": self.retraining_count,
                        "baseline_performance": self.baseline_performance,
                        "export_timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            export_paths["history"] = str(history_path)

            # Export performance buffer
            if self.performance_buffer:
                buffer_df = pd.DataFrame(self.performance_buffer)
                buffer_path = self.output_dir / "performance_buffer.parquet"
                buffer_df.to_parquet(buffer_path)
                export_paths["performance_buffer"] = str(buffer_path)

        logger.info(f"Retraining data exported to: {export_paths}")
        return export_paths
