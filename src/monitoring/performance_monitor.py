"""
Model performance monitoring for the Premier League prediction model.
Tracks model accuracy, prediction confidence, and performance degradation.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(
        self,
        model_path: str,
        performance_threshold: float = 0.05,  # 5% degradation threshold
        monitoring_window_days: int = 30,
        output_dir: str = "evaluation_reports",
    ):
        """
        Initialize performance monitor.

        Args:
            model_path: Path to trained model
            performance_threshold: Threshold for performance degradation alert
            monitoring_window_days: Rolling window for performance calculation
            output_dir: Directory to save monitoring reports
        """
        self.model_path = Path(model_path)
        self.performance_threshold = performance_threshold
        self.monitoring_window_days = monitoring_window_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load model
        self.model = self._load_model()

        # Performance history file
        self.performance_history_file = self.output_dir / "performance_history.json"
        self.performance_history = self._load_performance_history()

        # Baseline performance (from training)
        self.baseline_accuracy = self._get_baseline_performance()

    def _load_model(self) -> Any:
        """Load the trained model."""
        if self.model_path.exists():
            return joblib.load(self.model_path)
        else:
            logger.warning(f"Model not found at {self.model_path}")
            return None

    def _load_performance_history(self) -> list[dict[str, Any]]:
        """Load performance history from file."""
        if self.performance_history_file.exists():
            with open(self.performance_history_file) as f:
                return json.load(f)
        return []

    def _save_performance_history(self):
        """Save performance history to file."""
        with open(self.performance_history_file, "w") as f:
            json.dump(self.performance_history, f, indent=2)

    def _get_baseline_performance(self) -> float:
        """Get baseline accuracy from training evaluation."""
        # Try to read from evaluation report
        eval_report_path = self.output_dir / "classification_report.txt"
        if eval_report_path.exists():
            try:
                with open(eval_report_path) as f:
                    content = f.read()
                    # Extract accuracy from classification report
                    for line in content.split("\n"):
                        if "accuracy" in line.lower():
                            # Parse accuracy value
                            parts = line.split()
                            for part in parts:
                                try:
                                    accuracy = float(part)
                                    if 0 <= accuracy <= 1:
                                        return accuracy
                                except ValueError:
                                    continue
            except Exception as e:
                logger.warning(f"Could not parse baseline accuracy: {e}")

        # Default baseline if not found
        return 0.55  # Based on your README reported accuracy

    def evaluate_batch(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        batch_id: str = None,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate model performance on a batch of data.

        Args:
            X: Feature data
            y_true: True labels
            batch_id: Batch identifier
            save_results: Whether to save results to history

        Returns:
            Performance metrics dictionary
        """
        if self.model is None:
            logger.error("Model not loaded, cannot evaluate performance")
            return {"error": "Model not available"}

        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_pred_proba = None

            # Get prediction probabilities if available
            if hasattr(self.model, "predict_proba"):
                y_pred_proba = self.model.predict_proba(X)

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)

            # Add metadata
            metrics.update(
                {
                    "batch_id": batch_id,
                    "timestamp": datetime.now().isoformat(),
                    "batch_size": len(X),
                    "baseline_accuracy": self.baseline_accuracy,
                    "accuracy_change": metrics["accuracy"] - self.baseline_accuracy,
                    "performance_degraded": (
                        self.baseline_accuracy - metrics["accuracy"]
                    )
                    > self.performance_threshold,
                }
            )

            # Save to history
            if save_results:
                self.performance_history.append(metrics)
                self._save_performance_history()

            # Log results
            self._log_performance_results(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating batch {batch_id}: {str(e)}")
            return {"error": str(e), "batch_id": batch_id}

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray = None
    ) -> dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Class-specific metrics
        (
            precision_per_class,
            recall_per_class,
            f1_per_class,
            support_per_class,
        ) = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Get unique classes
        classes = sorted(y_true.unique())

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "confusion_matrix": cm.tolist(),
            "class_metrics": {
                str(cls): {
                    "precision": float(precision_per_class[i])
                    if i < len(precision_per_class)
                    else 0.0,
                    "recall": float(recall_per_class[i])
                    if i < len(recall_per_class)
                    else 0.0,
                    "f1": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
                    "support": int(support_per_class[i])
                    if i < len(support_per_class)
                    else 0,
                }
                for i, cls in enumerate(classes)
                if i < len(precision_per_class)
            },
        }

        # Add probabilistic metrics if available
        if y_pred_proba is not None:
            try:
                # Log loss (lower is better)
                logloss = log_loss(y_true, y_pred_proba)
                metrics["log_loss"] = float(logloss)

                # Prediction confidence statistics
                max_probabilities = np.max(y_pred_proba, axis=1)
                metrics["prediction_confidence"] = {
                    "mean": float(np.mean(max_probabilities)),
                    "std": float(np.std(max_probabilities)),
                    "min": float(np.min(max_probabilities)),
                    "max": float(np.max(max_probabilities)),
                }

                # Calibration (simplified)
                # High confidence predictions should be more accurate
                high_conf_mask = max_probabilities > 0.7
                if np.sum(high_conf_mask) > 0:
                    high_conf_accuracy = accuracy_score(
                        y_true[high_conf_mask], y_pred[high_conf_mask]
                    )
                    metrics["high_confidence_accuracy"] = float(high_conf_accuracy)

            except Exception as e:
                logger.warning(f"Error calculating probabilistic metrics: {e}")

        return metrics

    def _log_performance_results(self, metrics: dict[str, Any]):
        """Log performance monitoring results."""
        batch_id = metrics.get("batch_id", "unknown")
        accuracy = metrics.get("accuracy", 0)
        accuracy_change = metrics.get("accuracy_change", 0)

        if metrics.get("performance_degraded", False):
            logger.warning(
                f"PERFORMANCE ALERT for batch {batch_id}: "
                f"Accuracy dropped to {accuracy:.3f} "
                f"(change: {accuracy_change:+.3f}, "
                f"threshold: -{self.performance_threshold:.3f})"
            )
        else:
            logger.info(
                f"Performance check for batch {batch_id}: "
                f"Accuracy: {accuracy:.3f} "
                f"(change: {accuracy_change:+.3f})"
            )

    def get_performance_summary(self, days: int = None) -> dict[str, Any]:
        """
        Get performance summary for the last N days.

        Args:
            days: Number of days to include (default: monitoring window)

        Returns:
            Performance summary statistics
        """
        if days is None:
            days = self.monitoring_window_days

        # Filter recent performance data
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_performance = [
            p
            for p in self.performance_history
            if datetime.fromisoformat(p["timestamp"]) >= cutoff_date
        ]

        if not recent_performance:
            return {
                "period_days": days,
                "evaluations_count": 0,
                "message": "No performance data available for the specified period",
            }

        # Calculate summary statistics
        accuracies = [p["accuracy"] for p in recent_performance]
        accuracy_changes = [p.get("accuracy_change", 0) for p in recent_performance]

        summary = {
            "period_days": days,
            "evaluations_count": len(recent_performance),
            "baseline_accuracy": self.baseline_accuracy,
            "current_performance": {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "min_accuracy": float(np.min(accuracies)),
                "max_accuracy": float(np.max(accuracies)),
                "latest_accuracy": float(accuracies[-1]) if accuracies else None,
            },
            "performance_trend": {
                "mean_change": float(np.mean(accuracy_changes)),
                "total_degradations": sum(
                    1
                    for p in recent_performance
                    if p.get("performance_degraded", False)
                ),
                "degradation_rate": float(
                    sum(
                        1
                        for p in recent_performance
                        if p.get("performance_degraded", False)
                    )
                    / len(recent_performance)
                ),
            },
        }

        # Add latest detailed metrics if available
        if recent_performance:
            latest = recent_performance[-1]
            summary["latest_evaluation"] = {
                "timestamp": latest["timestamp"],
                "batch_id": latest["batch_id"],
                "accuracy": latest["accuracy"],
                "precision_macro": latest.get("precision_macro"),
                "recall_macro": latest.get("recall_macro"),
                "f1_macro": latest.get("f1_macro"),
            }

        return summary

    def check_model_health(self) -> dict[str, Any]:
        """
        Comprehensive model health check.

        Returns:
            Health status and recommendations
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.model is not None,
            "baseline_accuracy": self.baseline_accuracy,
            "health_score": 1.0,  # Start with perfect health
            "issues": [],
            "recommendations": [],
        }

        if not self.model:
            health_status["health_score"] = 0.0
            health_status["issues"].append("Model not loaded or not found")
            health_status["recommendations"].append(
                "Check model file path and retrain if necessary"
            )
            return health_status

        # Check recent performance
        recent_summary = self.get_performance_summary(days=7)  # Last week

        if recent_summary["evaluations_count"] == 0:
            health_status["health_score"] -= 0.3
            health_status["issues"].append("No recent performance evaluations")
            health_status["recommendations"].append(
                "Set up regular model evaluation on production data"
            )
        else:
            # Check for performance degradation
            degradation_rate = recent_summary["performance_trend"]["degradation_rate"]
            mean_accuracy = recent_summary["current_performance"]["mean_accuracy"]

            if degradation_rate > 0.5:  # More than 50% of evaluations show degradation
                health_status["health_score"] -= 0.4
                health_status["issues"].append(
                    f"High degradation rate: {degradation_rate:.1%}"
                )
                health_status["recommendations"].append(
                    "Investigate data drift and consider model retraining"
                )

            if mean_accuracy < (self.baseline_accuracy - self.performance_threshold):
                health_status["health_score"] -= 0.3
                health_status["issues"].append(
                    f"Mean accuracy below threshold: {mean_accuracy:.3f}"
                )
                health_status["recommendations"].append(
                    "Model performance significantly degraded - retraining recommended"
                )

        # Determine overall health status
        if health_status["health_score"] >= 0.8:
            health_status["status"] = "healthy"
        elif health_status["health_score"] >= 0.5:
            health_status["status"] = "warning"
        else:
            health_status["status"] = "critical"

        return health_status
