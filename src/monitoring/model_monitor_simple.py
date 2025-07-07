"""Simplified model monitoring for Premier League match prediction."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelMonitor:
    """Simple model monitoring without external dependencies."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_column: str = "result",
        prediction_column: str = "predicted_result",
    ):
        """Initialize the model monitor.

        Args:
            reference_data: Reference dataset for comparison
            target_column: Name of the target column
            prediction_column: Name of the prediction column
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.logger = logger
        self.last_performance = None  # Track last performance metric

    def calculate_data_drift(
        self, current_data: pd.DataFrame, threshold: float = 0.1
    ) -> dict[str, Any]:
        """Calculate simple data drift metrics.

        Args:
            current_data: Current dataset
            threshold: Threshold for drift detection

        Returns:
            Dictionary with drift metrics
        """
        drift_metrics: dict[str, Any] = {}

        # Compare feature distributions
        for column in self.reference_data.columns:
            if column in current_data.columns:
                # For numerical columns
                if self.reference_data[column].dtype in ["int64", "float64"]:
                    ref_mean = self.reference_data[column].mean()
                    current_mean = current_data[column].mean()
                    drift_score = abs(ref_mean - current_mean) / (ref_mean + 1e-8)
                    drift_metrics[column] = {
                        "drift_score": drift_score,
                        "is_drift": drift_score > threshold,
                        "reference_mean": ref_mean,
                        "current_mean": current_mean,
                    }
                # For categorical columns
                else:
                    ref_dist = self.reference_data[column].value_counts(normalize=True)
                    current_dist = current_data[column].value_counts(normalize=True)

                    # Calculate KL divergence approximation
                    common_values = set(ref_dist.index) & set(current_dist.index)
                    if len(common_values) > 0:
                        kl_div = 0
                        for val in common_values:
                            p = ref_dist.get(val, 1e-8)
                            q = current_dist.get(val, 1e-8)
                            kl_div += p * np.log(p / q)

                        drift_metrics[column] = {
                            "drift_score": kl_div,
                            "is_drift": kl_div > threshold,
                            "reference_distribution": ref_dist.to_dict(),
                            "current_distribution": current_dist.to_dict(),
                        }

        return drift_metrics

    def calculate_model_performance(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate model performance metrics.

        Args:
            current_data: Current dataset with predictions and actual values

        Returns:
            Dictionary with performance metrics
        """
        if self.target_column not in current_data.columns:
            logger.warning(
                f"Target column '{self.target_column}' not found in current data"
            )
            return {}

        if self.prediction_column not in current_data.columns:
            logger.warning(
                f"Prediction column '{self.prediction_column}' not found in "
                f"current data"
            )
            return {}

        y_true = current_data[self.target_column]
        y_pred = current_data[self.prediction_column]

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        # Store last performance for alerts
        self.last_performance = accuracy

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def generate_monitoring_report(
        self, current_data: pd.DataFrame, output_dir: str = "monitoring_reports"
    ) -> dict[str, Any]:
        """Generate comprehensive monitoring report.

        Args:
            current_data: Current dataset
            output_dir: Directory to save reports

        Returns:
            Dictionary with all monitoring metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate metrics
        drift_metrics = self.calculate_data_drift(current_data)
        performance_metrics = self.calculate_model_performance(current_data)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compile report
        report = {
            "timestamp": timestamp,
            "data_drift": drift_metrics,
            "model_performance": performance_metrics,
            "summary": {
                "total_features": len(drift_metrics),
                "features_with_drift": sum(
                    1 for m in drift_metrics.values() if m.get("is_drift", False)
                ),
                "accuracy": performance_metrics.get("accuracy", 0),
                "data_quality_score": self._calculate_data_quality_score(current_data),
            },
        }

        # Save report
        report_path = output_path / f"monitoring_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate visualizations
        self._generate_visualizations(
            current_data, drift_metrics, performance_metrics, output_path, timestamp
        )

        logger.info(f"Monitoring report saved to {report_path}")
        return report

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple data quality score.

        Args:
            data: Dataset to analyze

        Returns:
            Data quality score between 0 and 1
        """
        if data.empty:
            return 0.0

        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])

        # Check for duplicate rows
        duplicate_ratio = data.duplicated().sum() / len(data)

        # Simple quality score
        quality_score = 1.0 - (missing_ratio + duplicate_ratio)
        return float(max(0.0, min(1.0, quality_score)))

    def _generate_visualizations(
        self,
        current_data: pd.DataFrame,
        drift_metrics: dict[str, Any],
        performance_metrics: dict[str, Any],
        output_path: Path,
        timestamp: str,
    ) -> None:
        """Generate monitoring visualizations.

        Args:
            current_data: Current dataset
            drift_metrics: Data drift metrics
            performance_metrics: Model performance metrics
            output_path: Output directory
            timestamp: Timestamp for file naming
        """
        try:
            # Drift scores plot
            if drift_metrics:
                drift_scores = [m.get("drift_score", 0) for m in drift_metrics.values()]
                feature_names = list(drift_metrics.keys())

                plt.figure(figsize=(10, 6))
                plt.bar(feature_names, drift_scores)
                plt.title("Feature Drift Scores")
                plt.xlabel("Features")
                plt.ylabel("Drift Score")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path / f"drift_scores_{timestamp}.png")
                plt.close()

            # Confusion matrix
            if "confusion_matrix" in performance_metrics:
                cm = np.array(performance_metrics["confusion_matrix"])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.ylabel("Actual")
                plt.xlabel("Predicted")
                plt.tight_layout()
                plt.savefig(output_path / f"confusion_matrix_{timestamp}.png")
                plt.close()

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    def check_for_alerts(
        self, drift_metrics: dict[str, Any], performance_threshold: float = 0.4
    ) -> list[str]:
        """Check for monitoring alerts.

        Args:
            drift_metrics: Data drift metrics
            performance_threshold: Minimum acceptable performance

        Returns:
            List of alert messages
        """
        alerts = []

        # Check for drift alerts
        drifted_features = [
            name
            for name, metrics in drift_metrics.items()
            if metrics.get("is_drift", False)
        ]

        if drifted_features:
            alerts.append(
                f"Data drift detected in features: {', '.join(drifted_features)}"
            )

        # Check for performance alerts
        if hasattr(self, "last_performance"):
            if self.last_performance and self.last_performance < performance_threshold:
                alerts.append(
                    f"Model performance below threshold: {self.last_performance:.3f}"
                )

        return alerts


def create_model_monitor(
    reference_data_path: str,
    target_column: str = "result",
    prediction_column: str = "predicted_result",
) -> SimpleModelMonitor:
    """Create a model monitor instance.

    Args:
        reference_data_path: Path to reference dataset
        target_column: Name of the target column
        prediction_column: Name of the prediction column

    Returns:
        SimpleModelMonitor instance
    """
    reference_data = pd.read_csv(reference_data_path)
    return SimpleModelMonitor(reference_data, target_column, prediction_column)
