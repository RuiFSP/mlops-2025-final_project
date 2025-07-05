"""Model monitoring for Premier League match prediction.

This module provides simplified monitoring functionality due to Evidently API changes.
For production use, consider updating to the latest Evidently API or using the
simplified monitor in model_monitor_simple.py.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

# Import the simplified monitor
from .model_monitor_simple import SimpleModelMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Model monitoring class - currently using simplified approach."""

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

        # Use simplified monitor for now
        self.simple_monitor = SimpleModelMonitor(
            reference_data, target_column, prediction_column
        )

    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        output_dir: str = "monitoring_reports",
    ) -> Dict[str, Any]:
        """Generate data drift report using simplified approach.

        Args:
            current_data: Current dataset
            reference_data: Reference dataset
            output_dir: Directory to save reports

        Returns:
            Dictionary with drift metrics
        """
        logger.info("Generating data drift report using simplified approach")
        return self.simple_monitor.calculate_data_drift(current_data)

    def generate_target_drift_report(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        output_dir: str = "monitoring_reports",
    ) -> Dict[str, Any]:
        """Generate target drift report using simplified approach.

        Args:
            current_data: Current dataset
            reference_data: Reference dataset
            output_dir: Directory to save reports

        Returns:
            Dictionary with target drift metrics
        """
        logger.info("Generating target drift report using simplified approach")

        if (
            self.target_column not in current_data.columns
            or self.target_column not in reference_data.columns
        ):
            logger.warning(f"Target column '{self.target_column}' not found in data")
            return {}

        # Compare target distributions
        ref_dist = reference_data[self.target_column].value_counts(normalize=True)
        current_dist = current_data[self.target_column].value_counts(normalize=True)

        return {
            "target_drift": {
                "reference_distribution": ref_dist.to_dict(),
                "current_distribution": current_dist.to_dict(),
                "drift_detected": not ref_dist.equals(current_dist),
            }
        }

    def generate_data_quality_report(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        output_dir: str = "monitoring_reports",
    ) -> Dict[str, Any]:
        """Generate data quality report using simplified approach.

        Args:
            current_data: Current dataset
            reference_data: Reference dataset
            output_dir: Directory to save reports

        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Generating data quality report using simplified approach")

        quality_score = self.simple_monitor._calculate_data_quality_score(current_data)

        return {
            "data_quality": {
                "quality_score": quality_score,
                "missing_values": current_data.isnull().sum().to_dict(),
                "duplicate_rows": current_data.duplicated().sum(),
                "total_rows": len(current_data),
            }
        }

    def generate_model_performance_report(
        self,
        predictions: pd.DataFrame,
        actual_values: pd.DataFrame,
        output_dir: str = "monitoring_reports",
    ) -> Dict[str, Any]:
        """Generate model performance report using simplified approach.

        Args:
            predictions: Model predictions
            actual_values: Actual values
            output_dir: Directory to save reports

        Returns:
            Dictionary with performance metrics
        """
        logger.info("Generating model performance report using simplified approach")

        # Combine predictions and actual values
        combined_data = pd.DataFrame(
            {
                self.prediction_column: (
                    predictions.iloc[:, 0] if not predictions.empty else []
                ),
                self.target_column: (
                    actual_values.iloc[:, 0] if not actual_values.empty else []
                ),
            }
        )

        return self.simple_monitor.calculate_model_performance(combined_data)

    def generate_comprehensive_report(
        self,
        current_data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        actual_values: Optional[pd.DataFrame] = None,
        output_dir: str = "monitoring_reports",
    ) -> Dict[str, Any]:
        """Generate comprehensive monitoring report.

        Args:
            current_data: Current dataset
            predictions: Model predictions (optional)
            actual_values: Actual values (optional)
            output_dir: Directory to save reports

        Returns:
            Dictionary with all monitoring metrics
        """
        logger.info("Generating comprehensive monitoring report")

        # Add predictions and actual values to current data if provided
        if predictions is not None and actual_values is not None:
            current_data = current_data.copy()
            current_data[self.prediction_column] = (
                predictions.iloc[:, 0] if not predictions.empty else None
            )
            current_data[self.target_column] = (
                actual_values.iloc[:, 0] if not actual_values.empty else None
            )

        return self.simple_monitor.generate_monitoring_report(current_data, output_dir)

    def check_for_alerts(
        self, drift_metrics: Dict[str, Any], threshold: float = 0.5
    ) -> List[str]:
        """Check for monitoring alerts.

        Args:
            drift_metrics: Data drift metrics
            threshold: Threshold for alerts

        Returns:
            List of alert messages
        """
        return self.simple_monitor.check_for_alerts(drift_metrics, threshold)


def create_model_monitor(
    reference_data_path: str,
    target_column: str = "result",
    prediction_column: str = "predicted_result",
) -> ModelMonitor:
    """Create a model monitor instance.

    Args:
        reference_data_path: Path to reference dataset
        target_column: Name of the target column
        prediction_column: Name of the prediction column

    Returns:
        ModelMonitor instance
    """
    reference_data = pd.read_csv(reference_data_path)
    return ModelMonitor(reference_data, target_column, prediction_column)
