"""
Data drift detection for the Premier League prediction model.
Uses statistical tests for basic drift analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ModelDriftDetector:
    """Detect data drift in model inputs and outputs using statistical tests."""

    def __init__(
        self,
        reference_data_path: str,
        model_path: str,
        drift_threshold: float = 0.05,  # p-value threshold for statistical tests
        output_dir: str = "evidently_reports",
    ):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference/training data
            model_path: Path to trained model
            drift_threshold: P-value threshold for drift detection (0.0-1.0)
            output_dir: Directory to save drift reports
        """
        self.reference_data_path = Path(reference_data_path)
        self.model_path = Path(model_path)
        self.drift_threshold = drift_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load reference data and model
        self.reference_data = self._load_reference_data()
        self.model = self._load_model()

        # Define numerical and categorical features
        self.numerical_features = [
            "month",
            "home_odds",
            "draw_odds",
            "away_odds",
            "home_prob_margin_adj",
            "draw_prob_margin_adj",
            "away_prob_margin_adj",
        ]
        self.categorical_features = ["home_team", "away_team"]

    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference data for drift comparison."""
        if self.reference_data_path.suffix == ".parquet":
            return pd.read_parquet(self.reference_data_path)
        elif self.reference_data_path.suffix == ".csv":
            return pd.read_csv(self.reference_data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.reference_data_path.suffix}")

    def _load_model(self) -> Any:
        """Load the trained model."""
        if self.model_path.exists():
            return joblib.load(self.model_path)
        else:
            logger.warning(f"Model not found at {self.model_path}")
            return None

    def detect_drift(self, current_data: pd.DataFrame, save_report: bool = True) -> dict[str, Any]:
        """
        Detect drift between reference and current data using statistical tests.

        Args:
            current_data: Current production data
            save_report: Whether to save detailed drift report

        Returns:
            Dictionary with drift detection results
        """
        try:
            # Ensure current data has predictions if model is available
            if self.model is not None and "prediction" not in current_data.columns:
                current_data = self._add_predictions(current_data)

            # Perform drift tests
            drift_results = self._perform_drift_tests(current_data)

            # Generate summary
            results = self._summarize_drift_results(drift_results)

            # Save detailed report if requested
            if save_report:
                report_path = self._save_drift_report(drift_results)
                results["report_path"] = str(report_path)

            logger.info(
                f"Drift detection completed. " f"Dataset drift detected: {results['dataset_drift']}"
            )
            return results

        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            return {"error": str(e), "dataset_drift": False}

    def _add_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add model predictions to data."""
        try:
            # Prepare features for prediction
            features = self._prepare_features(data)
            predictions = self.model.predict(features)

            # Add predictions to data
            data_with_pred = data.copy()
            data_with_pred["prediction"] = predictions
            return data_with_pred

        except Exception as e:
            logger.error(f"Error adding predictions: {str(e)}")
            return data

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction."""
        required_columns = [
            "home_team",
            "away_team",
            "month",
            "home_odds",
            "draw_odds",
            "away_odds",
            "home_prob_margin_adj",
            "draw_prob_margin_adj",
            "away_prob_margin_adj",
        ]

        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns for prediction: {missing_cols}")

        # Return available features
        available_features = [col for col in required_columns if col in data.columns]
        return data[available_features]

    def _perform_drift_tests(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Perform statistical tests for drift detection."""
        drift_results = {}

        # Test numerical features
        for feature in self.numerical_features:
            if feature in self.reference_data.columns and feature in current_data.columns:
                try:
                    ref_values = self.reference_data[feature].dropna()
                    curr_values = current_data[feature].dropna()

                    if len(ref_values) > 0 and len(curr_values) > 0:
                        # Use Kolmogorov-Smirnov test for numerical features
                        ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)

                        drift_results[feature] = {
                            "test": "kolmogorov_smirnov",
                            "statistic": float(ks_stat),
                            "p_value": float(p_value),
                            "drift_detected": p_value < self.drift_threshold,
                            "feature_type": "numerical",
                        }
                except Exception as e:
                    logger.warning(f"Error testing feature {feature}: {e}")
                    drift_results[feature] = {"error": str(e)}

        # Test categorical features
        for feature in self.categorical_features:
            if feature in self.reference_data.columns and feature in current_data.columns:
                try:
                    ref_dist = self.reference_data[feature].value_counts(normalize=True)
                    curr_dist = current_data[feature].value_counts(normalize=True)

                    # Align distributions (fill missing categories with 0)
                    all_categories = set(ref_dist.index) | set(curr_dist.index)
                    ref_aligned = pd.Series([ref_dist.get(cat, 0) for cat in all_categories])
                    curr_aligned = pd.Series([curr_dist.get(cat, 0) for cat in all_categories])

                    if len(all_categories) > 1:
                        # Use Chi-square test for categorical features
                        chi2_stat, p_value = stats.chisquare(curr_aligned, ref_aligned)

                        drift_results[feature] = {
                            "test": "chi_square",
                            "statistic": float(chi2_stat),
                            "p_value": float(p_value),
                            "drift_detected": p_value < self.drift_threshold,
                            "feature_type": "categorical",
                        }
                except Exception as e:
                    logger.warning(f"Error testing feature {feature}: {e}")
                    drift_results[feature] = {"error": str(e)}

        return drift_results

    def _summarize_drift_results(self, drift_results: dict[str, Any]) -> dict[str, Any]:
        """Summarize drift test results."""
        try:
            # Count features with drift
            drifted_features = []
            total_features = 0

            for feature, result in drift_results.items():
                if "error" not in result:
                    total_features += 1
                    if result.get("drift_detected", False):
                        drifted_features.append(feature)

            drift_share = len(drifted_features) / total_features if total_features > 0 else 0
            dataset_drift = drift_share > 0.1  # More than 10% of features drifted

            return {
                "timestamp": datetime.now().isoformat(),
                "dataset_drift": dataset_drift,
                "drift_share": drift_share,
                "drifted_features": drifted_features,
                "total_features_tested": total_features,
                "drift_threshold": self.drift_threshold,
                "alert_triggered": dataset_drift,
                "feature_drift_details": drift_results,
            }

        except Exception as e:
            logger.error(f"Error summarizing drift results: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "dataset_drift": False,
                "error": str(e),
            }

    def _save_drift_report(self, drift_results: dict[str, Any]) -> Path:
        """Save drift report as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"drift_report_{timestamp}.json"

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "drift_threshold": self.drift_threshold,
            "feature_results": drift_results,
            "summary": self._summarize_drift_results(drift_results),
        }

        with open(report_path, "w") as f:
            import json

            json.dump(report_data, f, indent=2)

        logger.info(f"Drift report saved to {report_path}")
        return report_path

    def monitor_batch(
        self, batch_data: pd.DataFrame, batch_id: str | None = None
    ) -> dict[str, Any]:
        """
        Monitor a batch of new data for drift.

        Args:
            batch_data: New batch of data to check
            batch_id: Optional batch identifier

        Returns:
            Monitoring results with drift status and metrics
        """
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting drift monitoring for batch {batch_id}")

        # Detect drift
        drift_results = self.detect_drift(batch_data, save_report=True)
        drift_results["batch_id"] = batch_id
        drift_results["batch_size"] = len(batch_data)

        # Log results
        if drift_results.get("alert_triggered", False):
            logger.warning(
                f"DRIFT ALERT for batch {batch_id}: "
                f"{drift_results['drift_share']:.2%} features drifted "
                f"(threshold: 10% features)"
            )
        else:
            logger.info(f"No significant drift detected for batch {batch_id}")

        return drift_results
