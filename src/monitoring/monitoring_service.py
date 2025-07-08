"""
Unified monitoring service that combines drift detection and performance monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .drift_detector import ModelDriftDetector
from .performance_monitor import ModelPerformanceMonitor

logger = logging.getLogger(__name__)


class MLOpsMonitoringService:
    """
    Unified monitoring service for the Premier League prediction model.
    Combines drift detection and performance monitoring.
    """

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        reference_data_path: str = "data/real_data/premier_league_matches.parquet",
        output_dir: str = "evaluation_reports",
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.05,
    ):
        """
        Initialize monitoring service.

        Args:
            model_path: Path to trained model
            reference_data_path: Path to reference/training data
            output_dir: Directory for monitoring outputs
            drift_threshold: Threshold for drift alerts
            performance_threshold: Threshold for performance degradation alerts
        """
        self.model_path = model_path
        self.reference_data_path = reference_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize monitoring components
        self.drift_detector = ModelDriftDetector(
            reference_data_path=reference_data_path,
            model_path=model_path,
            drift_threshold=drift_threshold,
            output_dir=str(self.output_dir),
        )

        self.performance_monitor = ModelPerformanceMonitor(
            model_path=model_path,
            performance_threshold=performance_threshold,
            output_dir=str(self.output_dir),
        )

        # Monitoring summary file
        self.monitoring_summary_file = self.output_dir / "monitoring_summary.json"

    def monitor_production_batch(
        self,
        production_data: pd.DataFrame,
        true_labels: Union[pd.Series, None] = None,
        batch_id: Union[str, None] = None,
    ) -> Union[dict, Any]:
        """
        Comprehensive monitoring of a production data batch.

        Args:
            production_data: New production data
            true_labels: True labels (if available for performance evaluation)
            batch_id: Batch identifier

        Returns:
            Complete monitoring results
        """
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting comprehensive monitoring for batch {batch_id}")

        monitoring_results = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(production_data),
            "monitoring_type": "comprehensive",
        }

        # 1. Drift Detection
        logger.info("Running drift detection...")
        try:
            drift_results = self.drift_detector.monitor_batch(
                production_data, batch_id=batch_id
            )
            monitoring_results["drift_monitoring"] = drift_results
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            monitoring_results["drift_monitoring"] = {"error": str(e)}

        # 2. Performance Monitoring (if labels are available)
        if true_labels is not None:
            logger.info("Running performance evaluation...")
            try:
                # Prepare features for performance evaluation
                features = self._prepare_features_for_evaluation(production_data)

                performance_results = self.performance_monitor.evaluate_batch(
                    X=features, y_true=true_labels, batch_id=batch_id
                )
                monitoring_results["performance_monitoring"] = performance_results
            except Exception as e:
                logger.error(f"Performance monitoring failed: {e}")
                monitoring_results["performance_monitoring"] = {"error": str(e)}
        else:
            logger.info("No true labels provided, skipping performance evaluation")
            monitoring_results["performance_monitoring"] = {
                "message": "No labels provided for performance evaluation"
            }

        # 3. Generate alerts and recommendations
        alerts_and_recommendations = self._generate_alerts_and_recommendations(
            monitoring_results
        )
        monitoring_results.update(alerts_and_recommendations)

        # 4. Save monitoring summary
        self._save_monitoring_summary(monitoring_results)

        logger.info(f"Monitoring completed for batch {batch_id}")
        return monitoring_results

    def _prepare_features_for_evaluation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model evaluation."""
        # This should match your model's expected features
        required_features = [
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

        # Check for missing features
        missing_features = [col for col in required_features if col not in data.columns]
        if missing_features:
            logger.warning(f"Missing features for evaluation: {missing_features}")

        # Return available features
        available_features = [col for col in required_features if col in data.columns]
        return data[available_features]

    def _generate_alerts_and_recommendations(
        self, monitoring_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate alerts and recommendations based on monitoring results."""
        alerts = []
        recommendations = []
        overall_status = "healthy"

        # Check drift monitoring results
        drift_monitoring = monitoring_results.get("drift_monitoring", {})
        if drift_monitoring.get("alert_triggered", False):
            alerts.append(
                {
                    "type": "data_drift",
                    "severity": "high",
                    "message": (
                        f"Data drift detected: "
                        f"{drift_monitoring.get('drift_share', 0):.2%} "
                        f"of features have drifted"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            recommendations.append(
                "Investigate data drift causes and consider model retraining"
            )
            overall_status = "warning"

        # Check performance monitoring results
        performance_monitoring = monitoring_results.get("performance_monitoring", {})
        if performance_monitoring.get("performance_degraded", False):
            alerts.append(
                {
                    "type": "performance_degradation",
                    "severity": "high",
                    "message": (
                        f"Model performance degraded: "
                        f"accuracy {performance_monitoring.get('accuracy', 0):.3f}"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            recommendations.append(
                "Model performance below threshold - immediate retraining recommended"
            )
            overall_status = "critical"

        # Check for errors
        if drift_monitoring.get("error") or performance_monitoring.get("error"):
            alerts.append(
                {
                    "type": "monitoring_error",
                    "severity": "medium",
                    "message": "Errors occurred during monitoring",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            recommendations.append("Check monitoring system configuration and logs")
            if overall_status == "healthy":
                overall_status = "warning"

        return {
            "alerts": alerts,
            "recommendations": recommendations,
            "overall_status": overall_status,
            "alert_count": len(alerts),
        }

    def _save_monitoring_summary(self, monitoring_results: dict[str, Any]):
        """Save monitoring summary to file."""
        try:
            # Load existing summaries
            summaries = []
            if self.monitoring_summary_file.exists():
                with open(self.monitoring_summary_file) as f:
                    summaries = json.load(f)

            # Add new summary
            summaries.append(monitoring_results)

            # Keep only last 100 summaries to prevent file from growing too large
            summaries = summaries[-100:]

            # Save updated summaries
            with open(self.monitoring_summary_file, "w") as f:
                json.dump(summaries, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save monitoring summary: {e}")

    def get_monitoring_dashboard_data(self, days: int = 30) -> dict[str, Any]:
        """
        Get data for monitoring dashboard.

        Args:
            days: Number of days to include in dashboard

        Returns:
            Dashboard data structure
        """
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "model_status": self._get_model_status(),
        }

        # Get drift monitoring history
        try:
            dashboard_data["drift_monitoring"] = self._get_drift_history(days)
        except Exception as e:
            logger.error(f"Failed to get drift history: {e}")
            dashboard_data["drift_monitoring"] = {"error": str(e)}

        # Get performance monitoring history
        try:
            dashboard_data[
                "performance_monitoring"
            ] = self.performance_monitor.get_performance_summary(days)
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            dashboard_data["performance_monitoring"] = {"error": str(e)}

        # Get recent alerts
        try:
            dashboard_data["recent_alerts"] = self._get_recent_alerts(days)
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            dashboard_data["recent_alerts"] = {"error": str(e)}

        return dashboard_data

    def _get_model_status(self) -> dict[str, Any]:
        """Get current model status."""
        try:
            health_check = self.performance_monitor.check_model_health()
            return {
                "model_loaded": health_check["model_loaded"],
                "health_status": health_check.get("status", "unknown"),
                "health_score": health_check.get("health_score", 0),
                "last_check": health_check["timestamp"],
            }
        except Exception as e:
            return {"error": str(e), "model_loaded": False}

    def _get_drift_history(self, days: int) -> dict[str, Any]:
        """Get drift detection history."""
        # This would typically read from a drift history file
        # For now, return placeholder structure
        return {
            "total_checks": 0,
            "drift_detected_count": 0,
            "latest_drift_check": None,
            "drift_rate": 0.0,
        }

    def _get_recent_alerts(self, days: int) -> list[dict[str, Any]]:
        """Get recent alerts from monitoring summaries."""
        alerts = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            if self.monitoring_summary_file.exists():
                with open(self.monitoring_summary_file) as f:
                    summaries = json.load(f)

                for summary in summaries:
                    summary_date = datetime.fromisoformat(summary["timestamp"])
                    if summary_date >= cutoff_date:
                        alerts.extend(summary.get("alerts", []))

        except Exception as e:
            logger.error(f"Failed to load recent alerts: {e}")

        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return alerts

    def run_health_check(self) -> dict[str, Any]:
        """Run comprehensive health check of the monitoring system."""
        health_check = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_service_status": "healthy",
            "components": {},
        }

        # Check drift detector
        try:
            drift_check = {
                "status": "healthy"
                if self.drift_detector.model is not None
                else "warning",
                "reference_data_loaded": self.drift_detector.reference_data is not None,
                "model_loaded": self.drift_detector.model is not None,
            }
            health_check["components"]["drift_detector"] = drift_check
        except Exception as e:
            health_check["components"]["drift_detector"] = {
                "status": "error",
                "error": str(e),
            }

        # Check performance monitor
        try:
            perf_check = self.performance_monitor.check_model_health()
            health_check["components"]["performance_monitor"] = {
                "status": perf_check.get("status", "unknown"),
                "model_loaded": perf_check["model_loaded"],
                "baseline_accuracy": perf_check["baseline_accuracy"],
            }
        except Exception as e:
            health_check["components"]["performance_monitor"] = {
                "status": "error",
                "error": str(e),
            }

        # Determine overall status
        component_statuses = [
            comp.get("status", "error") for comp in health_check["components"].values()
        ]
        if "error" in component_statuses:
            health_check["monitoring_service_status"] = "error"
        elif "warning" in component_statuses:
            health_check["monitoring_service_status"] = "warning"

        return health_check
