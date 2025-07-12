"""
Prefect tasks for MLOps operations.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from prefect import task

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_integration.real_data_fetcher import RealDataFetcher
from monitoring.metrics_storage import MetricsStorage
from pipelines.prediction_pipeline import PredictionPipeline
from pipelines.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@task(name="check_model_performance", tags=["monitoring", "performance"])
def check_model_performance(
    model_name: str = "premier_league_predictor",
    threshold_accuracy: float = 0.55,
    threshold_f1: float = 0.50,
) -> dict[str, Any]:
    """
    Check current model performance against thresholds.

    Args:
        model_name: Name of the model to check
        threshold_accuracy: Minimum accuracy threshold
        threshold_f1: Minimum F1 score threshold

    Returns:
        Dictionary with performance metrics and retraining recommendation
    """
    logger.info(f"🔍 Checking performance for model: {model_name}")

    try:
        # Initialize components
        pipeline = PredictionPipeline()
        metrics_storage = MetricsStorage()

        # Get recent predictions and actual results
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        recent_metrics = metrics_storage.get_metrics_by_date_range(
            start_date=start_date,
            end_date=end_date,
            metric_types=["accuracy", "f1_score", "precision", "recall"],
        )

        if not recent_metrics:
            logger.warning("No recent metrics found, using model validation metrics")
            # Fallback to model validation metrics
            model_info = pipeline.get_model_info()
            current_accuracy = model_info.get("accuracy", 0.0)
            current_f1 = model_info.get("f1_score", 0.0)
        else:
            # Calculate average metrics from recent data
            accuracy_metrics = [m for m in recent_metrics if m["metric_type"] == "accuracy"]
            f1_metrics = [m for m in recent_metrics if m["metric_type"] == "f1_score"]

            current_accuracy = (
                sum(m["value"] for m in accuracy_metrics) / len(accuracy_metrics)
                if accuracy_metrics
                else 0.0
            )
            current_f1 = (
                sum(m["value"] for m in f1_metrics) / len(f1_metrics) if f1_metrics else 0.0
            )

        # Check if retraining is needed
        needs_retraining = current_accuracy < threshold_accuracy or current_f1 < threshold_f1

        performance_data = {
            "model_name": model_name,
            "current_accuracy": current_accuracy,
            "current_f1": current_f1,
            "threshold_accuracy": threshold_accuracy,
            "threshold_f1": threshold_f1,
            "needs_retraining": needs_retraining,
            "check_timestamp": datetime.now().isoformat(),
            "metrics_period_days": 30,
            "total_recent_predictions": len(recent_metrics),
        }

        logger.info(
            f"📊 Model performance check completed: accuracy={current_accuracy:.3f}, f1={current_f1:.3f}, needs_retraining={needs_retraining}"
        )

        return performance_data

    except Exception as e:
        logger.error(f"❌ Error checking model performance: {e}")
        return {
            "error": str(e),
            "needs_retraining": False,
            "check_timestamp": datetime.now().isoformat(),
        }


@task(name="analyze_model_drift", tags=["monitoring", "drift"])
def analyze_model_drift(
    reference_period_days: int = 90,
    current_period_days: int = 30,
) -> dict[str, Any]:
    """
    Analyze model drift using basic statistical methods.

    Args:
        reference_period_days: Days to look back for reference data
        current_period_days: Days to look back for current data

    Returns:
        Dictionary with drift analysis results
    """
    logger.info("🔍 Analyzing model drift")

    try:
        # Initialize components
        pipeline = PredictionPipeline()
        metrics_storage = MetricsStorage()

        # Get reference and current data
        end_date = datetime.now()
        current_start = end_date - timedelta(days=current_period_days)
        reference_start = end_date - timedelta(days=reference_period_days)
        reference_end = end_date - timedelta(days=current_period_days)

        # Get predictions data for both periods
        reference_data = metrics_storage.get_predictions_by_date_range(
            start_date=reference_start, end_date=reference_end
        )

        current_data = metrics_storage.get_predictions_by_date_range(
            start_date=current_start, end_date=end_date
        )

        if not reference_data or not current_data:
            logger.warning("Insufficient data for drift analysis")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "error": "Insufficient data for drift analysis",
                "analysis_timestamp": datetime.now().isoformat(),
            }

        # Simple drift analysis - compare prediction distributions
        reference_predictions = [d.get("prediction", "D") for d in reference_data]
        current_predictions = [d.get("prediction", "D") for d in current_data]

        # Calculate distribution differences
        ref_dist = {
            "H": reference_predictions.count("H") / len(reference_predictions),
            "D": reference_predictions.count("D") / len(reference_predictions),
            "A": reference_predictions.count("A") / len(reference_predictions),
        }

        cur_dist = {
            "H": current_predictions.count("H") / len(current_predictions),
            "D": current_predictions.count("D") / len(current_predictions),
            "A": current_predictions.count("A") / len(current_predictions),
        }

        # Calculate drift score using KL divergence approximation
        drift_score = 0.0
        for outcome in ["H", "D", "A"]:
            if cur_dist[outcome] > 0 and ref_dist[outcome] > 0:
                drift_score += (
                    cur_dist[outcome] * ((cur_dist[outcome] / ref_dist[outcome]) ** 0.5 - 1) ** 2
                )

        # Simple threshold for drift detection
        drift_detected = drift_score > 0.1

        drift_analysis = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "reference_distribution": ref_dist,
            "current_distribution": cur_dist,
            "analysis_timestamp": datetime.now().isoformat(),
            "reference_period_days": reference_period_days,
            "current_period_days": current_period_days,
            "reference_data_size": len(reference_data),
            "current_data_size": len(current_data),
        }

        # Save analysis for monitoring
        analysis_path = Path("data/reports/drift_analysis.json")
        analysis_path.parent.mkdir(parents=True, exist_ok=True)

        with open(analysis_path, "w") as f:
            json.dump(drift_analysis, f, indent=2, default=str)

        logger.info(
            f"📈 Drift analysis completed: drift_detected={drift_detected}, drift_score={drift_score:.3f}"
        )

        return drift_analysis

    except Exception as e:
        logger.error(f"❌ Error analyzing model drift: {e}")
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "error": str(e),
            "analysis_timestamp": datetime.now().isoformat(),
        }


@task(name="retrain_model", tags=["training", "model"])
def retrain_model(
    reason: str = "scheduled_retraining",
    performance_data: dict[str, Any] = None,
    drift_data: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Retrain the model with fresh data.

    Args:
        reason: Reason for retraining
        performance_data: Performance check results
        drift_data: Drift analysis results

    Returns:
        Dictionary with retraining results
    """
    logger.info(f"🔄 Starting model retraining: {reason}")

    try:
        # For demonstration, we'll skip actual retraining unless specifically requested
        if reason == "test_run":
            logger.info("⚠️  Test run - skipping actual retraining")
            return {
                "reason": reason,
                "skipped": True,
                "message": "Test run - actual retraining skipped",
                "retrain_timestamp": datetime.now().isoformat(),
            }

        # Initialize training pipeline
        training_pipeline = TrainingPipeline()

        # Run training pipeline
        training_results = training_pipeline.run()

        # Store retraining metadata
        retraining_metadata = {
            "reason": reason,
            "training_results": training_results,
            "performance_data": performance_data or {},
            "drift_data": drift_data or {},
            "retrain_timestamp": datetime.now().isoformat(),
            "success": True,
        }

        # Save metadata
        metadata_path = Path("data/retraining/metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(retraining_metadata, f, indent=2, default=str)

        logger.info(
            f"✅ Model retraining completed successfully. New accuracy: {training_results.get('accuracy', 'N/A')}"
        )

        return retraining_metadata

    except Exception as e:
        logger.error(f"❌ Error during model retraining: {e}")
        return {
            "reason": reason,
            "error": str(e),
            "success": False,
            "retrain_timestamp": datetime.now().isoformat(),
        }


@task(name="generate_predictions", tags=["prediction", "daily"])
def generate_predictions(days_ahead: int = 7) -> dict[str, Any]:
    """
    Generate predictions for upcoming matches.

    Args:
        days_ahead: Number of days to look ahead for matches

    Returns:
        Dictionary with prediction results
    """
    logger.info(f"🔮 Generating predictions for next {days_ahead} days")

    try:
        # Initialize components
        pipeline = PredictionPipeline()
        data_fetcher = RealDataFetcher()

        # Get upcoming matches
        upcoming_matches = data_fetcher.get_upcoming_matches(days_ahead=days_ahead)

        if not upcoming_matches:
            logger.warning("No upcoming matches found")
            return {
                "predictions_generated": 0,
                "matches_found": 0,
                "generation_timestamp": datetime.now().isoformat(),
            }

        # Generate predictions
        predictions = []
        for match in upcoming_matches:
            try:
                prediction = pipeline.predict_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    home_odds=match.get("home_odds"),
                    away_odds=match.get("away_odds"),
                    draw_odds=match.get("draw_odds"),
                )
                predictions.append(
                    {
                        "match": match,
                        "prediction": prediction,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error predicting match {match['home_team']} vs {match['away_team']}: {e}"
                )
                continue

        # Save predictions
        predictions_path = Path("data/predictions/daily_predictions.json")
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        with open(predictions_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "generation_timestamp": datetime.now().isoformat(),
                    "days_ahead": days_ahead,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(
            f"✅ Generated {len(predictions)} predictions for {len(upcoming_matches)} matches"
        )

        return {
            "predictions_generated": len(predictions),
            "matches_found": len(upcoming_matches),
            "generation_timestamp": datetime.now().isoformat(),
            "predictions": predictions,
        }

    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}")
        return {
            "error": str(e),
            "predictions_generated": 0,
            "generation_timestamp": datetime.now().isoformat(),
        }


@task(name="send_alerts", tags=["monitoring", "alerts"])
def send_alerts(
    alert_type: str,
    message: str,
    severity: str = "info",
    data: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Send alerts for monitoring events.

    Args:
        alert_type: Type of alert (performance, drift, error)
        message: Alert message
        severity: Alert severity (info, warning, error, critical)
        data: Additional alert data

    Returns:
        Dictionary with alert results
    """
    logger.info(f"🚨 Sending alert: {alert_type} - {severity}")

    try:
        # Create alert data
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
        }

        # Save alert to file (in production, this would send to monitoring system)
        alerts_path = Path("data/alerts/alerts.json")
        alerts_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing alerts
        existing_alerts = []
        if alerts_path.exists():
            with open(alerts_path) as f:
                existing_alerts = json.load(f)

        # Add new alert
        existing_alerts.append(alert_data)

        # Keep only last 100 alerts
        if len(existing_alerts) > 100:
            existing_alerts = existing_alerts[-100:]

        # Save alerts
        with open(alerts_path, "w") as f:
            json.dump(existing_alerts, f, indent=2, default=str)

        logger.info(f"✅ Alert sent successfully: {alert_type}")

        return {
            "alert_sent": True,
            "alert_type": alert_type,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error sending alert: {e}")
        return {
            "alert_sent": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
