"""
Prefect flows for orchestrating MLOps workflows.
"""

import logging
from datetime import datetime
from typing import Any

from prefect import flow, get_run_logger

from .tasks import (
    analyze_model_drift,
    check_model_performance,
    generate_predictions,
    retrain_model,
    send_alerts,
)

logger = logging.getLogger(__name__)


@flow(
    name="retraining_flow",
    description="Automated model retraining based on performance and drift analysis",
)
def retraining_flow(
    force_retrain: bool = False,
    performance_threshold_accuracy: float = 0.55,
    performance_threshold_f1: float = 0.50,
    drift_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Main retraining flow that orchestrates model monitoring and retraining.

    Args:
        force_retrain: Force retraining regardless of performance
        performance_threshold_accuracy: Minimum accuracy threshold
        performance_threshold_f1: Minimum F1 score threshold
        drift_threshold: Drift score threshold for retraining

    Returns:
        Dictionary with flow execution results
    """
    run_logger = get_run_logger()
    run_logger.info("🚀 Starting retraining flow")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_results = {
        "flow_name": "retraining_flow",
        "timestamp": timestamp,
        "force_retrain": force_retrain,
        "steps_completed": [],
        "alerts_sent": [],
    }

    try:
        # Step 1: Check model performance
        run_logger.info("📊 Checking model performance")
        performance_data = check_model_performance(
            threshold_accuracy=performance_threshold_accuracy,
            threshold_f1=performance_threshold_f1,
        )
        flow_results["performance_check"] = performance_data
        flow_results["steps_completed"].append("performance_check")

        # Step 2: Analyze model drift
        run_logger.info("🔍 Analyzing model drift")
        drift_data = analyze_model_drift()
        flow_results["drift_analysis"] = drift_data
        flow_results["steps_completed"].append("drift_analysis")

        # Step 3: Determine if retraining is needed
        needs_retraining = (
            force_retrain
            or performance_data.get("needs_retraining", False)
            or drift_data.get("drift_detected", False)
            or drift_data.get("drift_score", 0) > drift_threshold
        )

        flow_results["needs_retraining"] = needs_retraining

        # Step 4: Send alerts based on findings
        if performance_data.get("needs_retraining", False):
            alert_result = send_alerts(
                alert_type="performance",
                message=f"Model performance below threshold: accuracy={performance_data.get('current_accuracy', 0):.3f}, f1={performance_data.get('current_f1', 0):.3f}",
                severity="warning",
                data=performance_data,
            )
            flow_results["alerts_sent"].append(alert_result)

        if drift_data.get("drift_detected", False):
            alert_result = send_alerts(
                alert_type="drift",
                message=f"Model drift detected: drift_score={drift_data.get('drift_score', 0):.3f}",
                severity="warning",
                data=drift_data,
            )
            flow_results["alerts_sent"].append(alert_result)

        # Step 5: Retrain if needed
        if needs_retraining:
            run_logger.info("🔄 Retraining model")
            reason = "forced" if force_retrain else "performance_or_drift"
            retraining_result = retrain_model(
                reason=reason,
                performance_data=performance_data,
                drift_data=drift_data,
            )
            flow_results["retraining_result"] = retraining_result
            flow_results["steps_completed"].append("retraining")

            # Send success/failure alert
            if retraining_result.get("success", False):
                alert_result = send_alerts(
                    alert_type="retraining",
                    message=f"Model retraining completed successfully. New accuracy: {retraining_result.get('training_results', {}).get('accuracy', 'N/A')}",
                    severity="info",
                    data=retraining_result,
                )
                flow_results["alerts_sent"].append(alert_result)
            else:
                alert_result = send_alerts(
                    alert_type="retraining",
                    message=f"Model retraining failed: {retraining_result.get('error', 'Unknown error')}",
                    severity="error",
                    data=retraining_result,
                )
                flow_results["alerts_sent"].append(alert_result)
        else:
            run_logger.info("✅ No retraining needed")
            flow_results["retraining_result"] = {"message": "No retraining needed", "skipped": True}

        run_logger.info("✅ Retraining flow completed successfully")
        flow_results["success"] = True

        return flow_results

    except Exception as e:
        run_logger.error(f"❌ Error in retraining flow: {e}")

        # Send error alert
        alert_result = send_alerts(
            alert_type="flow_error",
            message=f"Retraining flow failed: {str(e)}",
            severity="critical",
            data={"error": str(e), "timestamp": timestamp},
        )
        flow_results["alerts_sent"].append(alert_result)
        flow_results["success"] = False
        flow_results["error"] = str(e)

        return flow_results


@flow(
    name="monitoring_flow",
    description="Continuous monitoring of model performance and system health",
)
def monitoring_flow(
    check_performance: bool = True,
    check_drift: bool = True,
    generate_daily_predictions: bool = True,
) -> dict[str, Any]:
    """
    Monitoring flow for regular health checks and predictions.

    Args:
        check_performance: Whether to check model performance
        check_drift: Whether to check for model drift
        generate_daily_predictions: Whether to generate daily predictions

    Returns:
        Dictionary with monitoring results
    """
    run_logger = get_run_logger()
    run_logger.info("🔍 Starting monitoring flow")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_results = {
        "flow_name": "monitoring_flow",
        "timestamp": timestamp,
        "checks_performed": [],
        "alerts_sent": [],
    }

    try:
        # Performance monitoring
        if check_performance:
            run_logger.info("📊 Checking model performance")
            performance_data = check_model_performance()
            flow_results["performance_check"] = performance_data
            flow_results["checks_performed"].append("performance")

            # Alert if performance is concerning
            if performance_data.get("needs_retraining", False):
                alert_result = send_alerts(
                    alert_type="performance",
                    message=f"Model performance monitoring alert: accuracy={performance_data.get('current_accuracy', 0):.3f}",
                    severity="info",
                    data=performance_data,
                )
                flow_results["alerts_sent"].append(alert_result)

        # Drift monitoring
        if check_drift:
            run_logger.info("🔍 Checking model drift")
            drift_data = analyze_model_drift()
            flow_results["drift_analysis"] = drift_data
            flow_results["checks_performed"].append("drift")

            # Alert if drift is detected
            if drift_data.get("drift_detected", False):
                alert_result = send_alerts(
                    alert_type="drift",
                    message=f"Model drift monitoring alert: drift_score={drift_data.get('drift_score', 0):.3f}",
                    severity="info",
                    data=drift_data,
                )
                flow_results["alerts_sent"].append(alert_result)

        # Daily predictions
        if generate_daily_predictions:
            run_logger.info("🔮 Generating daily predictions")
            predictions_data = generate_predictions(days_ahead=7)
            flow_results["predictions"] = predictions_data
            flow_results["checks_performed"].append("predictions")

        run_logger.info("✅ Monitoring flow completed successfully")
        flow_results["success"] = True

        return flow_results

    except Exception as e:
        run_logger.error(f"❌ Error in monitoring flow: {e}")

        # Send error alert
        alert_result = send_alerts(
            alert_type="flow_error",
            message=f"Monitoring flow failed: {str(e)}",
            severity="critical",
            data={"error": str(e), "timestamp": timestamp},
        )
        flow_results["alerts_sent"].append(alert_result)
        flow_results["success"] = False
        flow_results["error"] = str(e)

        return flow_results


@flow(
    name="daily_prediction_flow",
    description="Daily prediction generation and betting analysis",
)
def daily_prediction_flow(
    days_ahead: int = 7,
    run_betting_analysis: bool = True,
) -> dict[str, Any]:
    """
    Daily flow for generating predictions and betting analysis.

    Args:
        days_ahead: Number of days to look ahead for matches
        run_betting_analysis: Whether to run betting analysis

    Returns:
        Dictionary with prediction results
    """
    run_logger = get_run_logger()
    run_logger.info("🔮 Starting daily prediction flow")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_results = {
        "flow_name": "daily_prediction_flow",
        "timestamp": timestamp,
        "days_ahead": days_ahead,
    }

    try:
        # Generate predictions
        run_logger.info(f"🔮 Generating predictions for next {days_ahead} days")
        predictions_data = generate_predictions(days_ahead=days_ahead)
        flow_results["predictions"] = predictions_data

        run_logger.info("✅ Daily prediction flow completed successfully")
        flow_results["success"] = True

        return flow_results

    except Exception as e:
        run_logger.error(f"❌ Error in daily prediction flow: {e}")

        # Send error alert
        alert_result = send_alerts(
            alert_type="flow_error",
            message=f"Daily prediction flow failed: {str(e)}",
            severity="error",
            data={"error": str(e), "timestamp": timestamp},
        )
        flow_results["alerts_sent"] = [alert_result]
        flow_results["success"] = False
        flow_results["error"] = str(e)

        return flow_results
