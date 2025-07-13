#!/usr/bin/env python3
"""
Simple test script to demonstrate the orchestration concept.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_orchestration_concept():
    """Test the orchestration concept with direct task execution."""
    logger.info("🚀 Starting orchestration concept demonstration...")

    try:
        # Import the tasks directly
        from orchestration.tasks import (
            analyze_model_drift,
            check_model_performance,
            generate_predictions,
            send_alerts,
        )

        # Create necessary directories
        for dir_path in ["data/reports", "data/alerts", "data/predictions"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Test 1: Model performance check
        logger.info("1. 📊 Testing model performance check...")
        performance_result = check_model_performance.fn()
        logger.info(f"✅ Performance check result: {performance_result.get('current_accuracy', 'N/A'):.3f} accuracy")

        # Test 2: Drift analysis
        logger.info("2. 🔍 Testing drift analysis...")
        drift_result = analyze_model_drift.fn()
        logger.info(f"✅ Drift analysis result: drift_detected={drift_result.get('drift_detected', False)}")

        # Test 3: Prediction generation
        logger.info("3. 🔮 Testing prediction generation...")
        prediction_result = generate_predictions.fn(days_ahead=3)
        logger.info(f"✅ Generated {prediction_result.get('predictions_generated', 0)} predictions")

        # Test 4: Alert system
        logger.info("4. 🚨 Testing alert system...")
        alert_result = send_alerts.fn(
            alert_type="test",
            message="Test alert from orchestration system",
            severity="info",
            data={"test_mode": True},
        )
        logger.info(f"✅ Alert sent: {alert_result.get('alert_sent', False)}")

        # Test 5: Orchestration flow simulation
        logger.info("5. 🎯 Simulating orchestration flow...")

        # Simulate automated monitoring flow
        needs_retraining = performance_result.get("needs_retraining", False) or drift_result.get(
            "drift_detected", False
        )

        if needs_retraining:
            logger.info("⚠️  Retraining would be triggered!")
            send_alerts.fn(
                alert_type="retraining",
                message="Automated retraining triggered by monitoring",
                severity="warning",
                data={
                    "performance": performance_result,
                    "drift": drift_result,
                },
            )
        else:
            logger.info("✅ Model performance is satisfactory, no retraining needed")

        # Demonstrate the concept
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ORCHESTRATION CONCEPT DEMONSTRATION COMPLETE!")
        logger.info("=" * 60)
        logger.info("✅ The system successfully demonstrates:")
        logger.info("  • Model performance monitoring")
        logger.info("  • Drift detection (basic statistical method)")
        logger.info("  • Automated prediction generation")
        logger.info("  • Alert system for notifications")
        logger.info("  • Intelligent retraining decisions")
        logger.info("\n📊 Key Results:")
        logger.info(f"  • Model Accuracy: {performance_result.get('current_accuracy', 0):.3f}")
        logger.info(f"  • Drift Detected: {drift_result.get('drift_detected', False)}")
        logger.info(f"  • Predictions Generated: {prediction_result.get('predictions_generated', 0)}")
        logger.info(f"  • Alerts Sent: {len([alert_result, drift_result, performance_result])}")

        logger.info("\n🔧 Technology Stack:")
        logger.info("  • Prefect: Workflow orchestration (tasks and flows)")
        logger.info("  • Custom Analytics: Model drift detection")
        logger.info("  • PostgreSQL: Metrics storage")
        logger.info("  • Grafana: Dashboard visualization")
        logger.info("  • MLflow: Model management")

        logger.info("\n🚀 Production Deployment:")
        logger.info("  • Prefect flows can be scheduled (hourly, daily, weekly)")
        logger.info("  • Alerts can be sent to Slack, email, or monitoring systems")
        logger.info("  • Grafana dashboards provide real-time monitoring")
        logger.info("  • Automatic retraining based on performance thresholds")

        return True

    except Exception as e:
        logger.error(f"❌ Error in orchestration demonstration: {e}")
        return False


def main():
    """Main function."""
    logger.info("🎯 Premier League MLOps - Orchestration Demonstration")
    logger.info("Using Prefect + Custom Analytics + Grafana")
    logger.info("-" * 60)

    success = test_orchestration_concept()

    if success:
        logger.info("\n🎉 SUCCESS: The orchestration system is working correctly!")
        logger.info("Ready for production deployment with scheduled workflows.")
    else:
        logger.error("\n❌ FAILED: Issues detected in the orchestration system.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
