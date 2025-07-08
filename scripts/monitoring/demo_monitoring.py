#!/usr/bin/env python3
"""
Demo script for the new monitoring system.
Shows how to use drift detection and performance monitoring.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from monitoring import MLOpsMonitoringService, ModelDriftDetector, ModelPerformanceMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Create sample data for monitoring demo."""
    np.random.seed(42)

    teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester United", "Manchester City", "Tottenham"]

    data = pd.DataFrame({
        "home_team": np.random.choice(teams, n_samples),
        "away_team": np.random.choice(teams, n_samples),
        "month": np.random.randint(1, 13, n_samples),
        "home_odds": np.random.uniform(1.5, 4.0, n_samples),
        "draw_odds": np.random.uniform(2.5, 4.5, n_samples),
        "away_odds": np.random.uniform(1.5, 4.0, n_samples),
    })

    # Calculate margin-adjusted probabilities
    for outcome in ["home", "draw", "away"]:
        implied_prob = 1 / data[f"{outcome}_odds"]
        total_implied = (1/data["home_odds"] + 1/data["draw_odds"] + 1/data["away_odds"])
        data[f"{outcome}_prob_margin_adj"] = implied_prob / total_implied

    # Create synthetic results
    results = []
    for _, row in data.iterrows():
        probs = [row["home_prob_margin_adj"], row["draw_prob_margin_adj"], row["away_prob_margin_adj"]]
        result = np.random.choice(["Home Win", "Draw", "Away Win"], p=probs)
        results.append(result)

    data["result"] = results

    return data


def demo_drift_detection():
    """Demonstrate drift detection capabilities."""
    logger.info("=== Drift Detection Demo ===")

    # Check if we have real data
    reference_data_path = Path("data/real_data/premier_league_matches.parquet")
    model_path = Path("models/model.pkl")

    if not reference_data_path.exists():
        logger.warning(f"Reference data not found at {reference_data_path}")
        logger.info("Creating sample reference data for demo...")
        reference_data = create_sample_data(1000)
        reference_data_path = Path("data/sample_reference.parquet")
        reference_data_path.parent.mkdir(exist_ok=True)
        reference_data.to_parquet(reference_data_path)

    try:
        # Initialize drift detector
        drift_detector = ModelDriftDetector(
            reference_data_path=str(reference_data_path),
            model_path=str(model_path) if model_path.exists() else "models/dummy_model.pkl",
            drift_threshold=0.1
        )

        # Create new data that might have drift
        logger.info("Creating new data with potential drift...")
        new_data = create_sample_data(200)

        # Introduce some drift (change team distribution)
        new_data.loc[:50, "home_team"] = "Arsenal"  # Arsenal plays more home games
        new_data.loc[:30, "month"] = 12  # More December matches

        # Detect drift
        logger.info("Running drift detection...")
        drift_results = drift_detector.detect_drift(new_data, save_report=True)

        # Print results
        logger.info("Drift Detection Results:")
        logger.info(f"  Dataset drift detected: {drift_results.get('dataset_drift', False)}")
        logger.info(f"  Drift share: {drift_results.get('drift_share', 0):.2%}")
        logger.info(f"  Alert triggered: {drift_results.get('alert_triggered', False)}")
        logger.info(f"  Drifted features: {drift_results.get('drifted_features', [])}")

        if 'report_path' in drift_results:
            logger.info(f"  Detailed report saved to: {drift_results['report_path']}")

    except Exception as e:
        logger.error(f"Drift detection demo failed: {e}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    logger.info("=== Performance Monitoring Demo ===")

    model_path = Path("models/model.pkl")

    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        logger.info("Skipping performance monitoring demo (requires trained model)")
        return

    try:
        # Initialize performance monitor
        performance_monitor = ModelPerformanceMonitor(
            model_path=str(model_path),
            performance_threshold=0.05
        )

        # Create test data
        logger.info("Creating test data...")
        test_data = create_sample_data(150)

        # Extract features and labels
        feature_columns = [
            "home_team", "away_team", "month",
            "home_odds", "draw_odds", "away_odds",
            "home_prob_margin_adj", "draw_prob_margin_adj", "away_prob_margin_adj"
        ]

        X_test = test_data[feature_columns]
        y_test = test_data["result"]

        # Evaluate performance
        logger.info("Evaluating model performance...")
        performance_results = performance_monitor.evaluate_batch(
            X=X_test,
            y_true=y_test,
            batch_id="demo_batch_001"
        )

        # Print results
        logger.info("Performance Monitoring Results:")
        logger.info(f"  Accuracy: {performance_results.get('accuracy', 0):.3f}")
        logger.info(f"  Precision (macro): {performance_results.get('precision_macro', 0):.3f}")
        logger.info(f"  Recall (macro): {performance_results.get('recall_macro', 0):.3f}")
        logger.info(f"  F1 (macro): {performance_results.get('f1_macro', 0):.3f}")
        logger.info(f"  Performance degraded: {performance_results.get('performance_degraded', False)}")
        logger.info(f"  Accuracy change from baseline: {performance_results.get('accuracy_change', 0):+.3f}")

        # Get performance summary
        logger.info("\nGetting performance summary...")
        summary = performance_monitor.get_performance_summary(days=30)
        logger.info(f"  Evaluations in last 30 days: {summary['evaluations_count']}")
        if summary['evaluations_count'] > 0:
            logger.info(f"  Mean accuracy: {summary['current_performance']['mean_accuracy']:.3f}")
            logger.info(f"  Degradation rate: {summary['performance_trend']['degradation_rate']:.1%}")

    except Exception as e:
        logger.error(f"Performance monitoring demo failed: {e}")


def demo_unified_monitoring():
    """Demonstrate unified monitoring service."""
    logger.info("=== Unified Monitoring Demo ===")

    try:
        # Initialize monitoring service
        monitoring_service = MLOpsMonitoringService(
            model_path="models/model.pkl",
            reference_data_path="data/real_data/premier_league_matches.parquet",
            drift_threshold=0.1,
            performance_threshold=0.05
        )

        # Create production data
        logger.info("Creating production data batch...")
        production_data = create_sample_data(100)

        # Run comprehensive monitoring (without true labels)
        logger.info("Running comprehensive monitoring (drift only)...")
        monitoring_results = monitoring_service.monitor_production_batch(
            production_data=production_data,
            batch_id="prod_batch_001"
        )

        # Print results
        logger.info("Unified Monitoring Results:")
        logger.info(f"  Overall status: {monitoring_results.get('overall_status', 'unknown')}")
        logger.info(f"  Alert count: {monitoring_results.get('alert_count', 0)}")

        # Print alerts
        alerts = monitoring_results.get('alerts', [])
        if alerts:
            logger.info("  Alerts:")
            for alert in alerts:
                logger.info(f"    - {alert['type']}: {alert['message']}")

        # Print recommendations
        recommendations = monitoring_results.get('recommendations', [])
        if recommendations:
            logger.info("  Recommendations:")
            for rec in recommendations:
                logger.info(f"    - {rec}")

        # Run health check
        logger.info("\nRunning monitoring system health check...")
        health_check = monitoring_service.run_health_check()
        logger.info(f"  Monitoring service status: {health_check['monitoring_service_status']}")
        for component, status in health_check['components'].items():
            logger.info(f"  {component}: {status.get('status', 'unknown')}")

    except Exception as e:
        logger.error(f"Unified monitoring demo failed: {e}")


def main():
    """Run all monitoring demos."""
    logger.info("Starting Premier League Model Monitoring Demo")
    logger.info("=" * 50)

    # Demo 1: Drift Detection
    demo_drift_detection()
    print()

    # Demo 2: Performance Monitoring
    demo_performance_monitoring()
    print()

    # Demo 3: Unified Monitoring
    demo_unified_monitoring()

    logger.info("=" * 50)
    logger.info("Monitoring demo completed!")
    logger.info("\nNext steps:")
    logger.info("1. Set up automated monitoring with scheduled batch evaluations")
    logger.info("2. Integrate monitoring alerts with your notification system")
    logger.info("3. Create monitoring dashboards using the monitoring data")
    logger.info("4. Set up automated retraining triggers based on monitoring results")


if __name__ == "__main__":
    main()
