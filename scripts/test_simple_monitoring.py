#!/usr/bin/env python3
"""
Simple End-to-End Monitoring Test
This script demonstrates the complete MLOps monitoring pipeline with simplified imports
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Now import with proper path
from prefect import flow, get_run_logger, task

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@task
def initialize_monitoring():
    """Initialize monitoring components"""
    logger_task = get_run_logger()
    logger_task.info("🚀 Initializing MLOps Monitoring Stack")

    try:
        # Test database connection
        import psycopg2

        conn = psycopg2.connect(
            host="localhost", database="mlops_db", user="mlops_user", password="mlops_password"
        )
        conn.close()
        logger_task.info("✅ PostgreSQL connection verified")

        # Test metrics storage
        from monitoring.metrics_storage import MetricsStorage

        metrics_storage = MetricsStorage()
        logger_task.info("✅ Metrics storage initialized")

        # Test retraining monitor
        from retraining.retraining_monitor import RetrainingMonitor

        retraining_monitor = RetrainingMonitor()
        logger_task.info("✅ Retraining monitor initialized")

        return True

    except Exception as e:
        logger_task.error(f"❌ Initialization failed: {e}")
        return False


@task
def generate_metrics():
    """Generate sample metrics for monitoring"""
    logger_task = get_run_logger()
    logger_task.info("📊 Generating sample metrics for monitoring")

    try:
        from monitoring.metrics_storage import MetricsStorage

        metrics_storage = MetricsStorage()

        # Generate sample metrics
        sample_metrics = [
            ("accuracy", 0.6184),
            ("precision", 0.6250),
            ("recall", 0.6100),
            ("f1_score", 0.6174),
            ("auc_score", 0.6523),
        ]

        for metric_name, metric_value in sample_metrics:
            metrics_storage.store_model_metric(
                metric_type=metric_name, value=metric_value, model_name="premier_league_predictor"
            )

        logger_task.info(f"✅ Generated {len(sample_metrics)} metrics")
        return True

    except Exception as e:
        logger_task.error(f"❌ Metrics generation failed: {e}")
        return False


@task
def process_batch_data():
    """Process batch data for monitoring"""
    logger_task = get_run_logger()
    logger_task.info("🔄 Processing batch data for monitoring")

    try:
        from orchestration.batch_processor import WeeklyBatchProcessor

        batch_processor = WeeklyBatchProcessor()

        # Get recent weeks
        recent_weeks = batch_processor.get_recent_weeks(num_weeks=2)
        logger_task.info(f"📅 Processing {len(recent_weeks)} weeks of data")

        total_matches = 0
        for week_data in recent_weeks:
            matches = week_data.matches
            total_matches += len(matches)
            logger_task.info(f"📊 Week {week_data.week_start}: {len(matches)} matches")

            # Simulate processing
            time.sleep(1)

        logger_task.info(f"✅ Processed {total_matches} total matches")
        return True

    except Exception as e:
        logger_task.error(f"❌ Batch processing failed: {e}")
        return False


@task
def monitor_model_performance():
    """Monitor model performance and drift"""
    logger_task = get_run_logger()
    logger_task.info("🔍 Monitoring model performance and drift")

    try:
        from retraining.retraining_monitor import RetrainingMonitor

        monitor = RetrainingMonitor()

        # Get current model info
        model_info = monitor.get_current_model_info()
        logger_task.info(f"📈 Current model version: {model_info.get('version', 'unknown')}")

        # Run monitoring cycle
        monitoring_result = monitor.run_monitoring_cycle()
        logger_task.info(
            f"📊 Monitoring cycle completed: {monitoring_result.get('status', 'unknown')}"
        )

        # Check if retraining is needed
        needs_retraining = monitoring_result.get("needs_retraining", False)
        logger_task.info(f"🔄 Retraining needed: {'Yes' if needs_retraining else 'No'}")

        return True

    except Exception as e:
        logger_task.error(f"❌ Performance monitoring failed: {e}")
        return False


@task
def generate_alerts():
    """Generate monitoring alerts"""
    logger_task = get_run_logger()
    logger_task.info("🚨 Generating monitoring alerts")

    try:
        # Create alerts directory
        alerts_dir = Path(__file__).parent.parent / "alerts"
        alerts_dir.mkdir(exist_ok=True)

        # Generate alert
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": "monitoring_demo",
            "message": "End-to-end monitoring demonstration completed",
            "severity": "info",
            "components": ["prefect", "grafana", "postgresql", "mlflow"],
        }

        alert_file = alerts_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json

        with open(alert_file, "w") as f:
            json.dump(alert_data, f, indent=2)

        logger_task.info(f"✅ Alert generated: {alert_file}")
        return True

    except Exception as e:
        logger_task.error(f"❌ Alert generation failed: {e}")
        return False


@flow(name="MLOps Monitoring Demo")
def monitoring_demo_flow():
    """Complete monitoring demonstration flow"""
    logger_task = get_run_logger()
    logger_task.info("🎬 Starting MLOps Monitoring Demonstration")

    # Step 1: Initialize monitoring
    logger_task.info("🔧 Phase 1: Initializing monitoring stack")
    init_success = initialize_monitoring()

    if not init_success:
        logger_task.error("❌ Monitoring initialization failed")
        return False

    # Step 2: Generate metrics
    logger_task.info("🔧 Phase 2: Generating metrics")
    metrics_success = generate_metrics()

    if not metrics_success:
        logger_task.error("❌ Metrics generation failed")
        return False

    # Step 3: Process batch data
    logger_task.info("🔧 Phase 3: Processing batch data")
    batch_success = process_batch_data()

    if not batch_success:
        logger_task.error("❌ Batch processing failed")
        return False

    # Step 4: Monitor performance
    logger_task.info("🔧 Phase 4: Monitoring performance")
    monitor_success = monitor_model_performance()

    if not monitor_success:
        logger_task.error("❌ Performance monitoring failed")
        return False

    # Step 5: Generate alerts
    logger_task.info("🔧 Phase 5: Generating alerts")
    alert_success = generate_alerts()

    if not alert_success:
        logger_task.error("❌ Alert generation failed")
        return False

    # Success!
    logger_task.info("🎉 MLOps Monitoring Demo Completed Successfully!")
    logger_task.info("📊 Check Prefect UI at: http://localhost:4200")
    logger_task.info("📈 Check Grafana at: http://localhost:3000")
    logger_task.info("🔍 Check MLflow at: http://localhost:5000")
    logger_task.info("🚀 Check API at: http://localhost:8000")

    return True


def print_monitoring_info():
    """Print monitoring stack information"""
    print("\n" + "=" * 60)
    print("🎯 MLOPS MONITORING STACK - END-TO-END DEMO")
    print("=" * 60)
    print("🔧 Prefect UI:     http://localhost:4200")
    print("   └── Flow runs, task execution, scheduling")
    print("📈 Grafana:        http://localhost:3000")
    print("   └── Metrics visualization, dashboards")
    print("   └── Login: admin / admin")
    print("🔍 MLflow UI:      http://localhost:5000")
    print("   └── Model tracking, experiments, artifacts")
    print("🚀 API Server:     http://localhost:8000")
    print("   └── Live predictions, health checks")
    print("🗄️  PostgreSQL:    localhost:5432")
    print("   └── Metrics storage, monitoring data")
    print("=" * 60)
    print("📊 TO SET UP GRAFANA DASHBOARD:")
    print("1. Go to http://localhost:3000")
    print("2. Login with admin/admin")
    print("3. Add PostgreSQL data source:")
    print("   - Host: localhost:5432")
    print("   - Database: mlops_db")
    print("   - User: mlops_user")
    print("   - Password: mlops_password")
    print("4. Import: grafana/dashboards/mlops_dashboard.json")
    print("=" * 60)


def main():
    """Main execution function"""
    print_monitoring_info()

    print("\n🎬 Running MLOps Monitoring Demo Flow...")
    print("   This will generate metrics that you can see in:")
    print("   - Prefect UI (flow execution)")
    print("   - Grafana (metrics visualization)")
    print("   - PostgreSQL (stored metrics)")

    try:
        # Run the monitoring flow
        result = monitoring_demo_flow()

        if result:
            print("\n✅ SUCCESS! End-to-end monitoring demo completed!")
            print("\n🎯 Next steps:")
            print("1. Check Prefect UI at http://localhost:4200 to see flow execution")
            print("2. Set up Grafana dashboard at http://localhost:3000")
            print("3. Query metrics in PostgreSQL database")
            print("4. View model experiments in MLflow at http://localhost:5000")
            return True
        else:
            print("\n❌ FAILED! End-to-end monitoring demo failed")
            return False

    except Exception as e:
        print(f"\n❌ FAILED! Demo execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
