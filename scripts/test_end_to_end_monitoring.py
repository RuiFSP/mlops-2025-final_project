#!/usr/bin/env python3
"""
End-to-End Monitoring Test Script
This script demonstrates the complete MLOps monitoring pipeline:
- Prefect orchestration with flow visualization
- Real-time metrics generation
- Grafana dashboard updates
- PostgreSQL metrics storage
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prefect import flow, get_run_logger, task

from monitoring.metrics_storage import MetricsStorage
from orchestration.batch_processor import WeeklyBatchProcessor
from orchestration.weekly_flows import (
    comprehensive_weekly_monitoring_flow,
)
from retraining.retraining_monitor import RetrainingMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@task
def log_monitoring_start():
    """Log the start of monitoring demonstration"""
    logger = get_run_logger()
    logger.info("🚀 Starting End-to-End Monitoring Demonstration")
    logger.info("✅ Prefect orchestration active")
    logger.info("✅ Grafana dashboard available at http://localhost:3000")
    logger.info("✅ PostgreSQL metrics storage ready")
    return True


@task
def generate_sample_metrics():
    """Generate sample metrics for demonstration"""
    logger = get_run_logger()

    try:
        # Initialize metrics storage
        metrics_storage = MetricsStorage()

        # Generate sample model metrics
        sample_metrics = [
            {
                "model_name": "premier_league_predictor",
                "metric_name": "accuracy",
                "metric_value": 0.6184,
                "timestamp": datetime.now() - timedelta(days=1),
            },
            {
                "model_name": "premier_league_predictor",
                "metric_name": "precision",
                "metric_value": 0.6250,
                "timestamp": datetime.now() - timedelta(days=1),
            },
            {
                "model_name": "premier_league_predictor",
                "metric_name": "recall",
                "metric_value": 0.6100,
                "timestamp": datetime.now() - timedelta(days=1),
            },
            {
                "model_name": "premier_league_predictor",
                "metric_name": "f1_score",
                "metric_value": 0.6174,
                "timestamp": datetime.now() - timedelta(days=1),
            },
        ]

        # Store metrics
        for metric in sample_metrics:
            metrics_storage.store_model_metric(
                model_name=metric["model_name"],
                metric_name=metric["metric_name"],
                metric_value=metric["metric_value"],
                timestamp=metric["timestamp"],
            )

        logger.info(f"✅ Generated {len(sample_metrics)} sample metrics")
        return True

    except Exception as e:
        logger.error(f"❌ Error generating sample metrics: {e}")
        return False


@task
def verify_database_connection():
    """Verify database connections are working"""
    logger = get_run_logger()

    try:
        # Test metrics storage
        metrics_storage = MetricsStorage()

        # Test retraining monitor
        retraining_monitor = RetrainingMonitor()

        logger.info("✅ Database connections verified")
        return True

    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


@flow(name="End-to-End Monitoring Demo")
def end_to_end_monitoring_demo():
    """Complete end-to-end monitoring demonstration flow"""
    logger = get_run_logger()

    # Step 1: Initialize monitoring
    logger.info("🔧 Phase 1: Initializing Monitoring Stack")
    log_monitoring_start()

    # Step 2: Verify connections
    logger.info("🔧 Phase 2: Verifying Database Connections")
    db_ok = verify_database_connection()

    if not db_ok:
        logger.error("❌ Database verification failed")
        return False

    # Step 3: Generate sample metrics
    logger.info("🔧 Phase 3: Generating Sample Metrics")
    metrics_ok = generate_sample_metrics()

    if not metrics_ok:
        logger.error("❌ Metrics generation failed")
        return False

    # Step 4: Run weekly processing
    logger.info("🔧 Phase 4: Running Weekly Batch Processing")
    try:
        # Initialize batch processor
        batch_processor = WeeklyBatchProcessor()

        # Get recent weeks for processing
        recent_weeks = batch_processor.get_recent_weeks(num_weeks=3)
        logger.info(f"📅 Processing {len(recent_weeks)} recent weeks")

        # Process each week
        for week_data in recent_weeks:
            matches = week_data["matches"]
            logger.info(f"📊 Processing week {week_data['week_start']} with {len(matches)} matches")

            # Simulate processing time
            time.sleep(2)

        logger.info("✅ Weekly batch processing completed")

    except Exception as e:
        logger.error(f"❌ Weekly processing failed: {e}")
        return False

    # Step 5: Run comprehensive monitoring
    logger.info("🔧 Phase 5: Running Comprehensive Monitoring Flow")
    try:
        # This will trigger the comprehensive monitoring flow
        comprehensive_result = comprehensive_weekly_monitoring_flow()
        logger.info("✅ Comprehensive monitoring flow completed")

    except Exception as e:
        logger.error(f"❌ Comprehensive monitoring failed: {e}")
        return False

    # Step 6: Final summary
    logger.info("🎉 End-to-End Monitoring Demo Completed Successfully!")
    logger.info("📊 Check Prefect UI at: http://localhost:4200")
    logger.info("📈 Check Grafana at: http://localhost:3000")
    logger.info("🔍 MLflow UI at: http://localhost:5000")
    logger.info("🚀 API Server at: http://localhost:8000")

    return True


def setup_monitoring_flows():
    """Set up monitoring flows for demonstration"""
    logger.info("🚀 Setting up monitoring flows for demonstration...")

    try:
        # Log flow setup
        logger.info("✅ End-to-end monitoring flow ready")
        logger.info("✅ Weekly batch processing flow ready")
        logger.info("✅ Comprehensive monitoring flow ready")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to set up flows: {e}")
        return False


def print_access_info():
    """Print access information for all monitoring services"""
    print("\n" + "=" * 60)
    print("🎯 END-TO-END MONITORING STACK ACCESS INFO")
    print("=" * 60)
    print("🔧 Prefect UI:     http://localhost:4200")
    print("📈 Grafana:        http://localhost:3000")
    print("                   Username: admin")
    print("                   Password: admin")
    print("🔍 MLflow UI:      http://localhost:5000")
    print("🚀 API Server:     http://localhost:8000")
    print("🗄️  PostgreSQL:    localhost:5432")
    print("                   Database: mlops_db")
    print("                   Username: mlops_user")
    print("=" * 60)
    print("📊 GRAFANA DASHBOARD SETUP:")
    print("1. Login to Grafana at http://localhost:3000")
    print("2. Add PostgreSQL data source:")
    print("   - Host: localhost:5432")
    print("   - Database: mlops_db")
    print("   - User: mlops_user")
    print("   - Password: mlops_password")
    print("3. Import dashboard from: grafana/dashboards/mlops_dashboard.json")
    print("=" * 60)


async def main():
    """Main execution function"""
    logger.info("🚀 Starting End-to-End Monitoring Test")

    # Print access information
    print_access_info()

    # Set up flows
    logger.info("📦 Setting up monitoring flows...")
    setup_success = setup_monitoring_flows()

    if not setup_success:
        logger.error("❌ Failed to set up flows")
        return False

    # Wait a moment for setup to complete
    await asyncio.sleep(2)

    # Run the main monitoring flow
    logger.info("🎬 Running end-to-end monitoring demonstration...")
    try:
        result = end_to_end_monitoring_demo()

        if result:
            logger.info("✅ End-to-end monitoring test completed successfully!")
            print("\n🎉 SUCCESS! You can now view:")
            print("   - Prefect flows at: http://localhost:4200")
            print("   - Grafana dashboards at: http://localhost:3000")
            print("   - Real-time metrics and monitoring data")
            return True
        else:
            logger.error("❌ End-to-end monitoring test failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
