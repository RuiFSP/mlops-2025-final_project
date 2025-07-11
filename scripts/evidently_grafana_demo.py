"""
Example integration script showing how to use Evidently with Grafana for ML monitoring.

This script demonstrates:
1. Setting up Evidently monitoring with Prometheus and InfluxDB
2. Generating monitoring reports
3. Exporting metrics to dashboards
4. Automated monitoring workflows
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our monitoring modules
from src.monitoring.evidently import (
    EvidentlyMLMonitor,
    EvidentlyReportGenerator,
    EvidentlyMetricsExporter
)
from src.monitoring.grafana import (
    GrafanaDashboardConfig,
    GrafanaPanelFactory
)


def create_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data for demonstration."""
    np.random.seed(42)

    # Create reference data (training data simulation)
    n_samples = 1000
    reference_data = pd.DataFrame({
        'home_odds': np.random.uniform(1.5, 4.0, n_samples),
        'draw_odds': np.random.uniform(3.0, 5.0, n_samples),
        'away_odds': np.random.uniform(1.5, 4.0, n_samples),
        'home_prob_margin_adj': np.random.uniform(0.2, 0.8, n_samples),
        'draw_prob_margin_adj': np.random.uniform(0.1, 0.4, n_samples),
        'away_prob_margin_adj': np.random.uniform(0.2, 0.8, n_samples),
        'home_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'ManCity'], n_samples),
        'away_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'ManCity'], n_samples),
        'month': np.random.choice(['Aug', 'Sep', 'Oct', 'Nov', 'Dec'], n_samples),
        'result': np.random.choice(['home', 'draw', 'away'], n_samples),
        'predicted_result': np.random.choice(['home', 'draw', 'away'], n_samples),
    })

    # Create current data with some drift
    current_data = pd.DataFrame({
        'home_odds': np.random.uniform(1.8, 4.5, n_samples),  # Slight drift
        'draw_odds': np.random.uniform(3.2, 5.2, n_samples),  # Slight drift
        'away_odds': np.random.uniform(1.8, 4.5, n_samples),  # Slight drift
        'home_prob_margin_adj': np.random.uniform(0.15, 0.75, n_samples),  # Slight drift
        'draw_prob_margin_adj': np.random.uniform(0.08, 0.38, n_samples),  # Slight drift
        'away_prob_margin_adj': np.random.uniform(0.15, 0.75, n_samples),  # Slight drift
        'home_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'ManCity'], n_samples),
        'away_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'ManCity'], n_samples),
        'month': np.random.choice(['Aug', 'Sep', 'Oct', 'Nov', 'Dec'], n_samples),
        'result': np.random.choice(['home', 'draw', 'away'], n_samples),
        'predicted_result': np.random.choice(['home', 'draw', 'away'], n_samples),
    })

    # Add some missing values to simulate data quality issues
    current_data.loc[np.random.choice(current_data.index, 50), 'home_odds'] = np.nan
    current_data.loc[np.random.choice(current_data.index, 30), 'draw_odds'] = np.nan

    return reference_data, current_data


def demonstrate_evidently_monitoring():
    """Demonstrate Evidently monitoring capabilities."""
    logger.info("=== Evidently Monitoring Demo ===")

    # Create sample data
    reference_data, current_data = create_sample_data()
    logger.info(f"Created reference data: {len(reference_data)} samples")
    logger.info(f"Created current data: {len(current_data)} samples")

    # Initialize Evidently ML Monitor
    monitor = EvidentlyMLMonitor(
        reference_data=reference_data,
        output_dir="evidently_reports",
        enable_data_drift=True,
        enable_target_drift=True,
        enable_data_quality=True,
        drift_threshold=0.1
    )

    # Generate comprehensive report
    logger.info("Generating comprehensive monitoring report...")
    report_results = monitor.generate_comprehensive_report(
        current_data=current_data,
        include_predictions=True
    )

    if report_results["success"]:
        logger.info(f"Report generated successfully: {report_results['report_id']}")

        # Display key metrics
        summary = report_results.get("summary", {})
        logger.info(f"Total samples analyzed: {summary.get('total_samples', 'N/A')}")
        logger.info(f"Drift detected: {summary.get('drift_detected', 'N/A')}")
        logger.info(f"Quality issues: {summary.get('quality_issues', 'N/A')}")
        logger.info(f"Alert count: {summary.get('alert_count', 'N/A')}")

        # Show alerts
        for alert in report_results.get("alerts", []):
            logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
    else:
        logger.error(f"Report generation failed: {report_results.get('error', 'Unknown error')}")

    # Get monitoring summary
    logger.info("Getting monitoring summary...")
    monitoring_summary = monitor.get_monitoring_summary(days_back=7)
    logger.info(f"Monitoring summary: {monitoring_summary}")

    return monitor, report_results


def demonstrate_report_generation():
    """Demonstrate automated report generation."""
    logger.info("=== Report Generation Demo ===")

    # Create sample data
    reference_data, current_data = create_sample_data()

    # Initialize Report Generator
    report_generator = EvidentlyReportGenerator(
        output_dir="evidently_reports",
        archive_dir="evidently_reports/archive",
        max_reports=50
    )

    # Generate daily report
    logger.info("Generating daily report...")
    daily_report = report_generator.generate_daily_report(
        reference_data=reference_data,
        current_data=current_data,
        report_date=datetime.now()
    )

    if daily_report["success"]:
        logger.info(f"Daily report generated: {daily_report['report_id']}")
        logger.info(f"HTML report: {daily_report['html_path']}")
        logger.info(f"JSON report: {daily_report['json_path']}")

        # Display summary
        summary = daily_report.get("summary", {})
        logger.info(f"Date: {summary.get('date', 'N/A')}")
        logger.info(f"Alerts: {len(summary.get('alerts', []))}")
        logger.info(f"Recommendations: {len(summary.get('recommendations', []))}")
    else:
        logger.error(f"Daily report generation failed: {daily_report.get('error', 'Unknown error')}")

    # Generate comparison report
    logger.info("Generating comparison report...")
    # Create slightly different dataset for comparison
    _, current_data_2 = create_sample_data()

    comparison_report = report_generator.generate_comparison_report(
        reference_data=reference_data,
        dataset_a=current_data,
        dataset_b=current_data_2,
        comparison_name="week_comparison"
    )

    if comparison_report["success"]:
        logger.info(f"Comparison report generated: {comparison_report['report_id']}")

        # Display comparison analysis
        analysis = comparison_report.get("comparison_analysis", {})
        logger.info(f"Comparison: {analysis.get('comparison_name', 'N/A')}")
        logger.info(f"Recommendations: {len(analysis.get('recommendations', []))}")
    else:
        logger.error(f"Comparison report generation failed: {comparison_report.get('error', 'Unknown error')}")

    # Get report history
    logger.info("Getting report history...")
    history = report_generator.get_report_history(days_back=30)
    logger.info(f"Found {len(history)} reports in history")

    return report_generator, daily_report, comparison_report


def demonstrate_metrics_export():
    """Demonstrate metrics export to Prometheus and InfluxDB."""
    logger.info("=== Metrics Export Demo ===")

    # Create sample data
    reference_data, current_data = create_sample_data()

    # Initialize Metrics Exporter
    # Note: These URLs are examples - replace with your actual service URLs
    metrics_exporter = EvidentlyMetricsExporter(
        prometheus_gateway="http://localhost:9091",  # Prometheus pushgateway
        prometheus_job="evidently_ml_monitoring",
        influxdb_url="http://localhost:8086",
        influxdb_token="your-influxdb-token",
        influxdb_org="mlops",
        influxdb_bucket="ml_monitoring",
        export_interval=60
    )

    # Export metrics
    logger.info("Exporting metrics to monitoring backends...")
    export_results = metrics_exporter.export_metrics(
        reference_data=reference_data,
        current_data=current_data,
        tags={"environment": "demo", "model": "premier_league"}
    )

    if export_results["success"]:
        logger.info(f"Metrics exported successfully at {export_results['timestamp']}")
        logger.info(f"Export duration: {export_results['metrics']['export_duration']:.2f}s")

        # Show export results
        for backend, result in export_results.get("exports", {}).items():
            status = "SUCCESS" if result.get("success", False) else "FAILED"
            logger.info(f"{backend.upper()}: {status}")
            if not result.get("success", False):
                logger.error(f"  Error: {result.get('error', 'Unknown error')}")

        # Display key metrics
        metrics = export_results.get("metrics", {})
        logger.info(f"Dataset drift: {metrics.get('dataset_drift', 'N/A')}")
        logger.info(f"Drift share: {metrics.get('drift_share', 'N/A'):.2%}")
        logger.info(f"Missing values: {metrics.get('missing_values_percentage', 'N/A'):.1f}%")
    else:
        logger.error(f"Metrics export failed: {export_results.get('error', 'Unknown error')}")

    # Get export status
    logger.info("Getting export status...")
    status = metrics_exporter.get_export_status()
    logger.info(f"Prometheus enabled: {status.get('prometheus_enabled', 'N/A')}")
    logger.info(f"InfluxDB enabled: {status.get('influxdb_enabled', 'N/A')}")
    logger.info(f"Export history count: {status.get('export_history_count', 'N/A')}")
    logger.info(f"Recent errors: {status.get('recent_errors', 'N/A')}")

    return metrics_exporter, export_results


def demonstrate_grafana_dashboards():
    """Demonstrate Grafana dashboard creation."""
    logger.info("=== Grafana Dashboard Demo ===")

    # Initialize Dashboard Config
    dashboard_config = GrafanaDashboardConfig(
        dashboard_title="ML Monitoring Dashboard - Demo",
        dashboard_tags=["mlops", "monitoring", "evidently", "demo"],
        refresh_interval="30s",
        time_range="1h"
    )

    # Create ML monitoring dashboard
    logger.info("Creating ML monitoring dashboard...")
    dashboard = dashboard_config.create_ml_monitoring_dashboard(
        prometheus_datasource="prometheus",
        influxdb_datasource="influxdb"
    )

    # Export dashboard
    dashboard_path = "dashboards/demo_ml_monitoring_dashboard.json"
    success = dashboard_config.export_dashboard(dashboard_path)

    if success:
        logger.info(f"Dashboard exported to: {dashboard_path}")
        logger.info("Dashboard can be imported into Grafana")
    else:
        logger.error("Dashboard export failed")

    # Demonstrate panel factory
    logger.info("Creating custom panels...")
    panel_factory = GrafanaPanelFactory()

    # Create drift overview panel
    drift_panel = panel_factory.create_ml_drift_overview_panel(
        panel_id=100,
        datasource="prometheus",
        x=0, y=0, w=24, h=8
    )
    logger.info(f"Created drift overview panel: {drift_panel['title']}")

    # Create quality overview panel
    quality_panel = panel_factory.create_ml_quality_overview_panel(
        panel_id=101,
        datasource="prometheus",
        x=0, y=8, w=24, h=8
    )
    logger.info(f"Created quality overview panel: {quality_panel['title']}")

    # Create performance panel
    performance_panel = panel_factory.create_ml_performance_panel(
        panel_id=102,
        datasource="prometheus",
        x=0, y=16, w=24, h=8
    )
    logger.info(f"Created performance panel: {performance_panel['title']}")

    return dashboard_config, dashboard


def demonstrate_automated_workflow():
    """Demonstrate automated monitoring workflow."""
    logger.info("=== Automated Monitoring Workflow Demo ===")

    # Create sample data
    reference_data, current_data = create_sample_data()

    # Initialize all components
    monitor = EvidentlyMLMonitor(
        reference_data=reference_data,
        output_dir="evidently_reports",
        enable_data_drift=True,
        enable_target_drift=True,
        enable_data_quality=True,
        drift_threshold=0.1
    )

    report_generator = EvidentlyReportGenerator(
        output_dir="evidently_reports",
        archive_dir="evidently_reports/archive",
        max_reports=50
    )

    metrics_exporter = EvidentlyMetricsExporter(
        prometheus_gateway="http://localhost:9091",
        prometheus_job="evidently_ml_monitoring",
        influxdb_url="http://localhost:8086",
        influxdb_token="your-influxdb-token",
        influxdb_org="mlops",
        influxdb_bucket="ml_monitoring",
        export_interval=60
    )

    # Simulate automated workflow
    logger.info("Starting automated monitoring workflow...")

    workflow_results = []

    for i in range(3):  # Simulate 3 monitoring cycles
        logger.info(f"Monitoring cycle {i+1}/3")

        # Generate monitoring report
        report_results = monitor.generate_comprehensive_report(
            current_data=current_data,
            include_predictions=True
        )

        # Generate daily report
        daily_report = report_generator.generate_daily_report(
            reference_data=reference_data,
            current_data=current_data,
            report_date=datetime.now()
        )

        # Export metrics
        export_results = metrics_exporter.export_metrics(
            reference_data=reference_data,
            current_data=current_data,
            tags={"environment": "demo", "cycle": f"cycle_{i+1}"}
        )

        # Collect results
        cycle_results = {
            "cycle": i+1,
            "timestamp": datetime.now(),
            "monitoring_success": report_results.get("success", False),
            "report_success": daily_report.get("success", False),
            "export_success": export_results.get("success", False),
            "alerts": len(report_results.get("alerts", [])),
            "drift_detected": report_results.get("summary", {}).get("drift_detected", False),
            "quality_issues": report_results.get("summary", {}).get("quality_issues", False),
        }

        workflow_results.append(cycle_results)

        logger.info(f"Cycle {i+1} completed:")
        logger.info(f"  Monitoring: {'✓' if cycle_results['monitoring_success'] else '✗'}")
        logger.info(f"  Reporting: {'✓' if cycle_results['report_success'] else '✗'}")
        logger.info(f"  Exporting: {'✓' if cycle_results['export_success'] else '✗'}")
        logger.info(f"  Alerts: {cycle_results['alerts']}")

        # Small delay between cycles
        time.sleep(2)

    # Summary
    logger.info("=== Workflow Summary ===")
    successful_cycles = sum(1 for r in workflow_results if all([
        r["monitoring_success"], r["report_success"], r["export_success"]
    ]))

    logger.info(f"Total cycles: {len(workflow_results)}")
    logger.info(f"Successful cycles: {successful_cycles}")
    logger.info(f"Success rate: {successful_cycles/len(workflow_results)*100:.1f}%")

    total_alerts = sum(r["alerts"] for r in workflow_results)
    logger.info(f"Total alerts generated: {total_alerts}")

    return workflow_results


def main():
    """Main demonstration function."""
    logger.info("Starting Evidently + Grafana Integration Demo")
    logger.info("=" * 50)

    try:
        # Demonstrate each component
        monitor, report_results = demonstrate_evidently_monitoring()
        report_generator, daily_report, comparison_report = demonstrate_report_generation()
        metrics_exporter, export_results = demonstrate_metrics_export()
        dashboard_config, dashboard = demonstrate_grafana_dashboards()
        workflow_results = demonstrate_automated_workflow()

        # Final summary
        logger.info("=" * 50)
        logger.info("Demo completed successfully!")
        logger.info("=" * 50)

        logger.info("Generated files:")
        logger.info("- Evidently reports in: evidently_reports/")
        logger.info("- Grafana dashboards in: dashboards/")
        logger.info("- Configuration files in: config/grafana/")

        logger.info("\nNext steps:")
        logger.info("1. Set up Prometheus, InfluxDB, and Grafana")
        logger.info("2. Configure datasources using config/grafana/datasources.yaml")
        logger.info("3. Import dashboard JSON files into Grafana")
        logger.info("4. Set up automated monitoring pipelines")
        logger.info("5. Configure alerting rules and notifications")

        logger.info("\nFor production use:")
        logger.info("- Update service URLs and credentials")
        logger.info("- Set up proper authentication and security")
        logger.info("- Configure retention policies")
        logger.info("- Set up backup and disaster recovery")

    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
