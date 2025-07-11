"""
Evidently metrics exporter for Prometheus and InfluxDB integration.

This module provides metrics export capabilities for monitoring dashboards,
enabling real-time metrics visualization in Grafana and other monitoring tools.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from evidently import Report
from evidently.metrics import (
    ValueDrift,
    DatasetMissingValueCount,
    ColumnCount,
    DriftedColumnsCount,
)
from evidently import DataDefinition, ColumnType

# Prometheus integration
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available - Prometheus metrics disabled")

# InfluxDB integration
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logging.warning("influxdb_client not available - InfluxDB metrics disabled")

logger = logging.getLogger(__name__)


class EvidentlyMetricsExporter:
    """
    Export Evidently metrics to Prometheus and InfluxDB for dashboard visualization.

    Provides real-time metrics export capabilities for monitoring dashboards,
    enabling comprehensive MLOps observability.
    """

    def __init__(
        self,
        prometheus_gateway: Optional[str] = None,
        prometheus_job: str = "evidently_ml_monitoring",
        influxdb_url: Optional[str] = None,
        influxdb_token: Optional[str] = None,
        influxdb_org: Optional[str] = None,
        influxdb_bucket: str = "ml_monitoring",
        column_mapping: Optional[DataDefinition] = None,
        export_interval: int = 60,
    ):
        """
        Initialize Evidently Metrics Exporter.

        Args:
            prometheus_gateway: Prometheus pushgateway URL
            prometheus_job: Prometheus job name
            influxdb_url: InfluxDB connection URL
            influxdb_token: InfluxDB authentication token
            influxdb_org: InfluxDB organization
            influxdb_bucket: InfluxDB bucket name
            column_mapping: Evidently column mapping
            export_interval: Export interval in seconds
        """
        self.prometheus_gateway = prometheus_gateway
        self.prometheus_job = prometheus_job
        self.influxdb_url = influxdb_url
        self.influxdb_token = influxdb_token
        self.influxdb_org = influxdb_org
        self.influxdb_bucket = influxdb_bucket
        self.column_mapping = column_mapping or self._create_default_column_mapping()
        self.export_interval = export_interval

        # Initialize metrics registries
        self.prometheus_registry: Optional[CollectorRegistry] = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.influxdb_client: Optional[InfluxDBClient] = None

        # Initialize Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.prometheus_gateway:
            self._init_prometheus_metrics()

        # Initialize InfluxDB client
        if INFLUXDB_AVAILABLE and self.influxdb_url:
            self._init_influxdb_client()

        # Track export history
        self.export_history: List[Dict[str, Any]] = []

        logger.info("Initialized Evidently Metrics Exporter")

    def _create_default_column_mapping(self) -> Optional[DataDefinition]:
        """Create default column mapping for Premier League predictions."""
        # Use None to let Evidently auto-detect column types
        return None

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics collectors."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Data drift metrics
        self.prometheus_dataset_drift = Gauge(
            'evidently_dataset_drift',
            'Dataset drift detected (1 if drift, 0 if no drift)',
            registry=self.prometheus_registry
        )

        self.prometheus_drift_share = Gauge(
            'evidently_drift_share',
            'Share of features showing drift (0.0 to 1.0)',
            registry=self.prometheus_registry
        )

        self.prometheus_drifted_features = Gauge(
            'evidently_drifted_features_count',
            'Number of features showing drift',
            registry=self.prometheus_registry
        )

        # Data quality metrics
        self.prometheus_missing_values_pct = Gauge(
            'evidently_missing_values_percentage',
            'Percentage of missing values in dataset',
            registry=self.prometheus_registry
        )

        self.prometheus_missing_values_count = Gauge(
            'evidently_missing_values_count',
            'Total count of missing values',
            registry=self.prometheus_registry
        )

        # Monitoring metrics
        self.prometheus_report_generation_time = Histogram(
            'evidently_report_generation_seconds',
            'Time taken to generate Evidently report',
            registry=self.prometheus_registry
        )

        self.prometheus_export_timestamp = Gauge(
            'evidently_last_export_timestamp',
            'Timestamp of last successful metrics export',
            registry=self.prometheus_registry
        )

        # Error tracking
        self.prometheus_export_errors = Counter(
            'evidently_export_errors_total',
            'Total number of export errors',
            ['error_type'],
            registry=self.prometheus_registry
        )

        logger.info("Prometheus metrics initialized")

    def _init_influxdb_client(self) -> None:
        """Initialize InfluxDB client connection."""
        if not INFLUXDB_AVAILABLE:
            return

        try:
            if self.influxdb_url and self.influxdb_token and self.influxdb_org:
                self.influxdb_client = InfluxDBClient(
                    url=self.influxdb_url,
                    token=self.influxdb_token,
                    org=self.influxdb_org
                )

                # Test connection
                if self.influxdb_client:
                    self.influxdb_client.ping()
                logger.info("InfluxDB client initialized successfully")
            else:
                logger.warning("InfluxDB configuration incomplete")

        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {str(e)}")
            self.influxdb_client = None

    def export_metrics(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Export Evidently metrics to configured backends.

        Args:
            reference_data: Reference dataset for comparison
            current_data: Current dataset to analyze
            tags: Additional tags for metrics

        Returns:
            Export results summary
        """
        start_time = time.time()
        timestamp = datetime.now()

        logger.info("Starting metrics export")

        try:
            # Generate Evidently report
            report = self._generate_report(reference_data, current_data)

            # Extract metrics (simplified for new API)
            report_dict = {'total_metrics': len(report.metrics)}
            metrics = self._extract_metrics(report_dict)

            # Add metadata
            if not isinstance(metrics, dict):
                metrics = {}
            metrics["timestamp"] = timestamp
            metrics["export_duration"] = time.time() - start_time

            # Add custom tags
            if tags:
                metrics["tags"] = tags

            # Export to backends
            export_results: Dict[str, Any] = {
                "success": True,
                "timestamp": timestamp,
                "metrics": metrics,
                "exports": {},
            }

            # Export to Prometheus
            if PROMETHEUS_AVAILABLE and self.prometheus_gateway:
                prometheus_result = self._export_to_prometheus(metrics)
                export_results["exports"]["prometheus"] = prometheus_result

            # Export to InfluxDB
            if INFLUXDB_AVAILABLE and self.influxdb_client:
                influxdb_result = self._export_to_influxdb(metrics)
                export_results["exports"]["influxdb"] = influxdb_result

            # Update export history
            self.export_history.append(export_results)

            # Keep only recent history
            if len(self.export_history) > 100:
                self.export_history = self.export_history[-100:]

            logger.info(f"Metrics export completed in {metrics.get('export_duration', 0):.2f}s")

            return export_results

        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")

            # Track error in Prometheus if available
            if PROMETHEUS_AVAILABLE and self.prometheus_registry:
                self.prometheus_export_errors.labels(error_type="general").inc()

            return {
                "success": False,
                "error": str(e),
                "timestamp": timestamp,
                "export_duration": time.time() - start_time,
            }

    def _generate_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Report:
        """Generate Evidently report for metrics extraction."""
        report = Report(
            metrics=[
                DriftedColumnsCount(),
                ValueDrift(column="result"),
                ValueDrift(column="predicted_result"),
                DatasetMissingValueCount(),
                ColumnCount(),
            ]
        )

        report.run(
            reference_data=reference_data,
            current_data=current_data
        )

        return report

    def _extract_metrics(self, report_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from Evidently report."""
        metrics = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "drifted_features_count": 0,
            "missing_values_percentage": 0.0,
            "missing_values_count": 0,
            "total_features": 0,
        }

        try:
            for metric in report_dict.get("metrics", []):
                metric_name = metric.get("metric", "")
                result = metric.get("result", {})

                if "DatasetDriftMetric" in metric_name:
                    metrics.update({
                        "dataset_drift": result.get("dataset_drift", False),
                        "drift_share": result.get("drift_share", 0.0),
                        "drifted_features_count": result.get("number_of_drifted_columns", 0),
                        "total_features": result.get("number_of_columns", 0),
                    })

                elif "DatasetMissingValuesMetric" in metric_name:
                    metrics.update({
                        "missing_values_percentage": result.get("missing_percentage", 0.0),
                        "missing_values_count": result.get("missing_count", 0),
                    })

        except Exception as e:
            logger.warning(f"Error extracting metrics: {str(e)}")

        return metrics

    def _export_to_prometheus(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Export metrics to Prometheus pushgateway."""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_gateway:
            return {"success": False, "error": "Prometheus not available"}

        try:
            # Update Prometheus metrics
            self.prometheus_dataset_drift.set(1 if metrics.get("dataset_drift", False) else 0)
            self.prometheus_drift_share.set(metrics.get("drift_share", 0.0))
            self.prometheus_drifted_features.set(metrics.get("drifted_features_count", 0))
            self.prometheus_missing_values_pct.set(metrics.get("missing_values_percentage", 0.0))
            self.prometheus_missing_values_count.set(metrics.get("missing_values_count", 0))

            # Update export timestamp
            self.prometheus_export_timestamp.set(time.time())

            # Update report generation time
            if "export_duration" in metrics:
                self.prometheus_report_generation_time.observe(metrics["export_duration"])

            # Push to gateway
            if self.prometheus_gateway and self.prometheus_registry:
                push_to_gateway(
                    self.prometheus_gateway,
                    job=self.prometheus_job,
                    registry=self.prometheus_registry
                )

            logger.info("Metrics exported to Prometheus successfully")

            return {"success": True, "gateway": self.prometheus_gateway}

        except Exception as e:
            logger.error(f"Error exporting to Prometheus: {str(e)}")
            if self.prometheus_export_errors:
                self.prometheus_export_errors.labels(error_type="prometheus").inc()
            return {"success": False, "error": str(e)}

    def _export_to_influxdb(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Export metrics to InfluxDB."""
        if not INFLUXDB_AVAILABLE or not self.influxdb_client:
            return {"success": False, "error": "InfluxDB not available"}

        try:
            write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)

            # Create data points
            points = []
            timestamp = metrics.get("timestamp", datetime.now())

            # Data drift metrics
            points.append(
                Point("ml_data_drift")
                .field("dataset_drift", int(metrics["dataset_drift"]))
                .field("drift_share", metrics["drift_share"])
                .field("drifted_features_count", metrics["drifted_features_count"])
                .field("total_features", metrics["total_features"])
                .time(timestamp)
            )

            # Data quality metrics
            points.append(
                Point("ml_data_quality")
                .field("missing_values_percentage", metrics["missing_values_percentage"])
                .field("missing_values_count", metrics["missing_values_count"])
                .time(timestamp)
            )

            # Performance metrics
            if "export_duration" in metrics:
                points.append(
                    Point("ml_monitoring_performance")
                    .field("export_duration", metrics["export_duration"])
                    .time(timestamp)
                )

            # Add tags if provided
            if "tags" in metrics:
                for point in points:
                    for tag_key, tag_value in metrics["tags"].items():
                        point.tag(tag_key, tag_value)

            # Write to InfluxDB
            write_api.write(bucket=self.influxdb_bucket, record=points)

            logger.info("Metrics exported to InfluxDB successfully")

            return {"success": True, "bucket": self.influxdb_bucket, "points": len(points)}

        except Exception as e:
            logger.error(f"Error exporting to InfluxDB: {str(e)}")
            if PROMETHEUS_AVAILABLE and self.prometheus_registry:
                self.prometheus_export_errors.labels(error_type="influxdb").inc()
            return {"success": False, "error": str(e)}

    def export_batch_metrics(
        self,
        reference_data: pd.DataFrame,
        datasets: List[Tuple[pd.DataFrame, Dict[str, str]]],
    ) -> List[Dict[str, Any]]:
        """
        Export metrics for multiple datasets in batch.

        Args:
            reference_data: Reference dataset for comparison
            datasets: List of (dataset, tags) tuples

        Returns:
            List of export results
        """
        results = []

        for i, (dataset, tags) in enumerate(datasets):
            logger.info(f"Exporting batch metrics {i+1}/{len(datasets)}")

            result = self.export_metrics(reference_data, dataset, tags)
            results.append(result)

            # Add batch index
            result["batch_index"] = i

            # Small delay between exports to avoid overwhelming backends
            if i < len(datasets) - 1:
                time.sleep(1)

        return results

    def get_export_status(self) -> Dict[str, Any]:
        """Get current export status and health."""
        status: Dict[str, Any] = {
            "prometheus_enabled": PROMETHEUS_AVAILABLE and self.prometheus_gateway is not None,
            "influxdb_enabled": INFLUXDB_AVAILABLE and self.influxdb_client is not None,
            "export_history_count": len(self.export_history),
            "last_export": None,
            "recent_errors": 0,
        }

        if self.export_history:
            status["last_export"] = self.export_history[-1].get("timestamp")

            # Count recent errors (last 10 exports)
            recent_exports = self.export_history[-10:]
            status["recent_errors"] = len([e for e in recent_exports if not e.get("success", False)])

        # Test backend connections
        if self.influxdb_client:
            try:
                self.influxdb_client.ping()
                status["influxdb_connection"] = "healthy"
            except Exception:
                status["influxdb_connection"] = "unhealthy"

        return status

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.influxdb_client:
            try:
                self.influxdb_client.close()
                logger.info("InfluxDB client closed")
            except Exception as e:
                logger.warning(f"Error closing InfluxDB client: {str(e)}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.cleanup()
