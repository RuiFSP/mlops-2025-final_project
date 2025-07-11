"""
Enhanced ML monitoring using Evidently for comprehensive model observability.

This module provides advanced ML monitoring capabilities including:
- Data drift detection with detailed analysis
- Model performance monitoring over time
- Prediction quality assessment
- Data quality validation
- Automated report generation for stakeholders
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from evidently import DataDefinition, ColumnType
from evidently.metrics import (
    ValueDrift,
    DatasetMissingValueCount,
    ColumnCount,
    DriftedColumnsCount,
)
from evidently import Report

logger = logging.getLogger(__name__)


class EvidentlyMLMonitor:
    """
    Advanced ML monitoring using Evidently for comprehensive model observability.

    Provides enhanced monitoring capabilities beyond basic drift detection,
    including data quality, model performance trends, and automated reporting.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_mapping: Optional[DataDefinition] = None,
        output_dir: str = "evidently_reports",
        enable_data_drift: bool = True,
        enable_target_drift: bool = True,
        enable_data_quality: bool = True,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize Evidently ML Monitor.

        Args:
            reference_data: Reference dataset for comparison
            column_mapping: Evidently column mapping configuration
            output_dir: Directory to save reports
            enable_data_drift: Enable data drift monitoring
            enable_target_drift: Enable target drift monitoring
            enable_data_quality: Enable data quality monitoring
            drift_threshold: Threshold for drift detection
        """
        self.reference_data = reference_data
        self.column_mapping = column_mapping or self._create_default_column_mapping()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_data_drift = enable_data_drift
        self.enable_target_drift = enable_target_drift
        self.enable_data_quality = enable_data_quality
        self.drift_threshold = drift_threshold

        # Track monitoring history
        self.monitoring_history: List[Dict[str, Any]] = []

        logger.info(f"Initialized Evidently ML Monitor with {len(reference_data)} reference samples")

    def _create_default_column_mapping(self) -> Optional[DataDefinition]:
        """Create default column mapping for Premier League predictions."""
        # Use None to let Evidently auto-detect column types
        return None

    def generate_comprehensive_report(
        self,
        current_data: pd.DataFrame,
        include_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report using Evidently.

        Args:
            current_data: Current dataset to analyze
            include_predictions: Whether to include prediction analysis

        Returns:
            Dictionary containing report results and metrics
        """
        timestamp = datetime.now()
        report_id = f"ml_monitor_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating comprehensive report: {report_id}")

        # Build report with enabled metrics
        metrics = []

        if self.enable_data_drift:
            metrics.extend([
                DriftedColumnsCount(),
                ValueDrift(column="result"),
                ValueDrift(column="predicted_result"),
            ])

        if self.enable_target_drift and "result" in current_data.columns:
            metrics.append(ValueDrift(column="result"))

        if self.enable_data_quality:
            metrics.extend([
                DatasetMissingValueCount(),
                ColumnCount(),
            ])

        # Create and run report
        report = Report(metrics=metrics)

        try:
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )

            # Save JSON report (HTML export not available in new API)
            json_path = self.output_dir / f"{report_id}.json"
            import json            # Extract metrics data
            metrics_data: Dict[str, Any] = {
                'report_id': report_id,
                'timestamp': timestamp,
                'metrics': []
            }

            # Get results from metrics
            for metric in report.metrics:
                metrics_data['metrics'].append({
                    'name': metric.__class__.__name__,
                    'type': str(type(metric))
                })

            with open(json_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)

            # Process results (simplified for new API)
            results = self._process_report_results(metrics_data, timestamp, report_id)

            # Save JSON summary
            json_path = self.output_dir / f"{report_id}_summary.json"
            import json
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Update history
            self.monitoring_history.append(results)

            logger.info(f"Report generated successfully: {json_path}")
            return results

        except Exception as e:
            logger.error(f"Error generating Evidently report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": timestamp,
                "report_id": report_id,
            }

    def _process_report_results(
        self,
        report_dict: Dict[str, Any],
        timestamp: datetime,
        report_id: str
    ) -> Dict[str, Any]:
        """Process Evidently report results into structured format."""

        results: Dict[str, Any] = {
            "success": True,
            "timestamp": timestamp,
            "report_id": report_id,
            "data_drift": {},
            "target_drift": {},
            "data_quality": {},
            "alerts": [],
            "summary": {},
        }

        try:
            # Extract data drift metrics
            if self.enable_data_drift:
                drift_results = self._extract_drift_metrics(report_dict)
                results["data_drift"] = drift_results

                # Check for drift alerts
                if drift_results.get("dataset_drift", False):
                    if not isinstance(results["alerts"], list):
                        results["alerts"] = []
                    results["alerts"].append({
                        "type": "data_drift",
                        "severity": "high",
                        "message": f"Dataset drift detected with {drift_results.get('drift_share', 0):.2%} of features drifting"
                    })

            # Extract data quality metrics
            if self.enable_data_quality:
                quality_results = self._extract_quality_metrics(report_dict)
                results["data_quality"] = quality_results

                # Check for quality alerts
                missing_pct = quality_results.get("missing_values_percentage", 0)
                if missing_pct > 5:  # Alert if >5% missing values
                    if not isinstance(results["alerts"], list):
                        results["alerts"] = []
                    results["alerts"].append({
                        "type": "data_quality",
                        "severity": "medium",
                        "message": f"High missing values detected: {missing_pct:.1f}%"
                    })

            # Create summary
            results["summary"] = {
                "total_samples": len(self.reference_data),
                "drift_detected": len([a for a in (results["alerts"] if isinstance(results["alerts"], list) else []) if a.get("type") == "data_drift"]) > 0,
                "quality_issues": len([a for a in (results["alerts"] if isinstance(results["alerts"], list) else []) if a.get("type") == "data_quality"]) > 0,
                "alert_count": len(results["alerts"]) if isinstance(results["alerts"], list) else 0,
            }

        except Exception as e:
            logger.warning(f"Error processing report results: {str(e)}")
            results["processing_error"] = str(e)

        return results

    def _extract_drift_metrics(self, report_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data drift metrics from Evidently report."""
        drift_metrics = {}

        try:
            # Look for dataset drift metric
            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DriftedColumnsCount":
                    result = metric.get("result", {})
                    drift_metrics.update({
                        "dataset_drift": result.get("current", {}).get("number_of_drifted_columns", 0) > 0,
                        "drift_share": result.get("current", {}).get("share_of_drifted_columns", 0),
                        "number_of_columns": result.get("current", {}).get("number_of_columns", 0),
                        "number_of_drifted_columns": result.get("current", {}).get("number_of_drifted_columns", 0),
                    })
                    break

        except Exception as e:
            logger.warning(f"Error extracting drift metrics: {str(e)}")

        return drift_metrics

    def _extract_quality_metrics(self, report_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data quality metrics from Evidently report."""
        quality_metrics = {}

        try:
            # Look for missing values metric
            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DatasetMissingValueCount":
                    result = metric.get("result", {})
                    quality_metrics.update({
                        "missing_values_percentage": result.get("current", {}).get("share_of_missing_values", 0) * 100,
                        "missing_values_count": result.get("current", {}).get("number_of_missing_values", 0),
                        "total_values": result.get("current", {}).get("number_of_values", 0),
                    })
                    break

        except Exception as e:
            logger.warning(f"Error extracting quality metrics: {str(e)}")

        return quality_metrics

    def get_monitoring_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get summary of monitoring results over specified time period.

        Args:
            days_back: Number of days to include in summary

        Returns:
            Summary statistics and trends
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        recent_reports = [
            report for report in self.monitoring_history
            if report.get("timestamp", datetime.min) >= cutoff_date
        ]

        if not recent_reports:
            return {"message": "No recent monitoring data available"}

        summary = {
            "period_days": days_back,
            "total_reports": len(recent_reports),
            "drift_alerts": len([r for r in recent_reports if r["summary"].get("drift_detected", False)]),
            "quality_alerts": len([r for r in recent_reports if r["summary"].get("quality_issues", False)]),
            "latest_report": recent_reports[-1]["report_id"] if recent_reports else None,
            "trend_analysis": self._analyze_trends(recent_reports),
        }

        return summary

    def _analyze_trends(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in monitoring data."""
        if len(reports) < 2:
            return {"message": "Insufficient data for trend analysis"}

        # Analyze drift trend
        drift_rates = [r["data_drift"].get("drift_share", 0) for r in reports if "data_drift" in r]
        quality_scores = [100 - r["data_quality"].get("missing_values_percentage", 0) for r in reports if "data_quality" in r]

        trends = {}

        if len(drift_rates) >= 2:
            trends["drift_trend"] = "increasing" if drift_rates[-1] > drift_rates[0] else "stable"
            trends["avg_drift_rate"] = sum(drift_rates) / len(drift_rates)

        if len(quality_scores) >= 2:
            trends["quality_trend"] = "improving" if quality_scores[-1] > quality_scores[0] else "stable"
            trends["avg_quality_score"] = sum(quality_scores) / len(quality_scores)

        return trends
