"""
Evidently report generator for automated ML monitoring reports.

This module provides automated report generation capabilities for ML monitoring,
including scheduled reports, comparative analysis, and stakeholder notifications.
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)

logger = logging.getLogger(__name__)


class EvidentlyReportGenerator:
    """
    Automated report generator for ML monitoring using Evidently.

    Provides scheduled report generation, comparative analysis,
    and automated stakeholder notifications.
    """

    def __init__(
        self,
        output_dir: str = "evidently_reports",
        archive_dir: str = "evidently_reports/archive",
        max_reports: int = 50,
        column_mapping: Optional[ColumnMapping] = None,
    ):
        """
        Initialize Evidently Report Generator.

        Args:
            output_dir: Directory for current reports
            archive_dir: Directory for archived reports
            max_reports: Maximum number of reports to keep
            column_mapping: Evidently column mapping configuration
        """
        self.output_dir = Path(output_dir)
        self.archive_dir = Path(archive_dir)
        self.max_reports = max_reports
        self.column_mapping = column_mapping or self._create_default_column_mapping()

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Evidently Report Generator")

    def _create_default_column_mapping(self) -> ColumnMapping:
        """Create default column mapping for Premier League predictions."""
        return ColumnMapping(
            target="result",
            prediction="predicted_result",
            numerical_features=[
                "home_odds", "draw_odds", "away_odds",
                "home_prob_margin_adj", "draw_prob_margin_adj", "away_prob_margin_adj"
            ],
            categorical_features=["home_team", "away_team", "month"],
        )

    def generate_daily_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        report_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate daily monitoring report.

        Args:
            reference_data: Reference dataset for comparison
            current_data: Current dataset to analyze
            report_date: Date for the report (defaults to today)

        Returns:
            Report generation results
        """
        if report_date is None:
            report_date = datetime.now()

        report_id = f"daily_report_{report_date.strftime('%Y%m%d')}"

        logger.info(f"Generating daily report: {report_id}")

        try:
            # Create comprehensive daily report
            report = Report(
                metrics=[
                    DataDriftPreset(),
                    DataQualityPreset(),
                    TargetDriftPreset(),
                    ClassificationPreset(),
                ]
            )

            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )

            # Save HTML report
            html_path = self.output_dir / f"{report_id}.html"
            report.save_html(str(html_path))

            # Save JSON report
            json_path = self.output_dir / f"{report_id}.json"
            report.save_json(str(json_path))

            # Extract key metrics
            report_dict = report.as_dict()
            summary = self._create_daily_summary(report_dict, report_date)

            # Save summary
            summary_path = self.output_dir / f"{report_id}_summary.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            # Manage report retention
            self._manage_report_retention()

            logger.info(f"Daily report generated successfully: {html_path}")

            return {
                "success": True,
                "report_id": report_id,
                "html_path": str(html_path),
                "json_path": str(json_path),
                "summary": summary,
                "timestamp": report_date,
            }

        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "report_id": report_id,
                "timestamp": report_date,
            }

    def generate_weekly_summary(
        self,
        reference_data: pd.DataFrame,
        week_data: List[pd.DataFrame],
        week_start: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate weekly summary report comparing multiple days.

        Args:
            reference_data: Reference dataset for comparison
            week_data: List of daily datasets for the week
            week_start: Start date of the week (defaults to last Monday)

        Returns:
            Weekly summary report results
        """
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())

        week_id = f"weekly_summary_{week_start.strftime('%Y%m%d')}"

        logger.info(f"Generating weekly summary: {week_id}")

        try:
            # Combine week data
            combined_data = pd.concat(week_data, ignore_index=True)

            # Create weekly report
            report = Report(
                metrics=[
                    DataDriftPreset(),
                    DataQualityPreset(),
                    TargetDriftPreset(),
                ]
            )

            report.run(
                reference_data=reference_data,
                current_data=combined_data,
                column_mapping=self.column_mapping
            )

            # Save reports
            html_path = self.output_dir / f"{week_id}.html"
            json_path = self.output_dir / f"{week_id}.json"

            report.save_html(str(html_path))
            report.save_json(str(json_path))

            # Create weekly analysis
            weekly_analysis = self._create_weekly_analysis(
                report.as_dict(), week_data, week_start
            )

            # Save analysis
            analysis_path = self.output_dir / f"{week_id}_analysis.json"
            import json
            with open(analysis_path, 'w') as f:
                json.dump(weekly_analysis, f, indent=2, default=str)

            logger.info(f"Weekly summary generated successfully: {html_path}")

            return {
                "success": True,
                "report_id": week_id,
                "html_path": str(html_path),
                "json_path": str(json_path),
                "analysis": weekly_analysis,
                "week_start": week_start,
            }

        except Exception as e:
            logger.error(f"Error generating weekly summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "report_id": week_id,
                "week_start": week_start,
            }

    def generate_comparison_report(
        self,
        reference_data: pd.DataFrame,
        dataset_a: pd.DataFrame,
        dataset_b: pd.DataFrame,
        comparison_name: str,
    ) -> Dict[str, Any]:
        """
        Generate comparison report between two datasets.

        Args:
            reference_data: Reference dataset
            dataset_a: First dataset for comparison
            dataset_b: Second dataset for comparison
            comparison_name: Name for the comparison

        Returns:
            Comparison report results
        """
        timestamp = datetime.now()
        report_id = f"comparison_{comparison_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating comparison report: {report_id}")

        try:
            # Generate report for dataset A
            report_a = Report(
                metrics=[
                    DataDriftPreset(),
                    DataQualityPreset(),
                ]
            )

            report_a.run(
                reference_data=reference_data,
                current_data=dataset_a,
                column_mapping=self.column_mapping
            )

            # Generate report for dataset B
            report_b = Report(
                metrics=[
                    DataDriftPreset(),
                    DataQualityPreset(),
                ]
            )

            report_b.run(
                reference_data=reference_data,
                current_data=dataset_b,
                column_mapping=self.column_mapping
            )

            # Save individual reports
            html_path_a = self.output_dir / f"{report_id}_dataset_a.html"
            html_path_b = self.output_dir / f"{report_id}_dataset_b.html"

            report_a.save_html(str(html_path_a))
            report_b.save_html(str(html_path_b))

            # Create comparison analysis
            comparison_analysis = self._create_comparison_analysis(
                report_a.as_dict(),
                report_b.as_dict(),
                comparison_name
            )

            # Save comparison analysis
            analysis_path = self.output_dir / f"{report_id}_comparison.json"
            import json
            with open(analysis_path, 'w') as f:
                json.dump(comparison_analysis, f, indent=2, default=str)

            logger.info(f"Comparison report generated successfully: {report_id}")

            return {
                "success": True,
                "report_id": report_id,
                "html_path_a": str(html_path_a),
                "html_path_b": str(html_path_b),
                "comparison_analysis": comparison_analysis,
                "timestamp": timestamp,
            }

        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "report_id": report_id,
                "timestamp": timestamp,
            }

    def _create_daily_summary(
        self,
        report_dict: Dict[str, Any],
        report_date: datetime
    ) -> Dict[str, Any]:
        """Create daily summary from Evidently report."""
        summary: Dict[str, Any] = {
            "date": report_date.strftime('%Y-%m-%d'),
            "data_drift": {},
            "data_quality": {},
            "target_drift": {},
            "classification": {},
            "alerts": [],
            "recommendations": [],
        }

        try:
            # Extract key metrics
            for metric in report_dict.get("metrics", []):
                metric_name = metric.get("metric", "")
                result = metric.get("result", {})

                if "DatasetDriftMetric" in metric_name:
                    summary["data_drift"] = {
                        "dataset_drift": result.get("dataset_drift", False),
                        "drift_share": result.get("drift_share", 0),
                        "drifted_features": result.get("number_of_drifted_columns", 0),
                    }

                elif "DatasetMissingValuesMetric" in metric_name:
                    summary["data_quality"] = {
                        "missing_percentage": result.get("missing_percentage", 0),
                        "missing_count": result.get("missing_count", 0),
                    }

            # Generate alerts and recommendations
            summary["alerts"] = self._generate_alerts(summary)
            summary["recommendations"] = self._generate_recommendations(summary)

        except Exception as e:
            logger.warning(f"Error creating daily summary: {str(e)}")
            summary["error"] = str(e)

        return summary

    def _create_weekly_analysis(
        self,
        report_dict: Dict[str, Any],
        week_data: List[pd.DataFrame],
        week_start: datetime,
    ) -> Dict[str, Any]:
        """Create weekly analysis from combined data."""
        analysis: Dict[str, Any] = {
            "week_start": week_start.strftime('%Y-%m-%d'),
            "days_analyzed": len(week_data),
            "total_samples": sum(len(df) for df in week_data),
            "daily_sample_counts": [len(df) for df in week_data],
            "trends": {},
            "summary": {},
        }

        try:            # Analyze daily trends
            daily_counts = [len(df) for df in week_data]
            analysis["trends"] = {
                "sample_trend": "stable",  # Could be enhanced with actual trend analysis
                "data_volume_consistent": len(set(daily_counts)) == 1,
            }

            # Create summary
            total_samples = analysis["total_samples"]
            if isinstance(total_samples, int) and len(week_data) > 0:
                analysis["summary"] = {
                    "avg_daily_samples": total_samples / len(week_data),
                    "min_daily_samples": min(daily_counts),
                    "max_daily_samples": max(daily_counts),
                }

        except Exception as e:
            logger.warning(f"Error creating weekly analysis: {str(e)}")
            analysis["error"] = str(e)

        return analysis

    def _create_comparison_analysis(
        self,
        report_a_dict: Dict[str, Any],
        report_b_dict: Dict[str, Any],
        comparison_name: str,
    ) -> Dict[str, Any]:
        """Create comparison analysis between two reports."""
        analysis: Dict[str, Any] = {
            "comparison_name": comparison_name,
            "timestamp": datetime.now().isoformat(),
            "dataset_a": {},
            "dataset_b": {},
            "differences": {},
            "recommendations": [],
        }

        try:
            # Extract metrics for both datasets
            for report_dict, key in [(report_a_dict, "dataset_a"), (report_b_dict, "dataset_b")]:
                for metric in report_dict.get("metrics", []):
                    metric_name = metric.get("metric", "")
                    result = metric.get("result", {})

                    if "DatasetDriftMetric" in metric_name:
                        analysis[key]["drift_share"] = result.get("drift_share", 0)
                        analysis[key]["dataset_drift"] = result.get("dataset_drift", False)            # Calculate differences
            if "dataset_a" in analysis and "dataset_b" in analysis:
                drift_a = analysis["dataset_a"].get("drift_share", 0)
                drift_b = analysis["dataset_b"].get("drift_share", 0)

                analysis["differences"] = {
                    "drift_difference": abs(drift_a - drift_b),
                    "higher_drift": "dataset_a" if drift_a > drift_b else "dataset_b",
                }

                # Generate recommendations
                if analysis["differences"]["drift_difference"] > 0.1:
                    if not isinstance(analysis["recommendations"], list):
                        analysis["recommendations"] = []
                    analysis["recommendations"].append(
                        f"Significant drift difference detected between datasets "
                        f"({analysis['differences']['drift_difference']:.2%})"
                    )

        except Exception as e:
            logger.warning(f"Error creating comparison analysis: {str(e)}")
            analysis["error"] = str(e)

        return analysis

    def _generate_alerts(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on summary data."""
        alerts: List[Dict[str, Any]] = []

        # Data drift alerts
        if summary["data_drift"].get("dataset_drift", False):
            alerts.append({
                "type": "data_drift",
                "severity": "high",
                "message": f"Dataset drift detected ({summary['data_drift']['drift_share']:.2%} features drifting)",
            })

        # Data quality alerts
        missing_pct = summary["data_quality"].get("missing_percentage", 0)
        if missing_pct > 5:
            alerts.append({
                "type": "data_quality",
                "severity": "medium",
                "message": f"High missing values detected ({missing_pct:.1f}%)",
            })

        return alerts

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on summary data."""
        recommendations: List[str] = []

        # Data drift recommendations
        if summary["data_drift"].get("dataset_drift", False):
            recommendations.append(
                "Consider retraining the model due to detected data drift"
            )

        # Data quality recommendations
        missing_pct = summary["data_quality"].get("missing_percentage", 0)
        if missing_pct > 5:
            recommendations.append(
                "Investigate data collection pipeline for missing value issues"
            )

        return recommendations

    def _manage_report_retention(self) -> None:
        """Manage report retention by archiving old reports."""
        try:
            # Get all HTML reports
            html_files = list(self.output_dir.glob("*.html"))

            if len(html_files) > self.max_reports:
                # Sort by creation time
                html_files.sort(key=lambda f: float(f.stat().st_mtime))

                # Archive oldest files
                files_to_archive = html_files[:-self.max_reports]

                for file_path in files_to_archive:
                    archive_path = self.archive_dir / file_path.name
                    shutil.move(str(file_path), str(archive_path))

                    # Also archive associated JSON files
                    json_file = file_path.with_suffix('.json')
                    if json_file.exists():
                        json_archive_path = self.archive_dir / json_file.name
                        shutil.move(str(json_file), str(json_archive_path))

                logger.info(f"Archived {len(files_to_archive)} old reports")

        except Exception as e:
            logger.warning(f"Error managing report retention: {str(e)}")

    def get_report_history(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get history of generated reports.

        Args:
            days_back: Number of days to look back

        Returns:
            List of report information
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        reports = []

        try:
            # Check current reports
            for html_file in self.output_dir.glob("*.html"):
                file_time = datetime.fromtimestamp(html_file.stat().st_mtime)
                if file_time >= cutoff_date:
                    reports.append({
                        "name": html_file.stem,
                        "path": str(html_file),
                        "created": file_time,
                        "size": html_file.stat().st_size,
                    })

            # Check archived reports
            for html_file in self.archive_dir.glob("*.html"):
                file_time = datetime.fromtimestamp(html_file.stat().st_mtime)
                if file_time >= cutoff_date:
                    reports.append({
                        "name": html_file.stem,
                        "path": str(html_file),
                        "created": file_time,
                        "size": html_file.stat().st_size,
                        "archived": True,
                    })

            # Sort by creation time (handle missing or non-comparable values)
            reports.sort(key=lambda r: str(r.get("created", "")), reverse=True)

        except Exception as e:
            logger.error(f"Error getting report history: {str(e)}")

        return reports
