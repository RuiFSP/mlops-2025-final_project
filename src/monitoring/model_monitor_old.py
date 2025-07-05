"""Model monitoring for Premier League match prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and data drift using EvidentlyAI."""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """Initialize the model monitor.
        
        Args:
            reference_data: Reference dataset for drift detection
        """
        self.reference_data = reference_data
        self.reports_dir = Path("monitoring_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Define column mapping for football data
        self.column_mapping = ColumnMapping(
            target='result',
            prediction='predicted_result',
            numerical_features=['goal_difference', 'total_goals', 'month'],
            categorical_features=['home_team', 'away_team']
        )
    
    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set reference data for drift detection.
        
        Args:
            reference_data: Reference dataset
        """
        self.reference_data = reference_data
        logger.info(f"Reference data set with {len(reference_data)} samples")
    
    def generate_data_drift_report(self, current_data: pd.DataFrame, save_html: bool = True) -> Dict[str, Any]:
        """Generate data drift report comparing current data to reference.
        
        Args:
            current_data: Current dataset to compare
            save_html: Whether to save HTML report
            
        Returns:
            Dictionary containing drift metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        logger.info("Generating data drift report...")
        
        # Create data drift report
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        # Run the report
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.reports_dir / f"data_drift_report_{timestamp}.html"
            data_drift_report.save_html(str(html_path))
            logger.info(f"Data drift report saved to {html_path}")
        
        # Extract metrics
        report_dict = data_drift_report.as_dict()
        
        # Parse drift results
        drift_metrics = {}
        try:
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    drift_metrics['dataset_drift'] = result.get('dataset_drift', False)
                    drift_metrics['drift_score'] = result.get('drift_score', 0.0)
                    drift_metrics['number_of_drifted_columns'] = result.get('number_of_drifted_columns', 0)
                    
        except Exception as e:
            logger.error(f"Error parsing drift metrics: {e}")
            drift_metrics = {'error': str(e)}
        
        return drift_metrics
    
    def generate_target_drift_report(self, current_data: pd.DataFrame, save_html: bool = True) -> Dict[str, Any]:
        """Generate target drift report.
        
        Args:
            current_data: Current dataset to compare
            save_html: Whether to save HTML report
            
        Returns:
            Dictionary containing target drift metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        logger.info("Generating target drift report...")
        
        # Create target drift report
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        # Run the report
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.reports_dir / f"target_drift_report_{timestamp}.html"
            target_drift_report.save_html(str(html_path))
            logger.info(f"Target drift report saved to {html_path}")
        
        # Extract metrics
        report_dict = target_drift_report.as_dict()
        
        # Parse target drift results
        target_metrics = {}
        try:
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                if metric.get('metric') == 'TargetDriftMetric':
                    result = metric.get('result', {})
                    target_metrics['target_drift'] = result.get('drift_detected', False)
                    target_metrics['drift_score'] = result.get('drift_score', 0.0)
                    
        except Exception as e:
            logger.error(f"Error parsing target drift metrics: {e}")
            target_metrics = {'error': str(e)}
        
        return target_metrics
    
    def generate_data_quality_report(self, data: pd.DataFrame, save_html: bool = True) -> Dict[str, Any]:
        """Generate data quality report.
        
        Args:
            data: Dataset to analyze
            save_html: Whether to save HTML report
            
        Returns:
            Dictionary containing data quality metrics
        """
        logger.info("Generating data quality report...")
        
        # Create data quality report
        data_quality_report = Report(metrics=[
            DataQualityPreset(),
        ])
        
        # Run the report
        data_quality_report.run(
            reference_data=data,
            current_data=None,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.reports_dir / f"data_quality_report_{timestamp}.html"
            data_quality_report.save_html(str(html_path))
            logger.info(f"Data quality report saved to {html_path}")
        
        # Extract metrics
        report_dict = data_quality_report.as_dict()
        
        # Parse data quality results
        quality_metrics = {}
        try:
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                metric_name = metric.get('metric', '')
                if 'Missing' in metric_name:
                    result = metric.get('result', {})
                    quality_metrics['missing_values'] = result
                elif 'Duplicated' in metric_name:
                    result = metric.get('result', {})
                    quality_metrics['duplicated_rows'] = result
                    
        except Exception as e:
            logger.error(f"Error parsing data quality metrics: {e}")
            quality_metrics = {'error': str(e)}
        
        return quality_metrics
    
    def generate_model_performance_report(
        self, 
        data: pd.DataFrame, 
        predictions: np.ndarray,
        save_html: bool = True
    ) -> Dict[str, Any]:
        """Generate model performance report.
        
        Args:
            data: Dataset with actual results
            predictions: Model predictions
            save_html: Whether to save HTML report
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("Generating model performance report...")
        
        # Add predictions to data
        data_with_predictions = data.copy()
        data_with_predictions['predicted_result'] = predictions
        
        # Create performance report
        performance_report = Report(metrics=[
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(),
            ClassificationQualityMetric(),
        ])
        
        # Run the report
        performance_report.run(
            reference_data=data_with_predictions,
            current_data=None,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.reports_dir / f"model_performance_report_{timestamp}.html"
            performance_report.save_html(str(html_path))
            logger.info(f"Model performance report saved to {html_path}")
        
        # Extract metrics
        report_dict = performance_report.as_dict()
        
        # Parse performance results
        performance_metrics = {}
        try:
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                metric_name = metric.get('metric', '')
                result = metric.get('result', {})
                
                if 'Quality' in metric_name:
                    performance_metrics['accuracy'] = result.get('accuracy', 0.0)
                    performance_metrics['precision'] = result.get('precision', 0.0)
                    performance_metrics['recall'] = result.get('recall', 0.0)
                    performance_metrics['f1'] = result.get('f1', 0.0)
                    
        except Exception as e:
            logger.error(f"Error parsing performance metrics: {e}")
            performance_metrics = {'error': str(e)}
        
        return performance_metrics
    
    def check_for_alerts(self, drift_metrics: Dict[str, Any], threshold: float = 0.5) -> List[str]:
        """Check for alerts based on drift metrics.
        
        Args:
            drift_metrics: Drift metrics from report
            threshold: Threshold for triggering alerts
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        # Check dataset drift
        if drift_metrics.get('dataset_drift', False):
            alerts.append("🚨 Dataset drift detected!")
        
        # Check drift score
        drift_score = drift_metrics.get('drift_score', 0.0)
        if drift_score > threshold:
            alerts.append(f"⚠️ High drift score: {drift_score:.3f} (threshold: {threshold})")
        
        # Check number of drifted columns
        drifted_columns = drift_metrics.get('number_of_drifted_columns', 0)
        if drifted_columns > 0:
            alerts.append(f"📊 {drifted_columns} columns showing drift")
        
        return alerts
    
    def save_monitoring_summary(self, summary: Dict[str, Any]):
        """Save monitoring summary to JSON file.
        
        Args:
            summary: Monitoring summary dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.reports_dir / f"monitoring_summary_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Monitoring summary saved to {summary_path}")


def monitor_model_performance(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    predictions: np.ndarray
) -> Dict[str, Any]:
    """Main function to monitor model performance.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        predictions: Model predictions
        
    Returns:
        Monitoring summary
    """
    monitor = ModelMonitor(reference_data)
    
    # Generate reports
    data_drift = monitor.generate_data_drift_report(current_data)
    target_drift = monitor.generate_target_drift_report(current_data)
    data_quality = monitor.generate_data_quality_report(current_data)
    performance = monitor.generate_model_performance_report(current_data, predictions)
    
    # Check for alerts
    alerts = monitor.check_for_alerts(data_drift)
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_drift': data_drift,
        'target_drift': target_drift,
        'data_quality': data_quality,
        'model_performance': performance,
        'alerts': alerts,
        'reference_data_size': len(reference_data),
        'current_data_size': len(current_data)
    }
    
    # Save summary
    monitor.save_monitoring_summary(summary)
    
    return summary
