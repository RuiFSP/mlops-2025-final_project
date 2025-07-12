"""
Automated retraining monitor for the Premier League Match Predictor.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import mlflow
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_count: int
    evaluation_date: datetime
    model_version: str


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    min_accuracy_threshold: float = 0.55  # Retrain if accuracy drops below 55%
    min_prediction_count: int = 100  # Need at least 100 predictions to evaluate
    evaluation_window_days: int = 7  # Evaluate performance over last 7 days
    max_model_age_days: int = 30  # Force retrain if model is older than 30 days
    performance_degradation_threshold: float = 0.05  # Retrain if accuracy drops by 5%
    consecutive_poor_performance_limit: int = 3  # Retrain after 3 consecutive poor evaluations


class RetrainingMonitor:
    """Monitor model performance and trigger automated retraining."""
    
    def __init__(self, config: Optional[RetrainingConfig] = None):
        """Initialize the retraining monitor."""
        self.config = config or RetrainingConfig()
        
        # Set up database connection
        self.db_url = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
        )
        self.engine = create_engine(self.db_url)
        
        # Set up MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        
        # Initialize monitoring tables
        self._init_monitoring_tables()
        
        logger.info("✅ Retraining monitor initialized")
    
    def _init_monitoring_tables(self):
        """Initialize monitoring tables in the database."""
        try:
            with self.engine.connect() as conn:
                # Create performance_monitoring table
                monitoring_sql = """
                    CREATE TABLE IF NOT EXISTS performance_monitoring (
                        id SERIAL PRIMARY KEY,
                        model_version VARCHAR(100) NOT NULL,
                        accuracy FLOAT NOT NULL,
                        precision_score FLOAT NOT NULL,
                        recall_score FLOAT NOT NULL,
                        f1_score FLOAT NOT NULL,
                        prediction_count INTEGER NOT NULL,
                        evaluation_date TIMESTAMP NOT NULL,
                        evaluation_window_days INTEGER NOT NULL,
                        baseline_accuracy FLOAT,
                        performance_degradation FLOAT,
                        requires_retraining BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """
                
                # Create retraining_history table
                history_sql = """
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id SERIAL PRIMARY KEY,
                        trigger_reason TEXT NOT NULL,
                        old_model_version VARCHAR(100),
                        new_model_version VARCHAR(100),
                        old_accuracy FLOAT,
                        new_accuracy FLOAT,
                        training_duration_seconds INTEGER,
                        retraining_date TIMESTAMP NOT NULL,
                        success BOOLEAN DEFAULT FALSE,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """
                
                conn.execute(text(monitoring_sql))
                conn.execute(text(history_sql))
                conn.commit()
                
                logger.info("✅ Monitoring tables initialized")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring tables: {e}")
            raise
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            client = mlflow.tracking.MlflowClient()
            model_name = "premier_league_predictor"
            
            # Get latest model version
            latest_version = client.get_latest_versions(model_name, stages=["None"])
            if not latest_version:
                raise ValueError("No model versions found")
            
            model_version = latest_version[0]
            
            # Get model metadata
            run_info = client.get_run(model_version.run_id)
            
            return {
                "version": model_version.version,
                "run_id": model_version.run_id,
                "creation_date": datetime.fromtimestamp(run_info.info.start_time / 1000),
                "accuracy": run_info.data.metrics.get("accuracy", 0.0),
                "status": model_version.status
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get current model info: {e}")
            return {}
    
    def collect_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Collect recent predictions for performance evaluation."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.engine.connect() as conn:
                query = """
                    SELECT 
                        p.home_team,
                        p.away_team,
                        p.prediction,
                        p.confidence,
                        p.prediction_date,
                        p.model_version,
                        b.result as actual_result,
                        b.bet_date
                    FROM predictions p
                    LEFT JOIN bets b ON p.match_id = b.match_id
                    WHERE p.prediction_date >= %(cutoff_date)s
                    ORDER BY p.prediction_date DESC
                """
                
                df = pd.read_sql(query, conn, params={"cutoff_date": cutoff_date})
                
                logger.info(f"📊 Collected {len(df)} recent predictions for evaluation")
                return df
                
        except Exception as e:
            logger.error(f"❌ Failed to collect recent predictions: {e}")
            return pd.DataFrame()
    
    def evaluate_model_performance(self, predictions_df: pd.DataFrame) -> Optional[PerformanceMetrics]:
        """Evaluate model performance based on recent predictions."""
        if len(predictions_df) < self.config.min_prediction_count:
            logger.warning(f"⚠️ Insufficient predictions for evaluation: {len(predictions_df)} < {self.config.min_prediction_count}")
            return None
        
        # Filter predictions with actual results
        evaluated_df = predictions_df[predictions_df['actual_result'].notna()].copy()
        
        if len(evaluated_df) == 0:
            logger.warning("⚠️ No predictions with actual results for evaluation")
            return None
        
        # Convert actual results to match prediction format
        evaluated_df['actual_prediction'] = evaluated_df['actual_result'].map({
            'W': 'H',  # Win -> Home
            'L': 'A',  # Loss -> Away
            'D': 'D'   # Draw -> Draw
        })
        
        # Calculate metrics
        y_true = evaluated_df['actual_prediction']
        y_pred = evaluated_df['prediction']
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=report['macro avg']['precision'],
            recall=report['macro avg']['recall'],
            f1_score=report['macro avg']['f1-score'],
            prediction_count=len(evaluated_df),
            evaluation_date=datetime.now(),
            model_version=evaluated_df['model_version'].iloc[0] if len(evaluated_df) > 0 else "unknown"
        )
        
        logger.info(f"📈 Performance evaluation: Accuracy={accuracy:.3f}, Predictions={len(evaluated_df)}")
        return metrics
    
    def check_retraining_triggers(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Check if retraining should be triggered based on performance metrics."""
        triggers = []
        model_info = self.get_current_model_info()
        
        # Check accuracy threshold
        if metrics.accuracy < self.config.min_accuracy_threshold:
            triggers.append(f"Accuracy below threshold: {metrics.accuracy:.3f} < {self.config.min_accuracy_threshold}")
        
        # Check model age
        if model_info and 'creation_date' in model_info:
            model_age = (datetime.now() - model_info['creation_date']).days
            if model_age > self.config.max_model_age_days:
                triggers.append(f"Model too old: {model_age} days > {self.config.max_model_age_days} days")
        
        # Check performance degradation
        if model_info and 'accuracy' in model_info:
            baseline_accuracy = model_info['accuracy']
            degradation = baseline_accuracy - metrics.accuracy
            if degradation > self.config.performance_degradation_threshold:
                triggers.append(f"Performance degradation: {degradation:.3f} > {self.config.performance_degradation_threshold}")
        
        # Check consecutive poor performance
        consecutive_poor = self._count_consecutive_poor_performance()
        if consecutive_poor >= self.config.consecutive_poor_performance_limit:
            triggers.append(f"Consecutive poor performance: {consecutive_poor} >= {self.config.consecutive_poor_performance_limit}")
        
        return {
            "should_retrain": len(triggers) > 0,
            "triggers": triggers,
            "metrics": metrics,
            "model_info": model_info
        }
    
    def _count_consecutive_poor_performance(self) -> int:
        """Count consecutive poor performance evaluations."""
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT requires_retraining 
                    FROM performance_monitoring 
                    ORDER BY evaluation_date DESC 
                    LIMIT 10
                """
                
                result = conn.execute(text(query)).fetchall()
                
                consecutive_count = 0
                for row in result:
                    if row[0]:  # requires_retraining is True
                        consecutive_count += 1
                    else:
                        break
                
                return consecutive_count
                
        except Exception as e:
            logger.error(f"❌ Failed to count consecutive poor performance: {e}")
            return 0
    
    def record_performance_evaluation(self, metrics: PerformanceMetrics, triggers: Dict[str, Any]):
        """Record performance evaluation in the database."""
        try:
            with self.engine.connect() as conn:
                model_info = triggers.get('model_info', {})
                baseline_accuracy = model_info.get('accuracy', 0.0)
                degradation = baseline_accuracy - metrics.accuracy if baseline_accuracy > 0 else 0.0
                
                insert_sql = """
                    INSERT INTO performance_monitoring (
                        model_version, accuracy, precision_score, recall_score, f1_score,
                        prediction_count, evaluation_date, evaluation_window_days,
                        baseline_accuracy, performance_degradation, requires_retraining
                    ) VALUES (
                        :model_version, :accuracy, :precision_score, :recall_score, :f1_score,
                        :prediction_count, :evaluation_date, :evaluation_window_days,
                        :baseline_accuracy, :performance_degradation, :requires_retraining
                    )
                """
                
                conn.execute(text(insert_sql), {
                    "model_version": metrics.model_version,
                    "accuracy": metrics.accuracy,
                    "precision_score": metrics.precision,
                    "recall_score": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "prediction_count": metrics.prediction_count,
                    "evaluation_date": metrics.evaluation_date,
                    "evaluation_window_days": self.config.evaluation_window_days,
                    "baseline_accuracy": baseline_accuracy,
                    "performance_degradation": degradation,
                    "requires_retraining": triggers['should_retrain']
                })
                
                conn.commit()
                logger.info("✅ Performance evaluation recorded")
                
        except Exception as e:
            logger.error(f"❌ Failed to record performance evaluation: {e}")
    
    def trigger_retraining(self, trigger_info: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the retraining process."""
        start_time = datetime.now()
        logger.info("🔄 Starting automated retraining process...")
        
        try:
            # Import here to avoid circular imports
            from src.pipelines.training_pipeline import TrainingPipeline
            
            # Record retraining start
            retraining_record = {
                "trigger_reason": "; ".join(trigger_info['triggers']),
                "old_model_version": trigger_info['model_info'].get('version', 'unknown'),
                "old_accuracy": trigger_info['model_info'].get('accuracy', 0.0),
                "retraining_date": start_time,
                "success": False
            }
            
            # Start retraining
            training_pipeline = TrainingPipeline()
            run_id = training_pipeline.run_training()
            
            # Get the training results
            with mlflow.start_run(run_id=run_id):
                accuracy = mlflow.get_metric("accuracy")
                model_version = mlflow.get_artifact_uri("model")
            
            result = {
                "run_id": run_id,
                "accuracy": accuracy,
                "model_version": model_version
            }
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update record with success
            retraining_record.update({
                "new_model_version": result.get('model_version', 'unknown'),
                "new_accuracy": result.get('accuracy', 0.0),
                "training_duration_seconds": int(duration),
                "success": True
            })
            
            # Record in database
            self._record_retraining_history(retraining_record)
            
            logger.info(f"✅ Retraining completed successfully in {duration:.1f} seconds")
            logger.info(f"📊 New model accuracy: {result.get('accuracy', 0.0):.3f}")
            
            return {
                "success": True,
                "duration_seconds": duration,
                "new_model_version": result.get('model_version'),
                "new_accuracy": result.get('accuracy'),
                "message": "Retraining completed successfully"
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record failure
            retraining_record.update({
                "training_duration_seconds": int(duration),
                "success": False,
                "error_message": str(e)
            })
            
            self._record_retraining_history(retraining_record)
            
            logger.error(f"❌ Retraining failed after {duration:.1f} seconds: {e}")
            
            return {
                "success": False,
                "duration_seconds": duration,
                "error": str(e),
                "message": "Retraining failed"
            }
    
    def _record_retraining_history(self, record: Dict[str, Any]):
        """Record retraining history in the database."""
        try:
            with self.engine.connect() as conn:
                insert_sql = """
                    INSERT INTO retraining_history (
                        trigger_reason, old_model_version, new_model_version,
                        old_accuracy, new_accuracy, training_duration_seconds,
                        retraining_date, success, error_message
                    ) VALUES (
                        :trigger_reason, :old_model_version, :new_model_version,
                        :old_accuracy, :new_accuracy, :training_duration_seconds,
                        :retraining_date, :success, :error_message
                    )
                """
                
                conn.execute(text(insert_sql), record)
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to record retraining history: {e}")
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle."""
        logger.info("🔍 Starting monitoring cycle...")
        
        try:
            # Collect recent predictions
            predictions_df = self.collect_recent_predictions(self.config.evaluation_window_days)
            
            if len(predictions_df) == 0:
                logger.warning("⚠️ No recent predictions found for evaluation")
                return {"status": "no_data", "message": "No recent predictions for evaluation"}
            
            # Evaluate performance
            metrics = self.evaluate_model_performance(predictions_df)
            
            if metrics is None:
                logger.warning("⚠️ Unable to evaluate model performance")
                return {"status": "evaluation_failed", "message": "Unable to evaluate model performance"}
            
            # Check retraining triggers
            triggers = self.check_retraining_triggers(metrics)
            
            # Record evaluation
            self.record_performance_evaluation(metrics, triggers)
            
            # Trigger retraining if needed
            if triggers['should_retrain']:
                logger.info("🚨 Retraining triggered!")
                for trigger in triggers['triggers']:
                    logger.info(f"   - {trigger}")
                
                retraining_result = self.trigger_retraining(triggers)
                
                return {
                    "status": "retraining_triggered",
                    "metrics": {
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "prediction_count": metrics.prediction_count
                    },
                    "triggers": triggers['triggers'],
                    "retraining_result": retraining_result
                }
            else:
                logger.info("✅ Model performance is acceptable, no retraining needed")
                return {
                    "status": "performance_acceptable",
                    "metrics": {
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "prediction_count": metrics.prediction_count
                    },
                    "message": "Model performance is acceptable"
                }
                
        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of recent monitoring activities."""
        try:
            with self.engine.connect() as conn:
                # Get recent performance evaluations
                perf_query = """
                    SELECT 
                        model_version, accuracy, precision_score, recall_score, f1_score,
                        prediction_count, evaluation_date, requires_retraining
                    FROM performance_monitoring 
                    ORDER BY evaluation_date DESC 
                    LIMIT 10
                """
                
                perf_results = conn.execute(text(perf_query)).fetchall()
                
                # Get recent retraining history
                retrain_query = """
                    SELECT 
                        trigger_reason, old_model_version, new_model_version,
                        old_accuracy, new_accuracy, training_duration_seconds,
                        retraining_date, success
                    FROM retraining_history 
                    ORDER BY retraining_date DESC 
                    LIMIT 5
                """
                
                retrain_results = conn.execute(text(retrain_query)).fetchall()
                
                return {
                    "recent_evaluations": [
                        {
                            "model_version": row[0],
                            "accuracy": row[1],
                            "precision": row[2],
                            "recall": row[3],
                            "f1_score": row[4],
                            "prediction_count": row[5],
                            "evaluation_date": row[6],
                            "requires_retraining": row[7]
                        }
                        for row in perf_results
                    ],
                    "recent_retraining": [
                        {
                            "trigger_reason": row[0],
                            "old_model_version": row[1],
                            "new_model_version": row[2],
                            "old_accuracy": row[3],
                            "new_accuracy": row[4],
                            "training_duration_seconds": row[5],
                            "retraining_date": row[6],
                            "success": row[7]
                        }
                        for row in retrain_results
                    ]
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get monitoring summary: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run monitoring cycle
    monitor = RetrainingMonitor()
    result = monitor.run_monitoring_cycle()
    
    print(f"Monitoring result: {result}") 