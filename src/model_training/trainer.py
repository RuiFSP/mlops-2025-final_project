"""Model trainer for Premier League match prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of Premier League match prediction models."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('random_forest', 'xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        if df.empty:
            logger.warning("DataFrame is empty, returning as-is")
            return df
            
        # Select features (only use data available before the match)
        feature_columns = [
            'home_team', 'away_team', 'month'
        ]
        
        # Add odds if available (these are set before the match)
        if 'home_odds' in df.columns:
            feature_columns.append('home_odds')
        if 'draw_odds' in df.columns:
            feature_columns.append('draw_odds')
        if 'away_odds' in df.columns:
            feature_columns.append('away_odds')
        
        # Use only columns that exist in the DataFrame
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No feature columns available")
            return pd.DataFrame()
            
        features_df = df[available_columns].copy()
        
        # Encode categorical variables
        for col in ['home_team', 'away_team']:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    # Handle unseen categories
                    features_df[col] = self._encode_with_unknown(features_df[col], self.label_encoders[col])
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Prepared features shape: {features_df.shape}")
        return features_df
    
    def _encode_with_unknown(self, series: pd.Series, encoder: LabelEncoder) -> pd.Series:
        """Encode series with label encoder, handling unknown categories.
        
        Args:
            series: Series to encode
            encoder: Fitted label encoder
            
        Returns:
            Encoded series
        """
        # Map unknown categories to a special value
        known_categories = set(encoder.classes_)
        series_mapped = series.map(lambda x: x if x in known_categories else 'UNKNOWN')
        
        # Add 'UNKNOWN' to encoder if not present
        if 'UNKNOWN' not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, 'UNKNOWN')
        
        return pd.Series(encoder.transform(series_mapped.astype(str)), index=series.index)
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Any:
        """Train the prediction model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            
        Returns:
            Trained model
        """
        logger.info(f"Starting model training with {self.model_type}")
        
        # Start MLflow run
        with mlflow.start_run():
            # Prepare features
            X_train = self.prepare_features(train_data)
            X_val = self.prepare_features(val_data)
            
            # Prepare target
            y_train = train_data['result'] if 'result' in train_data.columns else pd.Series([], dtype=str)
            y_val = val_data['result'] if 'result' in val_data.columns else pd.Series([], dtype=str)
            
            if X_train.empty or y_train.empty:
                logger.warning("Training data is empty, returning None")
                return None
            
            # Initialize model
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Default to RandomForest
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if not X_val.empty else np.array([])
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            
            # Evaluate on validation set
            if not X_val.empty and not y_val.empty:
                val_predictions = self.model.predict(X_val_scaled)
                val_accuracy = (val_predictions == y_val).mean()
                
                # Log metrics
                mlflow.log_metric("val_accuracy", val_accuracy)
                
                # Log classification report
                report = classification_report(y_val, val_predictions, output_dict=True)
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(f"{class_name}_{metric_name}", value)
                
                logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Cross-validation on training set
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
            mlflow.log_metric("cv_std_accuracy", cv_scores.std())
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            # Save model locally
            self.save_model("models/")
            
            logger.info(f"Model training completed. CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return self.model
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model and preprocessing components.
        
        Args:
            save_path: Directory to save the model
        """
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = save_dir / "model.pkl"
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Save preprocessing components
        if self.label_encoders:
            encoders_path = save_dir / "label_encoders.pkl"
            joblib.dump(self.label_encoders, encoders_path)
            logger.info(f"Label encoders saved to {encoders_path}")
        
        scaler_path = save_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model and preprocessing components.
        
        Args:
            model_path: Path to the saved model directory
        """
        model_dir = Path(model_path)
        
        # Load model
        model_file = model_dir / "model.pkl"
        if model_file.exists():
            self.model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
        
        # Load preprocessing components
        encoders_file = model_dir / "label_encoders.pkl"
        if encoders_file.exists():
            self.label_encoders = joblib.load(encoders_file)
            logger.info(f"Label encoders loaded from {encoders_file}")
        
        scaler_file = model_dir / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info(f"Scaler loaded from {scaler_file}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        X = self.prepare_features(data)
        
        if X.empty:
            logger.warning("No features available for prediction")
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
