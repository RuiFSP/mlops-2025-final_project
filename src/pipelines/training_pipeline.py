"""
Training pipeline for Premier League match prediction model.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training pipeline for the Premier League prediction model."""

    def __init__(self, data_path: str = None):
        """Initialize the training pipeline."""
        # Use environment variable or default for data path
        self.data_path = data_path or os.getenv("TRAINING_DATA_PATH", "data/real_data/premier_league_matches.parquet")

        self.model = None
        self.feature_columns = [
            "B365H",
            "B365D",
            "B365A",  # Bet365 odds
            "HS",
            "AS",  # Shots
            "HST",
            "AST",  # Shots on target
            "HC",
            "AC",  # Corners
            "HF",
            "AF",  # Fouls
            "HY",
            "AY",  # Yellow cards
            "HR",
            "AR",  # Red cards
        ]
        self.target_column = "FTR"  # Full Time Result (H, D, A)

        # Set up MLflow with environment-aware configuration
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Set up MLflow tracking with environment-aware configuration."""
        # Get MLflow tracking URI from environment or use default
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # Set artifact root - use relative path that works in both local and Docker
        artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")
        os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root

        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow artifact root: {artifact_root}")

        # Test MLflow connectivity
        try:
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()
            logger.info("✅ MLflow connection successful")
        except Exception as e:
            logger.warning(f"⚠️ MLflow connection test failed: {e}")
            logger.info("Continuing with training - MLflow logging may fail")

    def _ensure_experiment_exists(self, experiment_name: str) -> None:
        """Ensure experiment exists by name. Create if not exists. Return None (always use name)."""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"Creating new experiment: {experiment_name}")
                client.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        except Exception as e:
            logger.error(f"Failed to ensure experiment exists: {e}")
            logger.info("Falling back to MLflow default experiment handling")
        return None

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the training data."""
        logger.info(f"Loading data from {self.data_path}")
        try:
            if self.data_path.endswith(".parquet"):
                df = pd.read_parquet(self.data_path)
            else:
                df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
            logger.debug(f"Sample data:\n{df.head()}\n...")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for training."""
        logger.info("Preprocessing data...")
        initial_shape = df.shape
        # Filter to only include matches with results and odds
        df = df.dropna(subset=[self.target_column, "B365H", "B365D", "B365A"])
        logger.info(f"Dropped rows with missing target/odds: {initial_shape[0] - df.shape[0]} rows removed")
        # Fill missing values in feature columns with 0
        for col in self.feature_columns:
            if col not in ["B365H", "B365D", "B365A"]:
                missing = df[col].isna().sum()
                if missing > 0:
                    logger.warning(f"Filling {missing} missing values in {col} with 0")
                df[col] = df[col].fillna(0)
        logger.info(f"Preprocessed data shape: {df.shape}")
        logger.debug(f"Preprocessed sample:\n{df.head()}\n...")
        return df

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train the prediction model."""
        logger.info("Training model...")
        logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        logger.debug(f"Feature columns: {list(X_train.columns)}")
        logger.debug(f"Target distribution: {y_train.value_counts().to_dict()}")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        return model

    def evaluate_model(self, model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        logger.info(f"Test samples: {X_test.shape[0]}")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        feature_importance = dict(zip(self.feature_columns, model.feature_importances_, strict=False))
        class_report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.debug(f"Classification report: {class_report}")
        logger.debug(f"Feature importance: {feature_importance}")
        metrics = {
            "accuracy": accuracy,
            "feature_importance": feature_importance,
            "classification_report": class_report,
        }
        return metrics

    def save_model_locally(self, model, run_id: str = None) -> str:
        """Save model to local filesystem with timestamp."""
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_suffix = f"_run_{run_id}" if run_id else ""
        model_filename = f"model_{timestamp}{run_suffix}.pkl"
        model_path = models_dir / model_filename

        # Save model
        import joblib

        joblib.dump(model, model_path)
        logger.info(f"Model saved locally to: {model_path}")

        return str(model_path)

    def run_training(self, experiment_name: str = "premier_league_prediction") -> str:
        """Run the complete training pipeline."""
        logger.info("Starting training pipeline...")
        logger.info(f"Experiment name: {experiment_name}")

        # Ensure experiment exists (by name)
        self._ensure_experiment_exists(experiment_name)

        # Always set experiment by name
        mlflow.set_experiment(experiment_name)

        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info(f"MLflow run started: {run_id}")

                # Load and preprocess data
                df = self.load_data()
                df = self.preprocess_data(df)

                # Prepare features and target
                logger.info("Preparing features and target...")
                X = df[self.feature_columns]
                y = df[self.target_column]
                logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

                # Split data
                logger.info("Splitting data into train and test sets...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

                # Train model
                model = self.train_model(X_train, y_train)

                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)

                # Log parameters and metrics to MLflow
                try:
                    logger.info("Logging parameters and metrics to MLflow...")
                    mlflow.log_params(
                        {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "random_state": 42,
                            "data_path": self.data_path,
                            "feature_count": len(self.feature_columns),
                        }
                    )

                    mlflow.log_metrics(
                        {
                            "accuracy": metrics["accuracy"],
                            "test_samples": len(X_test),
                            "train_samples": len(X_train),
                        }
                    )

                    # Log feature importance
                    for feature, importance in metrics["feature_importance"].items():
                        mlflow.log_metric(f"feature_importance_{feature}", importance)

                    # Log model artifact to MLflow
                    mlflow.sklearn.log_model(model, "model")
                    logger.info("Model artifact logged to MLflow")

                    # Register model if it's good enough
                    if metrics["accuracy"] > 0.6:
                        mlflow.register_model(f"runs:/{run_id}/model", "premier_league_predictor")
                        logger.info("Model registered in MLflow")
                    else:
                        logger.warning(f"Model accuracy {metrics['accuracy']:.4f} is below registration threshold.")

                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")
                    logger.info("Continuing with local model save...")

                # Always save model locally as backup
                local_model_path = self.save_model_locally(model, run_id)

                self.model = model
                logger.info(f"Training pipeline completed. Run ID: {run_id}")
                logger.info(f"Model saved locally at: {local_model_path}")

                return run_id

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            # Still try to save model locally if possible
            try:
                if hasattr(self, "model") and self.model is not None:
                    local_model_path = self.save_model_locally(self.model)
                    logger.info(f"Model saved locally as backup: {local_model_path}")
            except Exception as save_error:
                logger.error(f"Failed to save model locally: {save_error}")
            raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            # Uncomment below to also log to file
            # logging.FileHandler("training_pipeline.log")
        ],
    )

    # Run training pipeline
    logger.info("==== Premier League Training Pipeline START ====")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Environment: {'Docker' if os.path.exists('/.dockerenv') else 'Local'}")

    pipeline = TrainingPipeline()
    try:
        run_id = pipeline.run_training()
        logger.info(f"Training completed with run ID: {run_id}")
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
    logger.info("==== Premier League Training Pipeline END ====")
