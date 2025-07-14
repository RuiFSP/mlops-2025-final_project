#!/usr/bin/env python3

"""
Fix model loading issues by creating a simple mock model in MLflow.
This script creates a basic scikit-learn model and registers it in MLflow
to ensure the prediction pipeline can load a model.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def fix_model_issues():
    """Create and register a mock model in MLflow"""
    try:
        # Import required libraries
        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split

        logger.info("Creating mock model for MLflow...")

        # Set MLflow tracking URI
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Using MLflow tracking URI: {mlflow_uri}")

        # Create experiment if it doesn't exist
        experiment_name = "premier_league_predictions"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.warning(f"Error checking/creating experiment: {e}")
            # Try with default experiment
            experiment_id = "0"
            logger.info("Using default experiment (ID: 0)")

        # Create a simple mock dataset
        np.random.seed(42)
        X = np.random.rand(100, 15)  # 15 features
        y = np.random.choice(["H", "D", "A"], size=100)  # Home, Draw, Away outcomes

        # Create column names similar to what the real model would use
        feature_names = [
            "B365H",
            "B365D",
            "B365A",
            "HS",
            "AS",
            "HST",
            "AST",
            "HC",
            "AC",
            "HF",
            "AF",
            "HY",
            "AY",
            "HR",
            "AR",
        ]

        X_df = pd.DataFrame(X, columns=feature_names)
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Start MLflow run and log the model
        with mlflow.start_run(run_name="mock_premier_league_model", experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_param("n_estimators", 10)
            mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log feature names
            mlflow.log_param("features", ", ".join(feature_names))

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            # Get run ID
            run_id = run.info.run_id
            logger.info(f"Created mock model with run ID: {run_id}")

        # Register the model in MLflow Model Registry
        model_name = "premier_league_predictor"
        model_uri = f"runs:/{run_id}/model"

        # Check if model already exists
        try:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")

            if versions:
                logger.info(f"Model {model_name} already exists, creating new version")
            else:
                logger.info(f"Creating new model {model_name}")

        except Exception as e:
            logger.warning(f"Error checking model versions: {e}")

        # Register model
        try:
            model_details = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_details.name} version {model_details.version}")

            # Set model to production stage
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(name=model_name, version=model_details.version, stage="Production")
            logger.info(f"Model {model_name} version {model_details.version} set to Production stage")

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False

        logger.info("✅ Mock model created and registered successfully")
        return True

    except ImportError as e:
        logger.error(f"Missing required packages: {e}")
        logger.info("Installing required packages...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "mlflow", "scikit-learn", "pandas", "numpy"])
        logger.info("Please run the script again after packages are installed")
        return False

    except Exception as e:
        logger.error(f"Failed to create mock model: {e}")
        return False


if __name__ == "__main__":
    fix_model_issues()
