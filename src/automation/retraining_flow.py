"""
Automated Retraining Flow - Prefect-based automated model retraining workflow.

This module provides production-ready automated retraining flows that handle
data preparation, model training, validation, and deployment decisions.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.logging import get_run_logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Import configuration to ensure MLflow is properly configured
try:
    from src.config import config
    # Ensure MLflow environment is set
    if hasattr(config, 'mlflow_tracking_uri'):
        os.environ['MLFLOW_TRACKING_URI'] = config.mlflow_tracking_uri
except ImportError:
    # Fallback configuration
    os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

from src.data_preprocessing.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.model_training.trainer import ModelTrainer

# Import RetrainingConfig for type annotations
if sys.version_info >= (3, 10):
    from typing import TYPE_CHECKING
else:
    TYPE_CHECKING = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@task(name="backup-current-model")
def backup_current_model(
    model_path: str, backup_dir: str, backup_reason: str = "automated_retraining"
) -> str:
    """Backup the current model before retraining."""
    logger = get_run_logger()

    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return ""

    # Create backup directory
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem
    backup_filename = f"{model_name}_backup_{timestamp}_{backup_reason}.pkl"
    backup_file_path = backup_path / backup_filename

    # Copy model file
    import shutil

    shutil.copy2(model_path, backup_file_path)

    logger.info(f"Model backed up to: {backup_file_path}")
    return str(backup_file_path)


@task(name="prepare-retraining-data")
def prepare_retraining_data(
    original_data_path: str,
    new_data_path: str | None = None,
    data_window_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Prepare data for retraining with new observations."""
    logger = get_run_logger()

    # Load original training data
    data_loader = DataLoader(original_data_path)
    # Try to load data - first try as a data directory, then as a parquet file
    try:
        original_data = data_loader.load_raw_data()
        # If we get an empty DataFrame, try to read as parquet file directly
        if len(original_data) == 0:
            original_data = pd.read_parquet(original_data_path)
    except Exception:
        # Fallback: if original_data_path is actually a parquet file, load directly
        original_data = pd.read_parquet(original_data_path)

    logger.info(f"Original training data: {len(original_data)} samples")

    # Preprocess data: create 'result' column from 'FTR' if needed
    if "FTR" in original_data.columns and "result" not in original_data.columns:
        original_data["result"] = original_data["FTR"]
        logger.info("Created 'result' column from 'FTR' column")

    # Combine with new data if available
    if new_data_path and os.path.exists(new_data_path):
        new_data = pd.read_parquet(new_data_path)

        # Preprocess new data: create 'result' column from 'FTR' if needed
        if "FTR" in new_data.columns and "result" not in new_data.columns:
            new_data["result"] = new_data["FTR"]
            logger.info("Created 'result' column from 'FTR' column in new data")

        # Filter to recent data within window
        if "date" in new_data.columns:
            cutoff_date = datetime.now() - timedelta(days=data_window_days)
            new_data = new_data[pd.to_datetime(new_data["date"]) >= cutoff_date]

        # Combine datasets
        combined_data = pd.concat([original_data, new_data], ignore_index=True)
        logger.info(f"Added {len(new_data)} new samples")
    else:
        combined_data = original_data
        logger.info("No new data available, using original training data")

    # Remove duplicates and sort by date
    if "date" in combined_data.columns:
        combined_data = combined_data.drop_duplicates(subset=["date", "home_team", "away_team"])
        combined_data = combined_data.sort_values("date")

    # Split data for retraining
    # Use most recent 80% for training, 20% for validation
    split_idx = int(len(combined_data) * 0.8)
    train_data = combined_data.iloc[:split_idx].copy()
    val_data = combined_data.iloc[split_idx:].copy()

    # Prepare data statistics
    data_stats = {
        "total_samples": len(combined_data),
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "date_range": {
            "start": str(combined_data["date"].min()) if "date" in combined_data.columns else None,
            "end": str(combined_data["date"].max()) if "date" in combined_data.columns else None,
        },
        "class_distribution": combined_data["result"].value_counts().to_dict()
        if "result" in combined_data.columns
        else {},
    }

    logger.info(f"Prepared retraining data: {data_stats}")
    return train_data, val_data, data_stats


@task(name="train-new-model")
def train_new_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    model_type: str = "random_forest",
    hyperparameters: dict | None = None,
) -> tuple[str, dict]:
    """Train a new model with the prepared data."""
    logger = get_run_logger()

    logger.info(f"Training new {model_type} model with {len(train_data)} samples")

    # Initialize trainer
    trainer = ModelTrainer(model_type=model_type)

    # Apply custom hyperparameters if provided
    if hyperparameters:
        logger.info(f"Using custom hyperparameters: {hyperparameters}")
        # Update trainer's model configuration
        if hasattr(trainer, "model_params"):
            trainer.model_params.update(hyperparameters)

    # Train the model
    model = trainer.train(train_data, val_data)

    # Debug: Check the trainer state after training
    logger.info(f"After training - Trainer model is None: {trainer.model is None}")
    if trainer.model is not None:
        logger.info(f"Trainer model class: {type(trainer.model)}")

    # Save the trained model to a temporary directory using trainer's save method
    import tempfile

    temp_model_dir = tempfile.mkdtemp()

    # Use the trainer's built-in save method which properly saves all components
    trainer.save_model(temp_model_dir)
    logger.info(f"Trained model saved to temporary directory: {temp_model_dir}")

    # Get training metrics
    training_metrics = {
        "model_type": model_type,
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "training_time": datetime.now().isoformat(),
        "temp_model_path": temp_model_dir,
    }

    # Add model-specific metrics if available
    if hasattr(trainer, "training_history") and trainer.training_history:
        training_metrics.update(trainer.training_history)

    logger.info(f"Model training completed: {training_metrics}")
    return temp_model_dir, training_metrics


@task(name="validate-new-model")
def validate_new_model(
    temp_model_path: str,
    validation_data: pd.DataFrame,
    current_model_path: str,
    min_accuracy_threshold: float = 0.45,
    improvement_threshold: float = 0.01,
) -> tuple[dict, bool]:
    """Validate the new model against current model and thresholds."""
    logger = get_run_logger()

    logger.info("Validating new model performance")

    # Load the trained model using ModelTrainer's load method
    import os

    if not os.path.exists(temp_model_path):
        raise FileNotFoundError(f"Temporary model directory not found: {temp_model_path}")

    # Create a new trainer instance and load the saved model
    new_trainer = ModelTrainer()
    new_trainer.load_model(temp_model_path)
    logger.info(f"Loaded trained model from: {temp_model_path}")

    # Debug: Check the state of the loaded trainer
    logger.info(f"Trainer model type: {new_trainer.model_type}")
    logger.info(f"Trainer model is None: {new_trainer.model is None}")
    if new_trainer.model is not None:
        logger.info(f"Trainer model class: {type(new_trainer.model)}")
        if hasattr(new_trainer.model, "n_estimators"):
            logger.info(f"Model n_estimators: {new_trainer.model.n_estimators}")
    logger.info(f"Trainer scaler fitted: {hasattr(new_trainer.scaler, 'mean_')}")

    # Evaluate new model
    evaluator = ModelEvaluator()
    new_metrics = evaluator.evaluate(new_trainer, validation_data)

    new_accuracy = new_metrics.get("accuracy", 0.0)
    logger.info(f"New model accuracy: {new_accuracy:.4f}")

    # Load and evaluate current model if it exists
    current_accuracy = None
    if os.path.exists(current_model_path):
        try:
            current_trainer = ModelTrainer()
            current_trainer.load_model(current_model_path)
            current_metrics = evaluator.evaluate(current_trainer, validation_data)
            current_accuracy = current_metrics.get("accuracy", 0.0)
            logger.info(f"Current model accuracy: {current_accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate current model: {str(e)}")
            current_accuracy = None

    # Validation logic
    validation_results = {
        "new_accuracy": new_accuracy,
        "current_accuracy": current_accuracy,
        "min_threshold": min_accuracy_threshold,
        "improvement_threshold": improvement_threshold,
        "new_metrics": new_metrics,
    }

    # Check minimum accuracy threshold
    meets_min_threshold = new_accuracy >= min_accuracy_threshold
    logger.info(f"Meets minimum threshold ({min_accuracy_threshold}): {meets_min_threshold}")

    # Check improvement over current model
    improvement = 0.0
    improves_performance = True  # Default to True if no current model

    if current_accuracy is not None:
        improvement = new_accuracy - current_accuracy
        improves_performance = improvement >= improvement_threshold
        logger.info(
            f"Improvement over current: {improvement:.4f} (required: {improvement_threshold})"
        )
    else:
        logger.info("No current model for comparison")

    validation_results.update(
        {
            "improvement": improvement,
            "meets_min_threshold": meets_min_threshold,
            "improves_performance": improves_performance,
        }
    )

    # Overall deployment decision
    should_deploy = meets_min_threshold and improves_performance
    validation_results["should_deploy"] = should_deploy

    if should_deploy:
        logger.info("✅ New model passed validation and should be deployed")
    else:
        reasons = []
        if not meets_min_threshold:
            reasons.append(
                f"below minimum threshold ({new_accuracy:.4f} < {min_accuracy_threshold})"
            )
        if not improves_performance:
            reasons.append(
                f"insufficient improvement ({improvement:.4f} < {improvement_threshold})"
            )

        logger.warning(f"❌ New model failed validation: {', '.join(reasons)}")

    return validation_results, should_deploy


@task(name="deploy-new-model")
def deploy_new_model(
    model_file_path: str,
    model_path: str,
    validation_results: dict,
    deployment_metadata: dict | None = None,
) -> dict:
    """Deploy the new model if validation passed."""
    logger = get_run_logger()

    if not validation_results.get("should_deploy", False):
        logger.warning("Model deployment skipped due to validation failure")
        return {
            "deployed": False,
            "reason": "validation_failed",
            "validation_results": validation_results,
        }

    try:
        # Load the trained model using ModelTrainer's load method
        new_trainer = ModelTrainer()
        new_trainer.load_model(model_file_path)

        # Save the new model
        new_trainer.save_model(model_path)

        deployment_info = {
            "deployed": True,
            "deployment_time": datetime.now().isoformat(),
            "model_path": model_path,
            "validation_results": validation_results,
            "deployment_metadata": deployment_metadata or {},
        }

        # Create deployment record
        deployment_record_path = Path(model_path).parent / "deployment_history.json"

        import json

        deployment_history = []
        if deployment_record_path.exists():
            try:
                with open(deployment_record_path) as f:
                    deployment_history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read deployment history: {str(e)}")

        deployment_history.append(deployment_info)

        with open(deployment_record_path, "w") as f:
            json.dump(deployment_history, f, indent=2)

        logger.info(f"✅ New model deployed successfully to {model_path}")
        return deployment_info

    except Exception as e:
        error_msg = f"Model deployment failed: {str(e)}"
        logger.error(error_msg)

        return {
            "deployed": False,
            "reason": "deployment_error",
            "error": error_msg,
            "validation_results": validation_results,
        }


@task(name="generate-retraining-report")
def generate_retraining_report(
    triggers: list[str],
    data_stats: dict,
    training_metrics: dict,
    validation_results: dict,
    deployment_results: dict,
    backup_path: str,
) -> str:
    """Generate a comprehensive retraining report."""
    logger = get_run_logger()

    # Create markdown report
    report_content = f"""
# Automated Retraining Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Retraining Summary

### Triggers
- **Triggered by:** {', '.join(triggers)}

### Data Statistics
- **Total samples:** {data_stats.get('total_samples', 'N/A')}
- **Training samples:** {data_stats.get('training_samples', 'N/A')}
- **Validation samples:** {data_stats.get('validation_samples', 'N/A')}
- **Date range:** {data_stats.get('date_range', {}).get('start', 'N/A')} to {data_stats.get('date_range', {}).get('end', 'N/A')}

### Training Results
- **Model type:** {training_metrics.get('model_type', 'N/A')}
- **Training time:** {training_metrics.get('training_time', 'N/A')}

### Validation Results
- **New model accuracy:** {validation_results.get('new_accuracy', 'N/A'):.4f}
- **Current model accuracy:** {validation_results.get('current_accuracy', 'N/A') or 'N/A'}
- **Improvement:** {validation_results.get('improvement', 0):.4f}
- **Meets minimum threshold:** {'✅ Yes' if validation_results.get('meets_min_threshold', False) else '❌ No'}
- **Improves performance:** {'✅ Yes' if validation_results.get('improves_performance', False) else '❌ No'}

### Deployment
- **Deployed:** {'✅ Yes' if deployment_results.get('deployed', False) else '❌ No'}
- **Deployment time:** {deployment_results.get('deployment_time', 'N/A')}
- **Backup location:** {backup_path}

### Recommendations
"""

    # Add recommendations based on results
    if deployment_results.get("deployed", False):
        report_content += "- ✅ New model successfully deployed and is now serving predictions\n"
        report_content += "- 📊 Monitor performance closely for the next few days\n"
        report_content += "- 🔄 Continue regular monitoring and automated retraining\n"
    else:
        if not validation_results.get("meets_min_threshold", False):
            report_content += (
                "- ⚠️ New model accuracy below minimum threshold - investigate data quality\n"
            )
        if not validation_results.get("improves_performance", False):
            report_content += (
                "- ⚠️ New model didn't improve performance - consider hyperparameter tuning\n"
            )
        report_content += "- 🔍 Review training data and feature engineering\n"
        report_content += "- 📈 Consider collecting more diverse training data\n"

    # Create Prefect artifact
    create_markdown_artifact(
        key="retraining-report",
        markdown=report_content,
        description="Automated retraining execution report",
    )

    logger.info("Retraining report generated")
    return report_content


@flow(name="automated-retraining-flow")
def automated_retraining_flow(
    triggers: list[str],
    model_path: str = "models/model.pkl",
    training_data_path: str = "data/real_data/premier_league_matches.parquet",
    new_data_path: str | None = None,
    backup_dir: str = "models/backups",
    model_type: str = "random_forest",
    min_accuracy_threshold: float = 0.45,
    improvement_threshold: float = 0.01,
    hyperparameters: dict | None = None,
) -> dict[str, Any]:
    """
    Main automated retraining flow.

    Args:
        triggers: List of reasons that triggered retraining
        model_path: Path to current model file
        training_data_path: Path to original training data
        new_data_path: Path to new data (optional)
        backup_dir: Directory for model backups
        model_type: Type of model to train
        min_accuracy_threshold: Minimum accuracy for deployment
        improvement_threshold: Minimum improvement required for deployment
        hyperparameters: Custom hyperparameters for training

    Returns:
        Dictionary with retraining results
    """
    logger = get_run_logger()
    logger.info(f"Starting automated retraining flow. Triggers: {triggers}")

    try:
        # 0. Pre-flight validation
        logger.info("🔍 Performing pre-flight validation...")

        # Check if training data exists and is not empty
        if not os.path.exists(training_data_path):
            raise ValueError(f"Training data not found: {training_data_path}")

        # Quick check if data is loadable and not empty
        try:
            if training_data_path.endswith(".parquet"):
                df = pd.read_parquet(training_data_path)
            else:
                df = pd.read_csv(training_data_path)

            if len(df) == 0:
                raise ValueError("Training data is empty")

            logger.info(f"✅ Training data validation passed: {len(df)} rows")

        except Exception as e:
            raise ValueError(f"Invalid training data: {str(e)}")

        # 1. Backup current model
        backup_path = backup_current_model(
            model_path=model_path, backup_dir=backup_dir, backup_reason="_".join(triggers)
        )

        # 2. Prepare retraining data
        train_data, val_data, data_stats = prepare_retraining_data(
            original_data_path=training_data_path,
            new_data_path=new_data_path,
        )

        # 3. Train new model
        temp_model_path, training_metrics = train_new_model(
            train_data=train_data,
            val_data=val_data,
            model_type=model_type,
            hyperparameters=hyperparameters,
        )

        # 4. Validate new model
        validation_results, should_deploy = validate_new_model(
            temp_model_path=temp_model_path,
            validation_data=val_data,
            current_model_path=model_path,
            min_accuracy_threshold=min_accuracy_threshold,
            improvement_threshold=improvement_threshold,
        )

        # 5. Deploy if validation passed
        deployment_results = deploy_new_model(
            model_file_path=temp_model_path,
            model_path=model_path,
            validation_results=validation_results,
            deployment_metadata={
                "triggers": triggers,
                "retraining_flow_run": str(getattr(logger, "extra", {}).get("flow_run_id", "")),
            },
        )

        # 6. Generate report
        report_content = generate_retraining_report(
            triggers=triggers,
            data_stats=data_stats,
            training_metrics=training_metrics,
            validation_results=validation_results,
            deployment_results=deployment_results,
            backup_path=backup_path,
        )

        # Prepare final results
        flow_results = {
            "success": True,
            "triggers": triggers,
            "deployed": deployment_results.get("deployed", False),
            "backup_path": backup_path,
            "data_stats": data_stats,
            "training_metrics": training_metrics,
            "validation_results": validation_results,
            "deployment_results": deployment_results,
            "report": report_content,
            "execution_time": datetime.now().isoformat(),
        }

        if deployment_results.get("deployed", False):
            logger.info("🎉 Automated retraining completed successfully with deployment")
        else:
            logger.warning("⚠️ Automated retraining completed but model was not deployed")

        return flow_results

    except Exception as e:
        error_msg = f"Automated retraining flow failed: {str(e)}"
        logger.error(error_msg)

        return {
            "success": False,
            "error": error_msg,
            "triggers": triggers,
            "execution_time": datetime.now().isoformat(),
        }


def execute_automated_retraining(config: Any, triggers: list[str]) -> dict[str, Any]:
    """
    Execute automated retraining with given configuration.

    This is the main entry point called by the RetrainingScheduler.
    """
    return automated_retraining_flow(
        triggers=triggers,
        model_path=config.model_path,
        training_data_path=config.training_data_path,
        backup_dir=config.backup_model_dir,
        min_accuracy_threshold=config.min_validation_accuracy,
        improvement_threshold=config.validation_improvement_required,
    )


# Simple retraining flow for manual execution
@flow(name="manual-retraining")
def manual_retraining_flow(reason: str = "manual_execution", **kwargs: Any) -> dict[str, Any]:
    """Manual retraining flow for on-demand execution."""
    result = automated_retraining_flow(triggers=[reason], **kwargs)
    return dict(result)  # Ensure return type consistency


if __name__ == "__main__":
    # Allow running the flow directly for testing
    import argparse

    parser = argparse.ArgumentParser(description="Run automated retraining flow")
    parser.add_argument("--reason", default="manual_test", help="Reason for retraining")
    parser.add_argument("--model-path", default="models/model.pkl", help="Model path")
    parser.add_argument(
        "--data-path", default="data/real_data/premier_league_matches.parquet", help="Data path"
    )

    args = parser.parse_args()

    result = manual_retraining_flow(
        reason=args.reason,
        model_path=args.model_path,
        training_data_path=args.data_path,
    )

    print(f"Retraining completed: {result}")
