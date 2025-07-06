"""Prefect flow for Premier League match prediction pipeline."""

import os
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_preprocessing.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.model_training.trainer import ModelTrainer


@task
def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split data for training."""
    logger = get_run_logger()
    logger.info(f"Loading data from {data_path}")

    data_loader = DataLoader(data_path)
    train_data, val_data = data_loader.load_and_split()

    logger.info(
        f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}"
    )
    return train_data, val_data


@task
def train_model(
    train_data: pd.DataFrame, val_data: pd.DataFrame, model_type: str = "random_forest"
) -> ModelTrainer:
    """Train the prediction model."""
    logger = get_run_logger()
    logger.info(f"Training {model_type} model")

    trainer = ModelTrainer(model_type=model_type)
    model = trainer.train(train_data, val_data)

    logger.info("Model training completed")
    return trainer


@task
def evaluate_model(trainer: ModelTrainer, test_data: pd.DataFrame) -> dict:
    """Evaluate the trained model."""
    logger = get_run_logger()
    logger.info("Evaluating model performance")

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(trainer, test_data)

    logger.info(
        f"Model evaluation completed. Accuracy: {metrics.get('accuracy', 0):.4f}"
    )
    return metrics


@flow(name="training-pipeline")
def training_pipeline(
    data_path: str = "data/",
    model_type: str = "random_forest",
    experiment_name: str = "premier_league_prediction",
):
    """Main training pipeline for Premier League match prediction."""
    logger = get_run_logger()
    logger.info(f"Starting training pipeline for {experiment_name}")

    # Load data
    train_data, val_data = load_data(data_path)

    # Train model
    trainer = train_model(train_data, val_data, model_type)

    # Evaluate model
    metrics = evaluate_model(trainer, val_data)

    logger.info(f"Training pipeline completed successfully. Final metrics: {metrics}")
    return metrics


@flow(name="prediction-pipeline")
def prediction_pipeline(
    model_path: str, input_data_path: str, output_path: str = "predictions.csv"
):
    """Prediction pipeline for making match predictions."""
    logger = get_run_logger()
    logger.info(f"Starting prediction pipeline with model: {model_path}")

    # Load model
    trainer = ModelTrainer()
    trainer.load_model(model_path)

    # Load input data
    data_loader = DataLoader(input_data_path)
    input_data = data_loader.load_raw_data()
    processed_data = data_loader.preprocess_data(input_data)

    # Make predictions
    predictions = trainer.predict(processed_data)

    # Save predictions
    output_df = processed_data.copy()
    output_df["predicted_result"] = predictions
    output_df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Run training pipeline
    training_pipeline()
