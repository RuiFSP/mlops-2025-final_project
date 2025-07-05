"""Main entry point for the Premier League Match Predictor."""

import click
from dotenv import load_dotenv

from src.data_preprocessing.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.model_training.trainer import ModelTrainer

load_dotenv()


@click.group()
def cli() -> None:
    """Premier League Match Predictor CLI."""


@cli.command()
@click.option("--data-path", default="data/", help="Path to data directory")
def train(data_path: str) -> None:
    """Train the prediction model."""
    click.echo("Starting model training...")

    # Load data
    data_loader = DataLoader(data_path)
    train_data, val_data = data_loader.load_and_split()

    # Train model
    trainer = ModelTrainer()
    model = trainer.train(train_data, val_data)

    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, val_data)

    click.echo(f"Training completed! Metrics: {metrics}")


@cli.command()
@click.option("--model-path", required=True, help="Path to trained model")
@click.option("--data-path", default="data/", help="Path to data directory")
def predict(model_path: str, data_path: str) -> None:
    """Make predictions using trained model."""
    click.echo(f"Making predictions with model: {model_path}")
    # TODO: Implement prediction logic


@cli.command()
def serve() -> None:
    """Start the prediction API server."""
    click.echo("Starting API server...")
    try:
        import uvicorn

        from src.deployment.api import app

        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        click.echo("Error: uvicorn not installed. Install with: uv add uvicorn")
    except Exception as e:
        click.echo(f"Error starting server: {e}")


if __name__ == "__main__":
    cli()
