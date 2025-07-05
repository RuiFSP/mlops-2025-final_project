.PHONY: install test lint format run clean setup-dev

# Install dependencies
install:
	uv sync

# Install development dependencies
install-dev:
	uv sync --extra dev

# Run tests
test:
	uv run pytest

# Run linter
lint:
	uv run flake8 src tests
	uv run mypy src

# Format code
format:
	uv run black src tests
	uv run isort src tests

# Run the training pipeline
run:
	uv run python -m src.main

# Clean up generated files
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Setup development environment
setup-dev: install-dev
	uv run pre-commit install

# Run MLflow server
mlflow-server:
	uv run mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5000

# Run Prefect server
prefect-server:
	uv run prefect server start

# Build Docker image
docker-build:
	docker build -t premier-league-predictor .

# Run Docker container
docker-run:
	docker run -p 8001:8001 premier-league-predictor

# Collect real data
collect-data:
	uv run python scripts/collect_real_data.py

# Train model
train:
	uv run python -m src.main train --data-path data/real_data/

# Start API server
api:
	uv run python -m src.deployment.api

# Full pipeline: collect data, train model, start API
pipeline: collect-data train api
