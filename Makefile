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
	uv run ruff check src tests
	uv run mypy src

# Format code
format:
	uv run ruff format src tests

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
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

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

# Automated Retraining Commands
retraining-demo:
	uv run python scripts/automation/demo_automated_retraining.py --demo all

retraining-test:
	uv run python scripts/automation/test_retraining_system.py

retraining-start:
	uv run python scripts/automation/manage_retraining.py start

retraining-status:
	uv run python scripts/automation/manage_retraining.py status

retraining-trigger:
	uv run python scripts/automation/manage_retraining.py trigger --reason "manual_makefile"

retraining-config:
	uv run python scripts/automation/manage_retraining.py create-config

# Season Simulation Commands
simulation-demo:
	uv run python scripts/simulation/demo_simulation.py

simulation-run:
	uv run python scripts/simulation/run_simulation.py --mode interactive --weeks 5

# Monitoring Commands
monitoring-demo:
	uv run python scripts/monitoring/demo_monitoring.py

# Prefect deployment targets
prefect-deploy:
	@echo "🚀 Deploying Prefect retraining flows..."
	python deployments/deploy_retraining_flow.py

prefect-demo:
	@echo "🎯 Running Prefect deployment integration demo..."
	python scripts/automation/demo_prefect_deployments.py

prefect-demo-full:
	@echo "🎯 Running full Prefect deployment demo (requires deployments)..."
	python scripts/automation/demo_prefect_deployments.py --full

# Help target
help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  test             - Run tests"
	@echo "  lint             - Run linter"
	@echo "  format           - Format code"
	@echo "  run              - Run training pipeline"
	@echo "  clean            - Clean up generated files"
	@echo "  setup-dev        - Setup development environment"
	@echo ""
	@echo "  collect-data     - Collect real Premier League data"
	@echo "  train            - Train model"
	@echo "  api              - Start API server"
	@echo "  pipeline         - Full pipeline: collect data, train model, start API"
	@echo ""
	@echo "  retraining-demo  - Run automated retraining demo"
	@echo "  retraining-test  - Test automated retraining system"
	@echo "  retraining-start - Start automated retraining scheduler"
	@echo "  retraining-status- Check retraining system status"
	@echo "  retraining-trigger- Manually trigger retraining"
	@echo "  retraining-config- Create default retraining configuration"
	@echo ""
	@echo "  simulation-demo  - Run season simulation demo"
	@echo "  simulation-run   - Run interactive season simulation"
	@echo "  monitoring-demo  - Run monitoring system demo"
	@echo ""
	@echo "  prefect-deploy   - Deploy Prefect retraining flows"
	@echo "  prefect-demo     - Demo Prefect deployment integration"
	@echo "  mlflow-server    - Start MLflow server"
	@echo "  prefect-server   - Start Prefect server"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-run       - Run Docker container"
