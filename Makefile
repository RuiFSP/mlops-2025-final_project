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

# Fast tests for development (unit tests only)
test-fast:
	uv run pytest tests/unit/ -v --tb=short

# CI tests (unit tests with coverage - fast and reliable)
test-ci:
	uv run pytest tests/unit/ -v --tb=short --cov=src --cov-report=xml --timeout=60

# CI integration tests (matches GitHub workflow)
test-ci-integration:
	uv run pytest tests/integration/ -v --tb=short --cov=src --cov-append --cov-report=xml --timeout=900

# CI E2E tests (matches GitHub workflow)
test-ci-e2e:
	uv run pytest tests/e2e/ -v --tb=short --cov=src --cov-append --cov-report=xml --timeout=1800

# Full CI test suite (matches GitHub workflow exactly)
test-ci-full: test-ci test-ci-integration test-ci-e2e

# Local development: Integration tests (quick version - excludes slow tests)
test-integration:
	uv run pytest tests/integration/ -v --tb=short --timeout=60 -m "not slow"

# Local development: E2E tests
test-e2e:
	uv run pytest tests/e2e/ -v --tb=short --timeout=300

# Full integration tests (includes slow tests with real dependencies)
test-integration-full:
	uv run pytest tests/integration/ -v --tb=short --timeout=300

# Full test suite including E2E
test-full:
	uv run pytest tests/ -v --tb=short --cov=src --cov-report=xml

# Only slow/integration tests
test-slow:
	uv run pytest tests/integration/ tests/e2e/ -v --tb=short --timeout=300

# Parallel test execution (requires pytest-xdist)
test-parallel:
	uv run pytest tests/unit/ tests/integration/ -n auto -v --tb=short

# Run linter (matches CI)
lint:
	uv run ruff check src tests
	uv run mypy src

# Check formatting (matches CI - will fail if not formatted)
format-check:
	uv run ruff format --check src tests

# Format code (fix formatting issues)
format:
	uv run ruff format src tests

# Run all quality checks (lint + format-check + type checking)
quality-check: format-check lint

# CI-style complete check (matches GitHub workflow exactly)
ci-check: quality-check test-ci

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
	uv run python tests/integration/test_retraining_system.py

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
	@echo ""
	@echo "📦 Setup:"
	@echo "  install          - Install dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  setup-dev        - Setup development environment"
	@echo ""
	@echo "🧪 Testing (Local Development):"
	@echo "  test             - Run all tests"
	@echo "  test-fast        - Run unit tests only (quick)"
	@echo "  test-integration - Run integration tests (quick version)"
	@echo "  test-e2e         - Run E2E tests"
	@echo "  test-integration-full - Run full integration tests (includes slow tests)"
	@echo "  test-full        - Run full test suite including E2E"
	@echo "  test-slow        - Run slow tests (integration + E2E)"
	@echo "  test-parallel    - Run tests in parallel"
	@echo ""
	@echo "🚀 CI/CD Matching:"
	@echo "  test-ci          - Run CI unit tests (matches GitHub)"
	@echo "  test-ci-integration - Run CI integration tests (matches GitHub)"
	@echo "  test-ci-e2e      - Run CI E2E tests (matches GitHub)"
	@echo "  test-ci-full     - Run full CI test suite (matches GitHub exactly)"
	@echo "  quality-check    - Run linting and format checks"
	@echo "  ci-check         - Complete CI check (quality + tests)"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  lint             - Run linter and type checking"
	@echo "  format           - Format code"
	@echo "  format-check     - Check formatting (fails if not formatted)"
	@echo "  clean            - Clean up generated files"
	@echo ""
	@echo "🏃‍♂️ Running:"
	@echo "  run              - Run training pipeline"
	@echo "  api              - Start API server"
	@echo "  pipeline         - Full pipeline: collect data, train model, start API"
	@echo ""
	@echo "📊 Data & Training:"
	@echo "  collect-data     - Collect real Premier League data"
	@echo "  train            - Train model"
	@echo ""
	@echo "🤖 Automation:"
	@echo "  retraining-demo  - Run automated retraining demo"
	@echo "  retraining-test  - Test automated retraining system"
	@echo "  retraining-start - Start automated retraining scheduler"
	@echo "  retraining-status- Check retraining system status"
	@echo "  retraining-trigger- Manually trigger retraining"
	@echo "  retraining-config- Create default retraining configuration"
	@echo ""
	@echo "🎮 Simulation:"
	@echo "  simulation-demo  - Run season simulation demo"
	@echo "  simulation-run   - Run interactive season simulation"
	@echo "  monitoring-demo  - Run monitoring system demo"
	@echo ""
	@echo "🐳 Docker & Services:"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-run       - Run Docker container"
	@echo "  mlflow-server    - Start MLflow server"
	@echo "  prefect-server   - Start Prefect server"
	@echo "  prefect-deploy   - Deploy Prefect retraining flows"
	@echo "  prefect-demo     - Demo Prefect deployment integration"
