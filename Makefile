# MLOps Premier League Prediction System - Makefile
# ==================================================

.PHONY: help install setup start stop restart clean test logs status troubleshoot

# Default target
help:
	@echo "🏆 Premier League MLOps System - Available Commands:"
	@echo "=================================================="
	@echo "📦 Setup Commands:"
	@echo "  make install     - Install dependencies with uv"
	@echo "  make setup       - Complete initial setup (Docker + Database)"
	@echo ""
	@echo "🚀 Service Management:"
	@echo "  make start       - Start all services (5 terminals)"
	@echo "  make stop        - Stop all services"
	@echo "  make restart     - Restart all services"
	@echo "  make status      - Check status of all services"
	@echo ""
	@echo "🔧 Individual Services:"
	@echo "  make start-docker    - Start Docker services only"
	@echo "  make start-mlflow    - Start MLflow server"
	@echo "  make start-api       - Start API server"
	@echo "  make start-prefect   - Start Prefect server"
	@echo "  make start-dashboard - Start Streamlit dashboard"
	@echo "  make train           - Run training pipeline"
	@echo ""
	@echo "🧪 Testing & Monitoring:"
	@echo "  make test        - Run integration tests"
	@echo "  make test-orch   - Test orchestration components"
	@echo "  make logs        - Show logs from all services"
	@echo "  make troubleshoot - Run dashboard troubleshooting script"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  make lint        - Run Ruff linter (check critical issues)"
	@echo "  make format      - Format code with Ruff"
	@echo "  make lint-fix    - Auto-fix linting issues"
	@echo "  make check       - Run all code quality checks"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean       - Clean up all resources"
	@echo "  make clean-docker - Clean Docker resources only"

# Installation and Setup
install:
	@echo "📦 Installing dependencies..."
	uv sync

setup: install
	@echo "🔧 Setting up system..."
	@if [ ! -f .env ]; then cp config.env.example .env; echo "✅ Created .env file"; fi
	docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	uv run python scripts/setup_database.py
	@echo "✅ Setup complete!"

# Service Management
start-docker:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10

start-mlflow:
	@echo "📊 Starting MLflow server..."
	@if pgrep -f "mlflow server" > /dev/null; then \
		echo "⚠️  MLflow server already running"; \
	else \
		nohup uv run mlflow server --host 127.0.0.1 --port 5000 > logs/mlflow.log 2>&1 & \
		echo "✅ MLflow server started"; \
	fi

start-api:
	@echo "🚀 Starting API server..."
	@if pgrep -f "uvicorn.*main:app" > /dev/null; then \
		echo "⚠️  API server already running"; \
	else \
		cd src/api && nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../../logs/api.log 2>&1 & \
		echo "✅ API server started"; \
	fi

start-prefect:
	@echo "🔄 Starting Prefect server..."
	@if pgrep -f "prefect server" > /dev/null; then \
		echo "⚠️  Prefect server already running"; \
	else \
		nohup uv run prefect server start --host 0.0.0.0 --port 4200 > logs/prefect.log 2>&1 & \
		echo "✅ Prefect server started"; \
	fi

start-dashboard:
	@echo "📊 Starting Streamlit dashboard..."
	@if pgrep -f "streamlit.*streamlit_app" > /dev/null; then \
		echo "⚠️  Streamlit dashboard already running"; \
	else \
		nohup uv run streamlit run src/dashboard/streamlit_app.py --server.port 8501 > logs/streamlit.log 2>&1 & \
		echo "✅ Streamlit dashboard started"; \
	fi

train:
	@echo "🎯 Running training pipeline..."
	uv run python -m src.pipelines.training_pipeline

start: start-docker
	@echo "🚀 Starting all services..."
	@mkdir -p logs
	@$(MAKE) start-mlflow
	@sleep 5
	@$(MAKE) start-api
	@sleep 3
	@$(MAKE) start-prefect
	@sleep 3
	@$(MAKE) start-dashboard
	@echo ""
	@echo "🎉 All services started!"
	@echo "🌐 Access Points:"
	@echo "  • API: http://localhost:8000"
	@echo "  • API Docs: http://localhost:8000/docs"
	@echo "  • MLflow: http://127.0.0.1:5000"
	@echo "  • Prefect UI: http://localhost:4200"
	@echo "  • Dashboard: http://localhost:8501"
	@echo ""
	@echo "🧪 Run 'make test' to verify everything is working!"

# Stop services
stop:
	@echo "🛑 Stopping all services..."
	@echo "Stopping API server..."
	@-pkill -f "uvicorn.*main:app" || true
	@-pkill -f "uv run uvicorn" || true
	@-pkill -f "uvicorn main:app" || true
	@echo "Stopping MLflow server..."
	@-pkill -f "mlflow server" || true
	@-pkill -f "uv run mlflow" || true
	@-pkill -f "gunicorn.*mlflow" || true
	@echo "Stopping Prefect server..."
	@-pkill -f "prefect server" || true
	@-pkill -f "uv run prefect" || true
	@echo "Stopping training pipeline..."
	@-pkill -f "training_pipeline" || true
	@echo "Stopping Streamlit dashboard..."
	@-pkill -f "streamlit.*streamlit_app" || true
	@echo "Stopping Docker services..."
	@-docker-compose down || true
	@echo "Killing any remaining uv processes..."
	@-pkill -f "uv run" || true
	@echo "✅ All services stopped!"

# Restart services
restart: stop
	@sleep 3
	@$(MAKE) start

# Status check
status:
	@echo "📊 Service Status:"
	@echo "=================="
	@echo -n "🐳 Docker Services: "
	@if docker-compose ps | grep -q "Up"; then echo "✅ Running"; else echo "❌ Stopped"; fi
	@echo -n "📊 MLflow Server: "
	@if pgrep -f "mlflow server" > /dev/null; then echo "✅ Running (http://127.0.0.1:5000)"; else echo "❌ Stopped"; fi
	@echo -n "🚀 API Server: "
	@if pgrep -f "uvicorn.*main:app" > /dev/null; then echo "✅ Running (http://localhost:8000)"; else echo "❌ Stopped"; fi
	@echo -n "🔄 Prefect Server: "
	@if pgrep -f "prefect server" > /dev/null; then echo "✅ Running (http://localhost:4200)"; else echo "❌ Stopped"; fi
	@echo -n "📊 Streamlit Dashboard: "
	@if pgrep -f "streamlit.*streamlit_app" > /dev/null; then echo "✅ Running (http://localhost:8501)"; else echo "❌ Stopped"; fi

# Testing
test:
	@echo "🧪 Running integration tests..."
	uv run python scripts/test_simple_integration.py

test-orch:
	@echo "🔄 Testing orchestration components..."
	uv run python scripts/test_simple_orchestration.py

# Troubleshooting
troubleshoot:
	@echo "🔧 Running dashboard troubleshooting script..."
	uv run python scripts/dashboard_troubleshoot.py

# Logs
logs:
	@echo "📋 Service Logs:"
	@echo "==============="
	@if [ -f logs/mlflow.log ]; then echo "📊 MLflow Logs:"; tail -10 logs/mlflow.log; echo ""; fi
	@if [ -f logs/api.log ]; then echo "🚀 API Logs:"; tail -10 logs/api.log; echo ""; fi
	@if [ -f logs/prefect.log ]; then echo "🔄 Prefect Logs:"; tail -10 logs/prefect.log; echo ""; fi
	@if [ -f logs/streamlit.log ]; then echo "📊 Streamlit Logs:"; tail -10 logs/streamlit.log; echo ""; fi
	@echo "🐳 Docker Logs:"
	@docker-compose logs --tail=10

# Cleanup
clean-docker:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f

clean: stop clean-docker
	@echo "🧹 Cleaning up all resources..."
	@rm -rf logs/
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Quick development commands
dev: setup start train test
	@echo "🎯 Development environment ready!"

# Health check
health:
	@echo "🏥 Health Check:"
	@echo "==============="
	@echo -n "API Health: "
	@curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ Healthy" || echo "❌ Unhealthy"
	@echo -n "MLflow Health: "
	@curl -s http://127.0.0.1:5000/ | grep -q "MLflow" && echo "✅ Healthy" || echo "❌ Unhealthy"
	@echo -n "Prefect Health: "
	@curl -s http://localhost:4200/api/health | grep -q "true" && echo "✅ Healthy" || echo "❌ Unhealthy"
	@echo -n "Dashboard Health: "
	@curl -s http://localhost:8501/ | grep -q "html" && echo "✅ Healthy" || echo "❌ Unhealthy"

# Code Quality
lint:
	@echo "🔍 Running Ruff linter..."
	uv run ruff check .
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting code with Ruff..."
	uv run ruff format .
	@echo "✅ Formatting complete!"

lint-fix:
	@echo "🔧 Auto-fixing linting issues..."
	uv run ruff check --fix .
	@echo "✅ Auto-fix complete!"

check: lint
	@echo "✅ Code quality check complete!"
