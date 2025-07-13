# Premier League Match Predictor - MLOps System

A complete end-to-end MLOps pipeline for predicting Premier League match outcomes with automated monitoring, orchestration, and betting simulation.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ with `uv` package manager
- PostgreSQL (local or Docker)
- 10 minutes setup time

### Setup & Run
```bash
# 1. Clone and install
git clone <repo-url>
cd mlops-2025-final_project
uv sync

# 2. Setup database
sudo -u postgres psql
CREATE DATABASE mlops_db;
CREATE USER mlops_user WITH PASSWORD 'mlops_password';
GRANT ALL PRIVILEGES ON DATABASE mlops_db TO mlops_user;
\q

# 3. Initialize system
cp config.env.example .env
uv run python scripts/setup_database.py

# 4. Start services (3 terminals)
uv run mlflow server --host 127.0.0.1 --port 5000  # Terminal 1
uv run python -m src.pipelines.training_pipeline    # Terminal 2 (once)
cd src/api && uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload  # Terminal 3

# 5. Test system
uv run python scripts/test_simple_integration.py
```

### 🎯 Access Points
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://127.0.0.1:5000
- **Health Check**: http://localhost:8000/health

## ✅ What's Working

### Core Features
- **61.84% Model Accuracy** - Random Forest with 15 features
- **REST API** - FastAPI with comprehensive endpoints
- **Real-time Predictions** - Premier League match outcomes
- **Betting Simulation** - Automated betting strategy testing
- **MLflow Integration** - Model tracking and versioning
- **PostgreSQL Database** - Complete data persistence

### Monitoring & Orchestration
- **Prefect Workflows** - Automated orchestration
- **Grafana Dashboards** - Real-time monitoring
- **Performance Tracking** - Model drift detection
- **Alert System** - Automated notifications

## 🏗️ Architecture
