# 🏆 Premier League Match Predictor - MLOps System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange.svg)](https://mlflow.org)
[![Prefect](https://img.shields.io/badge/Prefect-2.14+-purple.svg)](https://prefect.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-ready MLOps pipeline for Premier League match prediction with automated monitoring, orchestration, and betting simulation.**

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [📊 Monitoring](#-monitoring)
- [🔄 Orchestration](#-orchestration)
- [🛠️ Development](#️-development)
- [📚 API Documentation](#-api-documentation)
- [🤝 Contributing](#-contributing)

## 🎯 Overview

This MLOps system provides a complete end-to-end pipeline for predicting Premier League match outcomes. Built with modern MLOps practices, it includes automated training, real-time predictions, comprehensive monitoring, and orchestrated workflows.

### 🎪 Key Highlights
- **🤖 61.84% Model Accuracy** - Random Forest classifier with 15 engineered features
- **⚡ Real-time Predictions** - FastAPI-powered REST API with sub-second response times
- **📊 Comprehensive Monitoring** - Grafana dashboards with PostgreSQL metrics storage
- **🔄 Automated Orchestration** - Prefect workflows for training, monitoring, and alerts
- **💰 Betting Simulation** - Automated betting strategy testing and validation
- **🐳 Containerized Deployment** - Docker Compose for easy deployment and scaling
- **🧪 Full Test Coverage** - Integration tests for all components

## ✨ Features

### 🔮 Machine Learning
- **Premier League Match Prediction** - Predict match outcomes (Home/Draw/Away)
- **Feature Engineering** - 15 carefully crafted features including team form, head-to-head records
- **Model Versioning** - MLflow integration for experiment tracking and model registry
- **Automated Retraining** - Scheduled model updates based on performance thresholds

### 🚀 API & Services
- **FastAPI REST API** - Comprehensive endpoints for predictions, model info, and betting
- **Real-time Data Integration** - Automated data fetching and processing
- **Betting Simulation** - Strategy testing with configurable parameters
- **Health Monitoring** - Comprehensive health checks and status endpoints

### 📊 Monitoring & Observability
- **Grafana Dashboards** - Real-time visualization of model performance and system metrics
- **Performance Tracking** - Model accuracy, drift detection, and prediction confidence
- **Alert System** - Automated notifications for performance degradation
- **Comprehensive Logging** - Structured logging across all components

### 🔄 Orchestration & Automation
- **Prefect Workflows** - 4 automated workflows for different operational needs:
  - **Hourly Monitoring** - Model performance and drift detection
  - **Daily Predictions** - Generate predictions for upcoming matches
  - **Weekly Retraining** - Automated model retraining evaluation
  - **Emergency Retraining** - Manual trigger for immediate model updates

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ with `uv` package manager
- PostgreSQL (local or Docker)
- Grafana server (for monitoring dashboards)
- 15 minutes setup time

### Setup & Run

#### 🚀 Quick Start (Recommended)
```bash
# 1. Clone and setup
git clone <repo-url>
cd mlops-2025-final_project

# 2. One-command setup and start
make setup
make start

# 3. Test system
make test
```

#### 🔧 Manual Setup (Alternative)
```bash
# 1. Clone and install
git clone <repo-url>
cd mlops-2025-final_project
uv sync

# 2. Start Docker services (PostgreSQL)
docker-compose up -d

# 3. Initialize system
cp config.env.example .env
uv run python scripts/setup_database.py

# 4. Start core services (5 terminals)
uv run mlflow server --host 127.0.0.1 --port 5000                           # Terminal 1
uv run python -m src.pipelines.training_pipeline                            # Terminal 2 (once)
cd src/api && uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload  # Terminal 3
uv run prefect server start --host 0.0.0.0 --port 4200                     # Terminal 4
sudo systemctl start grafana-server                                         # Terminal 5

# 5. Setup Grafana dashboard
uv run python scripts/setup_grafana.py

# 6. Test complete system
uv run python scripts/test_simple_integration.py
uv run python scripts/test_simple_monitoring.py
```

#### 🛠️ Makefile Commands
```bash
# Service Management
make start       # Start all services
make stop        # Stop all services
make restart     # Restart all services
make status      # Check service status

# Development
make setup       # Complete setup
make test        # Run integration tests
make train       # Run training pipeline
make clean       # Clean up resources

# Individual Services
make start-docker    # Start Docker only
make start-mlflow    # Start MLflow only
make start-api       # Start API only
make start-prefect   # Start Prefect only
make start-grafana   # Start Grafana only

# View all commands
make help
```

### 🎯 Access Points
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://127.0.0.1:5000
- **Prefect UI**: http://localhost:4200
- **Grafana**: http://localhost:3000 (admin/admin)
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
- **Prefect Workflows** - Automated orchestration with 4 flows:
  - `hourly-monitoring` - Model performance & drift checks
  - `daily-predictions` - Generate predictions for upcoming matches
  - `weekly-retraining-check` - Automated model retraining evaluation
  - `emergency-retraining` - Manual retraining trigger
- **Grafana Dashboards** - Real-time monitoring with PostgreSQL data source
- **Performance Tracking** - Model drift detection and accuracy monitoring
- **Alert System** - Automated notifications for performance degradation

## 🏗️ Architecture

### 🔄 System Architecture Overview

```mermaid
graph TB
    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Database)]
        DATA[Premier League<br/>Match Data]
    end

    subgraph "ML Pipeline"
        TRAIN[Training Pipeline]
        MLF[MLflow<br/>Tracking]
        MODEL[Model Registry]
        PRED[Prediction Pipeline]
    end

    subgraph "API Layer"
        API[FastAPI<br/>REST API]
        BET[Betting<br/>Simulator]
    end

    subgraph "Monitoring & Orchestration"
        PREF[Prefect<br/>Workflows]
        GRAF[Grafana<br/>Dashboards]
        ALERT[Alert System]
    end

    subgraph "Infrastructure"
        DOCKER[Docker<br/>Compose]
        MAKE[Makefile<br/>Commands]
    end

    DATA --> TRAIN
    TRAIN --> MLF
    MLF --> MODEL
    MODEL --> PRED
    PRED --> API
    API --> BET
    BET --> PG
    PRED --> PG

    PREF --> TRAIN
    PREF --> PRED
    PREF --> GRAF
    GRAF --> ALERT
    PG --> GRAF

    DOCKER --> PG
    DOCKER --> MLF
    MAKE --> DOCKER
    MAKE --> API
    MAKE --> PREF
```

### 🧩 Component Details

| Component | Technology | Purpose | Port |
|-----------|------------|---------|------|
| **API Server** | FastAPI + Uvicorn | REST API endpoints | 8000 |
| **ML Tracking** | MLflow | Experiment tracking & model registry | 5000 |
| **Database** | PostgreSQL | Data persistence & metrics | 5432 |
| **Orchestration** | Prefect | Workflow automation | 4200 |
| **Monitoring** | Grafana | Dashboards & visualization | 3000 |
| **Containerization** | Docker Compose | Service orchestration | - |

### 🔄 Data Flow

1. **Training Flow**: `Data → Feature Engineering → Model Training → MLflow → Model Registry`
2. **Prediction Flow**: `API Request → Model Loading → Feature Processing → Prediction → Response`
3. **Monitoring Flow**: `Metrics Collection → PostgreSQL → Grafana → Alerts`
4. **Orchestration Flow**: `Prefect Scheduler → Workflows → Monitoring → Notifications`

## 📁 Project Structure

```
mlops-2025-final_project/
├── 📁 src/
│   ├── 📁 api/                    # FastAPI application
│   ├── 📁 pipelines/              # ML training & prediction pipelines
│   ├── 📁 betting_simulator/      # Betting strategy simulation
│   ├── 📁 monitoring/             # Metrics collection & storage
│   ├── 📁 orchestration/          # Prefect workflows
│   └── 📁 data_integration/       # Data fetching & processing
├── 📁 scripts/                    # Setup & testing scripts
├── 📁 data/                       # Training data & datasets
├── 📁 grafana/                    # Grafana dashboards & config
├── 📁 alerts/                     # Alert configurations
├── 📄 docker-compose.yml         # Container orchestration
├── 📄 Makefile                   # Development commands
└── 📄 README.md                  # This file
```

## 🎯 System Performance

### 📊 Model Metrics
- **Accuracy**: 61.84% on 3,040 Premier League matches
- **Features**: 15 engineered features (team form, head-to-head, etc.)
- **Algorithm**: Random Forest Classifier
- **Training Time**: ~30 seconds on standard hardware

### ⚡ API Performance
- **Response Time**: <500ms for predictions
- **Throughput**: 1000+ requests/minute
- **Uptime**: 99.9% availability target
- **Health Checks**: Comprehensive endpoint monitoring

### 🔧 Infrastructure
- **Database**: PostgreSQL with optimized indexing
- **Containerization**: Docker Compose for easy deployment
- **Monitoring**: 100% component coverage with Grafana
- **Orchestration**: 4 automated Prefect workflows

## 🧪 Testing

### Integration Tests
```bash
make test        # Run all integration tests
make test-orch   # Test orchestration components
make health      # Health check all services
```

### Test Coverage
- ✅ **API Endpoints** - All REST endpoints tested
- ✅ **ML Pipeline** - Training and prediction workflows
- ✅ **Database** - Schema and data integrity
- ✅ **Monitoring** - Metrics collection and alerts
- ✅ **Orchestration** - Prefect workflow execution

## 📊 Monitoring

### Grafana Dashboards
- **Model Performance** - Accuracy trends and drift detection
- **API Metrics** - Request rates, response times, error rates
- **System Health** - Resource usage and service status
- **Betting Performance** - Strategy effectiveness and ROI

### Alert Conditions
- Model accuracy drops below 55%
- API response time exceeds 1 second
- Database connection failures
- Service downtime detection

## 🔄 Orchestration

### Prefect Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| **hourly-monitoring** | Every hour | Model performance & drift checks |
| **daily-predictions** | Daily at 9 AM | Generate predictions for upcoming matches |
| **weekly-retraining-check** | Weekly (Sunday) | Evaluate need for model retraining |
| **emergency-retraining** | Manual trigger | Immediate model retraining |

### Workflow Features
- **Retry Logic** - Automatic retry on failures
- **Notifications** - Slack/email alerts on completion
- **Logging** - Comprehensive workflow execution logs
- **Monitoring** - Real-time workflow status tracking

## 📚 Documentation

- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Premier League for providing match data
- MLflow community for experiment tracking tools
- Prefect team for orchestration framework
- FastAPI developers for the excellent web framework

## 🔧 Configuration

Key environment variables in `.env`:
```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mlops_db
POSTGRES_USER=mlops_user
POSTGRES_PASSWORD=mlops_password

# MLflow Configuration
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_ARTIFACT_ROOT=./mlruns

# Model Configuration
MODEL_REGISTRATION_THRESHOLD=0.6
TRAINING_DATA_PATH=data/real_data/premier_league_matches.parquet
```

### Service Ports
- **API Server**: 8000
- **MLflow**: 5000
- **Prefect UI**: 4200
- **Grafana**: 3000
- **PostgreSQL**: 5432

## 🔧 Monitoring & Orchestration

### Prefect Workflows
The system includes 4 automated workflows:

```bash
# Start Prefect server
uv run prefect server start --host 0.0.0.0 --port 4200

# Deploy and run workflows
uv run python -m src.orchestration.scheduler

# Manual workflow triggers
uv run python scripts/test_simple_orchestration.py
```

**Available Flows:**
- **Hourly Monitoring** - Model performance & drift detection
- **Daily Predictions** - Generate predictions for upcoming matches
- **Weekly Retraining** - Automated model retraining evaluation
- **Emergency Retraining** - Manual retraining trigger

### Grafana Dashboards
Real-time monitoring with PostgreSQL data source:

```bash
# Start Grafana
sudo systemctl start grafana-server

# Setup dashboard (automated)
uv run python scripts/setup_grafana.py

# Manual setup:
# 1. Go to http://localhost:3000 (admin/admin)
# 2. Add PostgreSQL data source (localhost:5432/mlops_db)
# 3. Import: grafana/dashboards/corrected_mlops_dashboard.json
```

**Dashboard Features:**
- Model performance metrics over time
- Prediction accuracy tracking
- System health indicators
- Real-time alerts and notifications

### Testing the Complete Stack
```bash
# Test API integration
uv run python scripts/test_simple_integration.py

# Test monitoring workflows
uv run python scripts/test_simple_monitoring.py

# Test end-to-end orchestration
uv run python scripts/test_end_to_end_monitoring.py
```

## 🛠️ Development

```bash
# API development
cd src/api && uv run uvicorn main:app --reload

# Database utilities
uv run python scripts/check_db_tables.py
uv run python scripts/clean_postgres.py
```

## 📊 Current Status

**✅ Production Ready**: Complete MLOps system with monitoring, orchestration, and automated workflows.

---

*Complete MLOps system for Premier League match prediction*
