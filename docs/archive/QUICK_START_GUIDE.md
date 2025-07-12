# 🚀 Premier League MLOps System - Quick Start Guide

## 📋 Prerequisites

### Required Software
- **Python 3.10+** (check with `python --version`)
- **PostgreSQL** (local installation or Docker)
- **Git** (for cloning)
- **uv** package manager (install with `pip install uv`)

### Optional (for advanced features)
- **Docker & Docker Compose** (for containerized deployment)
- **Grafana** (for monitoring dashboards)

## 🔧 Step 1: Environment Setup

### 1.1 Clone and Navigate
```bash
# Clone the repository
git clone https://github.com/RuiFSP/mlops-2025-final_project.git
cd mlops-2025-final_project
```

### 1.2 Install Dependencies
```bash
# Install all dependencies using uv
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 1.3 Environment Configuration
```bash
# Copy environment template
cp config.env.example .env

# Edit .env file with your settings
nano .env  # or use your preferred editor
```

**Key environment variables to configure:**
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
```

## 🗄️ Step 2: Database Setup

### 2.1 PostgreSQL Installation
**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download and install from: https://www.postgresql.org/download/windows/

### 2.2 Database Creation
```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE mlops_db;
CREATE USER mlops_user WITH PASSWORD 'mlops_password';
GRANT ALL PRIVILEGES ON DATABASE mlops_db TO mlops_user;
\q
```

### 2.3 Initialize Database Schema
```bash
# Run database setup script
uv run python scripts/setup_database.py
```

## 🤖 Step 3: MLflow Setup

### 3.1 Start MLflow Server
```bash
# Start MLflow tracking server (in a new terminal)
uv run mlflow server --host 127.0.0.1 --port 5000
```

**Keep this terminal open** - MLflow will run in the background

### 3.2 Verify MLflow
Open browser: http://127.0.0.1:5000
You should see the MLflow UI

## 🏋️ Step 4: Train the Model

### 4.1 Run Training Pipeline
```bash
# Train and register the model
uv run python -m src.pipelines.training_pipeline
```

**Expected output:**
```
✅ Model trained successfully
✅ Model registered in MLflow
✅ Model accuracy: 61.84%
```

### 4.2 Verify Model Registration
Check MLflow UI at http://127.0.0.1:5000 - you should see the "premier_league_predictor" model

## 🚀 Step 5: Start the API Server

### 5.1 Launch the API
```bash
# Start the FastAPI server
cd src/api
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
🚀 Starting Premier League Match Predictor API...
✅ All components initialized successfully
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 5.2 Verify API Health
Open browser: http://localhost:8000/health
You should see: `{"status": "healthy", "components": {...}}`

## 🔍 Step 6: Test the System

### 6.1 Interactive API Documentation
Open browser: http://localhost:8000/docs
- Try the `/predict` endpoint with: `{"home_team": "Arsenal", "away_team": "Chelsea"}`
- Explore all available endpoints

### 6.2 Run Integration Tests
```bash
# In a new terminal (keep API running)
cd /path/to/mlops-2025-final_project
uv run python scripts/test_simple_integration.py
```

**Expected output:**
```
✅ API Health: healthy
✅ Prediction: H (confidence: 0.48)
✅ Model: premier_league_predictor (accuracy: 0.62)
✅ Found 3 upcoming matches
✅ Betting simulation: 1 bets placed
🎉 ALL TESTS PASSED!
```

## 🔧 Step 7: Test Orchestration (Optional)

### 7.1 Test Orchestration System
```bash
# Test the Prefect orchestration
uv run python scripts/test_simple_orchestration.py
```

**Expected output:**
```
✅ Performance check result: 0.618 accuracy
✅ Drift analysis result: drift_detected=False
✅ Generated 3 predictions
✅ Alert system working
```

## 📊 Step 8: Access System Components

### 8.1 Main Access Points
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **MLflow UI**: http://127.0.0.1:5000

### 8.2 Key API Endpoints
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'

# Model information
curl "http://localhost:8000/model/info"

# Upcoming matches
curl "http://localhost:8000/matches/upcoming"

# Betting statistics
curl "http://localhost:8000/betting/statistics"
```

## 🐳 Step 9: Docker Deployment (Optional)

### 9.1 Using Docker Compose
```bash
# Start all services with Docker
docker-compose up -d

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000
# MLflow: http://localhost:5000
```

## 🔍 Step 10: Monitoring (Optional)

### 10.1 Grafana Dashboard
If using Docker:
1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Import dashboard from: `grafana/dashboards/mlops_dashboard.json`

## 🛠️ Troubleshooting

### Common Issues

**1. Database Connection Error**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -h localhost -U mlops_user -d mlops_db
```

**2. MLflow Connection Error**
```bash
# Check MLflow server is running
curl http://127.0.0.1:5000

# Restart MLflow server
uv run mlflow server --host 127.0.0.1 --port 5000
```

**3. API Import Errors**
```bash
# Check virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv sync
```

**4. Model Not Found**
```bash
# Re-run training pipeline
uv run python -m src.pipelines.training_pipeline

# Check MLflow UI for registered models
```

### Getting Help

**Check logs:**
```bash
# API logs are displayed in the terminal where you started the API
# Database logs: sudo journalctl -u postgresql
# MLflow logs: Check the terminal where MLflow is running
```

**Verify system:**
```bash
# Run comprehensive test
uv run python scripts/test_simple_integration.py

# Check database tables
uv run python scripts/check_db_tables.py
```

## 🎯 What You Should See

### Successfully Running System:
1. **PostgreSQL**: Running on port 5432
2. **MLflow**: Running on http://127.0.0.1:5000
3. **API**: Running on http://localhost:8000
4. **Model**: Registered in MLflow with 61.84% accuracy
5. **Database**: Tables created and populated
6. **Tests**: All integration tests passing

### Key Features Working:
- ✅ Match predictions with confidence scores
- ✅ Betting simulation with real-time statistics
- ✅ Model performance monitoring
- ✅ Automated workflows (orchestration)
- ✅ Health monitoring and alerts
- ✅ Real-time data integration

---

## 🎉 Success!

You now have a complete MLOps system running with:
- **Machine Learning**: Premier League match predictions
- **API**: FastAPI with comprehensive endpoints
- **Database**: PostgreSQL with betting simulation
- **Monitoring**: Performance tracking and alerts
- **Orchestration**: Automated workflows
- **Documentation**: Interactive API docs

**Next Steps:**
- Explore the API documentation at http://localhost:8000/docs
- Try different team predictions
- Monitor system performance
- Set up Grafana dashboards for advanced monitoring

**Need help?** Check the troubleshooting section above or review the logs for any error messages.
