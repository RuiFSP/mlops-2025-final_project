# Premier League Match Predictor - Complete MLOps System

A comprehensive end-to-end MLOps pipeline for predicting Premier League match outcomes with automated retraining, monitoring, and betting simulation.

## 🎉 **COMPLETE END-TO-END MONITORING SYSTEM**

This MLOps system features **full end-to-end monitoring** with real-time dashboards, automated orchestration, and complete observability:

- **🔮 FastAPI Live Predictions** → http://localhost:8000
- **📊 Grafana Metrics Dashboards** → http://localhost:3000
- **🎯 Prefect Flow Orchestration** → http://localhost:4200
- **🧪 MLflow Model Tracking** → http://localhost:5000

## 🚀 **Quick Start (5 Minutes)**

### Prerequisites
- **Python 3.10+** and **uv** package manager (`pip install uv`)
- **PostgreSQL** (local installation)
- **Git** for cloning

### 1. Setup
```bash
# Clone and setup
git clone https://github.com/RuiFSP/mlops-2025-final_project.git
cd mlops-2025-final_project
uv sync

# Configure environment
cp config.env.example .env
# Edit .env with your PostgreSQL settings
```

### 2. Database Setup
```bash
# Install PostgreSQL, then create database
sudo -u postgres psql
CREATE DATABASE mlops_db;
CREATE USER mlops_user WITH PASSWORD 'mlops_password';
GRANT ALL PRIVILEGES ON DATABASE mlops_db TO mlops_user;
\q

# Initialize database schema
uv run python scripts/setup_database.py
```

### 3. Start Services

**Terminal 1 - MLflow:**
```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2 - Train Model:**
```bash
uv run python -m src.pipelines.training_pipeline
```

**Terminal 3 - API:**
```bash
cd src/api
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test System
```bash
# Terminal 4 - Test everything
uv run python scripts/test_simple_integration.py
```

Expected output:
```
✅ API Health: healthy
✅ Prediction: H (confidence: 0.48)
✅ Model: premier_league_predictor (accuracy: 0.62)
✅ Found 3 upcoming matches
✅ Betting simulation: 1 bets placed
🎉 ALL TESTS PASSED!
```

## 🎯 **Access Points**

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **MLflow UI**: http://127.0.0.1:5000

## 📊 **Complete End-to-End Monitoring**

### Start Full Monitoring Stack
```bash
# Start Prefect orchestration
uv run prefect server start --host 0.0.0.0 --port 4200

# Start Grafana monitoring
sudo systemctl start grafana-server
# Setup: uv run python scripts/setup_grafana.py

# Test complete monitoring system
uv run python scripts/test_end_to_end_monitoring.py
```

### Access Monitoring
- **🎯 Prefect Orchestration**: http://localhost:4200
- **📊 Grafana Dashboards**: http://localhost:3000 (admin/admin)
- **Dashboard URL**: http://localhost:3000/d/388697c5-a3e7-43fb-a653-b90b7a86e703

### Monitoring Features
- **Live Metrics**: Real-time model performance tracking
- **Performance Tracking**: 46+ metrics stored with time-series visualization
- **Alert System**: Automated alerts for system events and performance issues
- **Database Integration**: PostgreSQL metrics storage with 22+ predictions tracked
- **Auto-refresh**: 30-second updates for real-time monitoring

## 🏗️ **Architecture**

### Complete MLOps Pipeline
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   MLflow        │    │   Model         │
│   Pipeline      │───▶│   Tracking      │───▶│   Registry      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prediction    │    │   PostgreSQL    │    │   Betting       │
│   Pipeline      │───▶│   Database      │───▶│   Simulation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ✅ REST API    │    │   Real Data     │    │   ✅ Prefect    │
│   (FastAPI)     │───▶│   Integration   │───▶│   Orchestration │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ✅ Monitoring  │    │   ✅ EvidentlyAI│    │   ✅ Grafana    │
│   & Metrics     │───▶│   Drift         │───▶│   Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 **Configuration**

### Environment Variables (.env)
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_ARTIFACT_ROOT=./mlruns

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mlops_db
POSTGRES_USER=mlops_user
POSTGRES_PASSWORD=mlops_password

# Data Paths
TRAINING_DATA_PATH=data/real_data/premier_league_matches.parquet

# Orchestration Configuration
PREFECT_API_URL=http://localhost:4200
ENABLE_MONITORING=true
ENABLE_RETRAINING=true
```

### Betting Configuration
- **Initial Balance**: £1000.0
- **Confidence Threshold**: 0.6 (production) ✅
- **Margin Threshold**: 0.1 (production) ✅
- **Max Bet Percentage**: 5% of balance

## 📈 **System Performance**

### Model Performance
- **Accuracy**: 61.84% (excellent for football prediction)
- **Model Type**: Random Forest with 15 features
- **Training Data**: 3,040 Premier League matches (2017-2023)
- **Features**: Betting odds + match statistics

### System Features ✅
- **✅ REST API**: FastAPI with 9 endpoints, full documentation
- **✅ Real Data Integration**: Live Premier League data via football-data.org API
- **✅ Automated Orchestration**: Prefect workflow orchestration with intelligent scheduling
- **✅ Advanced Monitoring**: EvidentlyAI + Grafana dashboards with drift detection
- **✅ Betting Simulation**: Real-time betting simulation with statistics
- **✅ Model Registry**: MLflow model versioning and experiment tracking

## 🔍 **Quick Test**

```bash
# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'

# Check model info
curl "http://localhost:8000/model/info"

# View upcoming matches
curl "http://localhost:8000/matches/upcoming"

# Check betting statistics
curl "http://localhost:8000/betting/statistics"
```

## 🛠️ **Troubleshooting**

### Common Issues

**Database Connection Error**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U mlops_user -d mlops_db
```

**MLflow Connection Error**
```bash
# Check MLflow server is running
curl http://127.0.0.1:5000

# Restart MLflow server
uv run mlflow server --host 127.0.0.1 --port 5000
```

**API Import Errors**
```bash
# Check virtual environment
source .venv/bin/activate

# Reinstall dependencies
uv sync
```

**Model Not Found**
```bash
# Re-run training pipeline
uv run python -m src.pipelines.training_pipeline

# Check MLflow UI for registered models
```

**Grafana Dashboard Issues**
- **Can't see data**: Check PostgreSQL data source configuration
- **Dashboard empty**: Run `uv run python scripts/setup_grafana.py`
- **Login issues**: Use admin/admin credentials

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

# Test end-to-end monitoring
uv run python scripts/test_end_to_end_monitoring.py
```

## 🐳 **Docker Deployment (Optional)**

```bash
# Start all services with Docker
docker-compose up -d

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000
# MLflow: http://localhost:5000
```

## 🎯 **What You Should See**

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

## 📚 **Documentation**

- **API Documentation**: See [API.md](API.md) for detailed API reference
- **🎯 Implementation Plan**: See [BETTING_SYSTEM_IMPLEMENTATION_PLAN.md](BETTING_SYSTEM_IMPLEMENTATION_PLAN.md) for the roadmap to transform this into a professional betting system
- **Advanced Configuration**: See [docs/](docs/) for additional documentation
- **Interactive API Docs**: http://localhost:8000/docs

## 🎉 **Success!**

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

**🎉 Premier League Match Predictor - Complete MLOps System with End-to-End Monitoring!**
