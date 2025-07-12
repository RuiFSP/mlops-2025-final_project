# Premier League Match Predictor - Complete MLOps System

A comprehensive end-to-end MLOps pipeline for predicting Premier League match outcomes with automated retraining, monitoring, and betting simulation.

## 🎉 **COMPLETE END-TO-END MONITORING SYSTEM ACHIEVED!**

This MLOps system now features **full end-to-end monitoring** with real-time dashboards, automated orchestration, and complete observability. You can simultaneously view:
- **Prefect Flow Orchestration** at http://localhost:4200
- **Grafana Metrics Dashboards** at http://localhost:3000
- **MLflow Model Tracking** at http://localhost:5000
- **FastAPI Live Predictions** at http://localhost:8000

## 🚀 **Quick Start (Local Development)**

### **Prerequisites**
- Python 3.10+
- `uv` package manager
- Local PostgreSQL server
- Local MLflow server
- Optional: Docker for containerized deployment

### **Setup**
1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp config.env.example .env
   # Edit .env with your local PostgreSQL and MLflow settings
   ```

3. **Start local services:**
   - PostgreSQL on `localhost:5432`
   - MLflow on `http://127.0.0.1:5000`

### **Run the Complete System**
1. **Train the model:**
   ```bash
   uv run python -m src.pipelines.training_pipeline
   ```

2. **Start the API server:**
   ```bash
   cd src/api
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Start Prefect orchestration:**
   ```bash
   uv run prefect server start --host 0.0.0.0 --port 4200
   ```

4. **Start Grafana monitoring:**
   ```bash
   sudo systemctl start grafana-server
   # Setup: uv run python scripts/setup_grafana.py
   ```

5. **Test the complete monitoring system:**
   ```bash
   uv run python scripts/test_end_to_end_monitoring.py
   ```

### **🆕 Access the Complete System**
- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **MLflow UI**: `http://127.0.0.1:5000`
- **🎯 Prefect Orchestration**: `http://localhost:4200`
- **📊 Grafana Dashboards**: `http://localhost:3000` (admin/admin)

## 📊 **Complete Monitoring Stack**

### **Real-Time Monitoring Dashboard**
- **Dashboard URL**: `http://localhost:3000/d/388697c5-a3e7-43fb-a653-b90b7a86e703`
- **Model Metrics Count**: Live count of all model metrics
- **Predictions Count**: Real-time prediction tracking
- **Metrics by Type**: Detailed breakdown of accuracy, precision, recall, F1-score, AUC
- **Time Series Visualization**: Historical performance trends
- **Recent Activity**: Live log of all monitoring activities
- **Auto-refresh**: 30-second updates for real-time monitoring

### **Orchestration & Automation**
- **Prefect Flow UI**: Complete workflow visualization and monitoring
- **Automated Retraining**: Intelligent model retraining based on performance thresholds
- **Performance Monitoring**: Continuous evaluation with drift detection
- **Alert System**: Automated alerts for system issues and performance degradation

## 🔧 **Configuration**

### **Environment Variables (.env)**
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

### **Betting Configuration**
- **Initial Balance**: £1000.0
- **Confidence Threshold**: 0.6 (production) ✅
- **Margin Threshold**: 0.1 (production) ✅
- **Max Bet Percentage**: 5% of balance

## 📊 **System Performance**

### **Model Performance**
- **Accuracy**: 61.84% (excellent for football prediction)
- **Model Type**: Random Forest with 15 features
- **Training Data**: 3,040 Premier League matches (2017-2023)
- **Features**: Betting odds + match statistics

### **Real Data Integration** ✅
- **Upcoming Matches**: Fetches real Premier League fixtures via football-data.org API
- **Fallback System**: Intelligent fallback with realistic team matchups when API unavailable
- **Team Mapping**: Normalized team names consistent with training data
- **Realistic Odds**: Generated based on team strength ratings

### **🆕 REST API** ✅
- **FastAPI Framework**: High-performance API with automatic OpenAPI documentation
- **Prediction Endpoints**: Single and batch match predictions
- **Betting Simulation**: API-based betting simulation with real-time statistics
- **Health Monitoring**: Comprehensive health checks and component status
- **Interactive Documentation**: Swagger UI and ReDoc available
- **Real-time Data**: Integration with live Premier League data

### **🆕 Automated Orchestration** ✅
- **Prefect Orchestration**: Complete workflow orchestration with intelligent scheduling
- **Performance Monitoring**: Continuous evaluation of model performance with drift detection
- **Intelligent Triggers**: Automatic retraining based on accuracy thresholds and model age
- **MLflow Integration**: Seamless model versioning and experiment tracking
- **Alert System**: Comprehensive alerting for system issues and performance degradation
- **Configurable Workflows**: Daily predictions, weekly monitoring, configurable retraining

### **🆕 Advanced Monitoring** ✅
- **EvidentlyAI Integration**: Statistical drift detection and model monitoring
- **Grafana Dashboards**: Comprehensive visualization of system metrics
- **Real-time Metrics**: Live monitoring of predictions, accuracy, and system health
- **Performance Analytics**: Detailed analysis of betting performance and model accuracy
- **Alert Management**: Configurable alerts for various system events

## 🏗️ **Architecture**

### **Complete MLOps Pipeline with End-to-End Monitoring**
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

## 🎯 **Complete End-to-End Monitoring**

### **Successfully Achieved:**
- ✅ **Real-time Prefect Flow Orchestration**: View live task execution at http://localhost:4200
- ✅ **Grafana Metrics Dashboards**: Real-time metrics visualization at http://localhost:3000
- ✅ **Automated Data Pipeline**: End-to-end data processing with live monitoring
- ✅ **Performance Tracking**: 46+ metrics stored with time-series visualization
- ✅ **Alert System**: Automated alerts for system events and performance issues
- ✅ **Database Integration**: PostgreSQL metrics storage with 22+ predictions tracked
- ✅ **Clean Resource Management**: Single working dashboard and data source

### **🆕 Monitoring Features**
- **Live Metrics**: Real-time model performance tracking
- **System Health**: Comprehensive health checks across all components
- **Drift Detection**: Statistical analysis of model performance over time
- **Automated Alerts**: Configurable alerts for various system events
- **Performance Analytics**: Detailed analysis of betting performance and accuracy

## 🔗 **API Endpoints**

### **Predictions**
- `POST /predict` - Single match prediction
- `POST /predict/batch` - Batch predictions
- `GET /predictions/upcoming` - Upcoming match predictions

### **Betting Simulation**
- `POST /betting/simulate` - Simulate betting on predictions
- `GET /betting/statistics` - Get betting statistics

### **Data & Status**
- `GET /health` - API health check
- `GET /model/info` - Model information
- `GET /matches/upcoming` - Upcoming matches

### **🆕 Automated Retraining**
- `GET /retraining/status` - Retraining system status
- `POST /retraining/check` - Run immediate performance check
- `POST /retraining/force` - Force immediate retraining
- `GET /retraining/history` - Retraining history and metrics

**📖 Full API Documentation**: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## 📝 **Available Scripts**

### **Core Pipeline**
- `uv run python -m src.pipelines.training_pipeline` - Train and register model
- `uv run python -m src.pipelines.prediction_pipeline` - Generate predictions

### **🆕 API Operations**
- `cd src/api && uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload` - Start API server
- `uv run python scripts/test_api.py` - Test all API endpoints

### **🆕 Orchestration & Monitoring**
- `uv run python scripts/test_simple_orchestration.py` - Test orchestration system
- `uv run python scripts/test_simple_integration.py` - Full integration test
- `uv run python scripts/test_retraining.py` - Test retraining system

### **Testing & Debugging**
- `uv run python scripts/test_betting_simulation.py` - Full betting simulation
- `uv run python scripts/debug_bets_table.py` - Check database state
- `uv run python scripts/clean_postgres.py` - Clean all tables

### **Database Management**
- `uv run python scripts/setup_database.py` - Initialize database schema
- `uv run python scripts/check_db_tables.py` - Verify table structure

## 🔧 **System Components**

### **Orchestration (Prefect)**
- **Automated Workflows**: Daily predictions, weekly monitoring, configurable retraining
- **Task Management**: Intelligent task scheduling and dependency management
- **Error Handling**: Comprehensive error handling and retry mechanisms
- **Monitoring Integration**: Real-time monitoring of workflow execution

### **Monitoring (EvidentlyAI + Grafana)**
- **Model Drift Detection**: Statistical analysis of prediction drift
- **Performance Monitoring**: Continuous accuracy and performance tracking
- **Visual Dashboards**: Comprehensive Grafana dashboards for system observability
- **Alert System**: Configurable alerts for various system events

### **Data Management**
- **PostgreSQL**: Centralized database for predictions, metrics, and betting data
- **MLflow**: Model versioning, experiment tracking, and model registry
- **Real-time Data**: Integration with live Premier League data

## 📊 **Monitoring & Dashboards**

### **Grafana Dashboard Features**
- **Performance Overview**: Model accuracy, prediction confidence, system health
- **Drift Detection**: Statistical drift analysis and visualization
- **Betting Analytics**: ROI tracking, win rates, and betting performance
- **System Metrics**: API response times, database performance, error rates
- **Alert Management**: Configurable alerts for various thresholds

### **Access Monitoring**
- **Grafana**: `http://localhost:3000` (when running via Docker)
- **MLflow**: `http://127.0.0.1:5000`
- **Prefect**: `http://localhost:4200` (when running Prefect server)

## 📄 **Documentation**

### **Implementation Details**
- **[ORCHESTRATION_IMPLEMENTATION.md](ORCHESTRATION_IMPLEMENTATION.md)**: Detailed orchestration architecture
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**: Complete API reference
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: Detailed system architecture

### **Configuration**
- **[pyproject.toml](pyproject.toml)**: Project dependencies and configuration
- **[config.env.example](config.env.example)**: Environment configuration template
- **[docker-compose.yml](docker-compose.yml)**: Container orchestration

## 🚀 **Deployment**

### **Docker Deployment**
```bash
# Start all services
docker-compose up -d

# Access services
- API: http://localhost:8000
- Grafana: http://localhost:3000
- MLflow: http://localhost:5000
```

### **Production Considerations**
- **Monitoring**: Comprehensive monitoring with Grafana dashboards
- **Scalability**: Prefect orchestration for workflow management
- **Reliability**: Database-backed metrics and alert system
- **Security**: Environment-based configuration management

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `uv run python scripts/test_simple_integration.py`
5. Submit a pull request

---

**🎉 Premier League Match Predictor - Complete MLOps System with Orchestration & Monitoring!**
