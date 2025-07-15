# Premier League MLOps System

A comprehensive MLOps system for Premier League match prediction using real-time data pipelines, machine learning workflows, and automated model training with MLflow tracking and Prefect orchestration.

## 🏆 System Overview

This production-ready MLOps system provides end-to-end machine learning workflows for Premier League match prediction:

- **Real Data Integration**: Fetches live Premier League data from football-data.co.uk and football-data.org APIs
- **Automated ML Pipelines**: Prefect-orchestrated workflows for data processing, model training, and prediction generation
- **Model Tracking**: MLflow integration for experiment tracking, model registry, and performance monitoring
- **Interactive Dashboard**: Streamlit web interface for triggering workflows and monitoring system health
- **Scalable Architecture**: Docker-containerized microservices with proper networking and data persistence

## ✅ Current System Status

**All Core Features Working:**
- ✅ **Prefect Workflows**: 3 active deployments running successfully
- ✅ **Real Data Pipelines**: Fetching actual Premier League match data (1000+ historical matches)
- ✅ **Model Training**: RandomForest classifier with MLflow experiment tracking
- ✅ **Live Predictions**: Generating predictions for upcoming Premier League matches
- ✅ **API Integration**: FastAPI backend with working Prefect deployment triggers
- ✅ **Dashboard Interface**: Streamlit UI for workflow management and monitoring

**System Health:**
- API: ✅ Healthy
- MLflow: ✅ Healthy  
- Prefect: ✅ Healthy
- Model: ✅ Available
- Database: ✅ Connected

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local development)

### 1. Start the System
```bash
git clone <repository-url>
cd mlops-2025-final_project
docker-compose up -d
```

### 2. Access the Interfaces
- **Main Dashboard**: http://localhost:8501
- **Prefect UI**: http://localhost:4200 (workflow monitoring)
- **MLflow UI**: http://localhost:5000 (experiment tracking)
- **API Documentation**: http://localhost:8000/docs

### 3. Trigger Workflows
Use the Streamlit dashboard buttons or API endpoints:

```bash
# Data Pipeline - Fetch latest Premier League data
curl -X POST http://localhost:8000/workflows/etl

# Training Pipeline - Train model with MLflow tracking
curl -X POST "http://localhost:8000/workflows/trigger" \
  -H "Content-Type: application/json" \
  -d '{"flow_name": "training_pipeline_flow", "parameters": {"force_retrain": true}}'

# Prediction Pipeline - Generate match predictions
curl -X POST http://localhost:8000/workflows/predictions
```

## 🏗️ Architecture

### Services
- **Prefect Server**: Workflow orchestration and scheduling
- **Prefect Worker**: Executes workflows from `premier-league-pool`
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: Backend API for workflow triggers and predictions
- **Streamlit**: Interactive dashboard and monitoring interface

### Active Deployments
- `data_pipeline_flow-deployment`: Real data fetching from Premier League APIs
- `training_pipeline_flow-deployment`: Model training with MLflow integration
- `prediction_pipeline_flow-deployment`: Live prediction generation

### Data Sources
- **Historical Data**: football-data.co.uk (3+ years of match data)
- **Live Data**: football-data.org API (upcoming fixtures)
- **Features**: Betting odds, shots, corners, fouls, cards, team statistics

## 🧪 Machine Learning Pipeline

### 1. Data Pipeline (`data_pipeline_flow`)
- Fetches real Premier League data from external APIs
- Processes and validates match statistics
- Stores data in structured format for training
- **Output**: `data/processed/premier_league_data_YYYYMMDD_HHMMSS.csv`

### 2. Training Pipeline (`training_pipeline_flow`)
- Loads processed Premier League data
- Trains RandomForestClassifier with hyperparameter tuning
- Logs experiments, metrics, and models to MLflow
- Registers models if accuracy > 60%
- **Output**: Trained model in `models/` and MLflow registry

### 3. Prediction Pipeline (`prediction_pipeline_flow`)
- Fetches upcoming Premier League fixtures
- Generates match outcome predictions (Home/Draw/Away)
- Returns confidence scores and probabilities
- **Output**: JSON predictions with confidence intervals

## 📊 Model Performance

**Current Model Metrics:**
- **Algorithm**: RandomForestClassifier (100 estimators, max_depth=10)
- **Features**: Betting odds (B365H/D/A), shots, corners, fouls, cards
- **Training Data**: 1000+ Premier League matches (2020-2024)
- **Validation**: Cross-validation with accuracy tracking
- **Tracking**: Full MLflow experiment logging

## 🔧 API Endpoints

### Workflow Management
- `POST /workflows/etl` - Trigger data fetching pipeline
- `POST /workflows/trigger` - Generic workflow trigger with parameters
- `POST /workflows/predictions` - Generate match predictions

### System Monitoring
- `GET /status` - System health check
- `GET /system/status` - Detailed service status (needs fix)
- `GET /model/info` - Current model information

### Predictions
- `POST /predictions/match` - Single match prediction
- `GET /predictions/latest` - Latest prediction results

## 🛠️ Development

### Local Setup
```bash
# Install dependencies
uv sync

# Run tests
python -m pytest tests/

# Start individual services
uv run streamlit run src/dashboard/app.py
uv run uvicorn src.api.enhanced_api:app --reload
```

### Docker Development
```bash
# Rebuild specific service
docker-compose build api
docker-compose restart api

# View logs
docker logs premier-league-api
docker logs premier-league-prefect-worker

# Execute commands in container
docker exec premier-league-api python -c "import requests; print('API works')"
```

## 📈 Monitoring & Observability

### Prefect UI (http://localhost:4200)
- Monitor workflow executions and logs
- View deployment status and run history
- Debug failed runs and performance metrics

### MLflow UI (http://localhost:5000)
- Track experiment runs and model performance
- Compare model versions and metrics
- Manage model registry and deployments

### Streamlit Dashboard (http://localhost:8501)
- Real-time system status monitoring
- One-click workflow triggering
- Manual refresh capability (autorefresh removed)

## 🚧 Known Issues & Next Steps

### Current Limitations
1. **System Status Endpoint**: `/system/status` returns 500 error (needs debugging)
2. **Error Handling**: Some API endpoints need better error responses
3. **Data Validation**: Additional data quality checks needed
4. **Model Versioning**: Automated model promotion pipeline missing

### Immediate Next Steps
1. **Fix System Status**: Debug and repair `/system/status` endpoint
2. **Add Data Validation**: Implement comprehensive data quality checks
3. **Enhanced Monitoring**: Add metrics for data freshness and model drift
4. **CI/CD Pipeline**: Automated testing and deployment workflows
5. **Performance Optimization**: Model serving optimization and caching

### Future Enhancements
1. **Real-time Inference**: WebSocket-based live prediction updates
2. **Advanced Models**: Implement deep learning models (LSTM, Transformer)
3. **Feature Engineering**: Advanced statistical features and team form analysis
4. **A/B Testing**: Model comparison and gradual rollout capabilities
5. **Alerting**: Slack/email notifications for workflow failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**System Last Updated**: 2025-07-15  
**Status**: ✅ Production Ready - All Core Workflows Operational