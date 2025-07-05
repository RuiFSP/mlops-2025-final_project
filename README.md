# Premier League Match Predictor

A complete end-to-end MLOps pipeline for predicting Premier League match outcomes using real historical data and modern Python tooling.

## 🎯 Project Overview

This project successfully demonstrates a production-ready MLOps pipeline that:
- ✅ **Collects real data** from football-data.co.uk (3,040+ matches from 8 seasons)
- ✅ **Trains ML models** with proper validation (46% accuracy - realistic for football)
- ✅ **Serves predictions** via FastAPI REST API
- ✅ **Tracks experiments** with MLflow
- ✅ **Uses modern tooling** (`uv`, `pyproject.toml`, Docker)

## 🏆 Key Achievements

- **Real Data Integration**: Successfully integrated 8 seasons of Premier League data
- **Working API**: FastAPI service running at `http://localhost:8001`
- **MLflow Tracking**: Complete experiment management with model versioning
- **Realistic Performance**: 46% accuracy (typical for football match prediction)
- **Production Ready**: Docker support, proper testing, CI/CD workflows

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
git clone <repository-url>
cd mlops-2025-final_project

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### 2. Collect Data
```bash
# Collect real Premier League data
python scripts/collect_real_data.py
```

### 3. Train Model
```bash
# Start MLflow server (in background)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

# Train the model
python -m src.main train --data-path data/real_data/
```

### 4. Start API Server
```bash
# Start the API server
python -m src.deployment.api
# API available at http://localhost:8001
```

## 📊 Data Pipeline

### Data Source
- **Primary**: football-data.co.uk (reliable, comprehensive)
- **Coverage**: 8 seasons (2016-2024), 3,040+ matches
- **Features**: Match results, team data, betting odds, statistics

### Data Quality
- ✅ Real historical match data
- ✅ Comprehensive team coverage (25+ teams)
- ✅ Time-series ready with proper date handling
- ✅ Betting odds for feature engineering

## 🤖 Machine Learning Pipeline

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~46% (realistic for football prediction)
- **Features**: Team names, match month, betting odds (home/draw/away)
- **Data Leakage**: Eliminated by removing post-match features

### Training Pipeline
- **Dataset Split**: Time-based (80/20 train/validation)
- **Feature Engineering**: Team encoding, date features, odds integration
- **Validation**: Cross-validation with temporal consistency
- **Tracking**: All experiments logged in MLflow

## 🌐 API Service

### Available Endpoints
```bash
# Health check
GET http://localhost:8001/health

# Single match prediction
POST http://localhost:8001/predict
{
  "home_team": "Arsenal",
  "away_team": "Manchester United", 
  "home_odds": 2.0,
  "draw_odds": 3.0,
  "away_odds": 2.5
}

# Response
{
  "home_team": "Arsenal",
  "away_team": "Manchester United",
  "predicted_result": "Draw"
}
```

### Example Usage
```bash
# Predict Arsenal vs Man United
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Manchester United",
    "home_odds": 2.0,
    "draw_odds": 3.0,
    "away_odds": 2.5
  }'
```

## 🛠 Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: uv (modern, fast)
- **ML Framework**: scikit-learn
- **API Framework**: FastAPI
- **Experiment Tracking**: MLflow
- **Data Processing**: pandas, numpy
- **Orchestration**: Prefect (configured)
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Monitoring**: EvidentlyAI (ready)

## 📁 Project Structure

```
mlops-2025-final_project/
├── src/
│   ├── data_collection/           # Data collection from football-data.co.uk
│   ├── data_preprocessing/        # Data loading and preprocessing  
│   ├── model_training/           # ML model training and validation
│   ├── evaluation/               # Model evaluation and metrics
│   ├── deployment/               # FastAPI application
│   └── monitoring/               # Model monitoring (EvidentlyAI)
├── data/
│   └── real_data/               # Real Premier League match data
├── models/                      # Trained model artifacts
├── notebooks/                   # Jupyter analysis notebooks
├── tests/                       # Unit and integration tests
├── scripts/                     # Utility scripts
├── evaluation_reports/          # Model evaluation results
├── mlruns/                      # MLflow experiment tracking
├── .github/workflows/           # CI/CD automation
├── pyproject.toml              # Modern Python dependencies
├── Dockerfile                  # Container configuration
└── Makefile                    # Development automation
```

## 🔄 Development Workflow

### Testing
```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src tests/

# Specific test suites
pytest tests/unit/
pytest tests/integration/
```

### Code Quality
```bash
# Format code
make format

# Lint code  
make lint

# Pre-commit checks
pre-commit run --all-files
```

### MLflow Experiment Tracking
```bash
# Start MLflow UI
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# View experiments at http://localhost:5000
```

## 📈 Model Performance

### Current Results
- **Accuracy**: 46.05%
- **Precision (macro)**: 27.6%
- **Recall (macro)**: 35.4%
- **F1 (macro)**: 30.1%

### Class Performance
- **Home Wins**: 50% precision, 81% recall (model favors home advantage)
- **Away Wins**: 33% precision, 25% recall  
- **Draws**: Very difficult to predict (realistic for football)

### Model Insights
- Home advantage is a strong signal (realistic)
- Betting odds provide valuable features
- Draw prediction remains challenging (typical in football)
- Performance is realistic for football match prediction

## 🚀 Deployment & Production

### Docker Deployment
```bash
# Build image
docker build -t premier-league-predictor .

# Run container
docker run -p 8001:8001 premier-league-predictor
```

### Production Checklist
- ✅ Real data pipeline established
- ✅ Model training automated
- ✅ API service working
- ✅ Experiment tracking (MLflow)
- ✅ Testing framework
- ✅ Docker containerization
- ✅ CI/CD workflows ready
- 🔄 Monitoring setup (EvidentlyAI configured)
- 🔄 Automated retraining (Prefect flows ready)

## 📊 Monitoring & Maintenance

### Data Quality
- Automated data collection from football-data.co.uk
- Data validation and quality checks
- Graceful handling of missing/malformed data

### Model Monitoring
- Model performance tracking in MLflow
- Evaluation reports with confusion matrices
- Ready for drift detection with EvidentlyAI

### Operational
- Health check endpoints
- Structured logging
- Error handling and graceful degradation

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Run pre-commit checks
5. Ensure CI/CD pipeline passes

## 📝 Next Steps

- [ ] Implement automated model retraining
- [ ] Add more sophisticated features (player data, injuries)
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
- [ ] Implement A/B testing for model versions
- [ ] Add real-time data streaming
- [ ] Enhance monitoring dashboards



