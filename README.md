# Premier League Match Predictor

[![CI/CD Pipeline](https://github.com/RuiFSP/mlops-2025-final_project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/RuiFSP/mlops-2025-final_project/actions/workflows/ci-cd.yml)

A complete end-to-end MLOps pipeline for predicting Premier League match outcomes using real historical data and modern Python tooling.

## 🎯 Project Overview

This project successfully demonstrates a production-ready MLOps pipeline that:
- ✅ **Collects real data** from football-data.co.uk (3,040+ matches from 8 seasons)
- ✅ **Trains ML models** with enhanced probability outputs (55% accuracy - excellent for football)
- ✅ **Serves predictions** via FastAPI REST API with probability distributions
- ✅ **Tracks experiments** with MLflow
- ✅ **Uses modern tooling** (`uv`, `pyproject.toml`, Docker)
- ✅ **Evaluates with Brier score** - professional probabilistic evaluation
- ✅ **Compares with betting market** - removes bookmaker margin for fair comparison

## 🏆 Key Achievements

- **Real Data Integration**: Successfully integrated 8 seasons of Premier League data
- **Enhanced Model**: 55% accuracy with probability outputs (9% improvement)
- **Professional Evaluation**: Brier score evaluation and market comparison
- **Working API**: FastAPI service with probability distributions at `http://localhost:8000`
- **MLflow Tracking**: Complete experiment management with model versioning
- **Market Competitive**: Within 6% of betting market performance
- **Production Ready**: Docker support, proper testing, CI/CD workflows
- **Code Quality**: Zero errors, comprehensive testing, security-hardened Docker container

## 📋 Current Project Status

### ✅ **Completed Components**
| Component | Status | Description |
|-----------|--------|-------------|
| **Data Pipeline** | ✅ Complete | 3,040+ matches from 8 seasons, automated collection |
| **Model Training** | ✅ Complete | Random Forest with 55% accuracy, probability outputs |
| **API Service** | ✅ Complete | FastAPI with health checks, prediction endpoints |
| **Experiment Tracking** | ✅ Complete | MLflow integration with model versioning |
| **Testing** | ✅ Complete | 25/25 tests passing, unit test coverage |
| **Containerization** | ✅ Complete | Security-hardened Docker container |
| **Documentation** | ✅ Complete | Comprehensive README and code documentation |
| **Code Quality** | ✅ Complete | Linting, formatting, type hints, pre-commit hooks |
| **Model Monitoring** | ✅ Complete | Statistical drift detection, performance monitoring |

### ❌ **Not Implemented**
| Component | Status | Priority | Effort |
|-----------|--------|----------|--------|
| **Automated Retraining** | ❌ Missing | High | Medium |
| **Cloud Deployment** | ❌ Missing | Medium | High |
| **Advanced Features** | ❌ Missing | Low | High |

### 🔄 **Partially Implemented**
| Component | Status | What's Done | What's Missing |
|-----------|--------|-------------|----------------|
| **Orchestration** | 🔄 Partial | Manual training pipeline | Automated Prefect workflows |

## 🔥 Latest Enhancements

- **Model Monitoring System**: Complete drift detection and performance monitoring
- **Statistical Drift Detection**: KS-test for numerical, Chi-square for categorical features
- **Performance Degradation Alerts**: Automated tracking of model accuracy decline
- **Unified Monitoring Service**: Integrated drift and performance monitoring
- **Probability Outputs**: Model returns full probability distributions for Home/Draw/Away
- **Brier Score Evaluation**: Industry-standard evaluation for probabilistic predictions
- **Bookmaker Margin Removal**: Proper comparison with betting market odds
- **Enhanced Model Architecture**: Improved Random Forest with balanced classes
- **Better Features**: 10 features including margin-adjusted probabilities
- **API Improvements**: Confidence scores and probability breakdowns

## ✨ Code Quality & Standards

- **Zero VS Code Errors**: All type annotations, imports, and linting issues resolved
- **15/15 Tests Passing**: Comprehensive unit test coverage with pytest
- **Security Hardened**: Docker container uses non-root user and secure Ubuntu base
- **Modern Python**: Proper type hints, async/await patterns, and best practices
- **Clean Codebase**: Removed unnecessary files and unused modules for focused architecture
- **Production Ready**: All components tested and verified for deployment

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
# API available at http://localhost:8000

# Note: Model will only be loaded if model artifacts exist in ./models/
# To train a model first, run: python -m src.main
```

## 🐳 Docker Quick Start

### Development (API only)
```bash
# Build the Docker image
docker build -t premier-league-predictor .

# Show help for available options
docker run --rm premier-league-predictor --help

# Run API container (default behavior)
docker run -p 8000:8000 premier-league-predictor

# Run API with custom host/port
docker run -p 8080:8080 premier-league-predictor --host 0.0.0.0 --port 8080

# Test health endpoint
curl http://localhost:8000/health
```

### Production (with trained model)
```bash
# Ensure you have a trained model first
python -m src.main  # This will train and save model artifacts

# Run with model mounted
docker run -p 8000:8000 -v $(pwd)/models:/app/models premier-league-predictor

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea", "month": 3, "goal_difference": 0, "total_goals": 0}'
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
- **Enhanced Features**: Team encoding, date features, margin-adjusted probabilities
- **Model Architecture**: Random Forest with balanced classes and probability calibration
- **Validation**: Cross-validation with temporal consistency
- **Tracking**: All experiments logged in MLflow

## 🌐 API Service

### Available Endpoints
```bash
# Health check (shows model loading status)
GET http://localhost:8000/health

# Model information
GET http://localhost:8000/model/info

# Available teams
GET http://localhost:8000/teams

# Match prediction with probabilities
POST http://localhost:8000/predict
{
  "home_team": "Arsenal",
  "away_team": "Manchester United",
  "home_odds": 2.1,
  "draw_odds": 3.2,
  "away_odds": 3.5
}

# Enhanced Response with Probabilities
{
  "home_team": "Arsenal",
  "away_team": "Manchester United",
  "predicted_result": "Home Win",
  "home_win_probability": 0.438,
  "draw_probability": 0.387,
  "away_win_probability": 0.175,
  "prediction_confidence": 0.438
}
```

### Example Usage
```bash
# Check API health and model status
curl http://localhost:8000/health

# Get available teams
curl http://localhost:8000/teams

# Predict Arsenal vs Man United with odds
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Manchester United",
    "home_odds": 2.1,
    "draw_odds": 3.2,
    "away_odds": 3.5
  }'

# Alternative: Predict without odds (using default features)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Manchester United",
    "month": 3
  }'
```

## 🛠 Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: uv (modern, fast)
- **ML Framework**: scikit-learn
- **API Framework**: FastAPI
- **Experiment Tracking**: MLflow
- **Data Processing**: pandas, numpy
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Code Quality**: ruff (linting & formatting), mypy (type checking), pre-commit

## 📁 Project Structure

```
mlops-2025-final_project/
├── src/
│   ├── data_collection/           # Data collection from football-data.co.uk
│   ├── data_preprocessing/        # Data loading and preprocessing
│   ├── model_training/           # ML model training and validation
│   ├── evaluation/               # Model evaluation and metrics
│   ├── deployment/               # FastAPI application
│   └── monitoring/               # Model monitoring (placeholder for future)
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

### Enhanced Model Testing
```bash
# Test the enhanced model with probability outputs
python scripts/test_enhanced_model.py

# Test the enhanced API with probability outputs
python scripts/test_enhanced_api.py

# Demo the new monitoring system
python scripts/demo_monitoring.py
```

## 🔍 Pre-Commit Hooks & Quality Checks

### Automatic Quality Checks
Pre-commit hooks are set up to automatically run quality checks before each commit:

```bash
# Install pre-commit hooks (one-time setup)
uv run pre-commit install

# Now every git commit will automatically run:
# ✅ Code linting and formatting (ruff)
# ✅ Type checking (mypy - optional)
# ✅ Unit tests (pytest)
```

### Manual Quality Checks
Run all quality checks manually before committing:

```bash
# Run all checks at once
python scripts/run_checks.py

# Or run individual checks
uv run ruff check src tests   # Linting
uv run ruff format src tests  # Format code
uv run mypy src              # Type checking
python -m pytest tests/     # Run tests
```

### Pre-Commit Workflow
With pre-commit hooks installed, your workflow becomes:

```bash
# Make your changes
git add .
git commit -m "your message"  # Hooks run automatically here!
# If hooks pass → commit succeeds
# If hooks fail → commit blocked, fix issues and try again
git push origin main
```

## 📈 Model Performance

### Enhanced Results (Latest)
- **Accuracy**: 55.26% (+9.21% improvement)
- **Precision (macro)**: 52.17% (+24.57% improvement)
- **Recall (macro)**: 53.46% (+18.06% improvement)
- **F1 (macro)**: 52.34% (+22.24% improvement)

### Class-Specific Performance
- **Home Wins**: 68% precision, 55% recall (balanced prediction)
- **Away Wins**: 58% precision, 72% recall (excellent detection)
- **Draws**: 31% precision, 33% recall (challenging but now detectable)

### Probabilistic Evaluation
- **Model Brier Score**: 0.1881 (lower is better)
- **Betting Market Brier Score**: 0.1778 (baseline)
- **Market Comparison**: Within 5.8% of professional bookmakers

### Previous Results (for comparison)
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

#### Basic API Container
```bash
# Build secure image (Ubuntu 22.04 base, non-root user)
docker build -t premier-league-predictor .

# View available command-line options
docker run --rm premier-league-predictor --help

# Run container on port 8000 (API only, no model loaded)
docker run -p 8000:8000 premier-league-predictor

# Run with custom configuration
docker run -p 8080:8080 premier-league-predictor --host 0.0.0.0 --port 8080
```

#### Full Prediction Service with Model
```bash
# Run container with trained model mounted
docker run -p 8000:8000 -v $(pwd)/models:/app/models premier-league-predictor

# Test the full prediction service
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea", "month": 3, "goal_difference": 0, "total_goals": 0}'
```

#### Available Docker Endpoints
```bash
# Health check (shows model status)
curl http://localhost:8000/health
# Returns: {"status":"healthy","model_loaded":true}

# Model information
curl http://localhost:8000/model/info
# Returns: {"model_type":"random_forest","model_loaded":true,"features":[...]}

# Team list
curl http://localhost:8000/teams
# Returns: {"teams": ["Arsenal", "Chelsea", ...]}

# Match prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea", "month": 3, "goal_difference": 0, "total_goals": 0}'
# Returns: {"home_team":"Arsenal","away_team":"Chelsea","predicted_result":"Home Win","prediction_confidence":null}
```

**Security Features:**
- Ubuntu 22.04 LTS base image (no known vulnerabilities)
- Non-root user execution (`appuser`)
- Minimal attack surface with clean package installation
- Optimized build with proper layer caching

### Production Checklist
- ✅ Real data pipeline established
- ✅ Model training automated
- ✅ API service working
- ✅ Experiment tracking (MLflow)
- ✅ Testing framework (25/25 tests passing)
- ✅ Docker containerization (security-hardened)
- ✅ CI/CD workflows ready
- ✅ Zero code errors or vulnerabilities
- ✅ Model monitoring implementation
- ❌ Automated retraining workflows

## 📊 Monitoring & Maintenance

### Model Monitoring (NEW!)
- ✅ **Statistical Drift Detection**: KS-test for numerical features, Chi-square for categorical
- ✅ **Performance Monitoring**: Automated accuracy tracking and degradation alerts
- ✅ **Unified Monitoring Service**: Combined drift and performance monitoring
- ✅ **Health Checks**: Comprehensive monitoring system health status
- ✅ **Alert System**: Configurable thresholds with detailed reporting

### Data Quality
- ✅ Automated data collection from football-data.co.uk
- ✅ Data validation and quality checks
- ✅ Graceful handling of missing/malformed data

### Model Monitoring
- ✅ Model performance tracking in MLflow
- ✅ Evaluation reports with confusion matrices
- ✅ Real-time drift detection with statistical tests

### Operational
- ✅ Health check endpoints
- ✅ Structured logging
- ✅ Error handling and graceful degradation

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Run pre-commit checks
5. Ensure CI/CD pipeline passes

## 📝 Next Steps & Future Enhancements

### 🎯 **Immediate Next Steps** (High Priority)

#### 1. **Automated Retraining Pipeline**
- **Goal**: Implement scheduled model retraining with fresh data
- **Tech**: Prefect workflows with time-based triggers
- **Effort**: 2-3 days
- **Impact**: Keeps model performance optimal with latest match data

#### 2. **Cloud Deployment**
- **Goal**: Deploy to AWS/GCP/Azure with full CI/CD integration
- **Tech**: Kubernetes, Terraform, GitHub Actions
- **Effort**: 1 week
- **Impact**: Production-scale availability and reliability

#### 3. **Advanced Evidently Integration**
- **Goal**: Upgrade to full Evidently AI reports with HTML dashboards
- **Tech**: Evidently AI, automated report generation
- **Effort**: 2-3 days
- **Impact**: Professional-grade monitoring visualizations

### 🚀 **Strategic Enhancements** (Medium Priority)

#### 4. **Enhanced Feature Engineering**
- **Goal**: Add player data, injuries, weather, team form
- **Tech**: Additional data sources, feature pipelines
- **Effort**: 1-2 weeks
- **Impact**: Potential accuracy improvement to 60%+

#### 5. **A/B Testing Framework**
- **Goal**: Compare model versions in production
- **Tech**: Feature flags, traffic splitting
- **Effort**: 3-4 days
- **Impact**: Data-driven model improvement decisions

#### 6. **Real-time Data Streaming**
- **Goal**: Live match odds and team news integration
- **Tech**: Apache Kafka, WebSocket APIs
- **Effort**: 1 week
- **Impact**: Real-time prediction updates

### 💡 **Innovation Opportunities** (Nice to Have)

#### 7. **Multi-League Expansion**
- **Goal**: Extend to La Liga, Serie A, Bundesliga
- **Impact**: 10x more data, broader market appeal

#### 8. **Advanced ML Models**
- **Goal**: Neural networks, gradient boosting ensembles
- **Impact**: Potential accuracy improvements

#### 9. **User Interface**
- **Goal**: Web dashboard for predictions and monitoring
- **Tech**: React/Vue.js frontend
- **Impact**: User-friendly access to predictions

#### 10. **Betting Strategy Engine**
- **Goal**: Kelly criterion stake sizing, bankroll management
- **Impact**: Practical application for sports betting

### 🔧 **Technical Infrastructure** (Lower Priority)

- **Database Migration**: PostgreSQL for scalable data storage
- **Caching Layer**: Redis for faster prediction serving
- **API Rate Limiting**: Request throttling and authentication
- **Performance Optimization**: Model inference speed improvements
- **Monitoring Dashboards**: Grafana visualizations for system health

### 📈 **Success Metrics**

| Enhancement | Target Metric | Timeline |
|-------------|---------------|----------|
| Automated Retraining | Weekly model updates | 1 month |
| Cloud Deployment | 99.9% uptime | 1 month |
| Enhanced Features | 58%+ accuracy | 2 months |
| Real-time Streaming | <1s prediction latency | 2 months |

---

**Current Status**: ✅ **Foundation Complete** - Ready for production deployment and strategic enhancements

## 🎯 Program Summary

This **Premier League Match Predictor** is a complete, production-ready MLOps pipeline that predicts football match outcomes using 8 seasons of real Premier League data. The system demonstrates enterprise-level ML engineering practices with a focus on reliability, monitoring, and maintainability.

### **🏆 What This Project Delivers**
- **Intelligent Predictions**: 55% accuracy rate competitive with professional bookmakers
- **Real-World Data**: 3,040+ matches from actual Premier League seasons (2016-2024)
- **Production API**: FastAPI service with probability distributions and health monitoring
- **Complete Monitoring**: Statistical drift detection and performance degradation alerts
- **Enterprise Standards**: Docker deployment, comprehensive testing, CI/CD automation

### **💻 Technical Architecture**
```
Data Collection → Model Training → API Deployment → Monitoring → Alerting
     ↓               ↓              ↓              ↓           ↓
Real PL Data → Random Forest → FastAPI Server → Drift Detection → Performance Alerts
```

### **🚀 Production Features**
- **Zero-Error Codebase**: 25/25 tests passing, comprehensive type coverage
- **Security Hardened**: Non-root Docker containers, secure dependency management
- **Monitoring Ready**: Statistical drift detection with KS-test and Chi-square analysis
- **Experiment Tracking**: MLflow integration with model versioning and metrics
- **Modern Tooling**: uv package management, async FastAPI, professional logging

### **🌟 Business Value**
This project showcases **real-world MLOps engineering capabilities** that bridge academic ML and production systems. It demonstrates proficiency in data engineering, model deployment, monitoring, and DevOps practices essential for enterprise ML teams.

**Perfect for**: Portfolio demonstration, production deployment, educational reference, or foundation for advanced football analytics platforms.
