# 🏈 Premier League Match Predictor - MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

A **production-ready MLOps pipeline** for predicting Premier League football match outcomes using real historical data and modern Python tooling.

## 🎯 What This Project Demonstrates

- ✅ **Real Data Integration** (3,040+ matches from 8 seasons)
- ✅ **End-to-End ML Pipeline** (data → training → serving)
- ✅ **RESTful API Service** (FastAPI with async support)
- ✅ **Experiment Tracking** (MLflow for reproducibility)
- ✅ **Modern Python Tooling** (uv, pyproject.toml, type hints)
- ✅ **Production Practices** (Docker, CI/CD, testing, monitoring)

## 🚀 Quick Demo

```bash
# 1. Setup (< 2 minutes)
git clone https://github.com/yourusername/mlops-2025-final_project.git
cd mlops-2025-final_project
uv sync

# 2. Get Data (automatic download)
python scripts/collect_real_data.py

# 3. Train Model
python -m src.main train --data-path data/real_data/

# 4. Start API & Make Predictions
python -m src.deployment.api &
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Manchester United", "home_odds": 2.0, "draw_odds": 3.0, "away_odds": 2.5}'
```

## 📊 Key Results

- **Model Performance**: 46% accuracy (realistic for football prediction)
- **Data Pipeline**: Automated collection from football-data.co.uk
- **API Response Time**: < 100ms for predictions
- **MLflow Experiments**: Complete tracking and reproducibility

## 🛠 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Collection** | football-data.co.uk API | Real match data (8 seasons) |
| **ML Framework** | scikit-learn | Random Forest classification |
| **API Service** | FastAPI | High-performance async API |
| **Experiment Tracking** | MLflow | Model versioning & metrics |
| **Package Manager** | uv | Fast, modern dependency management |
| **Containerization** | Docker | Reproducible deployment |
| **CI/CD** | GitHub Actions | Automated testing & deployment |

## 📁 Project Architecture

```
├── src/
│   ├── data_collection/     # Automated data pipeline
│   ├── model_training/      # ML training with MLflow
│   ├── deployment/          # FastAPI service
│   └── monitoring/          # Model performance tracking
├── data/real_data/          # 3,040 Premier League matches
├── tests/                   # Comprehensive test suite
├── .github/workflows/       # CI/CD automation
└── docker/                  # Production containers
```

## 📈 ML Pipeline Features

- **Data Validation**: Automated quality checks
- **Feature Engineering**: Team encoding, odds integration
- **Model Training**: Cross-validation with temporal splits
- **Evaluation**: Comprehensive metrics and visualizations
- **Serving**: RESTful API with health checks
- **Monitoring**: Performance tracking and drift detection

## 🎓 Learning Outcomes

This project demonstrates:
- Modern MLOps practices and tooling
- Real-world data collection and processing
- Production API development with FastAPI
- Experiment tracking and model management
- Docker containerization and CI/CD
- Testing strategies for ML systems

Perfect for **data scientists**, **ML engineers**, and **software developers** looking to understand production ML systems.

---

⭐ **Star this repo** if you find it useful for learning MLOps!
