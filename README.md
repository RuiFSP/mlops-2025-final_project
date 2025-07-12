# Premier League Match Predictor - MLOps System

A complete end-to-end MLOps pipeline for predicting Premier League match outcomes with automated betting simulation and monitoring.

## 🚀 **Quick Start (Local Development)**

### **Prerequisites**
- Python 3.10+
- `uv` package manager
- Local PostgreSQL server
- Local MLflow server

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

### **Run the Pipeline**
1. **Train the model:**
   ```bash
   uv run python -m src.pipelines.training_pipeline
   ```

2. **Generate predictions and run betting simulation:**
   ```bash
   uv run python scripts/test_betting_simulation.py
   ```

3. **Check results:**
   ```bash
   uv run python scripts/debug_bets_table.py
   ```

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
```

### **Betting Configuration**
- **Initial Balance**: £1000.0
- **Confidence Threshold**: 0.35 (testing) / 0.6 (production)
- **Margin Threshold**: 0.05 (testing) / 0.1 (production)
- **Max Bet Percentage**: 5% of balance

## 📊 **System Performance**

### **Model Performance**
- **Accuracy**: 61.84% (excellent for football prediction)
- **Model Type**: Random Forest with 15 features
- **Training Data**: 3,040 Premier League matches
- **Features**: Betting odds + match statistics

## 🏗️ **Architecture**

### **Local Development Pipeline**
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
```

## 📝 **Available Scripts**

### **Core Pipeline**
- `uv run python -m src.pipelines.training_pipeline` - Train and register model
- `uv run python -m src.pipelines.prediction_pipeline` - Generate predictions

### **Testing & Debugging**
- `uv run python scripts/test_betting_simulation.py` - Full betting simulation
- `uv run python scripts/debug_bets_table.py` - Check database state
- `uv run python scripts/clean_postgres.py` - Clean all tables

### **Database Management**
- `uv run python scripts/setup_database.py` - Initialize database schema
- `uv run python scripts/check_db_tables.py` - Verify table structure

## 📄 **Documentation**

- **[TODO.md](TODO.md)**: Current development priorities
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: Detailed architecture
- **[pyproject.toml](pyproject.toml)**: Project dependencies

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `uv run python scripts/test_betting_simulation.py`
5. Submit a pull request

---

**🎉 Premier League Match Predictor - Ready for Local Development!**
