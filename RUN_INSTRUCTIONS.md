# ⚡ Quick Run Instructions

## 🚀 5-Minute Setup

### 1. Prerequisites
```bash
# Install Python 3.10+ and uv
pip install uv
```

### 2. Setup
```bash
# Clone and setup
git clone https://github.com/RuiFSP/mlops-2025-final_project.git
cd mlops-2025-final_project
uv sync

# Copy environment config
cp config.env.example .env
```

### 3. Database (PostgreSQL required)
```bash
# Install PostgreSQL, then create database
sudo -u postgres psql
CREATE DATABASE mlops_db;
CREATE USER mlops_user WITH PASSWORD 'mlops_password';
GRANT ALL PRIVILEGES ON DATABASE mlops_db TO mlops_user;
\q

# Setup database schema
uv run python scripts/setup_database.py
```

### 4. Start Services

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

### 5. Test System
```bash
# Terminal 4 - Test everything
uv run python scripts/test_simple_integration.py
```

## 🎯 Access Points

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://127.0.0.1:5000
- **Health**: http://localhost:8000/health

## 🔍 Quick Test

```bash
# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea"}'
```

## 🛠️ Troubleshooting

- **Database issues**: Check PostgreSQL is running
- **MLflow issues**: Ensure MLflow server is running on port 5000
- **API issues**: Check all dependencies installed with `uv sync`
- **Model issues**: Re-run training pipeline

**For detailed instructions, see [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)**
