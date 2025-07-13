# Contributing to Premier League Match Predictor

## 🚀 Development Setup

### Prerequisites
- Python 3.10+
- PostgreSQL
- `uv` package manager

### Quick Setup
```bash
# 1. Fork and clone
git clone <your-fork-url>
cd mlops-2025-final_project

# 2. Install dependencies
uv sync

# 3. Setup environment
cp config.env.example .env
# Edit .env with your database settings

# 4. Initialize database
uv run python scripts/setup_database.py

# 5. Start development services
uv run mlflow server --host 127.0.0.1 --port 5000
```

## 🧪 Testing

### Run All Tests
```bash
# Integration tests
uv run python scripts/test_simple_integration.py

# API tests
uv run python scripts/test_api.py

# Orchestration tests
uv run python scripts/test_simple_orchestration.py
```

### Test Individual Components
```bash
# Database
uv run python scripts/check_db_tables.py

# Betting simulation
uv run python scripts/test_betting_simulation.py

# Model training
uv run python -m src.pipelines.training_pipeline
```

## 📁 Project Structure

```
src/
├── api/                 # FastAPI REST
```
