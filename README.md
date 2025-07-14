# Premier League Match Prediction System

A machine learning system for predicting Premier League match outcomes with MLOps best practices.

## Overview

This project implements an end-to-end MLOps system for Premier League match prediction, featuring:

1. **Data Pipeline** - Fetches and processes Premier League match data
2. **Training Pipeline** - Trains and registers ML models for match prediction
3. **Prediction Pipeline** - Generates predictions for upcoming matches
4. **Dashboard** - A Streamlit dashboard for monitoring and interacting with the system

## Project Structure

```
premier-league-mlops/
├── config/             # Configuration
├── data/               # Data directory
│   ├── predictions/    # Prediction storage
│   └── real_data/      # Real data storage
├── mlflow/             # MLflow structure
├── models/             # Model storage
├── scripts/            # Utility scripts
│   ├── start.py        # Startup script
│   └── run_tests.py    # Test runner script
├── src/                # Source code
│   ├── api/            # API code
│   │   └── api.py      # FastAPI application
│   ├── dashboard/      # Dashboard code
│   │   ├── app.py      # Streamlit dashboard
│   │   └── integrated_example.py # Integrated example
│   ├── orchestration/  # Workflow orchestration
│   │   └── flows.py    # Prefect flows
│   └── pipelines/      # ML pipelines
├── tests/              # Test suite
└── pyproject.toml      # Project configuration
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/premier-league-mlops.git
   cd premier-league-mlops
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the System

Run the startup script to launch all components:

```bash
python scripts/start.py
```

This will start:
- FastAPI backend on http://localhost:8000
- Streamlit dashboard on http://localhost:8501

You can also start individual components:

```bash
# Start only the API
python scripts/start.py --api-only

# Start only the dashboard
python scripts/start.py --dashboard-only
```

### API Endpoints

The API provides the following endpoints:

- `GET /health` - Health check endpoint
- `GET /model/info` - Get model information
- `POST /predictions/match` - Predict a specific match
- `GET /predictions/upcoming` - Get predictions for upcoming matches
- `POST /retraining/force` - Force model retraining

### Running the API Directly

```bash
uv run uvicorn src.api.api:app --host 0.0.0.0 --port 8000
```

### Running the Dashboard Directly

```bash
uv run streamlit run src/dashboard/app.py --server.port 8501
```

## Testing

Run the test suite using:

```bash
python scripts/run_tests.py
```

This will run all tests in the tests directory and provide a summary of the results.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 