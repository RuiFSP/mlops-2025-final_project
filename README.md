# Premier League Match Prediction System (Simplified)

A streamlined MLOps system for predicting Premier League football match outcomes.

## Tech Stack

- **MLflow**: Experiment tracking and model registry
- **Prefect**: Workflow orchestration
- **SQLite**: Lightweight database storage
- **FastAPI**: Backend API
- **Streamlit**: Frontend dashboard
- **uv**: Python package management

## System Components

1. **Streamlit Dashboard** - A web interface with:
   - Overview page with system information
   - Model performance metrics
   - Live prediction page for upcoming matches
   - Workflow page to trigger Prefect flows

2. **FastAPI Backend** - REST API with endpoints for:
   - Match prediction
   - Model information
   - Health checks
   - Upcoming matches data

3. **Prefect Flows** - Orchestration for:
   - Data fetching from football-data.uk
   - Data preprocessing
   - Model training
   - Match prediction

4. **Integrated Example** - A simplified dashboard that demonstrates all components working together:
   - System status overview (FastAPI, SQLite, MLflow, Prefect)
   - Match prediction form with visualization
   - Recent predictions from SQLite database
   - Model metrics visualization
   - Workflow triggering interface

## Setup

### Prerequisites

- Python 3.10+
- uv installed (`pip install uv`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Running the System

Run the startup script to launch all components:

```bash
python scripts/start_simplified.py
```

This will start:
- FastAPI server on port 8000
- Streamlit dashboard on port 8501
- Integrated example dashboard on port 8502
- Prefect server

Access the components:
- Dashboard: http://localhost:8501
- Integrated Example: http://localhost:8502
- API documentation: http://localhost:8000/docs
- Prefect dashboard: http://localhost:4200

### Manual Component Startup

If you prefer to start components individually:

1. Start the API:
   ```bash
   uv run uvicorn src.api.simplified_api:app --host 0.0.0.0 --port 8000
   ```

2. Start the Streamlit dashboard:
   ```bash
   uv run streamlit run src/dashboard/simplified_app.py --server.port 8501
   ```

3. Start the integrated example dashboard:
   ```bash
   uv run streamlit run src/dashboard/integrated_example.py --server.port 8502
   ```

4. Start the Prefect server:
   ```bash
   uv run prefect server start
   ```

## Project Structure

```
├── data/                  # Data storage
│   ├── real_data/         # Real match data
│   └── predictions/       # Model predictions
├── models/                # Trained models
├── logs/                  # Log files
├── config_minimal/        # Configuration
├── mlflow_simplified/     # MLflow structure
│   └── models/            # Model artifacts
├── scripts/
│   ├── start_simplified.py # Startup script
│   └── run_simplified_tests.py # Test runner script
├── src/
│   ├── api/
│   │   └── simplified_api.py # FastAPI application
│   ├── dashboard/
│   │   ├── simplified_app.py # Streamlit dashboard
│   │   └── integrated_example.py # Integrated example dashboard
│   ├── orchestration/
│   │   └── simplified_flows.py # Prefect flows
│   ├── pipelines/         # Model pipelines
│   └── data_integration/  # Data fetching
└── tests_simplified/      # Test suite
```

## Running Tests

To run the test suite:

```bash
python scripts/run_simplified_tests.py
```

This will run all tests in the tests_simplified directory and provide a summary of the results. 