# Premier League MLOps Dashboard

This Streamlit dashboard provides a comprehensive view of our Premier League prediction system, from model performance to betting simulations.

## Features

The dashboard includes the following sections:

### 🤖 Model Performance
- View accuracy, precision, recall, and F1 score trends
- Compare performance across different time periods
- Identify potential model drift
- Analyze performance by match type and team

### 🔮 Predictions
- View predictions for upcoming and past matches
- Analyze prediction confidence levels
- See probability distributions for different outcomes
- Compare predictions against actual results

### 💰 Betting Performance
- Monitor profit/loss over time
- View win rates and ROI metrics
- Analyze betting history and outcomes
- Compare different betting strategies

### 🎯 Live Prediction
- Enter team names and match details
- Get real-time predictions and confidence scores
- View detailed probability breakdowns
- Analyze feature importance for predictions

### 🔄 Workflows
- Schedule and trigger workflow runs
- Monitor workflow execution status
- View logs and error messages
- Configure workflow parameters

### 🎮 Simulation
- Configure simulation parameters (season, dates, betting strategy)
- Run, pause, and resume simulations
- View simulation progress and results
- Analyze long-term betting performance

## Running the Dashboard

You can start the dashboard using:

```bash
make start-dashboard
```

Or run the entire system with:

```bash
make start
```

The dashboard will be available at: http://localhost:8501

## Troubleshooting

If you encounter issues with the dashboard, you can run the troubleshooting script:

```bash
make troubleshoot
```

### Common Issues

#### Database Connection Issues
- PostgreSQL service is not running
- Connection parameters are incorrect
- Database tables haven't been created

Solution: Run `python scripts/fix_database.py`

#### Model Loading Issues
- MLflow tracking server is not running
- Model artifacts are missing or corrupted
- Model registry references are outdated

Solution: Check the MLflow UI for registered models and ensure the correct model version is set as active.

#### Port Conflicts
- Another application is using port 8501

Solution: Kill the process using the port or specify a different port:
```bash
streamlit run src/dashboard/streamlit_app.py --server.port 8502
```

#### Mock Data
The dashboard is designed to show mock data when real data is not available. This allows you to explore the interface even if some components are not working.
