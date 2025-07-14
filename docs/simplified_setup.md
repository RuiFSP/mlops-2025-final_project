# Simplified Setup Guide

This guide explains how to use the simplified database and model setup for the Premier League MLOps project.

## Database Simplification

We've simplified the database setup to avoid permission issues:

1. The `fix_database.py` script now:
   - Checks if the Docker container is running and starts it if needed
   - Creates the database and user if they don't exist
   - Grants superuser privileges to the database user (simplifies permissions)
   - Creates all required tables with proper permissions
   - Adds sample data for testing

2. The Makefile includes a `fix-db` command that runs this script:
   ```
   make fix-db
   ```

3. The `start` command now automatically runs `fix-db` to ensure the database is properly set up before starting other services.

## Model Simplification

To avoid model loading issues, we've created a simplified model setup:

1. The `fix_model.py` script:
   - Creates a simple scikit-learn model (RandomForestClassifier)
   - Logs it to MLflow with appropriate metrics
   - Registers it in the MLflow Model Registry
   - Sets it as the Production model

2. The Makefile includes a `fix-model` command that runs this script:
   ```
   make fix-model
   ```

3. The `start` command now automatically runs `fix-model` after starting MLflow to ensure a model is available for the prediction pipeline.

## Troubleshooting

The dashboard troubleshooting script has been enhanced to provide more options:

1. Run diagnostics to check connections and status
2. Fix database issues
3. Fix model loading issues
4. Restart all services
5. Run all fixes

To run the troubleshooting script:
```
make troubleshoot
```

## Common Issues and Solutions

### Database Issues

If you see database permission errors:
```
make fix-db
```

This will:
- Grant superuser privileges to the database user
- Create all required tables
- Add sample data

### Model Loading Issues

If you see model loading errors:
```
make fix-model
```

This will:
- Create a simple model
- Log it to MLflow
- Register it in the Model Registry

### All Services Not Starting

If some services fail to start:
```
make restart
```

This will:
- Stop all services
- Start them again in the correct order
- Fix database and model issues automatically

## Complete Setup

For a complete setup from scratch:
```
make setup
make start
```

This will:
1. Install dependencies
2. Set up Docker services
3. Fix database permissions and schema
4. Start MLflow
5. Create a mock model
6. Start the API
7. Start Prefect
8. Start the Streamlit dashboard
