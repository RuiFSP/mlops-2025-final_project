#!/usr/bin/env python3
"""
Script to check MLflow experiments and runs.
"""

import mlflow
from mlflow.tracking import MlflowClient


def check_mlflow():
    """Check MLflow experiments and runs."""

    # Set tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    print("🔍 Checking MLflow experiments...")

    # List experiments
    experiments = client.search_experiments()
    print(f"Found {len(experiments)} experiments:")

    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")

        # Get runs for this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=3)
        print(f"    Runs: {len(runs)}")

        for run in runs:
            accuracy = run.data.metrics.get("accuracy", "N/A")
            print(f"      - {run.info.run_id}: accuracy={accuracy}")

    print("\n✅ MLflow check completed!")


if __name__ == "__main__":
    check_mlflow()
