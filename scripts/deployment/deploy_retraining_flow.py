"""
Prefect Deployment Configuration for Automated Retraining Flow.

This script creates and manages Prefect deployments for the automated retraining
workflow, making it remotely triggerable via the Prefect API.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prefect import serve

from src.automation.retraining_flow import automated_retraining_flow


def create_retraining_deployments():
    """Create Prefect deployments for the automated retraining flows."""

    # Main automated retraining deployment
    main_deployment = automated_retraining_flow.to_deployment(
        name="automated-retraining",
        version="1.0.0",
        description="Production automated model retraining workflow",
        tags=["mlops", "retraining", "production"],
        work_pool_name="mlops-pool",
        parameters={
            "config_path": "config/retraining_config.yaml",
            "triggers": ["scheduled"],
            "force_retrain": False,
        },
    )

    # Simulation-triggered retraining deployment
    simulation_deployment = automated_retraining_flow.to_deployment(
        name="simulation-triggered-retraining",
        version="1.0.0",
        description="Retraining workflow triggered by season simulation",
        tags=["mlops", "retraining", "simulation", "triggered"],
        work_pool_name="mlops-pool",
        parameters={
            "config_path": "config/retraining_config.yaml",
            "triggers": ["simulation_performance_drop"],
            "force_retrain": False,
        },
    )

    return [main_deployment, simulation_deployment]


if __name__ == "__main__":
    # Create deployments
    deployments = create_retraining_deployments()

    print("🚀 Serving automated retraining flows with Prefect...")
    print(f"📋 Created {len(deployments)} deployments:")
    for deployment in deployments:
        print(f"  - {deployment.name}")

    print("\n🌐 Access via Prefect UI: http://localhost:4200")
    print("💡 Deployments will be served and available for triggering!")
    print("📊 Check the 'Deployments' tab in the Prefect UI")
    print("\n⏸️  Press Ctrl+C to stop serving deployments...")

    # Serve the deployments (this will run indefinitely)
    # Note: This connects to the main Prefect server via PREFECT_API_URL
    serve(*deployments)
