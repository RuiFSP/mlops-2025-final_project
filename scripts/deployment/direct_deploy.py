#!/usr/bin/env python3
"""
Direct Deployment Script for Prefect Flows

This script directly registers deployments with the running Prefect server
instead of using the serve() function.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
    else:
        print(f"⚠️  .env file not found, using defaults")
        os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

except ImportError:
    print("⚠️  python-dotenv not available, using defaults")
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Set Prefect API URL (fallback if not in .env)
if "PREFECT_API_URL" not in os.environ:
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prefect import get_client
from src.automation.retraining_flow import automated_retraining_flow

async def deploy_flows():
    """Deploy the retraining flows directly to the Prefect server."""
    client = get_client()

    try:
        # Get project root directory
        project_root = str(Path(__file__).parent.parent.parent.absolute())

        # Create deployments with proper working directory
        main_deployment = await automated_retraining_flow.to_deployment(
            name="automated-retraining",
            version="1.0.0",
            description="Production automated model retraining workflow",
            tags=["mlops", "retraining", "production"],
            work_pool_name="mlops-pool",
            job_variables={"working_dir": project_root},
            parameters={
                "triggers": ["scheduled"],
                "model_path": "models/model.pkl",
                "training_data_path": "data/real_data/premier_league_matches.parquet",
                "backup_dir": "models/backups",
                "model_type": "random_forest",
                "min_accuracy_threshold": 0.45,
                "improvement_threshold": 0.01,
            },
        )

        simulation_deployment = await automated_retraining_flow.to_deployment(
            name="simulation-triggered-retraining",
            version="1.0.0",
            description="Retraining workflow triggered by season simulation",
            tags=["mlops", "retraining", "simulation", "triggered"],
            work_pool_name="mlops-pool",
            job_variables={"working_dir": project_root},
            parameters={
                "triggers": ["simulation_performance_drop"],
                "model_path": "models/model.pkl",
                "training_data_path": "data/real_data/premier_league_matches.parquet",
                "backup_dir": "models/backups",
                "model_type": "random_forest",
                "min_accuracy_threshold": 0.45,
                "improvement_threshold": 0.01,
            },
        )

        # Apply deployments to the server
        print("📦 Applying deployments to Prefect server...")

        main_deployment_id = await main_deployment.apply()
        print(f"✅ Deployed: {main_deployment.name} (ID: {main_deployment_id})")

        simulation_deployment_id = await simulation_deployment.apply()
        print(f"✅ Deployed: {simulation_deployment.name} (ID: {simulation_deployment_id})")

        print(f"\n🌐 Deployments are now available in the Prefect UI!")
        print(f"📊 View at: http://localhost:4200/deployments")

        # List all deployments to verify
        deployments = await client.read_deployments()
        print(f"\n📋 Total deployments in server: {len(deployments)}")
        for deployment in deployments:
            print(f"  - {deployment.name} (Status: {deployment.status}, Work Pool: {deployment.work_pool_name})")

        return True

    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(deploy_flows())
    sys.exit(0 if success else 1)
