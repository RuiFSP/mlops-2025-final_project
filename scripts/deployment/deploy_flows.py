#!/usr/bin/env python3
"""
Deploy Retraining Flow to Prefect Server

This script deploys the automated retraining flow to a running Prefect server
so it can be triggered via the Prefect API and monitored in the UI.
"""

import sys
import os
import asyncio
from pathlib import Path

# Set Prefect API URL to connect to the main server
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.automation.retraining_flow import automated_retraining_flow


async def deploy_flows():
    """Deploy the retraining flows to the Prefect server."""
    print("🚀 Deploying automated retraining flows to Prefect server...")

    try:
        print("� Creating deployments using flow.deploy()...")

        # Deploy main automated retraining deployment
        print("   Deploying: automated-retraining")
        deployment_id_1 = await automated_retraining_flow.deploy(
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
        print(f"   ✅ Deployed with ID: {deployment_id_1}")

        # Deploy simulation-triggered retraining deployment
        print("   Deploying: simulation-triggered-retraining")
        deployment_id_2 = await automated_retraining_flow.deploy(
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
        print(f"   ✅ Deployed with ID: {deployment_id_2}")

        print(f"\n✅ Successfully deployed 2 retraining flows!")
        print(f"📊 View in Prefect UI: http://localhost:4200/deployments")
        print(f"🔧 Deployments can now be triggered via API or UI")

        return True

    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    try:
        success = asyncio.run(deploy_flows())
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Deployment interrupted")
        return 1
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
