#!/usr/bin/env python3
"""Test script to debug Prefect flow triggering."""

import asyncio
import os
import sys
from pathlib import Path

# Set environment
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
sys.path.append(str(Path(__file__).parent / "src"))

from src.automation.prefect_client import PrefectClient


async def test_trigger():
    """Test triggering a Prefect flow directly."""
    print("🧪 Testing Prefect flow triggering...")

    client = PrefectClient()

    try:
        print("📋 Attempting to trigger deployment...")

        # Test parameters that match our flow signature
        test_parameters = {
            "triggers": ["test_debug"],
            "model_path": "models/model.pkl",
            "training_data_path": "data/real_data/premier_league_matches.parquet",
            "backup_dir": "models/backups",
            "model_type": "random_forest",
            "min_accuracy_threshold": 0.45,
            "improvement_threshold": 0.01,
        }

        # Try to trigger the flow
        flow_run = await client.trigger_deployment_run(
            deployment_name="simulation-triggered-retraining",
            parameters=test_parameters,
            wait_for_completion=False,  # Don't wait
            timeout_seconds=30
        )

        if flow_run:
            print(f"✅ Flow triggered successfully! ID: {flow_run.id}")
            print(f"   State: {flow_run.state}")
            return True
        else:
            print("❌ Flow triggering returned None")
            return False

    except Exception as e:
        print(f"❌ Error triggering flow: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def main():
    """Main test function."""
    try:
        result = asyncio.run(test_trigger())
        print(f"\n🎯 Test result: {'SUCCESS' if result else 'FAILED'}")
        return 0 if result else 1
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
