#!/usr/bin/env python3
"""
Test script to check if Prefect deployments are available.
"""

import asyncio
import os
from prefect import get_client

# Set Prefect API URL
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

async def list_deployments():
    """List all available deployments."""
    client = get_client()

    try:
        deployments = await client.read_deployments()

        print("📋 Available Deployments:")
        for deployment in deployments:
            print(f"  - {deployment.name}")
            print(f"    Flow ID: {deployment.flow_id}")
            print(f"    Status: {deployment.status}")
            print(f"    Work Pool: {deployment.work_pool_name}")
            print()

        if not deployments:
            print("❌ No deployments found!")

    except Exception as e:
        print(f"❌ Error listing deployments: {e}")

if __name__ == "__main__":
    asyncio.run(list_deployments())
