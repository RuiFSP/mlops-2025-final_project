#!/usr/bin/env python3
"""
Deploy Prefect flows to the server
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_work_pool():
    """Create the work pool if it doesn't exist"""
    try:
        from prefect.client.orchestration import get_client
        
        async with get_client() as client:
            # Check if work pool exists
            try:
                work_pool = await client.read_work_pool("premier-league-pool")
                logger.info(f"Work pool 'premier-league-pool' already exists")
            except Exception:
                # Create work pool
                logger.info("Creating work pool 'premier-league-pool'...")
                from prefect.workers.process import ProcessWorkerPool
                
                work_pool = await client.create_work_pool(
                    work_pool={
                        "name": "premier-league-pool",
                        "type": "process",
                        "description": "Premier League MLOps work pool",
                        "is_paused": False,
                    }
                )
                logger.info(f"Created work pool: {work_pool.name}")
                
    except Exception as e:
        logger.error(f"Error creating work pool: {e}")


async def deploy_flows():
    """Deploy all flows to Prefect"""
    try:
        # Import flows
        from src.orchestration.flows import (
            data_pipeline_flow,
            training_pipeline_flow,
            prediction_pipeline_flow
        )
        
        # Deploy flows
        flows_to_deploy = [
            ("data_pipeline_flow", data_pipeline_flow),
            ("training_pipeline_flow", training_pipeline_flow), 
            ("prediction_pipeline_flow", prediction_pipeline_flow),
        ]
        
        for flow_name, flow_func in flows_to_deploy:
            try:
                logger.info(f"Deploying flow: {flow_name}")
                
                deployment = await flow_func.to_deployment(
                    name=f"{flow_name}-deployment",
                    work_pool_name="premier-league-pool",
                    description=f"Deployment for {flow_name}",
                    tags=["premier-league", "mlops"],
                    path="/app",
                    entrypoint=f"src/orchestration/flows.py:{flow_name}",
                )
                
                deployment_id = await deployment.apply()
                logger.info(f"✅ Deployed {flow_name} with ID: {deployment_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to deploy {flow_name}: {e}")
                
    except Exception as e:
        logger.error(f"Error importing flows: {e}")


async def main():
    """Main deployment function"""
    logger.info("🚀 Starting Prefect flow deployment...")
    
    # Set Prefect API URL
    prefect_url = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
    os.environ["PREFECT_API_URL"] = prefect_url
    logger.info(f"Using Prefect API URL: {prefect_url}")
    
    # Create work pool
    await create_work_pool()
    
    # Deploy flows
    await deploy_flows()
    
    logger.info("✅ Flow deployment completed!")


if __name__ == "__main__":
    asyncio.run(main()) 