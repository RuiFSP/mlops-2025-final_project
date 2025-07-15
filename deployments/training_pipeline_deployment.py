
from prefect import flow
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@flow(name="training_pipeline_flow")
def training_pipeline_flow(force_retrain: bool = False):
    """Training pipeline flow"""
    logger.info(f"Running training pipeline (force_retrain={force_retrain})")
    return {"status": "completed", "force_retrain": force_retrain, "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    training_pipeline_flow.deploy(
        name="training_pipeline_flow-deployment", 
        work_pool_name="premier-league-pool"
    )
