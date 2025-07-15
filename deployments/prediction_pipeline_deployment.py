
from prefect import flow
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@flow(name="prediction_pipeline_flow")
def prediction_pipeline_flow(days_ahead: int = 7):
    """Prediction pipeline flow"""
    logger.info(f"Running prediction pipeline for {days_ahead} days ahead")
    return {"status": "completed", "days_ahead": days_ahead, "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    prediction_pipeline_flow.deploy(
        name="prediction_pipeline_flow-deployment",
        work_pool_name="premier-league-pool"
    )
