
from prefect import flow
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@flow(name="data_pipeline_flow")
def data_pipeline_flow(season: str = "2023/24"):
    """Data pipeline flow"""
    logger.info(f"Running data pipeline for season {season}")
    return {"status": "completed", "season": season, "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    data_pipeline_flow.deploy(
        name="data_pipeline_flow-deployment",
        work_pool_name="premier-league-pool"
    )
