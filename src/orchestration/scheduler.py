"""
Prefect-based scheduler for orchestrating MLOps workflows.
"""

import logging
from datetime import datetime
from typing import Any

from prefect import serve
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from .flows import daily_prediction_flow, monitoring_flow, retraining_flow

logger = logging.getLogger(__name__)


class PrefectMLOpsScheduler:
    """Prefect-based scheduler for MLOps workflows."""

    def __init__(self):
        """Initialize the Prefect scheduler."""
        self.deployments = []
        self._create_deployments()

    def _create_deployments(self):
        """Create Prefect deployments for all flows."""

        # Daily monitoring deployment - runs every hour
        monitoring_deployment = Deployment.build_from_flow(
            flow=monitoring_flow,
            name="hourly-monitoring",
            description="Hourly monitoring of model performance and drift",
            schedule=CronSchedule(cron="0 * * * *"),  # Every hour
            parameters={
                "check_performance": True,
                "check_drift": True,
                "generate_daily_predictions": False,
            },
            tags=["mlops", "monitoring", "scheduled"],
        )

        # Daily predictions deployment - runs every morning at 6 AM
        prediction_deployment = Deployment.build_from_flow(
            flow=daily_prediction_flow,
            name="daily-predictions",
            description="Daily prediction generation for upcoming matches",
            schedule=CronSchedule(cron="0 6 * * *"),  # 6 AM daily
            parameters={
                "days_ahead": 7,
                "run_betting_analysis": True,
            },
            tags=["mlops", "predictions", "scheduled"],
        )

        # Weekly retraining check deployment - runs every Monday at 2 AM
        retraining_deployment = Deployment.build_from_flow(
            flow=retraining_flow,
            name="weekly-retraining-check",
            description="Weekly model retraining check based on performance",
            schedule=CronSchedule(cron="0 2 * * 1"),  # 2 AM every Monday
            parameters={
                "force_retrain": False,
                "performance_threshold_accuracy": 0.55,
                "performance_threshold_f1": 0.50,
                "drift_threshold": 0.5,
            },
            tags=["mlops", "retraining", "scheduled"],
        )

        # Emergency retraining deployment - can be triggered manually
        emergency_retraining_deployment = Deployment.build_from_flow(
            flow=retraining_flow,
            name="emergency-retraining",
            description="Emergency model retraining for critical issues",
            parameters={
                "force_retrain": True,
                "performance_threshold_accuracy": 0.0,
                "performance_threshold_f1": 0.0,
                "drift_threshold": 0.0,
            },
            tags=["mlops", "retraining", "emergency"],
        )

        self.deployments = [
            monitoring_deployment,
            prediction_deployment,
            retraining_deployment,
            emergency_retraining_deployment,
        ]

        logger.info(f"✅ Created {len(self.deployments)} Prefect deployments")

    def deploy_all(self):
        """Deploy all workflows to Prefect."""
        try:
            for deployment in self.deployments:
                deployment.apply()
                logger.info(f"✅ Deployed: {deployment.name}")

            logger.info("🚀 All MLOps workflows deployed successfully!")

        except Exception as e:
            logger.error(f"❌ Error deploying workflows: {e}")
            raise

    def serve_flows(self):
        """Serve all flows using Prefect serve."""
        try:
            logger.info("🚀 Starting Prefect flows server...")

            # Serve all deployments
            serve(*self.deployments, limit=10)

        except Exception as e:
            logger.error(f"❌ Error serving flows: {e}")
            raise

    def trigger_emergency_retraining(self) -> str:
        """Trigger emergency retraining manually."""
        try:
            from prefect.client.orchestration import PrefectClient

            client = PrefectClient()

            # Create a flow run for emergency retraining
            flow_run = client.create_flow_run_from_deployment(
                deployment_id="emergency-retraining",
                parameters={
                    "force_retrain": True,
                    "performance_threshold_accuracy": 0.0,
                    "performance_threshold_f1": 0.0,
                    "drift_threshold": 0.0,
                },
            )

            logger.info(f"🚨 Emergency retraining triggered: {flow_run.id}")
            return flow_run.id

        except Exception as e:
            logger.error(f"❌ Error triggering emergency retraining: {e}")
            raise

    def get_deployment_status(self) -> dict[str, Any]:
        """Get status of all deployments."""
        try:
            from prefect.client.orchestration import PrefectClient

            client = PrefectClient()

            status = {
                "total_deployments": len(self.deployments),
                "deployments": [],
                "last_updated": datetime.now().isoformat(),
            }

            for deployment in self.deployments:
                # Get recent flow runs for this deployment
                flow_runs = client.read_flow_runs(
                    deployment_filter={"name": {"any_": [deployment.name]}},
                    limit=5,
                )

                deployment_status = {
                    "name": deployment.name,
                    "description": deployment.description,
                    "schedule": str(deployment.schedule) if deployment.schedule else "Manual",
                    "tags": deployment.tags,
                    "recent_runs": len(flow_runs),
                    "last_run": flow_runs[0].start_time.isoformat() if flow_runs else None,
                    "last_state": flow_runs[0].state.name if flow_runs else "Never run",
                }

                status["deployments"].append(deployment_status)

            return status

        except Exception as e:
            logger.error(f"❌ Error getting deployment status: {e}")
            return {
                "error": str(e),
                "total_deployments": len(self.deployments),
                "last_updated": datetime.now().isoformat(),
            }


def main():
    """Main function to start the scheduler."""
    scheduler = PrefectMLOpsScheduler()

    try:
        # Deploy all workflows
        scheduler.deploy_all()

        # Start serving flows
        scheduler.serve_flows()

    except KeyboardInterrupt:
        logger.info("🔄 Scheduler stopped by user")
    except Exception as e:
        logger.error(f"❌ Scheduler error: {e}")
        raise


if __name__ == "__main__":
    main()
