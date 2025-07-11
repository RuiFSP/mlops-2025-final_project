"""
Prefect Client for Remote Flow Execution.

This module provides utilities for triggering Prefect flows remotely via API,
specifically for the automated retraining system integration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from prefect import get_client
from prefect.client.schemas import FlowRun

logger = logging.getLogger(__name__)


class PrefectClient:
    """Client for managing and triggering Prefect flows remotely."""

    def __init__(self, server_url: str = "http://localhost:4200/api"):
        """Initialize the Prefect client.

        Args:
            server_url: URL of the Prefect server API
        """
        self.server_url = server_url
        self._client = None

    async def get_client(self) -> Any:  # type: ignore
        """Get or create the Prefect client."""
        if self._client is None:
            self._client = get_client()  # type: ignore
        return self._client

    async def trigger_deployment_run(
        self,
        deployment_name: str,
        parameters: dict[str, Any] | None = None,
        wait_for_completion: bool = False,
        timeout_seconds: int = 300,
    ) -> FlowRun | None:
        """Trigger a deployment run.

        Args:
            deployment_name: Name of the deployment (flow/deployment format)
            parameters: Parameters to pass to the flow
            wait_for_completion: Whether to wait for the flow to complete
            timeout_seconds: Timeout for waiting if wait_for_completion is True

        Returns:
            FlowRun object if successful, None otherwise
        """
        try:
            client = await self.get_client()

            # Format deployment name if needed
            original_name = deployment_name
            logger.info(f"🔍 DEBUG: Received deployment_name: '{deployment_name}'")

            if "/" not in deployment_name:
                # If no '/' in deployment name, we need to find the flow name
                # For our MLOps project, we know the flow name is "automated-retraining-flow"
                full_deployment_name = f"automated-retraining-flow/{deployment_name}"
                logger.info(
                    f"🔧 Converting '{deployment_name}' to full path: '{full_deployment_name}'"
                )
                deployment_name = full_deployment_name
            else:
                # If it has a '/', it should be in the correct flow/deployment format
                logger.debug(f"Using full deployment path: {deployment_name}")

            logger.info(f"🚀 Triggering deployment: '{deployment_name}'")

            # Get the deployment
            try:
                deployment = await client.read_deployment_by_name(deployment_name)
                if not deployment:
                    raise ValueError(f"Deployment '{deployment_name}' not found")
            except Exception as e:
                # Don't include the original error message if it contains confusing info
                error_msg = str(e)
                if "Event loop is closed" in error_msg:
                    logger.warning(
                        f"Event loop error while accessing deployment '{deployment_name}': {error_msg}"
                    )
                    raise ValueError(
                        f"Event loop closed while accessing deployment '{deployment_name}'"
                    )
                else:
                    logger.error(f"Could not find deployment '{deployment_name}': {error_msg}")

                # List available deployments for debugging
                try:
                    deployments = await client.read_deployments()
                    available = [d.name for d in deployments]
                    logger.info(f"Available deployments: {available}")
                except Exception:
                    pass
                raise ValueError(f"Deployment '{deployment_name}' not found")

            # Create a flow run
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment.id,
                parameters=parameters or {},
                state=None,  # Let Prefect manage the state
            )

            logger.info(f"Created flow run: {flow_run.id}")

            if wait_for_completion:
                logger.info(f"Waiting for flow run {flow_run.id} to complete...")

                # Wait for completion with timeout
                start_time = datetime.now()
                while True:
                    # Check if timeout exceeded
                    if (datetime.now() - start_time).seconds > timeout_seconds:
                        logger.warning(f"Flow run {flow_run.id} timed out after {timeout_seconds}s")
                        break

                    # Check flow run state
                    updated_run = await client.read_flow_run(flow_run.id)
                    if updated_run.state.is_final():
                        logger.info(
                            f"Flow run {flow_run.id} completed with state: {updated_run.state.type}"
                        )
                        return updated_run  # type: ignore

                    # Wait a bit before checking again
                    await asyncio.sleep(2)

            return flow_run  # type: ignore

        except Exception as e:
            logger.error(f"Failed to trigger deployment {deployment_name}: {e}")
            return None

    async def trigger_retraining_flow(
        self,
        deployment_name: str = "automated-retraining",
        triggers: list[str] | None = None,
        config_path: str = "config/retraining_config.yaml",
        force_retrain: bool = False,
        **kwargs: Any,
    ) -> str | None:
        """Trigger the automated retraining flow.

        Args:
            deployment_name: Name of the deployment to trigger
            triggers: List of trigger reasons
            config_path: Path to the retraining configuration
            force_retrain: Whether to force retraining
            **kwargs: Additional parameters for the flow

        Returns:
            Flow run ID
        """
        try:
            client = await self.get_client()

            # Prepare flow parameters
            parameters = {
                "config_path": config_path,
                "triggers": triggers or ["api_trigger"],
                "force_retrain": force_retrain,
                **kwargs,
            }

            # Find the deployment
            deployments = await client.read_deployments(
                flow_filter={"name": {"any_": ["automated-retraining-flow"]}}
            )

            target_deployment = None
            for deployment in deployments:
                if deployment.name == deployment_name:
                    target_deployment = deployment
                    break

            if not target_deployment:
                available = [d.name for d in deployments]
                raise ValueError(
                    f"Deployment '{deployment_name}' not found. " f"Available: {available}"
                )

            # Create flow run
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=target_deployment.id,
                parameters=parameters,
                name=f"retraining-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                tags=["automated", "api-triggered"] + (triggers or []),
            )

            logger.info(f"Triggered flow run: {flow_run.id}")
            logger.info(f"Flow run name: {flow_run.name}")
            logger.info(f"Parameters: {parameters}")

            return flow_run.id  # type: ignore

        except Exception as e:
            logger.error(f"Failed to trigger retraining flow: {e}")
            raise

    async def get_flow_run_status(self, flow_run_id: str) -> dict[str, Any]:
        """Get the status of a flow run.

        Args:
            flow_run_id: ID of the flow run

        Returns:
            Flow run status information
        """
        try:
            client = await self.get_client()
            flow_run = await client.read_flow_run(flow_run_id)

            return {
                "id": flow_run.id,
                "name": flow_run.name,
                "state": flow_run.state.type if flow_run.state else "Unknown",
                "state_name": flow_run.state.name if flow_run.state else "Unknown",
                "start_time": flow_run.start_time,
                "end_time": flow_run.end_time,
                "total_run_time": flow_run.total_run_time,
                "tags": flow_run.tags,
                "parameters": flow_run.parameters,
            }

        except Exception as e:
            logger.error(f"Failed to get flow run status: {e}")
            raise

    async def list_recent_flow_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent flow runs for retraining flows.

        Args:
            limit: Maximum number of flow runs to return

        Returns:
            List of flow run information
        """
        try:
            client = await self.get_client()

            # Get recent flow runs for retraining flows
            flow_runs = await client.read_flow_runs(
                flow_filter={"name": {"any_": ["automated-retraining-flow"]}},
                limit=limit,
                sort="CREATED_DESC",
            )

            return [
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state.type if run.state else "Unknown",
                    "created": run.created,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "tags": run.tags,
                }
                for run in flow_runs
            ]

        except Exception as e:
            logger.error(f"Failed to list flow runs: {e}")
            raise

    def trigger_retraining_sync(
        self,
        deployment_name: str = "automated-retraining",
        triggers: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous wrapper for triggering retraining flow.

        Args:
            deployment_name: Name of the deployment to trigger
            triggers: List of trigger reasons
            **kwargs: Additional parameters for the flow

        Returns:
            Flow run ID
        """
        result = asyncio.run(
            self.trigger_retraining_flow(
                deployment_name=deployment_name, triggers=triggers, **kwargs
            )
        )
        return result or ""

    def get_flow_run_status_sync(self, flow_run_id: str) -> dict[str, Any]:
        """Synchronous wrapper for getting flow run status."""
        return asyncio.run(self.get_flow_run_status(flow_run_id))


# Convenience functions for direct use
def trigger_retraining_flow(
    triggers: list[str] | None = None, force_retrain: bool = False, **kwargs: Any
) -> str:
    """Trigger the automated retraining flow directly.

    Args:
        triggers: List of trigger reasons
        force_retrain: Whether to force retraining
        **kwargs: Additional parameters

    Returns:
        Flow run ID
    """
    client = PrefectClient()
    return client.trigger_retraining_sync(triggers=triggers, force_retrain=force_retrain, **kwargs)


def get_flow_run_status(flow_run_id: str) -> dict[str, Any]:
    """Get the status of a flow run.

    Args:
        flow_run_id: ID of the flow run

    Returns:
        Flow run status information
    """
    client = PrefectClient()
    return client.get_flow_run_status_sync(flow_run_id)
