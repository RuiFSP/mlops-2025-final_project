"""
Simple deployment trigger using subprocess to avoid event loop issues.

This module provides a clean way to trigger Prefect deployments from synchronous
contexts without dealing with async/await complexities.
"""

import json
import logging
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DeploymentTrigger:
    """Simple deployment trigger using subprocess calls."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the deployment trigger.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path

    def trigger_deployment(
        self,
        deployment_name: str,
        parameters: Optional[dict[str, Any]] = None,
        wait_for_completion: bool = False,
        timeout_seconds: int = 300,
    ) -> bool:
        """Trigger a Prefect deployment using subprocess.

        Args:
            deployment_name: Name of the deployment to trigger
            parameters: Parameters to pass to the deployment
            wait_for_completion: Whether to wait for completion
            timeout_seconds: Timeout for waiting

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build the command
            cmd = [
                sys.executable,
                "-c",
                f"""
import asyncio
import json
from prefect import get_client
from datetime import datetime

async def trigger_deployment():
    client = get_client()

    # Get deployment
    try:
        deployments = await client.read_deployments()
        deployment = None
        for d in deployments:
            if d.name == '{deployment_name}' or d.name.endswith('/{deployment_name}'):
                deployment = d
                break

        if not deployment:
            print(f"ERROR: Deployment '{deployment_name}' not found")
            return False

        # Create flow run
        parameters_str = '''{json.dumps(parameters or {})}'''
        flow_run = await client.create_flow_run_from_deployment(
            deployment_id=deployment.id,
            parameters=json.loads(parameters_str),
        )

        print(f"SUCCESS: Created flow run {{flow_run.id}}")

        if {str(wait_for_completion)}:
            start_time = datetime.now()
            while True:
                if (datetime.now() - start_time).seconds > {timeout_seconds}:
                    print(f"TIMEOUT: Flow run {{flow_run.id}} timed out")
                    return False

                updated_run = await client.read_flow_run(flow_run.id)
                if updated_run.state.is_final():
                    print(f"COMPLETED: Flow run {{flow_run.id}} finished with state {{updated_run.state.type}}")
                    return updated_run.state.type.value == "COMPLETED"

                await asyncio.sleep(2)

        return True

    except Exception as e:
        print(f"ERROR: {{str(e)}}")
        return False

if __name__ == "__main__":
    result = asyncio.run(trigger_deployment())
    exit(0 if result else 1)
""",
            ]

            logger.info(f"Triggering deployment: {deployment_name}")
            if parameters:
                logger.debug(f"Parameters: {parameters}")

            # Run the subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 60,  # Extra buffer
            )

            # Parse output
            if result.returncode == 0:
                # Look for success indicators in output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('SUCCESS:'):
                        logger.info(f"Deployment triggered successfully: {line}")
                        return True
                    elif line.startswith('COMPLETED:'):
                        logger.info(f"Deployment completed: {line}")
                        return True
                    elif line.startswith('ERROR:'):
                        logger.error(f"Deployment failed: {line}")
                        return False
                    elif line.startswith('TIMEOUT:'):
                        logger.warning(f"Deployment timed out: {line}")
                        return False

                # If no specific status found but return code is 0
                logger.info("Deployment triggered (no specific status)")
                return True
            else:
                # Log error output
                if result.stderr:
                    logger.error(f"Deployment trigger failed: {result.stderr}")
                if result.stdout:
                    logger.error(f"Stdout: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Deployment trigger timed out after {timeout_seconds}s")
            return False
        except Exception as e:
            logger.error(f"Failed to trigger deployment: {e}")
            return False

    def trigger_retraining(
        self,
        deployment_name: str = "simulation-triggered-retraining",
        week: Optional[int] = None,
        performance_drop: Optional[float] = None,
        baseline_score: Optional[float] = None,
        current_score: Optional[float] = None,
        training_data_path: Optional[str] = None,
        wait_for_completion: bool = True,
        timeout_seconds: int = 300,
    ) -> bool:
        """Trigger the retraining deployment with specific parameters.

        Args:
            deployment_name: Name of the retraining deployment
            week: Current simulation week
            performance_drop: Performance drop that triggered retraining
            baseline_score: Baseline model score
            current_score: Current model score
            training_data_path: Path to training data
            wait_for_completion: Whether to wait for completion
            timeout_seconds: Timeout for waiting

        Returns:
            True if successful, False otherwise
        """
        parameters: dict[str, Any] = {}

        if week is not None:
            parameters["week"] = week
        if performance_drop is not None:
            parameters["performance_drop"] = performance_drop
        if baseline_score is not None:
            parameters["baseline_score"] = baseline_score
        if current_score is not None:
            parameters["current_score"] = current_score
        if training_data_path is not None:
            parameters["training_data_path"] = training_data_path
        if self.config_path is not None:
            parameters["config_path"] = self.config_path

        return self.trigger_deployment(
            deployment_name=deployment_name,
            parameters=parameters,
            wait_for_completion=wait_for_completion,
            timeout_seconds=timeout_seconds,
        )


def main() -> None:
    """CLI interface for triggering deployments."""
    import argparse

    parser = argparse.ArgumentParser(description="Trigger Prefect deployments")
    parser.add_argument("deployment_name", help="Name of the deployment to trigger")
    parser.add_argument("--parameters", type=str, help="JSON parameters to pass")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--config", type=str, help="Configuration file path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Parse parameters
    parameters = None
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON parameters: {e}")
            sys.exit(1)

    # Create trigger and run
    trigger = DeploymentTrigger(config_path=args.config)
    success = trigger.trigger_deployment(
        deployment_name=args.deployment_name,
        parameters=parameters,
        wait_for_completion=args.wait,
        timeout_seconds=args.timeout,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
