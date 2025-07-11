#!/usr/bin/env python3
"""
Complete MLOps Simulation Demo

This script demonstrates the complete MLOps workflow:
1. Starts all required services (MLflow, Prefect)
2. Deploys the retraining flow to Prefect
3. Runs the real-time season simulation
4. Shows automated retraining in action

Usage:
    python scripts/simulation/complete_demo.py [--demo] [--weeks N]
"""

import subprocess
import sys
import time
import signal
from pathlib import Path
import logging

# Add src to path to import configuration
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.config import config
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Configuration loaded: {config}")

    # Setup environment variables
    config.setup_environment()

except ImportError as e:
    print(f"⚠️  Could not import configuration: {e}")
    print("Using fallback configuration...")

    # Fallback configuration
    import os
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

    class FallbackConfig:
        prefect_server_host = "127.0.0.1"
        prefect_server_port = 4200
        mlflow_server_host = "127.0.0.1"
        mlflow_server_port = 5000
        prefect_work_pool = "mlops-pool"

    config = FallbackConfig()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOpsDemo:
    """Complete MLOps demonstration with service management."""

    def __init__(self):
        self.services = []
        self.project_root = Path(__file__).parent.parent.parent

    def start_service(self, name: str, command: str, check_port: int = None):
        """Start a background service."""
        logger.info(f"🚀 Starting {name}...")

        try:
            # Start the service in background
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )

            self.services.append({
                'name': name,
                'process': process,
                'command': command
            })

            # Give the service time to start
            time.sleep(3)

            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✅ {name} started successfully")
                return True
            else:
                logger.error(f"❌ {name} failed to start")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return False

    def stop_services(self):
        """Stop all running services."""
        logger.info("🛑 Stopping services...")

        for service in self.services:
            try:
                if service['process'].poll() is None:
                    logger.info(f"   Stopping {service['name']}...")
                    service['process'].terminate()
                    service['process'].wait(timeout=5)
            except Exception as e:
                logger.warning(f"   Failed to stop {service['name']}: {e}")
                try:
                    service['process'].kill()
                except:
                    pass

        self.services.clear()
        logger.info("✅ All services stopped")

    def deploy_retraining_flow(self):
        """Deploy the retraining flow to Prefect."""
        logger.info("📦 Deploying retraining flow to Prefect...")

        try:
            # Run the direct deployment script
            result = subprocess.run([
                "python", "scripts/deployment/direct_deploy.py"
            ], cwd=self.project_root, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("✅ Retraining flow deployed successfully")
                logger.info(f"📊 Deployment output:\n{result.stdout}")
                return True
            else:
                logger.error(f"❌ Failed to deploy retraining flow: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Deployment timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            return False

    def run_simulation(self, demo_mode: bool = True, weeks: int = 10):
        """Run the real-time simulation."""
        logger.info("🏟️  Starting real-time simulation...")

        try:
            # Prepare command
            cmd = ["python", "scripts/simulation/realtime_season_simulation.py"]

            if demo_mode:
                cmd.append("--demo")
            
            # Always pass weeks parameter if specified
            cmd.extend(["--weeks", str(weeks)])

            # Run the simulation
            result = subprocess.run(cmd, cwd=self.project_root)

            if result.returncode == 0:
                logger.info("✅ Simulation completed successfully")
                return True
            else:
                logger.error("❌ Simulation failed")
                return False

        except KeyboardInterrupt:
            logger.info("⚠️  Simulation interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Simulation error: {e}")
            return False

    def run_complete_demo(self, demo_mode: bool = True, weeks: int = 10):
        """Run the complete MLOps demo."""
        print("="*80)
        print("🏟️  COMPLETE MLOPS AUTOMATION DEMO")
        print("="*80)
        print("This demo will:")
        print("1. 🚀 Start MLflow and Prefect servers")
        print("2. 🔧 Create work pool and start worker")
        print("3. 📦 Deploy the automated retraining flow")
        print("4. 🏟️  Run real-time season simulation")
        print("5. 🔄 Show automated retraining in action")
        print("6. 📊 Display results in Prefect UI")
        if demo_mode:
            print(f"🎮 Demo Mode: {weeks} weeks, 5 second intervals")
        else:
            print(f"📅 Full Mode: {weeks} weeks, 10 second intervals")
        print("="*80)

        try:
            # Step 1: Start services
            print(f"\n📋 STEP 1: Starting Services")
            print("-" * 40)

            # Start MLflow using configuration
            mlflow_cmd = f"uv run mlflow server --host {config.mlflow_server_host} --port {config.mlflow_server_port}"
            if not self.start_service("MLflow", mlflow_cmd):
                return False

            # Start Prefect using configuration
            prefect_cmd = f"uv run prefect server start --host {config.prefect_server_host} --port {config.prefect_server_port}"
            if not self.start_service("Prefect", prefect_cmd):
                return False

            # Give services more time to fully start
            logger.info("⏱️  Waiting for services to fully initialize...")
            time.sleep(10)

            # Step 1.5: Create work pool and start worker
            print(f"\n📋 STEP 1.5: Setting up Prefect Work Pool")
            print("-" * 40)

            # Create work pool using configuration
            logger.info("🔧 Creating work pool...")
            try:
                result = subprocess.run([
                    "prefect", "work-pool", "create", "--type", "process", config.prefect_work_pool
                ], cwd=self.project_root, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    logger.info("✅ Work pool created successfully")
                else:
                    logger.warning("⚠️  Work pool may already exist")
            except Exception as e:
                logger.warning(f"⚠️  Work pool creation issue: {e}")

            # Start worker using configuration
            worker_cmd = f"uv run prefect worker start --pool {config.prefect_work_pool}"
            if not self.start_service("Prefect Worker", worker_cmd):
                logger.warning("⚠️  Worker failed to start, deployments may not execute")

            # Step 2: Deploy retraining flow
            print(f"\n📋 STEP 2: Deploying Retraining Flow")
            print("-" * 40)

            if not self.deploy_retraining_flow():
                logger.warning("⚠️  Retraining flow deployment failed, continuing with simulation mode")

            # Step 3: Run simulation
            print(f"\n📋 STEP 3: Running Real-Time Simulation")
            print("-" * 40)
            print(f"💡 IMPORTANT: Open these URLs to monitor the demo:")
            print(f"   📊 Prefect UI: http://localhost:4200/deployments")
            print(f"   📈 MLflow UI: http://localhost:5000")
            print(f"   🔄 Flows will be visible in the Prefect UI when triggered!")
            print("-" * 40)

            # Give user time to open the UI
            print("⏱️  Waiting 5 seconds for you to open the Prefect UI...")
            time.sleep(5)

            success = self.run_simulation(demo_mode=demo_mode, weeks=weeks)

            # Step 4: Summary
            print(f"\n📋 DEMO SUMMARY")
            print("-" * 40)

            if success:
                print("✅ Demo completed successfully!")
                print("📊 Check the Prefect UI for retraining flow history")
                print("📈 Check MLflow for model tracking")
                print("📁 Simulation data saved to: data/realtime_simulation/")
            else:
                print("⚠️  Demo completed with issues")

            return success

        except KeyboardInterrupt:
            print(f"\n⚠️  Demo interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.exception("Demo error details:")
            return False
        finally:
            self.stop_services()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n⚠️  Received interrupt signal, stopping demo...")
    sys.exit(0)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete MLOps Demo")
    parser.add_argument("--demo", action="store_true", help="Run quick demo mode (fast 5 second intervals)")
    parser.add_argument("--weeks", type=int, default=10, help="Number of weeks to simulate (default: 10)")

    args = parser.parse_args()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        demo = MLOpsDemo()
        success = demo.run_complete_demo(demo_mode=args.demo, weeks=args.weeks)
        return 0 if success else 1

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
