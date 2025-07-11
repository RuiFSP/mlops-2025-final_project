#!/usr/bin/env python3
"""
MLOps Environment Setup Script

This script sets up the complete MLOps environment using configuration from .env file.
It properly starts all services and ensures they're connected.

Usage:
    python setup_mlops_env.py [--stop] [--status]
"""

import argparse
import os
import subprocess
import sys
import time
import signal
import psutil
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    from src.config import config
    print(f"✅ Configuration loaded from .env: {config.env_loaded}")
except ImportError as e:
    print(f"⚠️  Could not import configuration: {e}")
    # Use fallback
    import os
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

    class FallbackConfig:
        prefect_api_url = "http://127.0.0.1:4200/api"
        prefect_server_host = "127.0.0.1"
        prefect_server_port = 4200
        mlflow_server_host = "127.0.0.1"
        mlflow_server_port = 5000
        prefect_work_pool = "mlops-pool"
        mlflow_tracking_uri = "http://127.0.0.1:5000"

    config = FallbackConfig()


class MLOpsEnvironment:
    """Manages the complete MLOps environment."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services: List[Dict] = []

        # Setup environment variables
        os.environ["PREFECT_API_URL"] = config.prefect_api_url
        os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri

    def find_process_by_port(self, port: int) -> Optional[psutil.Process]:
        """Find process running on specific port."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def is_service_running(self, port: int) -> bool:
        """Check if service is running on port."""
        return self.find_process_by_port(port) is not None

    def start_service_background(self, name: str, command: str, port: int) -> bool:
        """Start a service in the background."""
        print(f"🚀 Starting {name}...")

        # Check if already running
        if self.is_service_running(port):
            print(f"✅ {name} already running on port {port}")
            return True

        try:
            # Start service
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                preexec_fn=os.setsid  # Create new process group
            )

            # Wait a bit and check if it's running
            time.sleep(3)

            if process.poll() is None and self.is_service_running(port):
                print(f"✅ {name} started successfully on port {port}")
                self.services.append({
                    'name': name,
                    'process': process,
                    'port': port,
                    'command': command
                })
                return True
            else:
                print(f"❌ {name} failed to start")
                return False

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False

    def stop_services(self):
        """Stop all managed services."""
        print("🛑 Stopping services...")

        # Stop managed services
        for service in self.services:
            try:
                print(f"   Stopping {service['name']}...")
                if service['process'].poll() is None:
                    # Kill process group
                    os.killpg(os.getpgid(service['process'].pid), signal.SIGTERM)
                    service['process'].wait(timeout=5)
            except Exception as e:
                print(f"   Warning: Could not stop {service['name']}: {e}")

        # Also stop any remaining services on known ports
        for port, name in [(config.prefect_server_port, "Prefect"),
                          (config.mlflow_server_port, "MLflow")]:
            proc = self.find_process_by_port(port)
            if proc:
                try:
                    print(f"   Stopping remaining {name} on port {port}...")
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass

        self.services.clear()
        print("✅ All services stopped")

    def setup_prefect_worker(self) -> bool:
        """Set up Prefect work pool and start worker."""
        print("🔧 Setting up Prefect worker...")

        # Create work pool
        try:
            result = subprocess.run([
                "uv", "run", "prefect", "work-pool", "create",
                "--type", "process", config.prefect_work_pool
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"✅ Work pool '{config.prefect_work_pool}' created")
            else:
                print(f"ℹ️  Work pool '{config.prefect_work_pool}' already exists")
        except Exception as e:
            print(f"⚠️  Work pool creation issue: {e}")

        # Start worker
        worker_cmd = f"uv run prefect worker start --pool {config.prefect_work_pool}"

        try:
            process = subprocess.Popen(
                worker_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                preexec_fn=os.setsid
            )

            # Give worker time to connect
            time.sleep(5)

            if process.poll() is None:
                print(f"✅ Worker started for pool '{config.prefect_work_pool}'")
                self.services.append({
                    'name': 'Prefect Worker',
                    'process': process,
                    'port': None,
                    'command': worker_cmd
                })
                return True
            else:
                print("❌ Worker failed to start")
                return False

        except Exception as e:
            print(f"❌ Failed to start worker: {e}")
            return False

    def check_status(self):
        """Check status of all services."""
        print("📊 MLOps Environment Status")
        print("=" * 50)

        # Check Prefect server
        prefect_running = self.is_service_running(config.prefect_server_port)
        print(f"Prefect Server: {'✅ Running' if prefect_running else '❌ Not running'} (port {config.prefect_server_port})")

        # Check MLflow server
        mlflow_running = self.is_service_running(config.mlflow_server_port)
        print(f"MLflow Server: {'✅ Running' if mlflow_running else '❌ Not running'} (port {config.mlflow_server_port})")

        # Check work pool status if Prefect is running
        if prefect_running:
            try:
                result = subprocess.run([
                    "uv", "run", "prefect", "work-pool", "inspect", config.prefect_work_pool
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    if "WorkPoolStatus.READY" in result.stdout:
                        print(f"Work Pool: ✅ {config.prefect_work_pool} is READY")
                    elif "WorkPoolStatus.NOT_READY" in result.stdout:
                        print(f"Work Pool: ⚠️  {config.prefect_work_pool} is NOT_READY")
                    else:
                        print(f"Work Pool: ❓ {config.prefect_work_pool} status unknown")
                else:
                    print(f"Work Pool: ❌ {config.prefect_work_pool} not found")
            except Exception as e:
                print(f"Work Pool: ❌ Error checking status: {e}")

        print(f"\n🔗 URLs:")
        print(f"   Prefect UI: http://{config.prefect_server_host}:{config.prefect_server_port}")
        print(f"   MLflow UI: http://{config.mlflow_server_host}:{config.mlflow_server_port}")

    def start_all(self) -> bool:
        """Start all MLOps services."""
        print("🚀 Starting MLOps Environment")
        print("=" * 50)

        success = True

        # Start MLflow
        mlflow_cmd = f"uv run mlflow server --host {config.mlflow_server_host} --port {config.mlflow_server_port}"
        if not self.start_service_background("MLflow", mlflow_cmd, config.mlflow_server_port):
            success = False

        # Start Prefect server
        prefect_cmd = f"uv run prefect server start --host {config.prefect_server_host} --port {config.prefect_server_port}"
        if not self.start_service_background("Prefect", prefect_cmd, config.prefect_server_port):
            success = False

        if success:
            # Wait for servers to be ready
            print("⏱️  Waiting for servers to initialize...")
            time.sleep(8)

            # Setup worker
            if not self.setup_prefect_worker():
                success = False

        if success:
            print("\n🎉 MLOps environment is ready!")
            print("🚀 Run demo: python scripts/simulation/complete_demo.py --demo")
        else:
            print("\n❌ Failed to start some services")

        return success


def main():
    parser = argparse.ArgumentParser(description="MLOps Environment Setup")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--status", action="store_true", help="Check service status")
    args = parser.parse_args()

    env = MLOpsEnvironment()

    if args.stop:
        env.stop_services()
    elif args.status:
        env.check_status()
    else:
        try:
            env.start_all()

            # Keep running until interrupted
            print("\n💡 Press Ctrl+C to stop all services")
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            env.stop_services()


if __name__ == "__main__":
    main()
