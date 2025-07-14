#!/usr/bin/env python3
"""
Startup script for the simplified Premier League MLOps System
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
from config_minimal.config import (
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    API_HOST,
    API_PORT,
    DASHBOARD_PORT,
    PREFECT_API_URL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)


def start_api():
    """Start the FastAPI server"""
    logger.info("Starting FastAPI server...")
    api_process = subprocess.Popen(
        ["uv", "run", "uvicorn", "src.api.simplified_api:app", "--host", API_HOST, "--port", str(API_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info(f"FastAPI server started on http://{API_HOST}:{API_PORT}")
    return api_process


def start_dashboard():
    """Start the Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard...")
    dashboard_process = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "src/dashboard/simplified_app.py", "--server.port", str(DASHBOARD_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info(f"Streamlit dashboard started on http://localhost:{DASHBOARD_PORT}")
    return dashboard_process


def start_integrated_example():
    """Start the integrated example dashboard"""
    logger.info("Starting integrated example dashboard...")
    example_port = DASHBOARD_PORT + 1  # Use the next port
    example_process = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "src/dashboard/integrated_example.py", "--server.port", str(example_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info(f"Integrated example dashboard started on http://localhost:{example_port}")
    return example_process


def start_prefect():
    """Start the Prefect server"""
    logger.info("Starting Prefect server...")
    prefect_process = subprocess.Popen(
        ["uv", "run", "prefect", "server", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info("Prefect server started")
    return prefect_process


def main():
    """Main entry point"""
    logger.info("Starting Premier League MLOps System...")
    
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Start services
    api_process = start_api()
    dashboard_process = start_dashboard()
    integrated_example_process = start_integrated_example()
    prefect_process = start_prefect()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        dashboard_process.terminate()
        integrated_example_process.terminate()
        prefect_process.terminate()
        logger.info("All services stopped")


if __name__ == "__main__":
    main() 