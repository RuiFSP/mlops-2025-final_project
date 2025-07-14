#!/usr/bin/env python3
"""
Startup script for the Premier League MLOps System
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration
try:
    from config.config import (
        API_HOST,
        API_PORT,
        DASHBOARD_PORT,
    )
except ImportError:
    # Default values if config cannot be imported
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DASHBOARD_PORT = 8501

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_api():
    """Start the API"""
    logger.info("Starting API...")
    
    # Run the API in a separate process
    api_process = subprocess.Popen(
        ["uv", "run", "uvicorn", "src.api.api:app", "--host", API_HOST, "--port", str(API_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    logger.info(f"API started at http://{API_HOST}:{API_PORT}")
    return api_process


def start_dashboard():
    """Start the Streamlit dashboard"""
    logger.info("Starting dashboard...")
    
    # Run the dashboard in a separate process
    dashboard_process = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "src/dashboard/app.py", "--server.port", str(DASHBOARD_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    logger.info(f"Dashboard started at http://localhost:{DASHBOARD_PORT}")
    return dashboard_process


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start the Premier League MLOps System")
    parser.add_argument("--api-only", action="store_true", help="Start only the API")
    parser.add_argument("--dashboard-only", action="store_true", help="Start only the dashboard")
    args = parser.parse_args()
    
    processes = []
    
    try:
        # Start the components based on arguments
        if args.api_only:
            processes.append(start_api())
        elif args.dashboard_only:
            processes.append(start_dashboard())
        else:
            # Start both by default
            processes.append(start_api())
            processes.append(start_dashboard())
        
        # Wait for any key to stop
        logger.info("Press Ctrl+C to stop all services...")
        
        # Wait for processes to complete (which they won't unless there's an error)
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        logger.info("Stopping all services...")
        
        # Terminate all processes
        for process in processes:
            process.terminate()
            
        logger.info("All services stopped.")
    except Exception as e:
        logger.error(f"Error: {e}")
        
        # Terminate all processes
        for process in processes:
            process.terminate()
            
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 