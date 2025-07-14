#!/usr/bin/env python3

"""
Start script for the Premier League MLOps Dashboard
Ensures proper environment setup before launching Streamlit
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """
    Main entry point for starting the dashboard
    Sets up environment and launches Streamlit
    """
    # Get the project root directory (2 levels up from this script)
    project_root = Path(__file__).parent.parent.absolute()

    # Set PYTHONPATH environment variable to include src directory
    os.environ["PYTHONPATH"] = f"{project_root}/src:{os.environ.get('PYTHONPATH', '')}"

    # Set database environment variables
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_DB"] = "mlops_db"
    os.environ["POSTGRES_USER"] = "mlops_user"
    os.environ["POSTGRES_PASSWORD"] = "mlops_password"

    # Dashboard path
    dashboard_path = project_root / "src" / "dashboard"

    # Change to dashboard directory
    os.chdir(str(dashboard_path))

    print("🚀 Starting Premier League MLOps Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")

    try:
        # Run Streamlit using uv
        cmd = [
            "uv",
            "run",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🔄 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
