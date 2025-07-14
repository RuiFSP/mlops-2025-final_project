#!/usr/bin/env python3
"""
Setup script for the Premier League MLOps Streamlit Dashboard
Ensures all necessary components are properly configured
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dashboard-setup")

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
DASHBOARD_DIR = SRC_DIR / "dashboard"


def check_dependencies():
    """Check if required Python packages are installed"""
    logger.info("Checking required dependencies...")

    required_packages = ["streamlit", "plotly", "pandas", "psycopg2-binary"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            logger.warning(f"❌ {package} is not installed")
            missing_packages.append(package)

    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        install = input("Would you like to install missing packages? (y/n): ")
        if install.lower() == "y":
            cmd = ["uv", "pip", "install"] + missing_packages
            subprocess.run(cmd, check=True)
            logger.info("✅ Installed missing packages")
        else:
            logger.warning("⚠️ Some required packages are missing")
    else:
        logger.info("✅ All required packages are installed")


def check_database_connection():
    """Check if the PostgreSQL database is accessible"""
    logger.info("Checking database connection...")

    try:
        import psycopg2

        # Get database credentials from environment or use defaults
        db_host = os.environ.get("POSTGRES_HOST", "localhost")
        db_port = os.environ.get("POSTGRES_PORT", "5432")
        db_name = os.environ.get("POSTGRES_DB", "mlops_db")
        db_user = os.environ.get("POSTGRES_USER", "mlops_user")
        db_pass = os.environ.get("POSTGRES_PASSWORD", "mlops_password")

        # Try to connect to the database
        conn = psycopg2.connect(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_pass)
        conn.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def check_streamlit_app():
    """Check if the Streamlit app file exists and is valid"""
    logger.info("Checking Streamlit app file...")

    app_path = DASHBOARD_DIR / "streamlit_app.py"

    if not app_path.exists():
        logger.error(f"❌ Streamlit app file not found at {app_path}")
        return False

    # Basic validation of the file content
    with open(app_path, "r") as f:
        content = f.read()

        if "import streamlit as st" not in content:
            logger.error("❌ Streamlit app file does not import streamlit")
            return False

        if '__name__ == "__main__"' not in content:
            logger.warning("⚠️ Streamlit app file does not have a main entry point")

    logger.info("✅ Streamlit app file is valid")
    return True


def setup_environment():
    """Set up environment variables for the dashboard"""
    logger.info("Setting up environment variables...")

    # Check if .env file exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        logger.warning("⚠️ No .env file found, creating from example")
        example_env = PROJECT_ROOT / "config.env.example"
        if example_env.exists():
            with open(example_env, "r") as src, open(env_file, "w") as dst:
                dst.write(src.read())
            logger.info("✅ Created .env file from example")
        else:
            logger.warning("⚠️ No config.env.example file found")

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
        logger.info("✅ Loaded environment variables from .env file")
    except ImportError:
        logger.warning("⚠️ python-dotenv not installed, skipping .env loading")

    # Set PYTHONPATH to include src directory
    os.environ["PYTHONPATH"] = f"{SRC_DIR}:{os.environ.get('PYTHONPATH', '')}"
    logger.info("✅ Set PYTHONPATH to include src directory")


def main():
    """Main function to set up the dashboard"""
    logger.info("🚀 Setting up Premier League MLOps Dashboard...")

    # Check dependencies
    check_dependencies()

    # Set up environment
    setup_environment()

    # Check database connection
    db_ok = check_database_connection()

    # Check Streamlit app
    app_ok = check_streamlit_app()

    if db_ok and app_ok:
        logger.info("✅ Dashboard setup complete!")
        logger.info("🎯 You can start the dashboard with:")
        logger.info("   ./scripts/start_dashboard.py")
        logger.info("   or")
        logger.info("   cd src/dashboard && uv run streamlit run streamlit_app.py")
        logger.info("📊 Dashboard will be available at: http://localhost:8501")
        return 0
    else:
        logger.error("❌ Dashboard setup failed!")
        if not db_ok:
            logger.error("   - Database connection issue needs to be resolved")
        if not app_ok:
            logger.error("   - Streamlit app file issue needs to be resolved")
        return 1


if __name__ == "__main__":
    sys.exit(main())
