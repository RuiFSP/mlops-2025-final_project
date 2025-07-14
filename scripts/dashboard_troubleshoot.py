#!/usr/bin/env python3

"""
Dashboard troubleshooting script for Premier League MLOps
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def check_database_connection():
    """Check database connection"""
    logger.info("Checking database connection...")

    try:
        import psycopg2

        from config.database import get_db_config

        db_config = get_db_config()
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result and result[0] == 1:
            logger.info("✅ Database connection successful")
            return True
        else:
            logger.error("❌ Database connection failed")
            return False

    except ImportError:
        logger.error("❌ psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def check_mlflow_connection():
    """Check MLflow connection"""
    logger.info("Checking MLflow connection...")

    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        logger.info(f"✅ MLflow connection successful. Found {len(experiments)} experiments")
        return True

    except ImportError:
        logger.error("❌ mlflow not installed. Install with: pip install mlflow")
        return False
    except Exception as e:
        logger.error(f"❌ MLflow connection failed: {e}")
        return False


def check_model_availability():
    """Check if model is available in MLflow"""
    logger.info("Checking model availability...")

    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        model_name = "premier_league_predictor"
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                logger.info(f"✅ Model '{model_name}' found with {len(versions)} versions")
                return True
            else:
                logger.error(f"❌ Model '{model_name}' not found in registry")
                return False
        except Exception as e:
            logger.error(f"❌ Error searching for model: {e}")
            return False

    except ImportError:
        logger.error("❌ mlflow not installed. Install with: pip install mlflow")
        return False
    except Exception as e:
        logger.error(f"❌ MLflow model check failed: {e}")
        return False


def check_api_health():
    """Check API health"""
    logger.info("Checking API health...")

    try:
        import requests

        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API health check successful")
            return True
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            return False

    except ImportError:
        logger.error("❌ requests not installed. Install with: pip install requests")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ API connection failed. Is the API running?")
        return False
    except Exception as e:
        logger.error(f"❌ API health check failed: {e}")
        return False


def fix_database_issues():
    """Fix database issues"""
    logger.info("Fixing database issues...")

    try:
        # Run the fix_database.py script
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "fix_database.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout)
        logger.info("✅ Database issues fixed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to fix database issues: {e}")
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"❌ Failed to fix database issues: {e}")
        return False


def fix_model_loading_issues():
    """Fix model loading issues"""
    logger.info("Fixing model loading issues...")

    try:
        # Run the fix_model.py script
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "fix_model.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout)
        logger.info("✅ Model loading issues fixed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to fix model loading issues: {e}")
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"❌ Failed to fix model loading issues: {e}")
        return False


def restart_services():
    """Restart services"""
    logger.info("Restarting services...")

    try:
        # Stop services
        subprocess.run(["make", "stop"], check=True)
        logger.info("Services stopped")

        # Start services
        subprocess.run(["make", "start"], check=True)
        logger.info("✅ Services restarted")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to restart services: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to restart services: {e}")
        return False


def main():
    """Main function"""
    print("\n🔧 Premier League MLOps Dashboard Troubleshooter 🔧\n")
    print("This tool will help diagnose and fix common issues with the dashboard.")
    print("\nSelect an option:")
    print("1. Run diagnostics (check connections and status)")
    print("2. Fix database issues")
    print("3. Fix model loading issues")
    print("4. Restart all services")
    print("5. Run all fixes")
    print("6. Exit")

    choice = input("\nEnter your choice (1-6): ")

    if choice == "1":
        print("\n🔍 Running diagnostics...\n")
        db_ok = check_database_connection()
        mlflow_ok = check_mlflow_connection()
        model_ok = check_model_availability()
        api_ok = check_api_health()

        print("\n📊 Diagnostic Summary:")
        print(f"Database Connection: {'✅' if db_ok else '❌'}")
        print(f"MLflow Connection: {'✅' if mlflow_ok else '❌'}")
        print(f"Model Availability: {'✅' if model_ok else '❌'}")
        print(f"API Health: {'✅' if api_ok else '❌'}")

        if not db_ok:
            print("\nTo fix database issues, run: make fix-db")
        if not mlflow_ok:
            print("\nTo fix MLflow connection, run: make start-mlflow")
        if not model_ok:
            print("\nTo fix model loading issues, run: make fix-model")
        if not api_ok:
            print("\nTo fix API issues, run: make start-api")

    elif choice == "2":
        print("\n🛠️ Fixing database issues...\n")
        fix_database_issues()

    elif choice == "3":
        print("\n🤖 Fixing model loading issues...\n")
        fix_model_loading_issues()

    elif choice == "4":
        print("\n🔄 Restarting all services...\n")
        restart_services()

    elif choice == "5":
        print("\n🚀 Running all fixes...\n")
        fix_database_issues()
        fix_model_loading_issues()
        restart_services()

    elif choice == "6":
        print("\nExiting troubleshooter.")
        return

    else:
        print("\n❌ Invalid choice. Please run the script again and select a valid option.")
        return

    print("\n✅ Troubleshooting complete!")


if __name__ == "__main__":
    main()
