#!/usr/bin/env python3

"""
Dashboard Troubleshooting Script

This script helps diagnose and fix common issues with the Premier League MLOps dashboard.
It checks for database connectivity, model availability, and other potential problems.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
import importlib.util
import socket
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

def check_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_environment():
    """Check if the Python environment is properly set up"""
    logger.info("Checking Python environment...")
    
    required_packages = [
        "streamlit", 
        "pandas", 
        "plotly", 
        "psycopg2", 
        "sqlalchemy", 
        "mlflow",
        "numpy",
        "streamlit_autorefresh"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
            logger.info("✅ Packages installed successfully")
        except subprocess.CalledProcessError:
            logger.error("❌ Failed to install packages. Try running: pip install -r requirements.txt")
            return False
    else:
        logger.info("✅ All required packages are installed")
    
    return True

def check_database_connection():
    """Check database connection"""
    logger.info("Checking database connection...")
    
    try:
        # Import here to ensure we've checked for the package first
        import psycopg2
        from config.database import get_db_config
        
        db_config = get_db_config()
        logger.info(f"Attempting to connect to database at {db_config['host']}:{db_config['port']}")
        
        conn = psycopg2.connect(**db_config)
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"✅ Successfully connected to PostgreSQL: {version}")
        conn.close()
        
        return True
    except ImportError:
        logger.error("❌ Failed to import database modules. Check your Python environment.")
        return False
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.info("Common causes:")
        logger.info("  - PostgreSQL service is not running")
        logger.info("  - Incorrect connection parameters")
        logger.info("  - Network issues")
        logger.info("\nTry running: python scripts/fix_database.py")
        return False

def check_database_tables():
    """Check if required database tables exist"""
    logger.info("Checking database tables...")
    
    try:
        import psycopg2
        from config.database import get_db_config
        
        db_config = get_db_config()
        conn = psycopg2.connect(**db_config)
        
        required_tables = ["model_metrics", "predictions", "bets", "wallet"]
        missing_tables = []
        
        with conn.cursor() as cursor:
            for table in required_tables:
                cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}');")
                exists = cursor.fetchone()[0]
                if not exists:
                    missing_tables.append(table)
        
        conn.close()
        
        if missing_tables:
            logger.error(f"❌ Missing required tables: {', '.join(missing_tables)}")
            logger.info("Try running: python scripts/fix_database.py")
            return False
        else:
            logger.info("✅ All required database tables exist")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to check database tables: {e}")
        return False

def check_mlflow_model():
    """Check if MLflow model is available"""
    logger.info("Checking MLflow model...")
    
    try:
        # Import here to ensure we've checked for the package first
        import mlflow
        
        # Try to get MLflow tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Check if MLflow server is running (if using remote server)
        if tracking_uri.startswith(("http://", "https://")):
            import requests
            try:
                response = requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/list")
                if response.status_code == 200:
                    logger.info("✅ MLflow server is accessible")
                else:
                    logger.error(f"❌ MLflow server returned status code {response.status_code}")
                    return False
            except requests.exceptions.ConnectionError:
                logger.error("❌ Cannot connect to MLflow server")
                logger.info("Make sure the MLflow server is running")
                return False
        
        # Try to load model from registry
        try:
            from src.pipelines.prediction_pipeline import PredictionPipeline
            pipeline = PredictionPipeline()
            if pipeline.model:
                logger.info("✅ Successfully loaded model from MLflow")
                return True
            else:
                logger.error("❌ Model is None")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.info("Check if the model is registered in MLflow and the path is correct")
            return False
            
    except ImportError:
        logger.error("❌ Failed to import MLflow. Check your Python environment.")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking MLflow model: {e}")
        return False

def check_streamlit_port():
    """Check if Streamlit port is available"""
    logger.info("Checking if Streamlit port is available...")
    
    port = 8501  # Default Streamlit port
    
    if check_port_in_use(port):
        logger.warning(f"⚠️ Port {port} is already in use")
        logger.info("This might be another Streamlit app or a previous instance of the dashboard")
        logger.info("You can:")
        logger.info(f"  1. Stop the process using port {port}")
        logger.info(f"  2. Use a different port by adding --server.port <port> to the Streamlit command")
        return False
    else:
        logger.info(f"✅ Port {port} is available for Streamlit")
        return True

def fix_common_issues():
    """Try to fix common issues automatically"""
    logger.info("Attempting to fix common issues...")
    
    # 1. Fix database issues
    try:
        logger.info("Running database fix script...")
        subprocess.run([sys.executable, "scripts/fix_database.py"], check=True)
        logger.info("✅ Database fix script completed")
    except subprocess.CalledProcessError:
        logger.error("❌ Failed to run database fix script")
    
    # 2. Kill any existing Streamlit processes
    try:
        logger.info("Checking for existing Streamlit processes...")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "streamlit.exe"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "streamlit"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("✅ Terminated any existing Streamlit processes")
    except:
        pass  # Ignore errors if no processes found
    
    # Wait a moment for ports to be released
    time.sleep(2)

def start_dashboard():
    """Start the dashboard"""
    logger.info("Starting the dashboard...")
    
    try:
        # Run in a new process so it doesn't block this script
        if sys.platform == "win32":
            subprocess.Popen([sys.executable, "scripts/start_dashboard.py"], 
                            creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen([sys.executable, "scripts/start_dashboard.py"], 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("✅ Dashboard started successfully")
        logger.info("📊 Dashboard will be available at: http://localhost:8501")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        return False

def fix_model_loading_issues():
    """Try to fix model loading issues"""
    logger.info("Attempting to fix model loading issues...")
    
    try:
        # Import here to ensure we've checked for the package first
        import mlflow
        from src.pipelines.prediction_pipeline import PredictionPipeline
        
        # Check if MLflow tracking URI is set
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri == "file:///home/ruifspinto/projects/mlops-2025-final_project/mlruns":
            logger.info("MLflow tracking URI is set to local directory")
            
            # Check if the model exists in the registry
            client = mlflow.tracking.MlflowClient()
            try:
                models = client.search_registered_models()
                if not models:
                    logger.error("❌ No models found in MLflow registry")
                    logger.info("You need to train a model first. Try running: make train")
                    return False
                
                logger.info(f"✅ Found {len(models)} models in MLflow registry")
                
                # Check if the model files exist
                for model in models:
                    model_name = model.name
                    logger.info(f"Checking model: {model_name}")
                    
                    try:
                        latest_version = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
                        if latest_version:
                            logger.info(f"Latest version: {latest_version[0].version}")
                            
                            # Check if model artifacts exist
                            run_id = latest_version[0].run_id
                            artifact_path = f"mlruns/{client.get_experiment_by_name('default').experiment_id}/"\
                                           f"{run_id}/artifacts/model"
                            
                            if os.path.exists(artifact_path):
                                logger.info(f"✅ Model artifacts found at: {artifact_path}")
                            else:
                                logger.error(f"❌ Model artifacts not found at: {artifact_path}")
                                logger.info("Try running: make train")
                        else:
                            logger.error(f"❌ No versions found for model: {model_name}")
                    except Exception as e:
                        logger.error(f"❌ Error checking model versions: {e}")
                
                return True
            except Exception as e:
                logger.error(f"❌ Error accessing MLflow registry: {e}")
                return False
        else:
            logger.info(f"MLflow tracking URI is set to: {tracking_uri}")
            logger.info("Please ensure the MLflow server is running at this URI")
            return False
    except ImportError:
        logger.error("❌ Failed to import MLflow. Check your Python environment.")
        return False
    except Exception as e:
        logger.error(f"❌ Error fixing model loading issues: {e}")
        return False

def main():
    """Main function"""
    logger.info("🔧 Premier League MLOps Dashboard Troubleshooter")
    logger.info("==============================================")
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        logger.error("❌ Environment check failed")
        return
    
    # Menu
    while True:
        logger.info("\n🔍 What would you like to check/fix?")
        logger.info("1. Check database connection")
        logger.info("2. Check database tables")
        logger.info("3. Check MLflow model")
        logger.info("4. Check Streamlit port")
        logger.info("5. Fix common issues automatically")
        logger.info("6. Fix model loading issues")
        logger.info("7. Start dashboard")
        logger.info("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == "1":
            check_database_connection()
        elif choice == "2":
            check_database_tables()
        elif choice == "3":
            check_mlflow_model()
        elif choice == "4":
            check_streamlit_port()
        elif choice == "5":
            fix_common_issues()
        elif choice == "6":
            fix_model_loading_issues()
        elif choice == "7":
            start_dashboard()
        elif choice == "8":
            logger.info("Exiting troubleshooter...")
            break
        else:
            logger.error("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 