#!/usr/bin/env python3
"""
Set up Prefect work pools and deploy flows
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_prefect():
    """Set up Prefect work pools and deployments"""
    try:
        import subprocess
        
        # Set environment variables
        prefect_url = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
        os.environ["PREFECT_API_URL"] = prefect_url
        
        logger.info(f"Setting up Prefect with API URL: {prefect_url}")
        
        # Create work pool
        logger.info("Creating work pool...")
        result = subprocess.run([
            "prefect", "work-pool", "create", "premier-league-pool", 
            "--type", "process"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Work pool created successfully")
        elif "already exists" in result.stderr or "already exists" in result.stdout:
            logger.info("✅ Work pool already exists")
        else:
            logger.warning(f"Work pool creation result: {result.stderr}")
        
        # Create real flows that integrate with existing pipelines
        logger.info("Creating real pipeline flow files...")
        
        # Create flows directory
        flows_dir = project_root / "flows"
        flows_dir.mkdir(exist_ok=True)
        
        # Create real flow files that use actual pipeline implementations
        flow_files = {
            "data_flow.py": '''
import sys
from pathlib import Path
from prefect import flow, serve
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

@flow(name="data_pipeline_flow", log_prints=True)
def data_pipeline_flow(years_back: int = 3):
    """Real data pipeline flow - fetches actual Premier League data"""
    print(f"🔄 Starting real data pipeline (fetching {years_back} years of data)")
    
    try:
        # Import the real data fetcher
        from src.data_integration.football_data_fetcher import FootballDataFetcher
        
        # Initialize fetcher
        fetcher = FootballDataFetcher()
        print(f"📡 Fetching Premier League data for last {years_back} years...")
        
        # Get historical data
        df = fetcher.get_historical_data(years_back=years_back)
        
        if df.empty:
            print("❌ No data fetched")
            return {"status": "failed", "error": "No data retrieved"}
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = fetcher.save_processed_data(df, f"premier_league_data_{timestamp}.csv")
        
        print(f"✅ Successfully fetched {len(df)} matches")
        print(f"📁 Data saved to: {filepath}")
        
        result = {
            "status": "completed",
            "matches_fetched": len(df),
            "file_path": str(filepath),
            "seasons": df['season'].unique().tolist() if 'season' in df.columns else [],
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"🎉 Data pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    serve(
        data_pipeline_flow.to_deployment(
            name="data_pipeline_flow-deployment",
            work_pool_name="premier-league-pool"
        )
    )
''',
            
            "training_flow.py": '''
import sys
from pathlib import Path
from prefect import flow, serve
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

@flow(name="training_pipeline_flow", log_prints=True)
def training_pipeline_flow(force_retrain: bool = False):
    """Real training pipeline flow - trains actual ML model"""
    print(f"🤖 Starting real training pipeline (force_retrain={force_retrain})")
    
    try:
        # Import the real training pipeline
        from src.pipelines.training_pipeline import TrainingPipeline
        
        # Initialize training pipeline
        pipeline = TrainingPipeline()
        print("🎯 Initializing training pipeline...")
        
        # Run training
        print("🔥 Starting model training...")
        run_id = pipeline.run_training()
        
        print(f"✅ Training completed with MLflow run ID: {run_id}")
        
        # Get model info
        model_info = {
            "run_id": run_id,
            "model_available": pipeline.model is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(pipeline, 'model') and pipeline.model:
            print("📊 Model training successful - model available for predictions")
        
        result = {
            "status": "completed",
            "mlflow_run_id": run_id,
            "model_available": pipeline.model is not None,
            "force_retrain": force_retrain,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"🎉 Training pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "force_retrain": force_retrain,
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    serve(
        training_pipeline_flow.to_deployment(
            name="training_pipeline_flow-deployment",
            work_pool_name="premier-league-pool"
        )
    )
''',
            
            "prediction_flow.py": '''
import sys
from pathlib import Path
from prefect import flow, serve
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

@flow(name="prediction_pipeline_flow", log_prints=True) 
def prediction_pipeline_flow(days_ahead: int = 7):
    """Real prediction pipeline flow - generates actual match predictions"""
    print(f"🔮 Starting real prediction pipeline (next {days_ahead} days)")
    
    try:
        # Import the real prediction pipeline and data fetcher
        from src.pipelines.prediction_pipeline import PredictionPipeline
        from src.data_integration.real_data_fetcher import RealDataFetcher
        
        # Initialize pipelines
        prediction_pipeline = PredictionPipeline()
        data_fetcher = RealDataFetcher()
        
        print("📡 Fetching upcoming matches...")
        
        # Get upcoming matches
        upcoming_matches = data_fetcher.get_upcoming_matches(days_ahead=days_ahead)
        
        if not upcoming_matches:
            print("⚠️ No upcoming matches found")
            return {
                "status": "completed",
                "predictions_generated": 0,
                "message": "No upcoming matches found",
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"🎯 Found {len(upcoming_matches)} upcoming matches")
        
        # Generate predictions for each match
        predictions = []
        for match in upcoming_matches:
            try:
                print(f"🤔 Predicting: {match['home_team']} vs {match['away_team']}")
                
                prediction = prediction_pipeline.predict_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    home_odds=match.get("home_odds"),
                    away_odds=match.get("away_odds"),
                    draw_odds=match.get("draw_odds"),
                )
                
                predictions.append({
                    "match": match,
                    "prediction": prediction,
                })
                
                outcome = prediction.get("prediction", "unknown")
                confidence = prediction.get("confidence", 0)
                print(f"✨ Prediction: {outcome} (confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"❌ Error predicting match {match['home_team']} vs {match['away_team']}: {e}")
                continue
        
        print(f"✅ Generated {len(predictions)} predictions")
        
        result = {
            "status": "completed",
            "matches_found": len(upcoming_matches),
            "predictions_generated": len(predictions),
            "predictions": predictions,
            "days_ahead": days_ahead,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"🎉 Prediction pipeline completed: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Prediction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "days_ahead": days_ahead,
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    serve(
        prediction_pipeline_flow.to_deployment(
            name="prediction_pipeline_flow-deployment",
            work_pool_name="premier-league-pool"
        )
    )
'''
        }
        
        # Write flow files
        for filename, content in flow_files.items():
            filepath = flows_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            logger.info(f"Created real flow file: {filepath}")
        
        # Start flow servers in background
        logger.info("Starting real flow servers...")
        processes = []
        
        for filename in flow_files.keys():
            filepath = flows_dir / filename
            
            # Start each flow server
            process = subprocess.Popen([
                "python", str(filepath)
            ], cwd=str(project_root))
            
            processes.append(process)
            logger.info(f"✅ Started real flow server for {filename}")
        
        # Wait a bit for servers to start
        import time
        time.sleep(5)
        
        logger.info("✅ Prefect setup completed with REAL pipelines!")
        logger.info("🎯 Data fetching, model training, and prediction generation are now available")
        logger.info("Flow deployments should be visible in Prefect UI at http://localhost:4200")
        
        # Keep processes running (this container should stay alive)
        logger.info("Keeping real flow servers running...")
        try:
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down flow servers...")
            for process in processes:
                process.terminate()
        
    except Exception as e:
        logger.error(f"Error setting up Prefect: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    setup_prefect() 