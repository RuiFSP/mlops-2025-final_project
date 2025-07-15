
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
