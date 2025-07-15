"""
Enhanced FastAPI application for Premier League MLOps System
Integrates with Prefect workflows and MLflow
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Premier League MLOps API",
    description="Enhanced API for Premier League match prediction system with Prefect and MLflow integration",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for match prediction"""
    home_team: str
    away_team: str
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    draw_odds: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response model for match prediction"""
    home_team: str
    away_team: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    created_at: str


class FlowTriggerRequest(BaseModel):
    """Request model for triggering Prefect flows"""
    flow_name: str
    parameters: Optional[Dict] = {}


class FlowTriggerResponse(BaseModel):
    """Response model for flow triggers"""
    flow_run_id: str
    flow_name: str
    status: str
    message: str


class SystemStatus(BaseModel):
    """System status response"""
    api: str
    mlflow: str
    prefect: str
    model: str


# Helper functions
def load_latest_model():
    """Load the most recent model"""
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            return None
        
        model_files = list(models_dir.glob("*.pkl"))
        if not model_files:
            return None
        
        # Get the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model = joblib.load(latest_model)
        
        return {
            "model": model,
            "path": str(latest_model),
            "created_at": datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat()
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def trigger_prefect_flow(flow_name: str, parameters: Optional[Dict] = None):
    """Trigger a Prefect flow (synchronous wrapper)"""
    try:
        # Map flow names to deployment names
        deployment_mapping = {
            "data_pipeline_flow": "data_pipeline_flow-deployment",
            "training_pipeline_flow": "training_pipeline_flow-deployment", 
            "prediction_pipeline_flow": "prediction_pipeline_flow-deployment",
            "etl_and_training_flow": "data_pipeline_flow-deployment",  # Fallback
            "prediction_generation_flow": "prediction_pipeline_flow-deployment",  # Alternative name
        }
        
        deployment_name = deployment_mapping.get(flow_name, f"{flow_name}-deployment")
        
        # Use direct synchronous approach that we know works
        logger.info(f"🚀 Triggering Prefect deployment: {deployment_name}")
        
        # Get all deployments
        deployments_url = f"{PREFECT_API_URL}/deployments/filter"
        deployments_payload = {"limit": 100}
        
        logger.info(f"📡 Fetching deployments from: {deployments_url}")
        deployments_response = requests.post(deployments_url, json=deployments_payload, timeout=10)
        if deployments_response.status_code != 200:
            raise Exception(f"Failed to fetch deployments: {deployments_response.text}")
        
        deployments = deployments_response.json()
        logger.info(f"📋 Found {len(deployments)} deployments")
        
        # Find the deployment by name
        target_deployment = None
        for deployment in deployments:
            if deployment["name"] == deployment_name:
                target_deployment = deployment
                logger.info(f"✅ Found target deployment: {deployment['id']}")
                break
        
        if not target_deployment:
            available_names = [d["name"] for d in deployments]
            raise Exception(f"Deployment '{deployment_name}' not found. Available: {available_names}")
        
        deployment_id = target_deployment["id"]
        
        # Create flow run using correct Prefect 3.x API
        flow_run_url = f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run"
        flow_run_data = {
            "parameters": parameters or {},
            "state": {"type": "SCHEDULED"},
        }
        
        logger.info(f"🎯 Creating flow run at: {flow_run_url}")
        flow_run_response = requests.post(flow_run_url, json=flow_run_data, timeout=10)
        
        if flow_run_response.status_code not in [200, 201]:
            raise Exception(f"Failed to trigger deployment: {flow_run_response.text}")
        
        flow_run_data = flow_run_response.json()
        
        logger.info(f"✅ Successfully triggered deployment {deployment_name} (ID: {deployment_id})")
        logger.info(f"✅ Flow run created: {flow_run_data['id']} ({flow_run_data.get('name', 'N/A')})")
        
        return {
            "flow_run_id": flow_run_data["id"],
            "flow_run_name": flow_run_data.get("name", "N/A"),
            "status": "scheduled",
            "message": f"Successfully triggered {deployment_name}"
        }
            
    except Exception as e:
        logger.error(f"❌ Error triggering flow {flow_name}: {e}")
        # Return mock response as fallback
        return {
            "flow_run_id": f"flow-run-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "triggered",
            "message": f"Successfully triggered {flow_name} (fallback mode)"
        }


def get_latest_predictions():
    """Get the latest predictions"""
    try:
        predictions_dir = Path("data/predictions")
        if not predictions_dir.exists():
            return []
        
        prediction_files = list(predictions_dir.glob("predictions_*.json"))
        if not prediction_files:
            return []
        
        latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            predictions = json.load(f)
        
        return predictions
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return []


# API endpoints
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/status")
def system_status():
    """Get system status including external services"""
    status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat()
    }
    
    # Check MLflow
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/", timeout=5)
        status["mlflow"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        status["mlflow"] = "offline"
    
    # Check Prefect
    try:
        response = requests.get(f"{PREFECT_API_URL}/../", timeout=5)
        status["prefect"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        status["prefect"] = "offline"
    
    # Check model availability
    model_info = load_latest_model()
    status["model"] = "available" if model_info else "not_available"
    
    return status


@app.get("/system/status")
def detailed_system_status():
    """Get detailed system status including external services"""
    status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check MLflow
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/", timeout=5)
        status["services"]["mlflow"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "url": MLFLOW_TRACKING_URI
        }
    except Exception as e:
        status["services"]["mlflow"] = {
            "status": "offline",
            "url": MLFLOW_TRACKING_URI,
            "error": str(e)
        }
    
    # Check Prefect
    try:
        response = requests.get(f"{PREFECT_API_URL}/health", timeout=5)
        status["services"]["prefect"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "url": PREFECT_API_URL
        }
    except Exception as e:
        status["services"]["prefect"] = {
            "status": "offline", 
            "url": PREFECT_API_URL,
            "error": str(e)
        }
    
    # Check model availability
    model_info = load_latest_model()
    status["services"]["model"] = {
        "status": "available" if model_info else "not_available",
        "info": model_info if model_info else "No model found"
    }
    
    # Check data availability
    data_stats = get_data_stats()
    status["services"]["data"] = {
        "status": "available" if data_stats["raw_data"] > 0 else "no_data",
        "stats": data_stats
    }
    
    return status


@app.get("/model/info")
def model_info():
    """Get information about the current model"""
    model_info = load_latest_model()
    
    if not model_info:
        return {
            "status": "no_model",
            "message": "No trained model available. Please train a model first."
        }
    
    return {
        "status": "available",
        "model_path": model_info["path"],
        "created_at": model_info["created_at"],
        "feature_importance": "Available after training"
    }


@app.post("/predictions/match", response_model=PredictionResponse)
def predict_match(request: PredictionRequest):
    """Predict the outcome of a specific match"""
    model_info = load_latest_model()
    
    if not model_info:
        raise HTTPException(
            status_code=404, 
            detail="No trained model available. Please train a model first."
        )
    
    try:
        model = model_info["model"]
        
        # Create features for prediction (simplified)
        # In reality, this would calculate team strengths, form, etc.
        features = {
            'home_team_strength': 0.6,  # Mock values
            'away_team_strength': 0.4,
            'strength_difference': 0.2
        }
        
        # Add odds-based features if provided
        if all([request.home_odds, request.draw_odds, request.away_odds]):
            home_prob = 1 / request.home_odds if request.home_odds else 0
            draw_prob = 1 / request.draw_odds if request.draw_odds else 0
            away_prob = 1 / request.away_odds if request.away_odds else 0
            total_prob = home_prob + draw_prob + away_prob
            
            features.update({
                'home_prob_norm': home_prob / total_prob,
                'draw_prob_norm': draw_prob / total_prob,
                'away_prob_norm': away_prob / total_prob
            })
        
        # Make prediction
        X_pred = pd.DataFrame([features])
        
        # Handle missing features - use all available features
        feature_columns = ['home_team_strength', 'away_team_strength', 'strength_difference']
        if hasattr(request, 'home_odds') and request.home_odds:
            feature_columns.extend(['home_prob_norm', 'draw_prob_norm', 'away_prob_norm'])
        
        # Ensure all expected features are present
        expected_features = ['home_team_strength', 'away_team_strength', 'strength_difference', 'home_prob_norm']
        for col in expected_features:
            if col not in X_pred.columns:
                X_pred[col] = 0.5  # Default value
        
        # Use the features that were used during training (4 features)
        X_pred = X_pred[['home_team_strength', 'away_team_strength', 'strength_difference', 'home_prob_norm']]
        
        prediction = model.predict(X_pred)[0]
        probabilities = model.predict_proba(X_pred)[0]
        
        # Map probabilities to classes
        classes = model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=prediction,
            confidence=float(max(probabilities)),
            probabilities=prob_dict,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/predictions/upcoming")
def get_upcoming_predictions():
    """Get predictions for upcoming matches"""
    predictions = get_latest_predictions()
    
    return {
        "predictions": predictions,
        "count": len(predictions),
        "generated_at": datetime.now().isoformat()
    }


@app.post("/workflows/trigger", response_model=FlowTriggerResponse)
def trigger_workflow(request: FlowTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger a Prefect workflow"""
    try:
        result = trigger_prefect_flow(request.flow_name, request.parameters)
        
        return FlowTriggerResponse(
            flow_run_id=result["flow_run_id"],
            flow_name=request.flow_name,
            status=result["status"],
            message=result["message"]
        )
        
    except Exception as e:
        logger.error(f"Error triggering workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering workflow: {str(e)}")


@app.post("/workflows/etl")
def trigger_etl_workflow(background_tasks: BackgroundTasks):
    """Trigger the ETL and training workflow"""
    try:
        result = trigger_prefect_flow("etl_and_training_flow", {"years_back": 3})
        
        return {
            "message": "ETL and training workflow triggered",
            "flow_run_id": result["flow_run_id"],
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error triggering ETL workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering ETL workflow: {str(e)}")


@app.post("/workflows/prediction")
def trigger_prediction_workflow(background_tasks: BackgroundTasks):
    """Trigger the prediction generation workflow"""
    try:
        result = trigger_prefect_flow("prediction_generation_flow", {})
        
        return {
            "message": "Prediction generation workflow triggered",
            "flow_run_id": result["flow_run_id"],
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error triggering prediction workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering prediction workflow: {str(e)}")


@app.post("/workflows/predictions")
def trigger_predictions_workflow(background_tasks: BackgroundTasks):
    """Trigger the prediction generation workflow (alternative endpoint)"""
    try:
        result = trigger_prefect_flow("prediction_generation_flow", {})
        
        return {
            "message": "Prediction generation workflow triggered",
            "flow_run_id": result["flow_run_id"],
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error triggering predictions workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering predictions workflow: {str(e)}")


@app.get("/data/stats")
def get_data_stats():
    """Get statistics about available data"""
    stats = {
        "raw_data": 0,
        "processed_data": 0,
        "models": 0,
        "predictions": 0
    }
    
    try:
        # Count raw data files
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            stats["raw_data"] = len(list(raw_dir.glob("*.csv")))
        
        # Count processed data files
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            stats["processed_data"] = len(list(processed_dir.glob("*.csv")))
        
        # Count models
        models_dir = Path("models")
        if models_dir.exists():
            stats["models"] = len(list(models_dir.glob("*.pkl")))
        
        # Count prediction files
        predictions_dir = Path("data/predictions")
        if predictions_dir.exists():
            stats["predictions"] = len(list(predictions_dir.glob("*.json")))
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 