"""
FastAPI application for Premier League MLOps System
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Premier League MLOps API",
    description="API for Premier League match prediction system",
    version="0.1.0",
)

# Add src to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
import sys
sys.path.append(str(project_root))

# Import project modules
try:
    from src.pipelines.prediction_pipeline import PredictionPipeline
    prediction_pipeline = PredictionPipeline()
except Exception as e:
    logger.error(f"Failed to initialize prediction pipeline: {e}")
    prediction_pipeline = None


# Define models
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
    created_at: datetime


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/model/info")
def model_info():
    """Get model information"""
    if not prediction_pipeline:
        raise HTTPException(status_code=503, detail="Prediction pipeline not available")
    
    return prediction_pipeline.get_model_info()


@app.post("/predictions/match", response_model=PredictionResponse)
def predict_match(request: PredictionRequest):
    """Predict the outcome of a match"""
    if not prediction_pipeline:
        raise HTTPException(status_code=503, detail="Prediction pipeline not available")
    
    try:
        prediction = prediction_pipeline.predict_match(
            home_team=request.home_team,
            away_team=request.away_team,
            home_odds=request.home_odds,
            away_odds=request.away_odds,
            draw_odds=request.draw_odds,
        )
        return prediction
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/predictions/upcoming")
def get_upcoming_predictions():
    """Get predictions for upcoming matches"""
    if not prediction_pipeline:
        raise HTTPException(status_code=503, detail="Prediction pipeline not available")
    
    try:
        predictions = prediction_pipeline.run_prediction()
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {str(e)}")


@app.post("/retraining/force")
def force_retraining():
    """Force model retraining"""
    # This would typically trigger a workflow in a production system
    return {
        "status": "triggered",
        "message": "Model retraining has been triggered",
        "timestamp": datetime.now().isoformat()
    } 