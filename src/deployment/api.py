"""FastAPI application for Premier League match prediction."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_training.trainer import ModelTrainer
from src.data_preprocessing.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Premier League Match Predictor API",
    description="API for predicting Premier League match outcomes",
    version="1.0.0"
)

# Pydantic models for request/response
class MatchInput(BaseModel):
    home_team: str
    away_team: str
    date: Optional[str] = None
    season: Optional[str] = "2023-24"
    home_odds: Optional[float] = 2.0
    draw_odds: Optional[float] = 3.0
    away_odds: Optional[float] = 2.5

class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    predicted_result: str
    prediction_confidence: Optional[float] = None

class BulkMatchInput(BaseModel):
    matches: List[MatchInput]

class BulkMatchPrediction(BaseModel):
    predictions: List[MatchPrediction]

# Global model trainer instance
trainer: Optional[ModelTrainer] = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global trainer
    try:
        model_path = Path("models")
        if model_path.exists() and any(model_path.glob("*.pkl")):
            trainer = ModelTrainer()
            trainer.load_model(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.warning("No trained model found. Please train a model first.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Premier League Match Predictor API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": trainer is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": trainer is not None
    }

@app.post("/predict", response_model=MatchPrediction)
async def predict_match(match: MatchInput):
    """Predict outcome for a single match."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame from input
        match_data = pd.DataFrame([{
            'home_team': match.home_team,
            'away_team': match.away_team,
            'date': match.date or '2024-01-01',
            'season': match.season,
            'home_score': 0,  # Placeholder
            'away_score': 0,  # Placeholder
            'home_odds': match.home_odds,
            'draw_odds': match.draw_odds,
            'away_odds': match.away_odds
        }])
        
        # Preprocess data
        data_loader = DataLoader("")
        processed_data = data_loader.preprocess_data(match_data)
        
        # Debug: Log the processed data columns
        logger.info(f"Processed data columns: {processed_data.columns.tolist()}")
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        # Make prediction
        predictions = trainer.predict(processed_data)
        
        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="Could not make prediction")
        
        prediction = predictions[0]
        
        # Convert prediction to readable format
        result_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
        readable_prediction = result_map.get(prediction, str(prediction))
        
        return MatchPrediction(
            home_team=match.home_team,
            away_team=match.away_team,
            predicted_result=readable_prediction
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/bulk", response_model=BulkMatchPrediction)
async def predict_matches_bulk(matches: BulkMatchInput):
    """Predict outcomes for multiple matches."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame from input
        match_data = pd.DataFrame([{
            'home_team': match.home_team,
            'away_team': match.away_team,
            'date': match.date or '2024-01-01',
            'season': match.season,
            'home_score': 0,  # Placeholder
            'away_score': 0,  # Placeholder
            'home_odds': match.home_odds,
            'draw_odds': match.draw_odds,
            'away_odds': match.away_odds
        } for match in matches.matches])
        
        # Preprocess data
        data_loader = DataLoader("")
        processed_data = data_loader.preprocess_data(match_data)
        
        # Make predictions
        predictions = trainer.predict(processed_data)
        
        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="Could not make predictions")
        
        # Convert predictions to readable format
        result_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
        
        prediction_results = []
        for i, (match, prediction) in enumerate(zip(matches.matches, predictions)):
            readable_prediction = result_map.get(prediction, str(prediction))
            prediction_results.append(MatchPrediction(
                home_team=match.home_team,
                away_team=match.away_team,
                predicted_result=readable_prediction
            ))
        
        return BulkMatchPrediction(predictions=prediction_results)
        
    except Exception as e:
        logger.error(f"Error making bulk predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk prediction error: {str(e)}")

@app.get("/teams")
async def get_teams():
    """Get list of available teams."""
    # This would ideally come from the trained model or a configuration file
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester United", 
        "Manchester City", "Tottenham", "Newcastle United", 
        "Brighton", "West Ham", "Aston Villa"
    ]
    return {"teams": teams}

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": trainer.model_type,
        "model_loaded": True,
        "features": ["home_team", "away_team", "month", "goal_difference", "total_goals"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
