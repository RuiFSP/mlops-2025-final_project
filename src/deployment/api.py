"""FastAPI application for Premier League match prediction."""

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after path setup
from src.data_preprocessing.data_loader import DataLoader  # noqa: E402
from src.model_training.trainer import ModelTrainer  # noqa: E402

# Initialize FastAPI app
app = FastAPI(
    title="Premier League Match Predictor API",
    description="API for predicting Premier League match outcomes",
    version="1.0.0",
)


# Pydantic models for request/response
class MatchInput(BaseModel):
    home_team: str
    away_team: str
    date: str | None = None
    season: str | None = "2023-24"
    home_odds: float | None = 2.0
    draw_odds: float | None = 3.0
    away_odds: float | None = 2.5


class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    predicted_result: str
    home_win_probability: float | None = None
    draw_probability: float | None = None
    away_win_probability: float | None = None
    prediction_confidence: float | None = None


class BulkMatchInput(BaseModel):
    matches: list[MatchInput]


class BulkMatchPrediction(BaseModel):
    predictions: list[MatchPrediction]


# Global model trainer instance
trainer: ModelTrainer | None = None


@app.on_event("startup")
async def load_model() -> None:
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
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Premier League Match Predictor API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": trainer is not None,
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": trainer is not None}


@app.post("/predict", response_model=MatchPrediction)
async def predict_match(match: MatchInput) -> MatchPrediction:
    """Predict outcome for a single match."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create DataFrame from input
        match_data = pd.DataFrame(
            [
                {
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "date": match.date or "2024-01-01",
                    "season": match.season,
                    "home_score": 0,  # Placeholder
                    "away_score": 0,  # Placeholder
                    "home_odds": match.home_odds,
                    "draw_odds": match.draw_odds,
                    "away_odds": match.away_odds,
                }
            ]
        )

        # Preprocess data
        data_loader = DataLoader("")
        processed_data = data_loader.preprocess_data(match_data)

        # Debug: Log the processed data columns
        logger.info(f"Processed data columns: {processed_data.columns.tolist()}")
        logger.info(f"Processed data shape: {processed_data.shape}")

        # Make prediction and get probabilities
        predictions = trainer.predict(processed_data)
        probabilities = trainer.predict_proba(processed_data)
        class_order = trainer.get_class_order()

        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="Could not make prediction")

        prediction = predictions[0]
        proba = probabilities[0] if len(probabilities) > 0 else None

        # Convert prediction to readable format
        result_map = {"H": "Home Win", "A": "Away Win", "D": "Draw"}
        readable_prediction = result_map.get(prediction, str(prediction))

        # Extract individual probabilities
        home_prob = None
        draw_prob = None
        away_prob = None
        confidence = None

        if proba is not None and class_order is not None:
            # Create mapping from class to probability
            class_to_prob = {cls: proba[i] for i, cls in enumerate(class_order)}

            home_prob = class_to_prob.get("H", 0.0)
            draw_prob = class_to_prob.get("D", 0.0)
            away_prob = class_to_prob.get("A", 0.0)

            # Confidence is the maximum probability
            confidence = float(max(proba))

        return MatchPrediction(
            home_team=match.home_team,
            away_team=match.away_team,
            predicted_result=readable_prediction,
            home_win_probability=home_prob,
            draw_probability=draw_prob,
            away_win_probability=away_prob,
            prediction_confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e


@app.post("/predict/bulk", response_model=BulkMatchPrediction)
async def predict_matches_bulk(matches: BulkMatchInput) -> BulkMatchPrediction:
    """Predict outcomes for multiple matches."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create DataFrame from input
        match_data = pd.DataFrame(
            [
                {
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "date": match.date or "2024-01-01",
                    "season": match.season,
                    "home_score": 0,  # Placeholder
                    "away_score": 0,  # Placeholder
                    "home_odds": match.home_odds,
                    "draw_odds": match.draw_odds,
                    "away_odds": match.away_odds,
                }
                for match in matches.matches
            ]
        )

        # Preprocess data
        data_loader = DataLoader("")
        processed_data = data_loader.preprocess_data(match_data)

        # Make predictions
        predictions = trainer.predict(processed_data)

        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="Could not make predictions")

        # Convert predictions to readable format
        result_map = {"H": "Home Win", "A": "Away Win", "D": "Draw"}

        prediction_results = []
        for match, prediction in zip(matches.matches, predictions, strict=False):
            readable_prediction = result_map.get(prediction, str(prediction))
            prediction_results.append(
                MatchPrediction(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    predicted_result=readable_prediction,
                )
            )

        return BulkMatchPrediction(predictions=prediction_results)

    except Exception as e:
        logger.error(f"Error making bulk predictions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Bulk prediction error: {str(e)}"
        ) from e


@app.get("/teams")
async def get_teams() -> dict[str, list[str]]:
    """Get list of available teams."""
    # This would ideally come from the trained model or a configuration file
    teams = [
        "Arsenal",
        "Chelsea",
        "Liverpool",
        "Manchester United",
        "Manchester City",
        "Tottenham",
        "Newcastle United",
        "Brighton",
        "West Ham",
        "Aston Villa",
    ]
    return {"teams": teams}


@app.get("/model/info")
async def get_model_info() -> dict[str, Any]:
    """Get information about the loaded model."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": trainer.model_type,
        "model_loaded": True,
        "features": [
            "home_team",
            "away_team",
            "month",
            "goal_difference",
            "total_goals",
        ],
    }


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Premier League Match Predictor API")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
