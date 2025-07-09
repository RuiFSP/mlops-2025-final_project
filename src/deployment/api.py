"""FastAPI application for Premier League match prediction."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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

# Import automation components
try:
    from src.automation.retraining_scheduler import AutomatedRetrainingScheduler, RetrainingConfig

    RETRAINING_AVAILABLE = True
except ImportError:
    RETRAINING_AVAILABLE = False
    logger.warning("Automated retraining components not available")

# Global variables
trainer: ModelTrainer | None = None
retraining_scheduler = None


async def load_model() -> None:
    """Load the trained model on startup."""
    global trainer, retraining_scheduler
    try:
        model_path = Path("models")
        if model_path.exists() and any(model_path.glob("*.pkl")):
            trainer = ModelTrainer()
            trainer.load_model(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.warning("No trained model found. Please train a model first.")

        # Initialize retraining scheduler if available
        if RETRAINING_AVAILABLE:
            try:
                config_path = "config/retraining_config.yaml"
                if Path(config_path).exists():
                    retraining_scheduler = AutomatedRetrainingScheduler(config_path=config_path)
                else:
                    retraining_scheduler = AutomatedRetrainingScheduler()

                # Don't auto-start scheduler on API startup for safety
                logger.info("Retraining scheduler initialized (not started)")
            except Exception as e:
                logger.error(f"Error initializing retraining scheduler: {e}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan."""
    # Startup
    await load_model()
    yield
    # Shutdown
    global retraining_scheduler
    if retraining_scheduler and hasattr(retraining_scheduler, "stop"):
        retraining_scheduler.stop()


# Initialize FastAPI app
app = FastAPI(
    title="Premier League Match Predictor API",
    description="API for predicting Premier League match outcomes",
    version="1.0.0",
    lifespan=lifespan,
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


# Additional Pydantic models for retraining endpoints
class RetrainingTriggerRequest(BaseModel):
    reason: str = "manual_trigger"
    force: bool = False


class RetrainingConfigUpdate(BaseModel):
    performance_threshold: float | None = None
    drift_threshold: float | None = None
    max_days_without_retraining: int | None = None
    min_days_between_retraining: int | None = None
    check_interval_minutes: int | None = None
    enable_automatic_deployment: bool | None = None


class RetrainingStatus(BaseModel):
    is_running: bool
    retraining_in_progress: bool
    last_check_time: str | None
    last_retraining_time: str | None
    prediction_count_since_retraining: int
    days_since_last_retraining: int | None
    total_trigger_events: int
    config: dict


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

        # Record prediction for retraining tracking
        if retraining_scheduler:
            try:
                retraining_scheduler.record_prediction(
                    {
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "prediction": predictions[0] if len(predictions) > 0 else None,
                    }
                )
            except Exception as e:
                logger.debug(f"Could not record prediction for retraining: {e}")

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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e


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
        raise HTTPException(status_code=500, detail=f"Bulk prediction error: {str(e)}") from e


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


# Automated Retraining Endpoints
@app.get("/retraining/status", response_model=RetrainingStatus)
async def get_retraining_status() -> RetrainingStatus:
    """Get current status of the automated retraining system."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    status = retraining_scheduler.get_status()
    return RetrainingStatus(**status)


@app.post("/retraining/start")
async def start_retraining_scheduler() -> dict[str, Any]:
    """Start the automated retraining scheduler."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    try:
        retraining_scheduler.start_scheduler()
        return {"message": "Automated retraining scheduler started", "status": "running"}
    except Exception as e:
        logger.error(f"Error starting retraining scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")


@app.post("/retraining/stop")
async def stop_retraining_scheduler() -> dict[str, Any]:
    """Stop the automated retraining scheduler."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    try:
        retraining_scheduler.stop_scheduler()
        return {"message": "Automated retraining scheduler stopped", "status": "stopped"}
    except Exception as e:
        logger.error(f"Error stopping retraining scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop scheduler: {str(e)}")


@app.post("/retraining/trigger")
async def trigger_retraining(request: RetrainingTriggerRequest) -> dict[str, Any]:
    """Manually trigger model retraining."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    try:
        if request.force:
            success = retraining_scheduler.force_retraining(request.reason)
        else:
            # Check if retraining should be triggered based on current conditions
            # For manual triggers, we'll force it unless explicitly told not to
            success = retraining_scheduler.force_retraining(request.reason)

        if success:
            return {
                "message": "Retraining triggered successfully",
                "reason": request.reason,
                "forced": request.force,
            }
        else:
            return {
                "message": "Retraining could not be triggered (already in progress)",
                "reason": request.reason,
                "forced": request.force,
            }
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


@app.get("/retraining/history")
async def get_retraining_history() -> dict[str, Any]:
    """Get history of retraining events."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    try:
        trigger_history = retraining_scheduler.get_trigger_history()
        retraining_history = retraining_scheduler.retraining_orchestrator.get_retraining_history()

        return {
            "trigger_events": trigger_history,
            "retraining_events": retraining_history,
            "total_triggers": len(trigger_history),
            "total_retrainings": len(retraining_history),
        }
    except Exception as e:
        logger.error(f"Error getting retraining history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.post("/retraining/config")
async def update_retraining_config(config_update: RetrainingConfigUpdate) -> dict[str, Any]:
    """Update retraining configuration."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    try:
        current_config = retraining_scheduler.config

        # Update only provided fields
        if config_update.performance_threshold is not None:
            current_config.performance_threshold = config_update.performance_threshold
        if config_update.drift_threshold is not None:
            current_config.drift_threshold = config_update.drift_threshold
        if config_update.max_days_without_retraining is not None:
            current_config.max_days_without_retraining = config_update.max_days_without_retraining
        if config_update.min_days_between_retraining is not None:
            current_config.min_days_between_retraining = config_update.min_days_between_retraining
        if config_update.check_interval_minutes is not None:
            current_config.check_interval_minutes = config_update.check_interval_minutes
        if config_update.enable_automatic_deployment is not None:
            current_config.enable_automatic_deployment = config_update.enable_automatic_deployment

        # Apply updated configuration
        retraining_scheduler.update_config(current_config)

        return {
            "message": "Retraining configuration updated successfully",
            "updated_config": retraining_scheduler.get_status()["config"],
        }
    except Exception as e:
        logger.error(f"Error updating retraining config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@app.get("/retraining/config")
async def get_retraining_config() -> dict[str, Any]:
    """Get current retraining configuration."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    return {
        "config": retraining_scheduler.get_status()["config"],
        "retraining_available": True,
    }


@app.post("/retraining/export")
async def export_retraining_report(
    output_path: str = "evaluation_reports/retraining_status_report.json",
) -> dict[str, Any]:
    """Export detailed retraining status report."""
    if not RETRAINING_AVAILABLE or retraining_scheduler is None:
        raise HTTPException(status_code=503, detail="Retraining system not available")

    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        retraining_scheduler.export_status_report(output_path)

        return {
            "message": "Retraining report exported successfully",
            "output_path": output_path,
            "export_time": pd.Timestamp.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error exporting retraining report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")


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
