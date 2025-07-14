"""
Simplified FastAPI application for Premier League MLOps System
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_integration.real_data_fetcher import RealDataFetcher
from pipelines.prediction_pipeline import PredictionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variables for singleton instances
prediction_pipeline = None
real_data_fetcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown."""
    global prediction_pipeline, real_data_fetcher

    # Startup
    logger.info("🚀 Starting Premier League Match Predictor API...")
    try:
        # Initialize components
        prediction_pipeline = PredictionPipeline()
        real_data_fetcher = RealDataFetcher()

        logger.info("✅ All components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise

    yield

    # Shutdown
    logger.info("🔄 Shutting down Premier League Match Predictor API...")


# Create FastAPI app
app = FastAPI(
    title="Premier League Match Predictor API",
    description="MLOps system for predicting Premier League match outcomes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for match prediction."""

    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    home_odds: float | None = Field(None, description="Home win odds")
    away_odds: float | None = Field(None, description="Away win odds")
    draw_odds: float | None = Field(None, description="Draw odds")


class PredictionResponse(BaseModel):
    """Response model for match prediction."""

    home_team: str
    away_team: str
    prediction: str = Field(..., description="Predicted outcome: H (Home), A (Away), D (Draw)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: dict[str, float] = Field(..., description="Outcome probabilities")
    created_at: datetime


class MatchData(BaseModel):
    """Match data for upcoming fixtures."""

    home_team: str
    away_team: str
    match_date: datetime
    home_odds: float
    away_odds: float
    draw_odds: float


class SystemStatus(BaseModel):
    """System status response."""

    status: str
    version: str
    components: dict[str, str]
    timestamp: datetime


# Dependency functions
def get_prediction_pipeline():
    """Get prediction pipeline instance."""
    if prediction_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction pipeline not initialized",
        )
    return prediction_pipeline


def get_real_data_fetcher():
    """Get real data fetcher instance."""
    if real_data_fetcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Real data fetcher not initialized",
        )
    return real_data_fetcher


# API endpoints
@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Premier League Match Predictor API"}


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint."""
    return SystemStatus(
        status="healthy",
        version="1.0.0",
        components={
            "prediction_pipeline": "healthy" if prediction_pipeline is not None else "unhealthy",
            "real_data_fetcher": "healthy" if real_data_fetcher is not None else "unhealthy",
        },
        timestamp=datetime.now(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest, pipeline: PredictionPipeline = Depends(get_prediction_pipeline)):
    """Predict the outcome of a match."""
    try:
        logger.info(f"🔮 Predicting match: {request.home_team} vs {request.away_team}")

        prediction = pipeline.predict_match(
            home_team=request.home_team,
            away_team=request.away_team,
            home_odds=request.home_odds,
            away_odds=request.away_odds,
            draw_odds=request.draw_odds,
        )

        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=prediction["prediction"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            created_at=prediction["created_at"],
        )

    except Exception as e:
        logger.error(f"❌ Failed to predict match: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict match: {str(e)}",
        )


@app.get("/predictions/upcoming", response_model=list[PredictionResponse])
async def get_upcoming_predictions(
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
    fetcher: RealDataFetcher = Depends(get_real_data_fetcher),
):
    """Get predictions for upcoming matches."""
    try:
        logger.info("🔮 Generating predictions for upcoming matches")

        # Get upcoming matches
        upcoming_matches = fetcher.get_upcoming_matches(days_ahead=7)

        if not upcoming_matches:
            return []

        # Generate predictions
        predictions = []
        for match in upcoming_matches:
            try:
                prediction = pipeline.predict_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    home_odds=match.get("home_odds"),
                    away_odds=match.get("away_odds"),
                    draw_odds=match.get("draw_odds"),
                )

                predictions.append(
                    PredictionResponse(
                        home_team=match["home_team"],
                        away_team=match["away_team"],
                        prediction=prediction["prediction"],
                        confidence=prediction["confidence"],
                        probabilities=prediction["probabilities"],
                        created_at=prediction["created_at"],
                    )
                )
            except Exception as e:
                logger.error(f"❌ Failed to predict match {match['home_team']} vs {match['away_team']}: {e}")
                continue

        return predictions

    except Exception as e:
        logger.error(f"❌ Failed to get upcoming predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upcoming predictions: {str(e)}",
        )


@app.get("/matches/upcoming", response_model=list[MatchData])
async def get_upcoming_matches(fetcher: RealDataFetcher = Depends(get_real_data_fetcher)):
    """Get upcoming Premier League matches."""
    try:
        logger.info("📅 Fetching upcoming matches")

        upcoming_matches = fetcher.get_upcoming_matches()

        matches = []
        for match in upcoming_matches:
            matches.append(
                MatchData(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    match_date=match["match_date"],
                    home_odds=match["home_odds"],
                    away_odds=match["away_odds"],
                    draw_odds=match["draw_odds"],
                )
            )

        return matches

    except Exception as e:
        logger.error(f"❌ Failed to get upcoming matches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upcoming matches: {str(e)}",
        )


@app.get("/model/info", response_model=dict[str, Any])
async def get_model_info(pipeline: PredictionPipeline = Depends(get_prediction_pipeline)):
    """Get model information."""
    try:
        logger.info("📊 Getting model info")

        model_info = pipeline.get_model_info()
        return model_info

    except Exception as e:
        logger.error(f"❌ Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@app.post("/retraining/force", response_model=dict[str, Any])
async def force_retraining():
    """Force model retraining."""
    try:
        logger.info("🔄 Forcing model retraining")

        # In a real implementation, this would trigger a Prefect flow
        # For now, just return a success message
        return {
            "success": True,
            "message": "Retraining triggered successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to force retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force retraining: {str(e)}",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) 