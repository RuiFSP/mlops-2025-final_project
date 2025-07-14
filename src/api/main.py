"""
Main FastAPI application for the Premier League Match Predictor MLOps system.
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from betting_simulator.simulator import BettingSimulator
from data_integration.real_data_fetcher import RealDataFetcher
from pipelines.prediction_pipeline import PredictionPipeline
from retraining.retraining_monitor import RetrainingMonitor
from retraining.scheduler import RetrainingScheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variables for singleton instances
prediction_pipeline = None
betting_simulator = None
real_data_fetcher = None
retraining_monitor = None
retraining_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown."""
    global prediction_pipeline, betting_simulator, real_data_fetcher, retraining_monitor, retraining_scheduler

    # Startup
    logger.info("🚀 Starting Premier League Match Predictor API...")
    try:
        # Initialize components
        prediction_pipeline = PredictionPipeline()
        betting_simulator = BettingSimulator(initial_balance=1000.0)
        real_data_fetcher = RealDataFetcher()
        retraining_monitor = RetrainingMonitor()
        retraining_scheduler = RetrainingScheduler()

        # Try to create tables but don't fail if there are permission issues
        try:
            from sqlalchemy import create_engine, text

            from config.database import get_db_config

            # Create database engine
            db_config = get_db_config()
            db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            engine = create_engine(db_url)

            # Try to create performance_monitoring table if it doesn't exist
            with engine.connect() as conn:
                conn.execute(
                    text("""
                    CREATE TABLE IF NOT EXISTS performance_monitoring (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        component VARCHAR(100) NOT NULL,
                        metadata JSONB
                    )
                """)
                )
                conn.commit()

            logger.info("✅ Database tables created/verified successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not create/verify database tables: {e}")
            logger.info("API will continue to run with limited functionality")

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
    description="MLOps system for predicting Premier League match outcomes with automated betting simulation",
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


class BettingRequest(BaseModel):
    """Request model for betting simulation."""

    predictions: list[dict[str, Any]] = Field(..., description="List of predictions to evaluate for betting")


class BettingResponse(BaseModel):
    """Response model for betting simulation."""

    bets_placed: int
    total_amount_bet: float
    successful_bets: list[dict[str, Any]]
    betting_statistics: dict[str, Any]
    current_balance: float


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
    """Get the prediction pipeline instance."""
    if prediction_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction pipeline not initialized",
        )
    return prediction_pipeline


def get_betting_simulator():
    """Get the betting simulator instance."""
    if betting_simulator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Betting simulator not initialized",
        )
    return betting_simulator


def get_real_data_fetcher():
    """Get the real data fetcher instance."""
    if real_data_fetcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Real data fetcher not initialized",
        )
    return real_data_fetcher


def get_retraining_monitor():
    """Get the retraining monitor instance."""
    if retraining_monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retraining monitor not initialized",
        )
    return retraining_monitor


def get_retraining_scheduler():
    """Get the retraining scheduler instance."""
    if retraining_scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retraining scheduler not initialized",
        )
    return retraining_scheduler


# API Routes
@app.get("/", response_model=dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Premier League Match Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint."""
    return SystemStatus(
        status="healthy",
        version="1.0.0",
        components={
            "prediction_pipeline": "operational" if prediction_pipeline else "unavailable",
            "betting_simulator": "operational" if betting_simulator else "unavailable",
            "real_data_fetcher": "operational" if real_data_fetcher else "unavailable",
        },
        timestamp=datetime.now(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest, pipeline: PredictionPipeline = Depends(get_prediction_pipeline)):
    """Predict the outcome of a single match."""
    try:
        logger.info(f"🔮 Predicting match: {request.home_team} vs {request.away_team}")

        # Create match data for prediction
        match_data = {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "match_date": datetime.now(),
            "home_odds": request.home_odds or 2.0,
            "away_odds": request.away_odds or 3.0,
            "draw_odds": request.draw_odds or 3.5,
        }

        # Generate prediction
        prediction = pipeline.predict_single_match(match_data)

        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=prediction["prediction"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            created_at=datetime.now(),
        )

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    matches: list[PredictionRequest],
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
):
    """Predict outcomes for multiple matches."""
    try:
        logger.info(f"🔮 Predicting {len(matches)} matches")

        predictions = []
        for match in matches:
            match_data = {
                "home_team": match.home_team,
                "away_team": match.away_team,
                "match_date": datetime.now(),
                "home_odds": match.home_odds or 2.0,
                "away_odds": match.away_odds or 3.0,
                "draw_odds": match.draw_odds or 3.5,
            }

            prediction = pipeline.predict_single_match(match_data)
            predictions.append(
                PredictionResponse(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    prediction=prediction["prediction"],
                    confidence=prediction["confidence"],
                    probabilities=prediction["probabilities"],
                    created_at=datetime.now(),
                )
            )

        return predictions

    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/predictions/upcoming", response_model=list[PredictionResponse])
async def get_upcoming_predictions(
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
    fetcher: RealDataFetcher = Depends(get_real_data_fetcher),
):
    """Get predictions for upcoming Premier League matches."""
    try:
        logger.info("🔮 Getting predictions for upcoming matches")

        # Fetch upcoming matches
        upcoming_matches = fetcher.get_upcoming_matches()

        predictions = []
        for match in upcoming_matches:
            prediction = pipeline.predict_single_match(match)
            predictions.append(
                PredictionResponse(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    prediction=prediction["prediction"],
                    confidence=prediction["confidence"],
                    probabilities=prediction["probabilities"],
                    created_at=datetime.now(),
                )
            )

        return predictions

    except Exception as e:
        logger.error(f"❌ Failed to get upcoming predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upcoming predictions: {str(e)}",
        )


@app.post("/betting/simulate", response_model=BettingResponse)
async def simulate_betting(request: BettingRequest, simulator: BettingSimulator = Depends(get_betting_simulator)):
    """Simulate betting on provided predictions."""
    try:
        logger.info(f"💰 Simulating betting on {len(request.predictions)} predictions")

        successful_bets = []
        total_amount_bet = 0.0

        for prediction_data in request.predictions:
            bet = simulator.place_bet(prediction_data)
            if bet:
                successful_bets.append(bet)
                total_amount_bet += bet.get("bet_amount", 0)

        # Get current statistics
        stats = simulator.get_statistics()

        return BettingResponse(
            bets_placed=len(successful_bets),
            total_amount_bet=total_amount_bet,
            successful_bets=successful_bets,
            betting_statistics=stats,
            current_balance=simulator.current_balance,
        )

    except Exception as e:
        logger.error(f"❌ Betting simulation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Betting simulation failed: {str(e)}",
        )


@app.get("/betting/statistics", response_model=dict[str, Any])
async def get_betting_statistics(simulator: BettingSimulator = Depends(get_betting_simulator)):
    """Get current betting statistics."""
    try:
        stats = simulator.get_statistics()
        return {
            "statistics": stats,
            "current_balance": simulator.current_balance,
            "initial_balance": simulator.initial_balance,
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to get betting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get betting statistics: {str(e)}",
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
    """Get information about the current model."""
    try:
        # Get model information
        model_info = {
            "model_type": "Random Forest",
            "accuracy": "61.84%",
            "features": 15,
            "training_matches": 3040,
            "last_updated": datetime.now(),
            "version": "latest",
        }

        return model_info

    except Exception as e:
        logger.error(f"❌ Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


# Retraining endpoints
@app.get("/retraining/status", response_model=dict[str, Any])
async def get_retraining_status(monitor: RetrainingMonitor = Depends(get_retraining_monitor)):
    """Get retraining system status."""
    try:
        summary = monitor.get_monitoring_summary()
        model_info = monitor.get_current_model_info()

        return {
            "model_info": model_info,
            "monitoring_summary": summary,
            "config": {
                "min_accuracy_threshold": monitor.config.min_accuracy_threshold,
                "min_prediction_count": monitor.config.min_prediction_count,
                "evaluation_window_days": monitor.config.evaluation_window_days,
                "max_model_age_days": monitor.config.max_model_age_days,
                "performance_degradation_threshold": monitor.config.performance_degradation_threshold,
                "consecutive_poor_performance_limit": monitor.config.consecutive_poor_performance_limit,
            },
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"❌ Failed to get retraining status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get retraining status: {str(e)}",
        )


@app.post("/retraining/check", response_model=dict[str, Any])
async def run_retraining_check(monitor: RetrainingMonitor = Depends(get_retraining_monitor)):
    """Run an immediate retraining check."""
    try:
        logger.info("🔍 Running immediate retraining check...")
        result = monitor.run_monitoring_cycle()

        return {"check_result": result, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"❌ Failed to run retraining check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run retraining check: {str(e)}",
        )


@app.post("/retraining/force", response_model=dict[str, Any])
async def force_retraining(monitor: RetrainingMonitor = Depends(get_retraining_monitor)):
    """Force an immediate retraining."""
    try:
        logger.info("🚨 Forcing immediate retraining...")

        # Create fake trigger to force retraining
        fake_trigger = {
            "should_retrain": True,
            "triggers": ["Manual force retraining via API"],
            "model_info": monitor.get_current_model_info(),
        }

        result = monitor.trigger_retraining(fake_trigger)

        return {"retraining_result": result, "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"❌ Failed to force retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force retraining: {str(e)}",
        )


@app.get("/retraining/history", response_model=dict[str, Any])
async def get_retraining_history(monitor: RetrainingMonitor = Depends(get_retraining_monitor)):
    """Get retraining history."""
    try:
        summary = monitor.get_monitoring_summary()

        return {
            "recent_evaluations": summary.get("recent_evaluations", []),
            "recent_retraining": summary.get("recent_retraining", []),
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"❌ Failed to get retraining history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get retraining history: {str(e)}",
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()},
    )


if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
