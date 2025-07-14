"""
Simplified Prefect flows for Premier League MLOps System
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_integration.real_data_fetcher import RealDataFetcher
from monitoring.metrics_storage import MetricsStorage
from pipelines.prediction_pipeline import PredictionPipeline
from pipelines.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@task(name="fetch_football_data", tags=["data", "fetch"])
def fetch_football_data(season: str = "2023/24") -> dict[str, Any]:
    """
    Fetch football data from football-data.uk
    
    Args:
        season: Season to fetch data for
        
    Returns:
        Dictionary with fetch results
    """
    run_logger = get_run_logger()
    run_logger.info(f"🔍 Fetching football data for season {season}")
    
    try:
        # Create data fetcher
        data_fetcher = RealDataFetcher()
        
        # Get upcoming matches
        upcoming_matches = data_fetcher.get_upcoming_matches(days_ahead=7)
        
        # Log results
        run_logger.info(f"✅ Successfully fetched {len(upcoming_matches)} upcoming matches")
        
        return {
            "success": True,
            "matches_fetched": len(upcoming_matches),
            "matches": upcoming_matches,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        run_logger.error(f"❌ Error fetching football data: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task(name="preprocess_data", tags=["data", "preprocess"])
def preprocess_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Preprocess football data
    
    Args:
        data: Dictionary with fetch results
        
    Returns:
        Dictionary with preprocessing results
    """
    run_logger = get_run_logger()
    run_logger.info("🔍 Preprocessing football data")
    
    try:
        # Check if data fetch was successful
        if not data.get("success", False):
            run_logger.error("❌ Data fetch was not successful, cannot preprocess")
            return {
                "success": False,
                "error": "Data fetch was not successful",
                "timestamp": datetime.now().isoformat(),
            }
        
        # Get matches
        matches = data.get("matches", [])
        
        if not matches:
            run_logger.warning("⚠️ No matches to preprocess")
            return {
                "success": True,
                "matches_processed": 0,
                "timestamp": datetime.now().isoformat(),
            }
        
        # Preprocess matches
        processed_matches = []
        for match in matches:
            # Ensure all required fields are present
            if "home_team" in match and "away_team" in match:
                # Add any additional preprocessing here
                processed_match = match.copy()
                processed_matches.append(processed_match)
        
        # Log results
        run_logger.info(f"✅ Successfully preprocessed {len(processed_matches)} matches")
        
        return {
            "success": True,
            "matches_processed": len(processed_matches),
            "processed_matches": processed_matches,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        run_logger.error(f"❌ Error preprocessing data: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task(name="train_model", tags=["training", "model"])
def train_model(force_retrain: bool = False) -> dict[str, Any]:
    """
    Train the prediction model
    
    Args:
        force_retrain: Whether to force retraining
        
    Returns:
        Dictionary with training results
    """
    run_logger = get_run_logger()
    run_logger.info(f"🔍 Training model (force_retrain={force_retrain})")
    
    try:
        # Create training pipeline
        training_pipeline = TrainingPipeline()
        
        # Run training
        run_id = training_pipeline.run_training()
        
        # Log results
        run_logger.info(f"✅ Successfully trained model with run ID: {run_id}")
        
        return {
            "success": True,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        run_logger.error(f"❌ Error training model: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task(name="generate_predictions", tags=["prediction"])
def generate_predictions(days_ahead: int = 7) -> dict[str, Any]:
    """
    Generate predictions for upcoming matches
    
    Args:
        days_ahead: Number of days to look ahead for matches
        
    Returns:
        Dictionary with prediction results
    """
    run_logger = get_run_logger()
    run_logger.info(f"🔍 Generating predictions for next {days_ahead} days")
    
    try:
        # Create prediction pipeline
        prediction_pipeline = PredictionPipeline()
        
        # Get upcoming matches
        data_fetcher = RealDataFetcher()
        upcoming_matches = data_fetcher.get_upcoming_matches(days_ahead=days_ahead)
        
        if not upcoming_matches:
            run_logger.warning("⚠️ No upcoming matches found")
            return {
                "success": True,
                "predictions_generated": 0,
                "timestamp": datetime.now().isoformat(),
            }
        
        # Generate predictions
        predictions = []
        for match in upcoming_matches:
            try:
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
            except Exception as e:
                run_logger.error(f"❌ Error predicting match {match['home_team']} vs {match['away_team']}: {e}")
                continue
        
        # Log results
        run_logger.info(f"✅ Successfully generated {len(predictions)} predictions")
        
        return {
            "success": True,
            "predictions_generated": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        run_logger.error(f"❌ Error generating predictions: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@flow(name="data_pipeline_flow", description="Fetch and process football data")
def data_pipeline_flow(season: str = "2023/24") -> dict[str, Any]:
    """
    Flow for fetching and processing football data
    
    Args:
        season: Season to fetch data for
        
    Returns:
        Dictionary with flow results
    """
    run_logger = get_run_logger()
    run_logger.info(f"🚀 Starting data pipeline flow for season {season}")
    
    flow_results = {
        "flow_name": "data_pipeline_flow",
        "timestamp": datetime.now().isoformat(),
        "season": season,
        "steps_completed": [],
    }
    
    try:
        # Step 1: Fetch football data
        run_logger.info("📊 Fetching football data")
        fetch_results = fetch_football_data(season=season)
        flow_results["fetch_results"] = fetch_results
        flow_results["steps_completed"].append("fetch_data")
        
        # Step 2: Preprocess data
        if fetch_results.get("success", False):
            run_logger.info("📊 Preprocessing data")
            preprocess_results = preprocess_data(fetch_results)
            flow_results["preprocess_results"] = preprocess_results
            flow_results["steps_completed"].append("preprocess_data")
        
        run_logger.info("✅ Data pipeline flow completed successfully")
        flow_results["success"] = True
        
        return flow_results
    except Exception as e:
        run_logger.error(f"❌ Error in data pipeline flow: {e}")
        flow_results["success"] = False
        flow_results["error"] = str(e)
        
        return flow_results


@flow(name="training_pipeline_flow", description="Train prediction model")
def training_pipeline_flow(force_retrain: bool = False) -> dict[str, Any]:
    """
    Flow for training prediction model
    
    Args:
        force_retrain: Whether to force retraining
        
    Returns:
        Dictionary with flow results
    """
    run_logger = get_run_logger()
    run_logger.info(f"🚀 Starting training pipeline flow (force_retrain={force_retrain})")
    
    flow_results = {
        "flow_name": "training_pipeline_flow",
        "timestamp": datetime.now().isoformat(),
        "force_retrain": force_retrain,
        "steps_completed": [],
    }
    
    try:
        # Train model
        run_logger.info("📊 Training model")
        training_results = train_model(force_retrain=force_retrain)
        flow_results["training_results"] = training_results
        flow_results["steps_completed"].append("train_model")
        
        run_logger.info("✅ Training pipeline flow completed successfully")
        flow_results["success"] = True
        
        return flow_results
    except Exception as e:
        run_logger.error(f"❌ Error in training pipeline flow: {e}")
        flow_results["success"] = False
        flow_results["error"] = str(e)
        
        return flow_results


@flow(name="prediction_pipeline_flow", description="Generate predictions for upcoming matches")
def prediction_pipeline_flow(days_ahead: int = 7) -> dict[str, Any]:
    """
    Flow for generating predictions for upcoming matches
    
    Args:
        days_ahead: Number of days to look ahead for matches
        
    Returns:
        Dictionary with flow results
    """
    run_logger = get_run_logger()
    run_logger.info(f"🚀 Starting prediction pipeline flow for next {days_ahead} days")
    
    flow_results = {
        "flow_name": "prediction_pipeline_flow",
        "timestamp": datetime.now().isoformat(),
        "days_ahead": days_ahead,
        "steps_completed": [],
    }
    
    try:
        # Generate predictions
        run_logger.info("📊 Generating predictions")
        prediction_results = generate_predictions(days_ahead=days_ahead)
        flow_results["prediction_results"] = prediction_results
        flow_results["steps_completed"].append("generate_predictions")
        
        run_logger.info("✅ Prediction pipeline flow completed successfully")
        flow_results["success"] = True
        
        return flow_results
    except Exception as e:
        run_logger.error(f"❌ Error in prediction pipeline flow: {e}")
        flow_results["success"] = False
        flow_results["error"] = str(e)
        
        return flow_results


def register_flows():
    """Register all flows with Prefect"""
    logger.info("Registering flows with Prefect")
    
    # Nothing to do here - flows are registered automatically when imported
    pass


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Register flows
    register_flows()
    
    # Run data pipeline flow
    data_pipeline_flow() 