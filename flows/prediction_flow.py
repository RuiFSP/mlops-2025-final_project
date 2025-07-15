
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
