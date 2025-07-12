"""
Main application entry point for the MLOps betting simulation system.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    logger.info("🚀 Starting MLOps Betting Simulation System")
    
    try:
        # Import components
        from pipelines.training_pipeline import TrainingPipeline
        from pipelines.prediction_pipeline import PredictionPipeline
        from betting_simulator.simulator import BettingSimulator
        
        logger.info("✅ All components loaded successfully")
        
        # Initialize components
        training_pipeline = TrainingPipeline()
        prediction_pipeline = PredictionPipeline()
        betting_simulator = BettingSimulator(initial_balance=1000.0)
        
        logger.info("✅ Components initialized successfully")
        
        # For now, just run a simple test
        logger.info("Running system test...")
        
        # Test prediction pipeline
        predictions = prediction_pipeline.run_prediction()
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Test betting simulator
        for prediction in predictions:
            bet = betting_simulator.place_bet(prediction)
            if bet:
                logger.info(f"Placed bet: £{bet['bet_amount']} on {bet['home_team']} vs {bet['away_team']}")
        
        # Get statistics
        stats = betting_simulator.get_statistics()
        logger.info(f"Betting statistics: {stats}")
        
        logger.info("✅ System test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in main application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
