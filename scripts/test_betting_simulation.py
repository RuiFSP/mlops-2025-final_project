#!/usr/bin/env python3
"""
Test script to run betting simulation with detailed logging.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_betting_simulation():
    """Test the betting simulation with localhost connection."""

    # Set environment variables for localhost connection
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5432"

    try:
        from betting_simulator.simulator import BettingSimulator
        from pipelines.prediction_pipeline import PredictionPipeline

        logger.info("🚀 Starting betting simulation test...")

        # Initialize components
        prediction_pipeline = PredictionPipeline()
        betting_simulator = BettingSimulator(initial_balance=1000.0)

        logger.info("✅ Components initialized successfully")

        # Generate predictions
        logger.info("📊 Generating predictions...")
        predictions = prediction_pipeline.run_prediction()
        logger.info(f"Generated {len(predictions)} predictions")

        # Show predictions
        for i, pred in enumerate(predictions):
            logger.info(
                f"Prediction {i + 1}: {pred['home_team']} vs {pred['away_team']} -> {pred['prediction']} (confidence: {pred['confidence']:.2%})"
            )

        # Test betting simulation
        logger.info("🎲 Testing betting simulation...")
        bets_placed = 0

        for prediction in predictions:
            logger.info(f"\n--- Processing prediction: {prediction['home_team']} vs {prediction['away_team']} ---")

            # Check if we should place a bet
            should_bet = betting_simulator.should_place_bet(prediction)
            logger.info(f"Should place bet: {should_bet}")

            if should_bet:
                bet_amount = betting_simulator.calculate_bet_amount(prediction)
                logger.info(f"Calculated bet amount: £{bet_amount}")

                if bet_amount > 0:
                    bet = betting_simulator.place_bet(prediction)
                    if bet:
                        bets_placed += 1
                        logger.info(f"✅ Bet placed successfully: £{bet['bet_amount']} on {bet['bet_type']}")
                    else:
                        logger.warning("❌ Bet placement failed")
                else:
                    logger.info("Bet amount is 0, skipping")
            else:
                logger.info("Betting conditions not met")

        logger.info("\n📊 Betting simulation completed:")
        logger.info(f"  - Predictions processed: {len(predictions)}")
        logger.info(f"  - Bets placed: {bets_placed}")

        # Get statistics
        stats = betting_simulator.get_statistics()
        logger.info(f"📈 Betting statistics: {stats}")

        # Check database directly
        logger.info("\n🔍 Checking database directly...")
        try:
            from sqlalchemy import create_engine, text

            db_url = "postgresql://mlops_user:mlops_password@localhost:5432/mlops_db"
            engine = create_engine(db_url)

            with engine.connect() as conn:
                bet_count = conn.execute(text("SELECT COUNT(*) FROM bets")).scalar()
                logger.info(f"Total bets in database: {bet_count}")

                if bet_count > 0:
                    recent_bets = conn.execute(
                        text("""
                        SELECT match_id, home_team, away_team, bet_type, bet_amount,
                               prediction_confidence, result, created_at
                        FROM bets
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    ).fetchall()

                    logger.info("Recent bets in database:")
                    for bet in recent_bets:
                        logger.info(f"  {bet}")

        except Exception as e:
            logger.error(f"Failed to check database: {e}")

        logger.info("✅ Betting simulation test completed!")

    except Exception as e:
        logger.error(f"❌ Error in betting simulation test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    test_betting_simulation()
