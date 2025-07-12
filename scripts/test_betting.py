#!/usr/bin/env python3
"""
Test script to directly test the betting simulator.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_betting_simulator():
    """Test the betting simulator directly."""
    try:
        from betting_simulator.simulator import BettingSimulator

        logger.info("Testing betting simulator...")

        # Create betting simulator
        simulator = BettingSimulator(initial_balance=1000.0)

        # Create a test prediction
        test_prediction = {
            "match_id": "test_match_001",
            "home_team": "Test Home",
            "away_team": "Test Away",
            "prediction": "H",
            "confidence": 0.6,
            "home_win_prob": 0.6,
            "draw_prob": 0.2,
            "away_win_prob": 0.2,
            "home_odds": 2.0,
            "draw_odds": 3.5,
            "away_odds": 3.0,
        }

        logger.info(f"Test prediction: {test_prediction}")

        # Try to place a bet
        bet = simulator.place_bet(test_prediction)

        if bet:
            logger.info(f"Bet placed successfully: {bet}")
        else:
            logger.info("No bet placed")

        # Check statistics
        stats = simulator.get_statistics()
        logger.info(f"Statistics: {stats}")

    except Exception as e:
        logger.error(f"Error testing betting simulator: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_betting_simulator()
