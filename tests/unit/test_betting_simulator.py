"""
Unit tests for BettingSimulator class.
"""

from unittest.mock import patch

import numpy as np

from src.betting_simulator.simulator import BettingSimulator


class TestBettingSimulator:
    """Test cases for BettingSimulator class."""

    def test_init_with_default_balance(self):
        """Test initialization with default balance."""
        with (
            patch("src.betting_simulator.simulator.create_engine"),
            patch.object(BettingSimulator, "_init_tables"),
            patch.object(BettingSimulator, "_init_wallet"),
        ):
            simulator = BettingSimulator()
            assert simulator.initial_balance == 1000.0

    def test_init_with_custom_balance(self):
        """Test initialization with custom balance."""
        with (
            patch("src.betting_simulator.simulator.create_engine"),
            patch.object(BettingSimulator, "_init_tables"),
            patch.object(BettingSimulator, "_init_wallet"),
        ):
            simulator = BettingSimulator(initial_balance=500.0)
            assert simulator.initial_balance == 500.0

    def test_calculate_bet_amount_with_good_prediction(self):
        """Test bet amount calculation with good prediction."""
        with (
            patch("src.betting_simulator.simulator.create_engine"),
            patch.object(BettingSimulator, "_init_tables"),
            patch.object(BettingSimulator, "_init_wallet"),
        ):
            simulator = BettingSimulator(initial_balance=1000.0)

            prediction = {
                "prediction": "H",
                "confidence": 0.7,
                "home_win_prob": 0.7,
                "home_odds": 2.0,
                "draw_odds": 3.0,
                "away_odds": 4.0,
            }

            bet_amount = simulator.calculate_bet_amount(prediction)
            assert bet_amount > 0
            assert bet_amount <= 1000.0 * simulator.max_bet_percentage

    def test_calculate_bet_amount_with_poor_prediction(self):
        """Test bet amount calculation with poor prediction."""
        with patch("src.betting_simulator.simulator.create_engine"):
            simulator = BettingSimulator(initial_balance=1000.0)

            prediction = {
                "prediction": "H",
                "confidence": 0.4,
                "home_win_prob": 0.4,
                "home_odds": 1.5,
                "draw_odds": 3.0,
                "away_odds": 4.0,
            }

            bet_amount = simulator.calculate_bet_amount(prediction)
            assert bet_amount == 0.0

    def test_to_native_conversion(self):
        """Test conversion of numpy types to native Python types."""
        with patch("src.betting_simulator.simulator.create_engine"):
            simulator = BettingSimulator()

            # Test numpy scalar conversion
            numpy_float = np.float64(3.14)
            native_value = simulator._to_native(numpy_float)
            assert isinstance(native_value, float)
            assert native_value == 3.14

            # Test dict with numpy values
            numpy_dict = {"accuracy": np.float64(0.85), "count": np.int64(42)}
            native_dict = simulator._to_native(numpy_dict)
            assert isinstance(native_dict["accuracy"], float)
            assert isinstance(native_dict["count"], int)

    def test_betting_parameters(self):
        """Test betting parameter defaults."""
        with patch("src.betting_simulator.simulator.create_engine"):
            simulator = BettingSimulator()

            assert simulator.min_confidence == 0.6
            assert simulator.min_margin == 0.1
            assert simulator.max_bet_percentage == 0.05

    def test_should_place_bet_logic(self):
        """Test the should_place_bet method logic."""
        with patch("src.betting_simulator.simulator.create_engine"):
            simulator = BettingSimulator()

            # Good prediction should return True
            good_prediction = {
                "confidence": 0.7,
                "home_win_prob": 0.7,
                "draw_prob": 0.2,
                "away_win_prob": 0.1,
                "home_odds": 2.0,
                "draw_odds": 3.5,
                "away_odds": 4.0,
                "prediction": "H",
            }

            should_bet = simulator.should_place_bet(good_prediction)
            assert should_bet is True

            # Poor confidence should return False
            poor_prediction = {
                "confidence": 0.4,
                "home_win_prob": 0.4,
                "draw_prob": 0.3,
                "away_win_prob": 0.3,
                "home_odds": 2.0,
                "draw_odds": 3.0,
                "away_odds": 4.0,
                "prediction": "H",
            }

            should_bet = simulator.should_place_bet(poor_prediction)
            assert should_bet is False
