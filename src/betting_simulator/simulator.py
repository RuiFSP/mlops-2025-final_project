"""
Betting simulator for managing bets and wallet balance.
"""

import logging
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class BettingSimulator:
    """Simulates betting decisions and tracks wallet balance."""

    def __init__(self, initial_balance: float = 1000.0):
        """Initialize the betting simulator."""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Set up database connection
        self.db_url = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
            f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
        )
        logger.info(f"[DEBUG] Betting simulator connecting to: {self.db_url}")
        self.engine = create_engine(self.db_url)

        # Test connection and show current database
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT current_database()")).scalar()
                logger.info(f"[DEBUG] Connected to database: {result}")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to get current database: {e}")

        # Betting parameters - Production thresholds
        self.min_confidence = 0.6  # Production threshold for confidence
        self.min_margin = 0.1  # Production threshold for margin
        self.max_bet_percentage = 0.05  # 5% of balance per bet

        # Initialize tables
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables."""
        try:
            # Create bets table
            bets_table_sql = """
                CREATE TABLE IF NOT EXISTS bets (
                    id SERIAL PRIMARY KEY,
                    match_id VARCHAR(100) NOT NULL,
                    home_team VARCHAR(100) NOT NULL,
                    away_team VARCHAR(100) NOT NULL,
                    bet_type VARCHAR(10) NOT NULL, -- 'H', 'D', 'A'
                    bet_amount FLOAT NOT NULL,
                    odds FLOAT NOT NULL,
                    prediction_confidence FLOAT NOT NULL,
                    prediction_probability FLOAT NOT NULL,
                    bet_date TIMESTAMP NOT NULL,
                    result VARCHAR(10), -- 'W', 'L', 'P' (pending)
                    payout FLOAT,
                    roi FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """

            # Create wallet table
            wallet_table_sql = """
                CREATE TABLE IF NOT EXISTS wallet (
                    id SERIAL PRIMARY KEY,
                    balance FLOAT NOT NULL,
                    total_bets FLOAT NOT NULL,
                    total_wins FLOAT NOT NULL,
                    total_losses FLOAT NOT NULL,
                    roi FLOAT NOT NULL,
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """

            with self.engine.connect() as conn:
                conn.execute(text(bets_table_sql))
                conn.execute(text(wallet_table_sql))
                conn.commit()
                logger.info("[DEBUG] Tables checked/created successfully.")

            # Initialize wallet if empty
            self._init_wallet()

        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")

    def _init_wallet(self):
        """Initialize wallet with starting balance."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM wallet")).scalar()

                if result == 0:
                    conn.execute(
                        text("""
                        INSERT INTO wallet (balance, total_bets, total_wins, total_losses, roi)
                        VALUES (:balance, 0, 0, 0, 0)
                    """),
                        {"balance": self.initial_balance},
                    )
                    conn.commit()
                    logger.info(f"Initialized wallet with balance: £{self.initial_balance}")
                else:
                    # Get current balance
                    result = conn.execute(
                        text("SELECT balance FROM wallet ORDER BY id DESC LIMIT 1")
                    ).scalar()
                    self.current_balance = result or self.initial_balance

        except Exception as e:
            logger.error(f"Failed to initialize wallet: {e}")

    def should_place_bet(self, prediction: dict[str, Any]) -> bool:
        """Determine if we should place a bet based on prediction confidence and odds."""
        confidence = prediction["confidence"]
        prediction_prob = prediction.get("home_win_prob", 0)

        # Get the relevant odds based on prediction
        if prediction["prediction"] == "H":
            odds = prediction["home_odds"]
        elif prediction["prediction"] == "D":
            odds = prediction["draw_odds"]
        else:  # 'A'
            odds = prediction["away_odds"]

        # Calculate implied probability from odds
        implied_prob = 1 / odds

        # Check if our prediction is significantly better than market odds
        margin = prediction_prob - implied_prob

        return confidence >= self.min_confidence and margin >= self.min_margin

    def calculate_bet_amount(self, prediction: dict[str, Any]) -> float:
        """Calculate bet amount based on Kelly Criterion and risk management."""
        confidence = prediction["confidence"]
        prediction_prob = prediction.get("home_win_prob", 0)

        # Get the relevant odds
        if prediction["prediction"] == "H":
            odds = prediction["home_odds"]
        elif prediction["prediction"] == "D":
            odds = prediction["draw_odds"]
        else:  # 'A'
            odds = prediction["away_odds"]

        # Kelly Criterion: f = (bp - q) / b
        # where b = odds - 1, p = our probability, q = 1 - p
        b = odds - 1
        p = prediction_prob
        q = 1 - p
        kelly_fraction = (b * p - q) / b if b > 0 else 0
        # Ensure kelly_fraction is positive
        kelly_fraction = max(kelly_fraction, 0.0)
        # Apply risk management: cap at max_bet_percentage
        kelly_fraction = min(kelly_fraction, self.max_bet_percentage)
        # Only bet if Kelly fraction is positive
        if kelly_fraction > 0:
            bet_amount = self.current_balance * kelly_fraction
            return round(bet_amount, 2)
        return 0.0

    def _to_native(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif hasattr(obj, "item") and callable(obj.item):
            try:
                return obj.item()
            except Exception:
                return obj
        else:
            return obj

    def place_bet(self, prediction: dict[str, Any]) -> dict[str, Any] | None:
        """Place a bet if conditions are met."""
        if not self.should_place_bet(prediction):
            return None
        bet_amount = self.calculate_bet_amount(prediction)
        if bet_amount <= 0:
            return None
        # Get the relevant odds
        if prediction["prediction"] == "H":
            odds = prediction["home_odds"]
        elif prediction["prediction"] == "D":
            odds = prediction["draw_odds"]
        else:  # 'A'
            odds = prediction["away_odds"]
        bet = {
            "match_id": prediction["match_id"],
            "home_team": prediction["home_team"],
            "away_team": prediction["away_team"],
            "bet_type": prediction["prediction"],
            "bet_amount": bet_amount,
            "odds": odds,
            "prediction_confidence": prediction["confidence"],
            "prediction_probability": prediction.get("home_win_prob", 0),
            "bet_date": datetime.now(),
            "result": "P",  # Pending
            "payout": None,
            "roi": None,
        }
        # Debug: print types before conversion
        print("[DEBUG] Bet dict types before conversion:")
        for k, v in bet.items():
            print(f"  {k}: {type(v)} -> {v}")
        # Convert all values to native types
        bet = self._to_native(bet)
        # Explicitly cast critical fields to float
        bet["prediction_confidence"] = float(bet["prediction_confidence"])
        bet["prediction_probability"] = float(bet["prediction_probability"])
        # Debug: print bet after conversion
        print("[DEBUG] Bet dict after conversion:")
        for k, v in bet.items():
            print(f"  {k}: {type(v)} -> {v}")
        # Save bet to database
        self._save_bet(bet)
        # Update wallet
        self.current_balance -= bet_amount
        self._update_wallet()
        logger.info(
            f"Placed bet: £{bet_amount} on {prediction['home_team']} vs {prediction['away_team']} - {prediction['prediction']} @ {odds}"
        )
        return bet

    def _save_bet(self, bet: dict[str, Any]):
        """Save bet to database."""
        try:
            logger.info(f"[DEBUG] Attempting to save bet: {bet}")

            # Explicitly cast all float fields
            for key in ["bet_amount", "odds", "prediction_confidence", "prediction_probability"]:
                if key in bet:
                    bet[key] = float(bet[key])

            logger.info(f"[DEBUG] Bet after explicit float cast: {bet}")
            for k, v in bet.items():
                logger.info(f"[DEBUG] {k}: {type(v)} -> {v}")

            with self.engine.connect() as conn:
                sql = """
                    INSERT INTO bets (
                        match_id, home_team, away_team, bet_type, bet_amount, odds,
                        prediction_confidence, prediction_probability, bet_date, result
                    ) VALUES (
                        :match_id, :home_team, :away_team, :bet_type, :bet_amount, :odds,
                        :prediction_confidence, :prediction_probability, :bet_date, :result
                    )
                """
                logger.info(f"[DEBUG] Executing SQL: {sql}")
                logger.info(f"[DEBUG] Parameters: {bet}")

                # Execute the insert
                result = conn.execute(text(sql), bet)
                logger.info(f"[DEBUG] Insert result: {result}")

                # Commit the transaction
                conn.commit()
                logger.info("[DEBUG] Transaction committed successfully!")

                # Verify the bet was actually inserted
                verify_sql = "SELECT COUNT(*) FROM bets WHERE match_id = :match_id"
                count = conn.execute(text(verify_sql), {"match_id": bet["match_id"]}).scalar()
                logger.info(
                    f"[DEBUG] Verification: Found {count} bets with match_id {bet['match_id']}"
                )

        except Exception as e:
            logger.error(f"Failed to save bet: {e}")
            logger.error(f"[DEBUG] Bet data that failed: {bet}")
            import traceback

            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            # Don't re-raise the exception to avoid stopping the pipeline

    def _update_wallet(self):
        """Update wallet balance in database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                    UPDATE wallet
                    SET balance = :balance, last_updated = NOW()
                    WHERE id = (SELECT id FROM wallet ORDER BY id DESC LIMIT 1)
                """),
                    {"balance": self.current_balance},
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to update wallet: {e}")

    def process_results(self, match_results: list[dict[str, Any]]):
        """Process match results and update bet outcomes."""
        logger.info("Processing match results...")

        for result in match_results:
            match_id = result["match_id"]
            actual_result = result["result"]  # 'H', 'D', 'A'

            # Get pending bets for this match
            try:
                with self.engine.connect() as conn:
                    bets = conn.execute(
                        text("""
                        SELECT * FROM bets
                        WHERE match_id = :match_id AND result = 'P'
                    """),
                        {"match_id": match_id},
                    ).fetchall()

                    for bet in bets:
                        # Determine if bet won
                        bet_won = bet.bet_type == actual_result

                        if bet_won:
                            payout = bet.bet_amount * bet.odds
                            roi = (payout - bet.bet_amount) / bet.bet_amount
                            self.current_balance += payout
                        else:
                            payout = 0
                            roi = -1.0

                        # Update bet record
                        conn.execute(
                            text("""
                            UPDATE bets
                            SET result = :result, payout = :payout, roi = :roi
                            WHERE id = :bet_id
                        """),
                            {
                                "result": "W" if bet_won else "L",
                                "payout": payout,
                                "roi": roi,
                                "bet_id": bet.id,
                            },
                        )

                        logger.info(
                            f"Bet result: {bet.home_team} vs {bet.away_team} - {bet.bet_type} -> {'WON' if bet_won else 'LOST'} (ROI: {roi:.2%})"
                        )

                    conn.commit()

            except Exception as e:
                logger.error(f"Failed to process results for match {match_id}: {e}")

        # Update wallet
        self._update_wallet()
        logger.info(f"Updated wallet balance: £{self.current_balance:.2f}")

    def get_statistics(self) -> dict[str, Any]:
        """Get betting statistics."""
        try:
            with self.engine.connect() as conn:
                # Get total bets
                total_bets = conn.execute(text("SELECT COUNT(*) FROM bets")).scalar()

                # Get winning bets
                winning_bets = conn.execute(
                    text("SELECT COUNT(*) FROM bets WHERE result = 'W'")
                ).scalar()

                # Get total amount bet
                total_amount = conn.execute(
                    text("SELECT COALESCE(SUM(bet_amount), 0) FROM bets")
                ).scalar()

                # Get total winnings
                total_winnings = conn.execute(
                    text("SELECT COALESCE(SUM(payout), 0) FROM bets WHERE result = 'W'")
                ).scalar()

                # Calculate overall ROI
                overall_roi = (
                    (total_winnings - total_amount) / total_amount if total_amount > 0 else 0
                )

                # Get win rate
                win_rate = winning_bets / total_bets if total_bets > 0 else 0

                return {
                    "total_bets": total_bets,
                    "winning_bets": winning_bets,
                    "win_rate": win_rate,
                    "total_amount_bet": total_amount,
                    "total_winnings": total_winnings,
                    "overall_roi": overall_roi,
                    "current_balance": self.current_balance,
                    "profit_loss": self.current_balance - self.initial_balance,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test betting simulator
    simulator = BettingSimulator(initial_balance=1000.0)
    stats = simulator.get_statistics()
    print(f"Betting statistics: {stats}")
