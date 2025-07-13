#!/usr/bin/env python3
"""
Debug script to check bets table schema and diagnose betting issues.
"""

import logging
import os

from sqlalchemy import create_engine, inspect, text

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_bets_table():
    """Check the bets table schema and data."""

    # Database connection - use localhost for testing outside Docker
    db_url = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
        f"localhost:5432/"
        f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
    )

    logger.info(f"Connecting to database: {db_url}")
    engine = create_engine(db_url)

    try:
        with engine.connect() as conn:
            # Check if bets table exists
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            logger.info(f"Available tables: {tables}")

            if "bets" not in tables:
                logger.error("❌ Bets table does not exist!")
                return

            # Get table schema
            columns = inspector.get_columns("bets")
            logger.info("📋 Bets table schema:")
            for col in columns:
                logger.info(f"  {col['name']}: {col['type']} (nullable: {col['nullable']})")

            # Check table constraints
            constraints = inspector.get_unique_constraints("bets")
            logger.info(f"Unique constraints: {constraints}")

            # Check indexes
            indexes = inspector.get_indexes("bets")
            logger.info(f"Indexes: {indexes}")

            # Check row count
            count = conn.execute(text("SELECT COUNT(*) FROM bets")).scalar()
            logger.info(f"📊 Total bets in table: {count}")

            if count > 0:
                # Show recent bets
                recent_bets = conn.execute(
                    text("""
                    SELECT id, match_id, home_team, away_team, bet_type, bet_amount,
                           prediction_confidence, prediction_probability, bet_date, result
                    FROM bets
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                ).fetchall()

                logger.info("📋 Recent bets:")
                for bet in recent_bets:
                    logger.info(f"  {bet}")

            # Check for any errors in the table
            try:
                # Test inserting a simple bet
                test_bet = {
                    "match_id": "TEST_MATCH_001",
                    "home_team": "Test Home",
                    "away_team": "Test Away",
                    "bet_type": "H",
                    "bet_amount": 10.0,
                    "odds": 2.0,
                    "prediction_confidence": 0.5,
                    "prediction_probability": 0.6,
                    "bet_date": "2024-01-01 12:00:00",
                    "result": "P",
                }

                logger.info("🧪 Testing bet insertion...")
                logger.info(f"Test bet data: {test_bet}")

                # Try to insert test bet
                insert_sql = """
                    INSERT INTO bets (
                        match_id, home_team, away_team, bet_type, bet_amount, odds,
                        prediction_confidence, prediction_probability, bet_date, result
                    ) VALUES (
                        :match_id, :home_team, :away_team, :bet_type, :bet_amount, :odds,
                        :prediction_confidence, :prediction_probability, :bet_date, :result
                    )
                """

                conn.execute(text(insert_sql), test_bet)
                conn.commit()
                logger.info("✅ Test bet inserted successfully!")

                # Check if it was actually inserted
                new_count = conn.execute(text("SELECT COUNT(*) FROM bets")).scalar()
                logger.info(f"📊 Bets count after test insert: {new_count}")

                # Clean up test bet
                conn.execute(text("DELETE FROM bets WHERE match_id = 'TEST_MATCH_001'"))
                conn.commit()
                logger.info("🧹 Test bet cleaned up")

            except Exception as e:
                logger.error(f"❌ Failed to insert test bet: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    check_bets_table()
