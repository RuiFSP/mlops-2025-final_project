#!/usr/bin/env python3
"""
Database setup script for the Premier League MLOps system.
Creates all required tables and initializes the database schema.
"""

import os
import logging
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Set up the database schema with all required tables."""
    
    # Database connection
    db_url = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
        f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
    )
    
    engine = create_engine(db_url)
    
    # Table creation SQL statements
    tables = {
        'matches': """
            CREATE TABLE IF NOT EXISTS matches (
                id SERIAL PRIMARY KEY,
                match_id VARCHAR(100) UNIQUE NOT NULL,
                home_team VARCHAR(100) NOT NULL,
                away_team VARCHAR(100) NOT NULL,
                match_date DATE NOT NULL,
                season VARCHAR(20),
                home_odds FLOAT,
                draw_odds FLOAT,
                away_odds FLOAT,
                home_goals INTEGER,
                away_goals INTEGER,
                result VARCHAR(10), -- 'H', 'D', 'A'
                home_shots INTEGER,
                away_shots INTEGER,
                home_shots_target INTEGER,
                away_shots_target INTEGER,
                home_corners INTEGER,
                away_corners INTEGER,
                home_fouls INTEGER,
                away_fouls INTEGER,
                home_yellow_cards INTEGER,
                away_yellow_cards INTEGER,
                home_red_cards INTEGER,
                away_red_cards INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """,
        
        'predictions': """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                match_id VARCHAR(100) NOT NULL,
                home_team VARCHAR(100) NOT NULL,
                away_team VARCHAR(100) NOT NULL,
                prediction VARCHAR(10) NOT NULL, -- 'H', 'D', 'A'
                confidence FLOAT NOT NULL,
                home_win_prob FLOAT,
                draw_prob FLOAT,
                away_win_prob FLOAT,
                home_odds FLOAT,
                draw_odds FLOAT,
                away_odds FLOAT,
                prediction_date TIMESTAMP NOT NULL,
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """,
        
        'bets': """
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
        """,
        
        'wallet': """
            CREATE TABLE IF NOT EXISTS wallet (
                id SERIAL PRIMARY KEY,
                balance FLOAT NOT NULL,
                total_bets FLOAT NOT NULL,
                total_wins FLOAT NOT NULL,
                total_losses FLOAT NOT NULL,
                roi FLOAT NOT NULL,
                last_updated TIMESTAMP DEFAULT NOW()
            )
        """,
        
        'metrics': """
            CREATE TABLE IF NOT EXISTS metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(255) NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                metric_type VARCHAR(50) DEFAULT 'gauge',
                labels JSONB,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """
    }
    
    try:
        with engine.begin() as conn:
            # Create all tables
            for table_name, create_sql in tables.items():
                try:
                    logger.info(f"Creating table: {table_name}")
                    conn.execute(text(create_sql))
                except Exception as e:
                    logger.error(f"❌ Failed to create table {table_name}: {e}\nSQL: {create_sql}")
            
            # Create indexes for better performance
            indexes = [
                ("idx_matches_date", "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)"),
                ("idx_matches_teams", "CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team)"),
                ("idx_predictions_match_id", "CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id)"),
                ("idx_predictions_date", "CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)"),
                ("idx_bets_match_id", "CREATE INDEX IF NOT EXISTS idx_bets_match_id ON bets(match_id)"),
                ("idx_bets_date", "CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(bet_date)")
            ]
            
            for index_name, index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                except Exception as e:
                    logger.error(f"❌ Failed to create index {index_name}: {e}\nSQL: {index_sql}")
            
            # Create metrics indexes only if metrics table exists and has the right columns
            try:
                conn.execute(text("SELECT metric_name FROM metrics LIMIT 1"))
                metrics_indexes = [
                    ("idx_metrics_name_timestamp", "CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(metric_name, timestamp)"),
                    ("idx_metrics_labels", "CREATE INDEX IF NOT EXISTS idx_metrics_labels ON metrics USING GIN(labels)")
                ]
                for index_name, index_sql in metrics_indexes:
                    try:
                        conn.execute(text(index_sql))
                    except Exception as e:
                        logger.error(f"❌ Failed to create metrics index {index_name}: {e}\nSQL: {index_sql}")
            except Exception as e:
                logger.warning(f"Metrics table not found or has different schema, skipping metrics indexes: {e}")
        # End of first transaction

        # Short delay to ensure all tables are committed
        import time
        time.sleep(1)

        # Wallet initialization in a separate transaction
        with engine.begin() as conn:
            wallet_init_sql = """
                INSERT INTO wallet (balance, total_bets, total_wins, total_losses, roi)
                SELECT 1000.0, 0, 0, 0, 0
                WHERE NOT EXISTS (SELECT 1 FROM wallet)
            """
            try:
                conn.execute(text(wallet_init_sql))
                logger.info("✅ Wallet initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to initialize wallet: {e}\nSQL: {wallet_init_sql}")

        logger.info("✅ Database schema setup completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Failed to setup database: {e}")
        raise

if __name__ == "__main__":
    setup_database() 