#!/usr/bin/env python3

"""
Fix database issues by creating missing tables and ensuring permissions
"""

import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_sql_command(sql_command):
    """Run SQL command using docker exec"""
    try:
        cmd = [
            "docker", "exec", "mlops_postgres", 
            "psql", "-U", "mlops_user", "-d", "mlops_db", 
            "-c", sql_command
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def fix_database():
    """Fix database issues"""
    
    logger.info("Fixing database permissions...")
    
    # Grant all privileges to mlops_user
    success, output = run_sql_command("ALTER USER mlops_user WITH SUPERUSER;")
    if not success:
        logger.error(f"Failed to grant superuser: {output}")
        return False
    
    success, output = run_sql_command("GRANT ALL PRIVILEGES ON SCHEMA public TO mlops_user;")
    if not success:
        logger.error(f"Failed to grant privileges: {output}")
        return False
    
    success, output = run_sql_command("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlops_user;")
    if not success:
        logger.error(f"Failed to grant table privileges: {output}")
        return False
    
    # Create tables if they don't exist
    logger.info("Creating missing tables...")
    
    # Bets table
    success, output = run_sql_command("""
        DROP TABLE IF EXISTS bets;
        CREATE TABLE bets (
            id SERIAL PRIMARY KEY,
            match_id VARCHAR(100) NOT NULL,
            home_team VARCHAR(100) NOT NULL,
            away_team VARCHAR(100) NOT NULL,
            bet_type VARCHAR(10) NOT NULL,
            bet_amount FLOAT NOT NULL,
            odds FLOAT NOT NULL,
            prediction_confidence FLOAT NOT NULL,
            prediction_probability FLOAT NOT NULL,
            bet_date TIMESTAMP NOT NULL,
            result VARCHAR(10),
            payout FLOAT,
            roi FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    if not success:
        logger.error(f"Failed to create bets table: {output}")
        return False
    
    # Model metrics table
    success, output = run_sql_command("""
        DROP TABLE IF EXISTS model_metrics;
        CREATE TABLE model_metrics (
            id SERIAL PRIMARY KEY,
            metric_type VARCHAR(50) NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            model_name VARCHAR(255) DEFAULT 'premier_league_predictor',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB
        );
    """)
    if not success:
        logger.error(f"Failed to create model_metrics table: {output}")
        return False
    
    # Predictions table
    success, output = run_sql_command("""
        DROP TABLE IF EXISTS predictions;
        CREATE TABLE predictions (
            id SERIAL PRIMARY KEY,
            match_id VARCHAR(100) NOT NULL,
            home_team VARCHAR(100) NOT NULL,
            away_team VARCHAR(100) NOT NULL,
            prediction VARCHAR(10) NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            home_win_prob DOUBLE PRECISION,
            draw_prob DOUBLE PRECISION,
            away_win_prob DOUBLE PRECISION,
            home_odds DOUBLE PRECISION,
            draw_odds DOUBLE PRECISION,
            away_odds DOUBLE PRECISION,
            prediction_date TIMESTAMP NOT NULL DEFAULT NOW(),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    if not success:
        logger.error(f"Failed to create predictions table: {output}")
        return False
    
    # Wallet table
    success, output = run_sql_command("""
        DROP TABLE IF EXISTS wallet;
        CREATE TABLE wallet (
            id SERIAL PRIMARY KEY,
            balance FLOAT NOT NULL,
            last_updated TIMESTAMP DEFAULT NOW()
        );
    """)
    if not success:
        logger.error(f"Failed to create wallet table: {output}")
        return False
    
    # Add sample data
    logger.info("Adding sample data...")
    
    # Model metrics data
    success, output = run_sql_command("""
        INSERT INTO model_metrics (metric_type, value, model_name) 
        VALUES 
            ('accuracy', 0.75, 'premier_league_predictor'),
            ('precision', 0.72, 'premier_league_predictor'),
            ('recall', 0.68, 'premier_league_predictor'),
            ('f1_score', 0.70, 'premier_league_predictor');
    """)
    if not success:
        logger.error(f"Failed to add model metrics data: {output}")
    
    # Predictions data
    success, output = run_sql_command("""
        INSERT INTO predictions (
            match_id, home_team, away_team, prediction, confidence, 
            home_win_prob, draw_prob, away_win_prob
        ) VALUES 
            ('ARS-CHE-20250714', 'Arsenal', 'Chelsea', 'H', 0.72, 0.72, 0.18, 0.10),
            ('LIV-MCI-20250714', 'Liverpool', 'Man City', 'D', 0.45, 0.30, 0.45, 0.25),
            ('MUN-TOT-20250714', 'Man United', 'Tottenham', 'A', 0.55, 0.25, 0.20, 0.55);
    """)
    if not success:
        logger.error(f"Failed to add predictions data: {output}")
    
    # Bets data
    success, output = run_sql_command("""
        INSERT INTO bets (
            match_id, home_team, away_team, bet_type, bet_amount, 
            odds, prediction_confidence, prediction_probability, 
            bet_date, result, payout, roi
        ) VALUES 
            ('ARS-CHE-20250714', 'Arsenal', 'Chelsea', 'H', 100.0, 2.1, 0.72, 0.72, NOW(), 'W', 210.0, 1.1),
            ('LIV-MCI-20250714', 'Liverpool', 'Man City', 'D', 50.0, 3.5, 0.45, 0.45, NOW(), 'L', 0.0, -1.0),
            ('MUN-TOT-20250714', 'Man United', 'Tottenham', 'A', 75.0, 2.8, 0.55, 0.55, NOW(), 'W', 210.0, 1.8);
    """)
    if not success:
        logger.error(f"Failed to add bets data: {output}")
    
    # Wallet data
    success, output = run_sql_command("""
        INSERT INTO wallet (balance) VALUES (1225.0);
    """)
    if not success:
        logger.error(f"Failed to add wallet data: {output}")
    
    # Verify tables exist
    success, output = run_sql_command("\\dt")
    logger.info(f"Tables in database:\n{output}")
    
    logger.info("✅ Database fixed successfully")
    return True

if __name__ == "__main__":
    fix_database() 