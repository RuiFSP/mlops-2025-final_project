#!/usr/bin/env python3

"""
Database setup script for Premier League MLOps
Creates necessary tables and populates with sample data
"""

import os
import sys
import subprocess
from pathlib import Path
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

def setup_database():
    """Set up the database with required tables and sample data"""
    
    logger.info("Setting up database tables...")
    
    # Model metrics table
    success, output = run_sql_command("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            metric_type VARCHAR(50) NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            model_name VARCHAR(255) DEFAULT 'premier_league_predictor',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB
        )
    """)
    if not success:
        logger.error(f"Failed to create model_metrics table: {output}")
        return False
    
    # Predictions table
    success, output = run_sql_command("""
        CREATE TABLE IF NOT EXISTS predictions (
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
        )
    """)
    if not success:
        logger.error(f"Failed to create predictions table: {output}")
        return False
    
    # Bets table
    success, output = run_sql_command("""
        CREATE TABLE IF NOT EXISTS bets (
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
        )
    """)
    if not success:
        logger.error(f"Failed to create bets table: {output}")
        return False
    
    # Wallet table
    success, output = run_sql_command("""
        CREATE TABLE IF NOT EXISTS wallet (
            id SERIAL PRIMARY KEY,
            balance FLOAT NOT NULL,
            last_updated TIMESTAMP DEFAULT NOW()
        )
    """)
    if not success:
        logger.error(f"Failed to create wallet table: {output}")
        return False
    
    # Check if tables are empty and insert sample data
    success, output = run_sql_command("SELECT COUNT(*) FROM model_metrics")
    if success and "0" in output:
        logger.info("Adding sample metrics data...")
        run_sql_command("""
            INSERT INTO model_metrics (metric_type, value, model_name) 
            VALUES 
                ('accuracy', 0.75, 'premier_league_predictor'),
                ('precision', 0.72, 'premier_league_predictor'),
                ('recall', 0.68, 'premier_league_predictor'),
                ('f1_score', 0.70, 'premier_league_predictor')
        """)
    
    success, output = run_sql_command("SELECT COUNT(*) FROM predictions")
    if success and "0" in output:
        logger.info("Adding sample predictions data...")
        run_sql_command("""
            INSERT INTO predictions (
                match_id, home_team, away_team, prediction, confidence, 
                home_win_prob, draw_prob, away_win_prob
            ) VALUES 
                ('ARS-CHE-20250714', 'Arsenal', 'Chelsea', 'H', 0.72, 0.72, 0.18, 0.10),
                ('LIV-MCI-20250714', 'Liverpool', 'Man City', 'D', 0.45, 0.30, 0.45, 0.25),
                ('MUN-TOT-20250714', 'Man United', 'Tottenham', 'A', 0.55, 0.25, 0.20, 0.55)
        """)
    
    success, output = run_sql_command("SELECT COUNT(*) FROM bets")
    if success and "0" in output:
        logger.info("Adding sample betting data...")
        run_sql_command("""
            INSERT INTO bets (
                match_id, home_team, away_team, bet_type, bet_amount, 
                odds, prediction_confidence, prediction_probability, 
                bet_date, result, payout, roi
            ) VALUES 
                ('ARS-CHE-20250714', 'Arsenal', 'Chelsea', 'H', 100.0, 2.1, 0.72, 0.72, NOW(), 'W', 210.0, 1.1),
                ('LIV-MCI-20250714', 'Liverpool', 'Man City', 'D', 50.0, 3.5, 0.45, 0.45, NOW(), 'L', 0.0, -1.0),
                ('MUN-TOT-20250714', 'Man United', 'Tottenham', 'A', 75.0, 2.8, 0.55, 0.55, NOW(), 'W', 210.0, 1.8)
        """)
    
    success, output = run_sql_command("SELECT COUNT(*) FROM wallet")
    if success and "0" in output:
        logger.info("Adding wallet data...")
        run_sql_command("INSERT INTO wallet (balance) VALUES (1225.0)")
    
    logger.info("✅ Database setup completed successfully")
    return True

if __name__ == "__main__":
    setup_database()
