#!/usr/bin/env python3
import os
import sqlalchemy
from sqlalchemy import create_engine, text

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
)

def debug_predictions():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        # Get latest predictions
        result = conn.execute(text("""
            SELECT home_team, away_team, prediction, confidence, home_win_prob, draw_prob, away_win_prob, 
                   home_odds, draw_odds, away_odds, prediction_date
            FROM predictions 
            ORDER BY prediction_date DESC 
            LIMIT 5
        """)).fetchall()
        
        print("Latest 5 predictions:")
        print("-" * 80)
        for row in result:
            print(f"{row.home_team} vs {row.away_team}")
            print(f"  Prediction: {row.prediction}, Confidence: {row.confidence:.3f}")
            print(f"  Probs: H={row.home_win_prob:.3f}, D={row.draw_prob:.3f}, A={row.away_win_prob:.3f}")
            print(f"  Odds: H={row.home_odds}, D={row.draw_odds}, A={row.away_odds}")
            
            # Calculate margin for the predicted outcome
            if row.prediction == 'H':
                implied_prob = 1 / row.home_odds
                margin = row.home_win_prob - implied_prob
                print(f"  Margin (H): {margin:.3f} (need > 0.01)")
            elif row.prediction == 'D':
                implied_prob = 1 / row.draw_odds
                margin = row.draw_prob - implied_prob
                print(f"  Margin (D): {margin:.3f} (need > 0.01)")
            else:  # A
                implied_prob = 1 / row.away_odds
                margin = row.away_win_prob - implied_prob
                print(f"  Margin (A): {margin:.3f} (need > 0.01)")
            
            print(f"  Meets criteria: Confidence >= 0.3: {row.confidence >= 0.3}, Margin >= 0.01: {margin >= 0.01}")
            print()

if __name__ == "__main__":
    debug_predictions() 