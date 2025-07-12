"""
Prediction pipeline for daily Premier League match predictions.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

import mlflow
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Prediction pipeline for daily match predictions."""
    
    def __init__(self):
        """Initialize the prediction pipeline."""
        # Set up MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        
        # Set up database connection
        self.db_url = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'mlops_user')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'mlops_password')}@"
            f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'mlops_db')}"
        )
        self.engine = create_engine(self.db_url)
        
        # Load the latest model
        self.model = self._load_latest_model()
        
    def _load_latest_model(self):
        """Load the latest model from MLflow."""
        try:
            # Get the latest model version
            model_name = "premier_league_predictor"
            model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def get_todays_matches(self) -> pd.DataFrame:
        """Get today's matches from the database or generate simulated data."""
        logger.info("Getting today's matches...")
        
        try:
            # Try to get from database first
            query = """
                SELECT * FROM matches 
                WHERE match_date = CURRENT_DATE
                AND home_odds IS NOT NULL
            """
            
            df = pd.read_sql(query, self.engine)
            
            if len(df) > 0:
                logger.info(f"Found {len(df)} matches for today")
                return df
                
        except Exception as e:
            logger.warning(f"Could not get matches from database: {e}")
        
        # Generate simulated matches if none found
        logger.info("Generating simulated matches for today")
        return self._generate_simulated_matches()
    
    def _generate_simulated_matches(self) -> pd.DataFrame:
        """Generate simulated matches for today."""
        teams = [
            "Arsenal", "Chelsea", "Manchester United", "Liverpool", "Manchester City",
            "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham"
        ]
        
        import random
        
        matches = []
        for i in range(5):  # 5 matches today
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            
            # Generate realistic odds
            home_odds = round(random.uniform(1.5, 4.0), 2)
            draw_odds = round(random.uniform(2.5, 4.5), 2)
            away_odds = round(random.uniform(1.8, 5.0), 2)
            
            match = {
                'match_id': f"match_{datetime.now().strftime('%Y%m%d')}_{i}",
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                'match_date': datetime.now().date(),
                'season': '2024/25'
            }
            matches.append(match)
        
        return pd.DataFrame(matches)
    
    def prepare_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        logger.info("Preparing features for prediction...")
        
        # Map the simulated data to the features used in training
        # The model was trained with: B365H, B365D, B365A, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR
        features_df = pd.DataFrame()
        
        # Map odds (these are the same)
        features_df['B365H'] = matches['home_odds']
        features_df['B365D'] = matches['draw_odds']
        features_df['B365A'] = matches['away_odds']
        
        # Generate simulated match statistics (since we don't have real data for today's matches)
        import random
        for i in range(len(matches)):
            # Shots
            features_df.loc[i, 'HS'] = random.randint(8, 20)  # Home shots
            features_df.loc[i, 'AS'] = random.randint(6, 18)  # Away shots
            
            # Shots on target
            features_df.loc[i, 'HST'] = random.randint(3, 8)  # Home shots on target
            features_df.loc[i, 'AST'] = random.randint(2, 7)  # Away shots on target
            
            # Corners
            features_df.loc[i, 'HC'] = random.randint(3, 12)  # Home corners
            features_df.loc[i, 'AC'] = random.randint(2, 10)  # Away corners
            
            # Fouls
            features_df.loc[i, 'HF'] = random.randint(8, 16)  # Home fouls
            features_df.loc[i, 'AF'] = random.randint(8, 16)  # Away fouls
            
            # Yellow cards
            features_df.loc[i, 'HY'] = random.randint(0, 4)  # Home yellow cards
            features_df.loc[i, 'AY'] = random.randint(0, 4)  # Away yellow cards
            
            # Red cards
            features_df.loc[i, 'HR'] = random.randint(0, 1)  # Home red cards
            features_df.loc[i, 'AR'] = random.randint(0, 1)  # Away red cards
        
        return features_df
    
    def make_predictions(self, matches: pd.DataFrame) -> List[Dict[str, Any]]:
        """Make predictions for today's matches."""
        logger.info("Making predictions...")
        
        if self.model is None:
            logger.error("No model available for predictions")
            return []
        
        # Prepare features
        features = self.prepare_features(matches)
        
        # Make predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Format results
        results = []
        for i, (_, match) in enumerate(matches.iterrows()):
            prediction = predictions[i]
            proba = probabilities[i]
            
            result = {
                'match_id': match['match_id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'prediction': prediction,
                'confidence': max(proba),
                'home_win_prob': proba[0] if len(proba) > 0 else 0,
                'draw_prob': proba[1] if len(proba) > 1 else 0,
                'away_win_prob': proba[2] if len(proba) > 2 else 0,
                'home_odds': match['home_odds'],
                'draw_odds': match['draw_odds'],
                'away_odds': match['away_odds'],
                'prediction_date': datetime.now(),
                'model_version': 'latest'
            }
            results.append(result)
            
            logger.info(f"Prediction: {match['home_team']} vs {match['away_team']} -> {prediction} (confidence: {max(proba):.3f})")
        
        return results
    
    def save_predictions(self, predictions: List[Dict[str, Any]]):
        """Save predictions to the database."""
        logger.info("Saving predictions to database...")
        
        try:
            # Create predictions table if it doesn't exist
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    match_id VARCHAR(100) NOT NULL,
                    home_team VARCHAR(100) NOT NULL,
                    away_team VARCHAR(100) NOT NULL,
                    prediction VARCHAR(10) NOT NULL,
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
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            # Insert predictions
            df = pd.DataFrame(predictions)
            df.to_sql('predictions', self.engine, if_exists='append', index=False)
            
            logger.info(f"Saved {len(predictions)} predictions to database")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    
    def run_prediction(self) -> List[Dict[str, Any]]:
        """Run the complete prediction pipeline."""
        logger.info("Starting prediction pipeline...")
        
        # Get today's matches
        matches = self.get_todays_matches()
        
        if len(matches) == 0:
            logger.warning("No matches found for today")
            return []
        
        # Make predictions
        predictions = self.make_predictions(matches)
        
        # Save predictions
        self.save_predictions(predictions)
        
        logger.info(f"Prediction pipeline completed. Generated {len(predictions)} predictions")
        return predictions


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run prediction pipeline
    pipeline = PredictionPipeline()
    predictions = pipeline.run_prediction()
    print(f"Generated {len(predictions)} predictions") 