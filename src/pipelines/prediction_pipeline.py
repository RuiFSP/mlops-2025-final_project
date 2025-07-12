"""
Prediction pipeline for daily Premier League match predictions.
"""

import logging
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Prediction pipeline for daily match predictions."""

    def __init__(self):
        """Initialize the prediction pipeline."""
        # Set up MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

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
        self.model_info = self._get_model_metadata()

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

    def _get_model_metadata(self) -> dict[str, Any]:
        """Get model metadata from MLflow."""
        try:
            model_name = "premier_league_predictor"
            client = mlflow.MlflowClient()

            # Get the latest model version
            latest_version = client.get_latest_versions(
                model_name, stages=["Production", "Staging", "None"]
            )

            if latest_version:
                version = latest_version[0]
                # Get run info
                run = client.get_run(version.run_id)

                return {
                    "model_name": model_name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "accuracy": run.data.metrics.get("accuracy", 0.0),
                    "f1_score": run.data.metrics.get("f1_score", 0.0),
                    "precision": run.data.metrics.get("precision", 0.0),
                    "recall": run.data.metrics.get("recall", 0.0),
                    "created_at": version.creation_timestamp,
                    "run_id": version.run_id,
                }
            else:
                return {
                    "model_name": model_name,
                    "version": "unknown",
                    "stage": "unknown",
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }

        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            return {
                "model_name": "premier_league_predictor",
                "version": "unknown",
                "stage": "unknown",
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "error": str(e),
            }

    def get_model_info(self) -> dict[str, Any]:
        """Get model information for monitoring."""
        return self.model_info

    def get_todays_matches(self) -> pd.DataFrame:
        """Get today's matches from real data or database."""
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

        # Get real upcoming matches
        logger.info("Fetching real upcoming matches...")
        return self._get_real_upcoming_matches()

    def _get_real_upcoming_matches(self) -> pd.DataFrame:
        """Get real upcoming matches using the data fetcher."""
        try:
            # Import here to avoid circular imports
            from src.data_integration.real_data_fetcher import RealDataFetcher

            fetcher = RealDataFetcher()
            matches = fetcher.get_upcoming_matches(days_ahead=7)

            if matches:
                logger.info(f"Successfully fetched {len(matches)} real upcoming matches")
                return pd.DataFrame(matches)
            else:
                logger.warning("No real matches found, using fallback")
                return self._generate_fallback_matches()

        except Exception as e:
            logger.error(f"Error fetching real matches: {e}")
            return self._generate_fallback_matches()

    def _generate_fallback_matches(self) -> pd.DataFrame:
        """Generate fallback matches when real data is unavailable."""
        logger.info("Generating fallback matches...")

        # Use realistic Premier League teams
        teams = [
            "Arsenal",
            "Manchester City",
            "Liverpool",
            "Chelsea",
            "Manchester United",
            "Newcastle",
            "Tottenham",
            "Brighton",
            "Aston Villa",
            "West Ham",
        ]

        import random

        matches = []
        today = datetime.now()

        for i in range(3):  # Generate 3 realistic matches
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])

            # Generate realistic odds
            home_odds = round(random.uniform(1.5, 4.0), 2)
            draw_odds = round(random.uniform(2.5, 4.5), 2)
            away_odds = round(random.uniform(1.8, 5.0), 2)

            match = {
                "match_id": f"fallback_{today.strftime('%Y%m%d')}_{i}",
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "match_date": today.date(),
                "season": "2024/25",
            }
            matches.append(match)

        logger.info(f"Generated {len(matches)} fallback matches")
        return pd.DataFrame(matches)

    def prepare_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        logger.info("Preparing features for prediction...")

        # Map the simulated data to the features used in training
        # The model was trained with: B365H, B365D, B365A, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR
        features_df = pd.DataFrame()

        # Map odds (these are the same)
        features_df["B365H"] = matches["home_odds"]
        features_df["B365D"] = matches["draw_odds"]
        features_df["B365A"] = matches["away_odds"]

        # Generate simulated match statistics (since we don't have real data for today's matches)
        import random

        for i in range(len(matches)):
            # Shots
            features_df.loc[i, "HS"] = random.randint(8, 20)  # Home shots
            features_df.loc[i, "AS"] = random.randint(6, 18)  # Away shots

            # Shots on target
            features_df.loc[i, "HST"] = random.randint(3, 8)  # Home shots on target
            features_df.loc[i, "AST"] = random.randint(2, 7)  # Away shots on target

            # Corners
            features_df.loc[i, "HC"] = random.randint(3, 12)  # Home corners
            features_df.loc[i, "AC"] = random.randint(2, 10)  # Away corners

            # Fouls
            features_df.loc[i, "HF"] = random.randint(8, 16)  # Home fouls
            features_df.loc[i, "AF"] = random.randint(8, 16)  # Away fouls

            # Yellow cards
            features_df.loc[i, "HY"] = random.randint(0, 4)  # Home yellow cards
            features_df.loc[i, "AY"] = random.randint(0, 4)  # Away yellow cards

            # Red cards
            features_df.loc[i, "HR"] = random.randint(0, 1)  # Home red cards
            features_df.loc[i, "AR"] = random.randint(0, 1)  # Away red cards

        return features_df

    def _to_native(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        home_odds: float = None,
        away_odds: float = None,
        draw_odds: float = None,
    ) -> dict[str, Any]:
        """Predict a single match outcome."""
        logger.info(f"Predicting match: {home_team} vs {away_team}")

        # Create match data
        match_data = {
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": home_odds or 2.0,  # Default odds if not provided
            "away_odds": away_odds or 2.0,
            "draw_odds": draw_odds or 3.0,
        }

        # Convert to DataFrame
        matches_df = pd.DataFrame([match_data])

        # Prepare features
        features = self.prepare_features(matches_df)

        # Make prediction
        if self.model is not None:
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            # Map prediction to outcome
            outcome_map = {0: "A", 1: "D", 2: "H"}  # Away, Draw, Home
            predicted_outcome = outcome_map[prediction]

            # Calculate confidence (max probability)
            confidence = float(max(probabilities))

            # Create probability dictionary
            prob_dict = {
                "H": float(probabilities[2]),
                "D": float(probabilities[1]),
                "A": float(probabilities[0]),
            }

            return {
                "home_team": home_team,
                "away_team": away_team,
                "prediction": predicted_outcome,
                "confidence": confidence,
                "probabilities": prob_dict,
                "created_at": datetime.now(),
            }
        else:
            logger.error("Model not loaded, cannot make prediction")
            return {
                "home_team": home_team,
                "away_team": away_team,
                "prediction": "D",  # Default to draw
                "confidence": 0.33,
                "probabilities": {"H": 0.33, "D": 0.33, "A": 0.33},
                "created_at": datetime.now(),
                "error": "Model not loaded",
            }

    def predict_single_match(self, match_data: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single match."""
        if self.model is None:
            logger.error("No model available for predictions")
            return {}

        # Convert match data to DataFrame
        match_df = pd.DataFrame([match_data])

        # Prepare features
        features = self.prepare_features(match_df)

        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Get class order from model
        class_order = list(self.model.classes_)

        # Map probabilities to correct outcomes
        proba_dict = {c: float(probabilities[j]) for j, c in enumerate(class_order)}

        result = {
            "prediction": str(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                "H": proba_dict.get("H", 0.0),
                "D": proba_dict.get("D", 0.0),
                "A": proba_dict.get("A", 0.0),
            },
        }

        return self._to_native(result)

    def make_predictions(self, matches: pd.DataFrame) -> list[dict[str, Any]]:
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

        # Get class order from model
        class_order = list(self.model.classes_)

        # Format results
        results = []
        for i, (_, match) in enumerate(matches.iterrows()):
            prediction = predictions[i]
            proba = probabilities[i]
            # Map probabilities to correct outcomes
            proba_dict = {c: float(proba[j]) for j, c in enumerate(class_order)}
            result = {
                "match_id": match["match_id"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "prediction": str(prediction),
                "confidence": float(max(proba)),
                "home_win_prob": proba_dict.get("H", 0.0),
                "draw_prob": proba_dict.get("D", 0.0),
                "away_win_prob": proba_dict.get("A", 0.0),
                "home_odds": float(match["home_odds"]),
                "draw_odds": float(match["draw_odds"]),
                "away_odds": float(match["away_odds"]),
                "prediction_date": datetime.now(),
                "model_version": "latest",
            }
            # Convert all values to native types
            result = self._to_native(result)
            results.append(result)

            logger.info(
                f"Prediction: {match['home_team']} vs {match['away_team']} -> {prediction} (confidence: {max(proba):.3f})"
            )

        return results

    def save_predictions(self, predictions: list[dict[str, Any]]):
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
            df.to_sql("predictions", self.engine, if_exists="append", index=False)

            logger.info(f"Saved {len(predictions)} predictions to database")

        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")

    def run_prediction(self) -> list[dict[str, Any]]:
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
