"""
Prefect flows for ETL operations and ML workflows
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner

logger = logging.getLogger(__name__)


@task(name="fetch_football_data", log_prints=True)
def fetch_football_data(years_back: int = 3) -> str:
    """Fetch football data from football-data.co.uk"""
    try:
        from src.data_integration.football_data_fetcher import FootballDataFetcher
        
        fetcher = FootballDataFetcher()
        df = fetcher.get_historical_data(years_back=years_back)
        
        if df.empty:
            raise ValueError("No data fetched")
        
        # Save the data
        filepath = fetcher.save_processed_data(df, f"premier_league_{datetime.now().strftime('%Y%m%d')}.csv")
        
        print(f"✅ Successfully fetched {len(df)} matches")
        print(f"📁 Data saved to: {filepath}")
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error fetching football data: {e}")
        raise


@task(name="feature_engineering", log_prints=True)
def create_features(data_path: str) -> str:
    """Create features for ML model"""
    try:
        df = pd.read_csv(data_path)
        print(f"📊 Processing {len(df)} matches for feature engineering")
        
        # Create features
        features_df = df.copy()
        
        # Basic features
        features_df['total_goals'] = features_df['home_goals'] + features_df['away_goals']
        features_df['goal_difference'] = features_df['home_goals'] - features_df['away_goals']
        
        # Odds-based features (if available)
        if all(col in features_df.columns for col in ['home_odds', 'draw_odds', 'away_odds']):
            features_df['home_prob'] = 1 / features_df['home_odds']
            features_df['draw_prob'] = 1 / features_df['draw_odds']
            features_df['away_prob'] = 1 / features_df['away_odds']
            
            # Normalize probabilities
            total_prob = features_df['home_prob'] + features_df['draw_prob'] + features_df['away_prob']
            features_df['home_prob_norm'] = features_df['home_prob'] / total_prob
            features_df['draw_prob_norm'] = features_df['draw_prob'] / total_prob
            features_df['away_prob_norm'] = features_df['away_prob'] / total_prob
        
        # Team strength indicators (simplified)
        team_stats = {}
        for team in pd.concat([features_df['home_team'], features_df['away_team']]).unique():
            home_matches = features_df[features_df['home_team'] == team]
            away_matches = features_df[features_df['away_team'] == team]
            
            home_wins = len(home_matches[home_matches['result'] == 'H'])
            away_wins = len(away_matches[away_matches['result'] == 'A'])
            total_matches = len(home_matches) + len(away_matches)
            
            win_rate = (home_wins + away_wins) / total_matches if total_matches > 0 else 0
            team_stats[team] = win_rate
        
        # Add team strength features
        features_df['home_team_strength'] = features_df['home_team'].map(team_stats).fillna(0.5)
        features_df['away_team_strength'] = features_df['away_team'].map(team_stats).fillna(0.5)
        features_df['strength_difference'] = features_df['home_team_strength'] - features_df['away_team_strength']
        
        # Save features
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_path = processed_dir / f"features_{datetime.now().strftime('%Y%m%d')}.csv"
        features_df.to_csv(features_path, index=False)
        
        print(f"🔧 Created features for {len(features_df)} matches")
        print(f"📁 Features saved to: {features_path}")
        
        return str(features_path)
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


@task(name="train_model", log_prints=True)
def train_prediction_model(features_path: str) -> Dict:
    """Train the prediction model"""
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        import os
        
        # Set MLflow tracking URI
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
        
        # Load features
        df = pd.read_csv(features_path)
        print(f"🎯 Training model on {len(df)} matches")
        
        # Prepare features and target
        feature_columns = [
            'home_team_strength', 'away_team_strength', 'strength_difference'
        ]
        
        # Add odds features if available
        if 'home_prob_norm' in df.columns:
            feature_columns.extend(['home_prob_norm', 'draw_prob_norm', 'away_prob_norm'])
        
        X = df[feature_columns].fillna(0)
        y = df['result']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} matches")
        print(f"📊 Test set: {len(X_test)} matches")
        
        # Train model
        with mlflow.start_run(run_name=f"premier_league_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_params({
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "features": feature_columns
            })
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="premier_league_predictor"
            )
            
            # Save model locally too
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"premier_league_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(model, model_path)
            
            print(f"🤖 Model accuracy: {accuracy:.3f}")
            print(f"💾 Model saved to: {model_path}")
            
            return {
                "accuracy": accuracy,
                "model_path": str(model_path),
                "run_id": mlflow.active_run().info.run_id,
                "feature_columns": feature_columns
            }
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


@task(name="generate_predictions", log_prints=True)
def generate_upcoming_predictions(model_info: Dict) -> List[Dict]:
    """Generate predictions for upcoming matches"""
    try:
        import joblib
        
        # Load model
        model = joblib.load(model_info["model_path"])
        feature_columns = model_info["feature_columns"]
        
        # Mock upcoming matches (in reality, this would fetch from an API)
        upcoming_matches = [
            {"home_team": "Arsenal", "away_team": "Chelsea"},
            {"home_team": "Manchester City", "away_team": "Liverpool"},
            {"home_team": "Manchester United", "away_team": "Tottenham"},
        ]
        
        predictions = []
        for match in upcoming_matches:
            # Create mock features (in reality, these would be calculated)
            features = {col: 0.5 for col in feature_columns}  # Default values
            
            # Make prediction
            X_pred = pd.DataFrame([features])
            prediction = model.predict(X_pred)[0]
            probabilities = model.predict_proba(X_pred)[0]
            
            # Map probabilities to classes
            classes = model.classes_
            prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
            
            predictions.append({
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "prediction": prediction,
                "probabilities": prob_dict,
                "confidence": max(probabilities),
                "created_at": datetime.now().isoformat()
            })
        
        print(f"🔮 Generated {len(predictions)} predictions")
        
        # Save predictions
        predictions_dir = Path("data/predictions")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"💾 Predictions saved to: {predictions_path}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise


@flow(name="etl_and_training_flow", task_runner=ThreadPoolTaskRunner(workers=3))
def etl_and_training_flow(years_back: int = 3):
    """Complete ETL and training flow"""
    print("🚀 Starting ETL and Training Flow")
    
    # ETL Pipeline
    print("📥 Step 1: Fetching data...")
    data_path = fetch_football_data(years_back)
    
    print("🔧 Step 2: Feature engineering...")
    features_path = create_features(data_path)
    
    print("🎯 Step 3: Training model...")
    model_info = train_prediction_model(features_path)
    
    print("🔮 Step 4: Generating predictions...")
    predictions = generate_upcoming_predictions(model_info)
    
    print("✅ ETL and Training Flow completed successfully!")
    
    return {
        "data_path": data_path,
        "features_path": features_path,
        "model_info": model_info,
        "predictions": predictions
    }


@flow(name="daily_data_update_flow")
def daily_data_update_flow():
    """Daily data update flow"""
    print("📅 Starting daily data update...")
    
    # Fetch latest data
    data_path = fetch_football_data(years_back=1)
    
    print("✅ Daily data update completed!")
    return {"data_path": data_path}


@flow(name="prediction_generation_flow")
def prediction_generation_flow():
    """Generate predictions using the latest model"""
    print("🔮 Starting prediction generation...")
    
    # Load latest model info (in practice, this would be loaded from MLflow)
    models_dir = Path("models")
    if not models_dir.exists():
        raise ValueError("No models directory found. Please train a model first.")
    
    # Find latest model file
    model_files = list(models_dir.glob("*.pkl"))
    if not model_files:
        raise ValueError("No model files found. Please train a model first.")
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    mock_model_info = {
        "model_path": str(latest_model),
        "feature_columns": ['home_team_strength', 'away_team_strength', 'strength_difference']
    }
    
    predictions = generate_upcoming_predictions(mock_model_info)
    
    print("✅ Prediction generation completed!")
    return {"predictions": predictions}


if __name__ == "__main__":
    # Example usage
    result = etl_and_training_flow()
    print("Flow completed:", result) 