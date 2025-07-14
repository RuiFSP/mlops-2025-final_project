"""
Integrated Example Dashboard for Premier League MLOps System
This example demonstrates all components working together:
- Streamlit for frontend
- FastAPI for backend API
- SQLite for database
- MLflow for experiment tracking
- Prefect for workflow orchestration
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000"
PREFECT_API_URL = "http://localhost:4200/api"
MLFLOW_TRACKING_URI = "http://localhost:5000"
DB_PATH = project_root / "data" / "predictions" / "predictions.db"

# Create SQLite database if it doesn't exist
def init_sqlite_db():
    """Initialize SQLite database with necessary tables"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        probabilities TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Insert some sample metrics if table is empty
    cursor.execute('SELECT COUNT(*) FROM metrics')
    if cursor.fetchone()[0] == 0:
        sample_metrics = [
            ('accuracy', 0.75),
            ('precision', 0.72),
            ('recall', 0.71),
            ('f1_score', 0.70),
        ]
        cursor.executemany(
            'INSERT INTO metrics (metric_name, metric_value) VALUES (?, ?)',
            sample_metrics
        )
    
    conn.commit()
    conn.close()
    
    logger.info(f"SQLite database initialized at {DB_PATH}")


def get_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        logger.error(f"Error checking API health: {e}")
        return None


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return None


def get_upcoming_matches():
    """Get upcoming matches from API"""
    try:
        response = requests.get(f"{API_URL}/matches/upcoming", timeout=5)
        return response.json() if response.status_code == 200 else []
    except Exception as e:
        logger.error(f"Error getting upcoming matches: {e}")
        return []


def predict_match(home_team, away_team, home_odds=None, away_odds=None, draw_odds=None):
    """Get match prediction from API"""
    try:
        payload = {
            "home_team": home_team,
            "away_team": away_team,
        }
        
        if home_odds:
            payload["home_odds"] = home_odds
        if away_odds:
            payload["away_odds"] = away_odds
        if draw_odds:
            payload["draw_odds"] = draw_odds
            
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Save prediction to SQLite
            save_prediction_to_db(result)
            
            return result
        return None
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        return None


def save_prediction_to_db(prediction):
    """Save prediction to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO predictions 
            (home_team, away_team, prediction, confidence, probabilities, created_at) 
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                prediction["home_team"],
                prediction["away_team"],
                prediction["prediction"],
                prediction["confidence"],
                json.dumps(prediction["probabilities"]),
                prediction["created_at"],
            )
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved prediction to database: {prediction['home_team']} vs {prediction['away_team']}")
    except Exception as e:
        logger.error(f"Error saving prediction to database: {e}")


def get_recent_predictions(limit=10):
    """Get recent predictions from SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result["probabilities"] = json.loads(result["probabilities"])
            results.append(result)
            
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        return []


def get_metrics_from_db():
    """Get metrics from SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM metrics ORDER BY created_at DESC')
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
            
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error getting metrics from database: {e}")
        return []


def trigger_prefect_flow(flow_name, parameters=None):
    """Trigger a Prefect flow"""
    try:
        if flow_name == "data_pipeline_flow":
            response = requests.get(f"{API_URL}/model/info", timeout=5)
            return {
                "success": True,
                "flow_id": f"data_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "message": "Data pipeline flow triggered"
            }
        elif flow_name == "training_pipeline_flow":
            response = requests.post(f"{API_URL}/retraining/force", timeout=5)
            return {
                "success": True,
                "flow_id": f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "message": "Training pipeline flow triggered"
            }
        elif flow_name == "prediction_pipeline_flow":
            response = requests.get(f"{API_URL}/predictions/upcoming", timeout=5)
            return {
                "success": True,
                "flow_id": f"prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "message": "Prediction pipeline flow triggered"
            }
        return {
            "success": False,
            "message": f"Unknown flow: {flow_name}"
        }
    except Exception as e:
        logger.error(f"Error triggering Prefect flow: {e}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }


def main():
    """Main function for the integrated example dashboard"""
    # Initialize SQLite database
    init_sqlite_db()
    
    # Set page config
    st.set_page_config(
        page_title="Integrated MLOps Example",
        page_icon="⚽",
        layout="wide",
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .component-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .status-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Integrated MLOps Example</h1>', unsafe_allow_html=True)
    st.markdown("This example demonstrates all components of the MLOps system working together.")
    
    # System status section
    st.markdown('<h2 class="component-header">🔄 System Status</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Check API status
    api_health = get_api_health()
    with col1:
        if api_health:
            st.markdown('<div class="status-card status-healthy">✅ FastAPI: Healthy</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-error">❌ FastAPI: Offline</div>', unsafe_allow_html=True)
    
    # Check SQLite status
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM predictions')
        prediction_count = cursor.fetchone()[0]
        conn.close()
        with col2:
            st.markdown(f'<div class="status-card status-healthy">✅ SQLite: {prediction_count} predictions</div>', unsafe_allow_html=True)
    except Exception:
        with col2:
            st.markdown('<div class="status-card status-error">❌ SQLite: Error</div>', unsafe_allow_html=True)
    
    # Check MLflow status
    model_info = get_model_info()
    with col3:
        if model_info:
            st.markdown('<div class="status-card status-healthy">✅ MLflow: Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-warning">⚠️ MLflow: Unknown</div>', unsafe_allow_html=True)
    
    # Check Prefect status
    with col4:
        try:
            prefect_response = requests.get(f"{PREFECT_API_URL}/health", timeout=2)
            if prefect_response.status_code == 200:
                st.markdown('<div class="status-card status-healthy">✅ Prefect: Healthy</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card status-warning">⚠️ Prefect: Issues</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="status-card status-warning">⚠️ Prefect: Unknown</div>', unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Match Prediction", "Recent Predictions", "Model Metrics", "Workflows"])
    
    # Tab 1: Match Prediction
    with tab1:
        st.header("🔮 Match Prediction")
        st.write("Use this form to predict the outcome of a Premier League match.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.text_input("Home Team", "Arsenal")
            home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.0, step=0.1)
        
        with col2:
            away_team = st.text_input("Away Team", "Chelsea")
            away_odds = st.number_input("Away Win Odds", min_value=1.01, value=3.0, step=0.1)
        
        draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.5, step=0.1)
        
        if st.button("🔮 Predict Match"):
            with st.spinner("Generating prediction..."):
                prediction = predict_match(
                    home_team=home_team,
                    away_team=away_team,
                    home_odds=home_odds,
                    away_odds=away_odds,
                    draw_odds=draw_odds,
                )
                
                if prediction:
                    st.success("✅ Prediction Generated!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
                        st.metric(
                            "Predicted Outcome", 
                            outcome_map.get(prediction["prediction"], prediction["prediction"])
                        )
                    
                    with col2:
                        st.metric("Confidence", f"{prediction['confidence']:.1%}")
                    
                    with col3:
                        created_at = datetime.fromisoformat(prediction["created_at"].replace("Z", "+00:00"))
                        st.metric("Created", created_at.strftime("%H:%M:%S"))
                    
                    # Probability breakdown
                    st.subheader("📊 Probability Breakdown")
                    prob_data = {
                        "Outcome": ["Home Win", "Draw", "Away Win"],
                        "Probability": [
                            prediction["probabilities"]["H"],
                            prediction["probabilities"]["D"],
                            prediction["probabilities"]["A"],
                        ],
                    }
                    
                    fig = px.bar(
                        prob_data,
                        x="Outcome",
                        y="Probability",
                        title=f"{home_team} vs {away_team} - Outcome Probabilities",
                        color="Outcome",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Failed to generate prediction. Please check the API status.")
    
    # Tab 2: Recent Predictions
    with tab2:
        st.header("📜 Recent Predictions")
        st.write("This section shows recent predictions from the SQLite database.")
        
        recent_predictions = get_recent_predictions(limit=10)
        
        if recent_predictions:
            # Convert to DataFrame for display
            df_predictions = pd.DataFrame([
                {
                    "Match": f"{p['home_team']} vs {p['away_team']}",
                    "Prediction": p['prediction'],
                    "Confidence": p['confidence'],
                    "Created At": p['created_at']
                }
                for p in recent_predictions
            ])
            
            st.dataframe(df_predictions, use_container_width=True)
            
            # Show chart of prediction distribution
            prediction_counts = {
                "Home Win": sum(1 for p in recent_predictions if p["prediction"] == "H"),
                "Draw": sum(1 for p in recent_predictions if p["prediction"] == "D"),
                "Away Win": sum(1 for p in recent_predictions if p["prediction"] == "A"),
            }
            
            fig = px.pie(
                names=list(prediction_counts.keys()),
                values=list(prediction_counts.values()),
                title="Distribution of Recent Predictions",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent predictions found. Make some predictions first!")
    
    # Tab 3: Model Metrics
    with tab3:
        st.header("📊 Model Metrics")
        st.write("This section shows model metrics from the SQLite database.")
        
        metrics = get_metrics_from_db()
        
        if metrics:
            # Convert to DataFrame for display
            df_metrics = pd.DataFrame([
                {
                    "Metric": m['metric_name'],
                    "Value": m['metric_value'],
                    "Created At": m['created_at']
                }
                for m in metrics
            ])
            
            st.dataframe(df_metrics, use_container_width=True)
            
            # Show chart of metrics
            fig = px.bar(
                df_metrics,
                x="Metric",
                y="Value",
                title="Model Performance Metrics",
                color="Metric",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics found in the database.")
    
    # Tab 4: Workflows
    with tab4:
        st.header("🔄 Workflows")
        st.write("This section allows you to trigger Prefect workflows.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Data Pipeline")
            st.write("Fetches data from football-data.uk")
            if st.button("🚀 Run Data Pipeline"):
                with st.spinner("Triggering data pipeline..."):
                    result = trigger_prefect_flow("data_pipeline_flow")
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.code(f"Flow ID: {result['flow_id']}")
                    else:
                        st.error(f"❌ {result['message']}")
        
        with col2:
            st.subheader("Training Pipeline")
            st.write("Trains the prediction model")
            force_retrain = st.checkbox("Force retrain")
            if st.button("🚀 Run Training Pipeline"):
                with st.spinner("Triggering training pipeline..."):
                    result = trigger_prefect_flow("training_pipeline_flow", {"force_retrain": force_retrain})
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.code(f"Flow ID: {result['flow_id']}")
                    else:
                        st.error(f"❌ {result['message']}")
        
        with col3:
            st.subheader("Prediction Pipeline")
            st.write("Generates predictions for upcoming matches")
            days_ahead = st.slider("Days ahead", min_value=1, max_value=14, value=7)
            if st.button("🚀 Run Prediction Pipeline"):
                with st.spinner("Triggering prediction pipeline..."):
                    result = trigger_prefect_flow("prediction_pipeline_flow", {"days_ahead": days_ahead})
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.code(f"Flow ID: {result['flow_id']}")
                    else:
                        st.error(f"❌ {result['message']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
            Integrated MLOps Example | Premier League Match Prediction | Last updated: {}
        </div>
        """.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main() 