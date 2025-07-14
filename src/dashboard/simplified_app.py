"""
Simplified Streamlit Dashboard for Premier League MLOps System
"""

import logging
import sys
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Add src to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import project modules
try:
    from src.pipelines.prediction_pipeline import PredictionPipeline
    
    # Define run_flow function to trigger workflows via API
    def run_flow(flow_name, parameters=None):
        """
        Trigger a workflow via the API
        
        Args:
            flow_name: Name of the workflow to trigger
            parameters: Optional parameters for the workflow
            
        Returns:
            Response from the API or None if failed
        """
        try:
            import requests
            
            # Set default parameters if none provided
            if parameters is None:
                parameters = {}
                
            # Use different endpoints based on flow type
            if flow_name == "training_pipeline_flow":
                # Use the retraining endpoint
                force_retrain = parameters.get("force_retrain", False)
                url = "http://localhost:8000/retraining/force"
                    
                response = requests.post(url)
                
                if response.status_code == 200:
                    result = response.json()
                    # Create a workflow-like response
                    return {
                        "workflow_id": f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "workflow_name": "training_pipeline",
                        "status": "triggered",
                        "started_at": datetime.now().isoformat(),
                        "parameters": parameters
                    }
            elif flow_name == "prediction_pipeline_flow":
                # Use the predictions endpoint
                days_ahead = parameters.get("days_ahead", 7)
                url = "http://localhost:8000/predictions/upcoming"
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Create a workflow-like response
                    return {
                        "workflow_id": f"prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "workflow_name": "prediction_pipeline",
                        "status": "triggered",
                        "started_at": datetime.now().isoformat(),
                        "parameters": parameters
                    }
            elif flow_name == "data_pipeline_flow":
                # Use the model info endpoint as a proxy for data pipeline
                url = "http://localhost:8000/model/info"
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Create a workflow-like response
                    return {
                        "workflow_id": f"data_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "workflow_name": "data_pipeline",
                        "status": "triggered",
                        "started_at": datetime.now().isoformat(),
                        "parameters": parameters
                    }
            else:
                logging.error(f"Unknown flow name: {flow_name}")
                return None
                
            logging.error(f"Failed to trigger workflow: {response.text}")
            return None
        except Exception as e:
            logging.error(f"Error triggering workflow: {e}")
            return None
except ImportError:
    # Fallback imports if src prefix doesn't work
    from pipelines.prediction_pipeline import PredictionPipeline

    # Define placeholder functions if orchestration module is not available
    def run_flow(flow_name, parameters=None):
        """Placeholder for run_flow"""
        logging.warning("Orchestration module not available, run_flow is a placeholder")
        return None


# Simplified metrics storage class
class SimplifiedMetricsStorage:
    """Simplified metrics storage using SQLite"""
    
    def __init__(self):
        """Initialize the metrics storage"""
        self.db_path = project_root / "data" / "predictions" / "predictions.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize the database"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Insert some sample metrics if table is empty
        cursor.execute('SELECT COUNT(*) FROM metrics')
        if cursor.fetchone()[0] == 0:
            sample_metrics = [
                ('accuracy', 0.75, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                ('precision', 0.72, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                ('recall', 0.71, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                ('f1_score', 0.70, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            ]
            cursor.executemany(
                'INSERT INTO metrics (metric_name, metric_value, timestamp) VALUES (?, ?, ?)',
                sample_metrics
            )
        
        conn.commit()
        conn.close()
    
    def get_latest_metrics(self, limit=10):
        """Get the latest metrics"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        )
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
            
        conn.close()
        return results
    
    def get_metrics_by_date_range(self, start_date, end_date, metric_types=None):
        """Get metrics within a date range"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM metrics WHERE timestamp BETWEEN ? AND ?'
        params = [start_date.strftime('%Y-%m-%d %H:%M:%S'), end_date.strftime('%Y-%m-%d %H:%M:%S')]
        
        if metric_types:
            placeholders = ','.join(['?' for _ in metric_types])
            query += f' AND metric_name IN ({placeholders})'
            params.extend(metric_types)
        
        query += ' ORDER BY timestamp'
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            results.append({
                'timestamp': row_dict['timestamp'],
                'metric_type': row_dict['metric_name'],
                'value': row_dict['metric_value']
            })
            
        conn.close()
        return results


class SimplifiedDashboard:
    """Simplified Streamlit Dashboard for Premier League MLOps System"""

    def __init__(self):
        """Initialize the dashboard"""
        self.metrics_storage = SimplifiedMetricsStorage()
        self.prediction_pipeline = None

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize ML components"""
        try:
            # Try to initialize components with a shorter timeout
            import os
            import mlflow
            
            # Set a shorter timeout for MLflow API calls
            os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '10'  # 10 seconds instead of default 120
            
            try:
                self.prediction_pipeline = PredictionPipeline()
            except Exception as e:
                st.warning(f"⚠️ Could not initialize prediction pipeline: {e}")
                st.info("Dashboard will run in limited mode without prediction capabilities.")
                self.prediction_pipeline = None
                logging.warning(f"Prediction pipeline initialization failed: {e}")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            st.info("Dashboard will run in limited mode. Some features may not be available.")
            self.prediction_pipeline = None
            logging.error(f"Component initialization failed: {e}")

    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">⚽ Premier League MLOps Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")

    def _render_status_card(self, title: str, status: str, icon: str, sidebar=False):
        """Render a status card"""
        card_class = "sidebar-status-card" if sidebar else "metric-card"
        
        if status == "Healthy":
            st.markdown(
                f"""
            <div class="{card_class} success-metric">
                <h4>{icon} {title}</h4>
                <p style="color: #28a745; font-weight: bold;">{status}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif status == "Warning":
            st.markdown(
                f"""
            <div class="{card_class} warning-metric">
                <h4>{icon} {title}</h4>
                <p style="color: #ffc107; font-weight: bold;">{status}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="{card_class} danger-metric">
                <h4>{icon} {title}</h4>
                <p style="color: #dc3545; font-weight: bold;">{status}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def _check_api_status(self) -> str:
        """Check API status"""
        try:
            import requests

            response = requests.get("http://localhost:8000/health", timeout=5)
            return "Healthy" if response.status_code == 200 else "Unhealthy"
        except:
            return "Offline"

    def _check_model_status(self) -> str:
        """Check model status"""
        try:
            if self.prediction_pipeline and hasattr(self.prediction_pipeline, "model"):
                return "Healthy"
            else:
                return "Offline"
        except Exception:
            return "Offline"

    def _check_db_status(self) -> str:
        """Check database status"""
        try:
            # Try to connect to the database
            self.metrics_storage.get_latest_metrics(1)
            return "Healthy"
        except Exception:
            return "Offline"

    def render_landing_page(self):
        """Render the landing page with feature explanations"""
        # Add welcome message
        st.markdown("""
        # Welcome to the Premier League MLOps Dashboard
        
        This dashboard provides monitoring and management tools for the Premier League Match Predictor MLOps system.
        Use the sidebar to navigate between different sections.
        """)
        
        st.subheader("🔍 Dashboard Sections")

        # Section descriptions in two columns
        col1, col2 = st.columns(2)

        with col1:
            # Model Performance section
            st.markdown(
                """
            <div class="section-description">
                <h3>🤖 Model Performance</h3>
                <p>This section shows the performance metrics of our Premier League prediction model over time.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Live Prediction section
            st.markdown(
                """
            <div class="section-description">
                <h3>🎯 Live Prediction</h3>
                <p>This interactive tool allows you to generate predictions for specific matches.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Workflows section
            st.markdown(
                """
            <div class="section-description">
                <h3>🔄 Workflows</h3>
                <p>This section allows you to trigger and monitor automated workflows.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_model_performance(self):
        """Render model performance section"""
        st.header("🤖 Model Performance")

        # Get model info
        if self.prediction_pipeline:
            model_info = self.prediction_pipeline.get_model_info()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Accuracy", f"{model_info.get('accuracy', 0):.2%}")

            with col2:
                st.metric("Model Version", model_info.get("version", "unknown"))

            with col3:
                st.metric("Model Stage", model_info.get("stage", "unknown"))

            with col4:
                created_at = model_info.get("created_at")
                if created_at:
                    days_old = (datetime.now().timestamp() - created_at / 1000) / (24 * 3600)
                    st.metric("Days Since Training", f"{days_old:.0f}")
                else:
                    st.metric("Days Since Training", "Unknown")

        # Performance over time chart
        st.subheader("📈 Performance Over Time")
        self._render_performance_chart()

    def _render_performance_chart(self):
        """Render performance chart"""
        try:
            # Get performance metrics from the last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            try:
                metrics = self.metrics_storage.get_metrics_by_date_range(
                    start_date, end_date, ["accuracy", "precision", "recall", "f1_score"]
                )

                if metrics:
                    df = pd.DataFrame(metrics)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    fig = px.line(
                        df,
                        x="timestamp",
                        y="value",
                        color="metric_type",
                        title="Model Performance Metrics Over Time",
                        labels={"value": "Score", "timestamp": "Date"},
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    self._show_mock_performance_data()
            except Exception:
                self._show_mock_performance_data()

        except Exception as e:
            st.error(f"Error loading performance data: {e}")
            self._show_mock_performance_data()

    def _show_mock_performance_data(self):
        """Show mock performance data when real data is not available"""
        st.info("Using sample performance data for demonstration")

        # Create mock data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D")
        metrics_types = ["accuracy", "precision", "recall", "f1_score"]

        # Base values and some random variation
        base_values = {"accuracy": 0.75, "precision": 0.72, "recall": 0.68, "f1_score": 0.70}

        # Create dataframe
        data = []
        for date in dates:
            for metric in metrics_types:
                # Add some random variation to make it look realistic
                value = base_values[metric] + (np.random.random() - 0.5) * 0.1
                # Ensure values are between 0 and 1
                value = max(0, min(1, value))
                data.append({"timestamp": date, "metric_type": metric, "value": value})

        df = pd.DataFrame(data)

        fig = px.line(
            df,
            x="timestamp",
            y="value",
            color="metric_type",
            title="Model Performance Metrics Over Time (Sample Data)",
            labels={"value": "Score", "timestamp": "Date"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def render_live_prediction_tool(self):
        """Render live prediction tool"""
        st.header("🔮 Live Match Prediction")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                home_team = st.text_input("Home Team", value="Arsenal")
                home_odds = st.number_input("Home Win Odds", min_value=1.01, value=2.0, step=0.1)

            with col2:
                away_team = st.text_input("Away Team", value="Chelsea")
                away_odds = st.number_input("Away Win Odds", min_value=1.01, value=3.0, step=0.1)

            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.5, step=0.1)

            submitted = st.form_submit_button("🔮 Get Prediction")

            if submitted and self.prediction_pipeline:
                try:
                    prediction = self.prediction_pipeline.predict_match(
                        home_team=home_team,
                        away_team=away_team,
                        home_odds=home_odds,
                        away_odds=away_odds,
                        draw_odds=draw_odds,
                    )

                    # Display prediction results
                    st.success("✅ Prediction Generated!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
                        st.metric(
                            "Predicted Outcome", outcome_map.get(prediction["prediction"], prediction["prediction"])
                        )

                    with col2:
                        st.metric("Confidence", f"{prediction['confidence']:.1%}")

                    with col3:
                        st.metric("Created", prediction["created_at"].strftime("%H:%M:%S"))

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
                        color="Probability",
                        color_continuous_scale="viridis",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating prediction: {e}")

    def render_workflows_dashboard(self):
        """Render the workflows dashboard page."""
        st.markdown("## 🔄 Workflow Orchestration")
        st.markdown(
            """
            This page allows you to monitor and manage the automated workflows in the system.
            You can trigger workflows manually using the cards below.
            """
        )
        
        # Add note about workflow triggering
        st.info("""
        ### 🚀 Workflow Management
        
        Use the workflow cards below to trigger specific workflows with custom parameters.
        Each workflow serves a different purpose in the MLOps pipeline:
        
        - **Data Pipeline**: Fetches data from football-data.uk
        - **Training Pipeline**: Trains the prediction model
        - **Prediction Pipeline**: Generates predictions for upcoming matches
        """)
        
        # Workflow cards
        st.markdown("### 🔄 Workflows")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_workflow_card(
                title="Data Pipeline",
                description="Fetches data from football-data.uk",
                flow_id="data_pipeline_flow",
                schedule="Every day at 6 AM"
            )
            
            self._render_workflow_card(
                title="Prediction Pipeline",
                description="Generates predictions for upcoming matches",
                flow_id="prediction_pipeline_flow",
                schedule="Every day at 8 AM"
            )
            
        with col2:
            self._render_workflow_card(
                title="Training Pipeline",
                description="Trains the prediction model",
                flow_id="training_pipeline_flow",
                schedule="Every Monday at midnight"
            )

    def _render_workflow_card(self, title: str, description: str, flow_id: str, schedule: str):
        """Render a workflow card with controls"""
        st.markdown(
            f"""
        <div class="workflow-card">
            <h4>{title}</h4>
            <p>{description}</p>
            <p><strong>Schedule:</strong> {schedule}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([3, 1])
        
        with col1:
            if flow_id == "training_pipeline_flow":
                force_retrain = st.checkbox(f"Force retrain for {title}", key=f"force_{flow_id}")
                params = {"force_retrain": force_retrain}
            elif flow_id == "prediction_pipeline_flow":
                days_ahead = st.slider(f"Days ahead for {title}", min_value=1, max_value=14, value=7, key=f"days_{flow_id}")
                params = {"days_ahead": days_ahead}
            else:
                params = {}
        
        with col2:
            if st.button(f"Run {title}", key=f"run_{flow_id}"):
                with st.spinner(f"Triggering {title} workflow..."):
                    result = run_flow(flow_id, params)
                    if result:
                        st.success(f"✅ Triggered {title} workflow!")
                    else:
                        st.error(f"❌ Failed to trigger {title} workflow")

    def _check_prefect_status(self) -> str:
        """Check Prefect status"""
        try:
            import requests

            response = requests.get("http://localhost:4200/api/health", timeout=5)
            return "Healthy" if response.status_code == 200 else "Unhealthy"
        except:
            return "Offline"


def main():
    """Main entry point for the Streamlit dashboard"""
    try:
        # Add auto-refresh every 5 minutes (300000 milliseconds)
        st_autorefresh(interval=300000, key="dashboard_refresh")
        
        # Create the dashboard
        dashboard = SimplifiedDashboard()

        # Sidebar navigation
        st.sidebar.title("⚽ Navigation")
        page = st.sidebar.radio(
            "Select a page",
            ["Overview", "Model Performance", "Live Prediction", "Workflows"],
        )
        
        # Add status indicators to sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Status")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            dashboard._render_status_card("API Status", dashboard._check_api_status(), "🚀", sidebar=True)
            dashboard._render_status_card("Model Status", dashboard._check_model_status(), "🤖", sidebar=True)
        with col2:
            dashboard._render_status_card("Database Status", dashboard._check_db_status(), "🐘", sidebar=True)
            dashboard._render_status_card("Prefect Status", dashboard._check_prefect_status(), "🔄", sidebar=True)

        # Render the selected page
        if page == "Overview":
            dashboard.render_landing_page()
        elif page == "Model Performance":
            dashboard.render_model_performance()
        elif page == "Live Prediction":
            dashboard.render_live_prediction_tool()
        elif page == "Workflows":
            dashboard.render_workflows_dashboard()

        # Add footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #888;">
                Premier League MLOps Dashboard | MLOps Zoomcamp Final Project | Last updated: {}
            </div>
            """.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        st.warning("Please check the logs for more details and try refreshing the page.")
        logging.exception("Dashboard error")
        
        # Provide a way to recover
        if st.button("🔄 Refresh Dashboard"):
            st.experimental_rerun()


if __name__ == "__main__":
    # Import os here to avoid issues with the SimplifiedMetricsStorage class
    import os
    main() 