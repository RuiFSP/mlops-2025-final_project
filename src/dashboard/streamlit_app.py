"""
Streamlit Dashboard for Premier League MLOps System
Replaces Grafana with an interactive web dashboard
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np  # Added for mock data generation
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Add src to path - more robust approach
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import project modules
try:
    from src.betting_simulator.simulator import BettingSimulator
    from src.monitoring.metrics_storage import MetricsStorage
    from src.orchestration.flows import register_flows, run_flow
    from src.pipelines.prediction_pipeline import PredictionPipeline
    from src.simulation import (
        SeasonSimulator,
        get_simulation_status,
        get_simulator,
        pause_simulation,
        resume_simulation,
        start_simulation,
        stop_simulation,
    )
except ImportError:
    # Fallback imports if src prefix doesn't work
    from betting_simulator.simulator import BettingSimulator
    from monitoring.metrics_storage import MetricsStorage
    from pipelines.prediction_pipeline import PredictionPipeline

    try:
        from orchestration.flows import register_flows, run_flow
    except ImportError:
        # Mock functions if orchestration module is not available
        def register_flows():
            return None

        def run_flow(flow_name):
            return None

    try:
        from simulation import (
            SeasonSimulator,
            get_simulation_status,
            get_simulator,
            pause_simulation,
            resume_simulation,
            start_simulation,
            stop_simulation,
        )
    except ImportError:
        # Mock functions if simulation module is not available
        class SeasonSimulator:
            pass

        def get_simulator():
            return None

        def start_simulation(*args, **kwargs):
            return False

        def pause_simulation():
            return False

        def resume_simulation():
            return False

        def stop_simulation():
            return False

        def get_simulation_status():
            return {"is_running": False}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Premier League MLOps Dashboard", page_icon="⚽", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: rgba(38, 39, 48, 0.8);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: white;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .workflow-card {
        background-color: rgba(38, 39, 48, 0.8);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6f42c1;
    }
    .simulation-controls {
        background-color: rgba(38, 39, 48, 0.8);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(250, 250, 250, 0.2);
    }
    .feature-card {
        background-color: rgba(38, 39, 48, 0.8);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #17a2b8;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .section-description {
        background-color: rgba(38, 39, 48, 0.8);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #17a2b8;
    }
    .stButton>button {
        background-color: #4e8df5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitDashboard:
    """Main Streamlit Dashboard class"""

    def __init__(self):
        """Initialize the dashboard"""
        self.metrics_storage = MetricsStorage()
        self.prediction_pipeline = None
        self.betting_simulator = None

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize ML components"""
        try:
            self.prediction_pipeline = PredictionPipeline()
            self.betting_simulator = BettingSimulator(initial_balance=1000.0)
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            logger.error(f"Component initialization failed: {e}")

    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">⚽ Premier League MLOps Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")

        # System status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            self._render_status_card("API Status", self._check_api_status(), "🚀")

        with col2:
            self._render_status_card("Model Status", self._check_model_status(), "🤖")

        with col3:
            self._render_status_card("Database Status", self._check_db_status(), "🐘")

        with col4:
            self._render_status_card("Predictions Today", self._get_predictions_count(), "📊")

    def _render_status_card(self, title: str, status: str, icon: str):
        """Render a status card"""
        if status == "Healthy":
            st.markdown(
                f"""
            <div class="metric-card success-metric">
                <h4>{icon} {title}</h4>
                <p style="color: #28a745; font-weight: bold;">{status}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif status == "Warning":
            st.markdown(
                f"""
            <div class="metric-card warning-metric">
                <h4>{icon} {title}</h4>
                <p style="color: #ffc107; font-weight: bold;">{status}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="metric-card danger-metric">
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
            if self.prediction_pipeline and self.prediction_pipeline.model:
                return "Healthy"
            return "No Model"
        except:
            return "Error"

    def _check_db_status(self) -> str:
        """Check database status"""
        try:
            with self.metrics_storage._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return "Healthy"
        except:
            return "Offline"

    def _get_predictions_count(self) -> str:
        """Get today's predictions count"""
        try:
            with self.metrics_storage._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) FROM predictions
                        WHERE DATE(prediction_date) = CURRENT_DATE
                    """)
                    count = cursor.fetchone()[0]
                    return str(count)
        except:
            return "0"

    def render_landing_page(self):
        """Render the landing page with feature explanations"""
        st.subheader("🔍 Dashboard Sections")

        # Section descriptions in two columns
        col1, col2 = st.columns(2)

        with col1:
            # Model Performance section
            st.markdown(
                """
            <div class="section-description">
                <h3>🤖 Model Performance</h3>
                <p>This section shows the performance metrics of our Premier League prediction model over time. You can:</p>
                <ul>
                    <li>View accuracy, precision, recall, and F1 score trends</li>
                    <li>Compare performance across different time periods</li>
                    <li>Identify potential model drift</li>
                    <li>Analyze performance by match type and team</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Predictions section
            st.markdown(
                """
            <div class="section-description">
                <h3>🔮 Predictions</h3>
                <p>This section displays recent match predictions generated by our model. You can:</p>
                <ul>
                    <li>View predictions for upcoming and past matches</li>
                    <li>Analyze prediction confidence levels</li>
                    <li>See probability distributions for different outcomes</li>
                    <li>Compare predictions against actual results</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Live Prediction section
            st.markdown(
                """
            <div class="section-description">
                <h3>🎯 Live Prediction</h3>
                <p>This interactive tool allows you to generate predictions for specific matches. You can:</p>
                <ul>
                    <li>Enter team names and match details</li>
                    <li>Get real-time predictions and confidence scores</li>
                    <li>View detailed probability breakdowns</li>
                    <li>Analyze feature importance for predictions</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Betting Performance section
            st.markdown(
                """
            <div class="section-description">
                <h3>💰 Betting Performance</h3>
                <p>This section tracks the performance of betting strategies based on our model's predictions. You can:</p>
                <ul>
                    <li>Monitor profit/loss over time</li>
                    <li>View win rates and ROI metrics</li>
                    <li>Analyze betting history and outcomes</li>
                    <li>Compare different betting strategies</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Workflows section
            st.markdown(
                """
            <div class="section-description">
                <h3>🔄 Workflows</h3>
                <p>This section manages Prefect workflows for data pipelines and model training. You can:</p>
                <ul>
                    <li>Schedule and trigger workflow runs</li>
                    <li>Monitor workflow execution status</li>
                    <li>View logs and error messages</li>
                    <li>Configure workflow parameters</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Simulation section
            st.markdown(
                """
            <div class="section-description">
                <h3>🎮 Simulation</h3>
                <p>This section allows you to run season simulations with different parameters. You can:</p>
                <ul>
                    <li>Configure simulation parameters (season, dates, betting strategy)</li>
                    <li>Run, pause, and resume simulations</li>
                    <li>View simulation progress and results</li>
                    <li>Analyze long-term betting performance</li>
                </ul>
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

    def render_predictions_dashboard(self):
        """Render predictions dashboard"""
        st.header("🔮 Recent Predictions")

        try:
            # Get recent predictions
            with self.metrics_storage._get_connection() as conn:
                query = """
                    SELECT home_team, away_team, prediction, confidence,
                           home_win_prob, draw_prob, away_win_prob,
                           prediction_date
                    FROM predictions
                    ORDER BY prediction_date DESC
                    LIMIT 10
                """
                try:
                    df = pd.read_sql(query, conn)

                    if not df.empty:
                        # Display recent predictions table
                        st.subheader("📋 Latest Predictions")

                        # Format the dataframe for display
                        display_df = df.copy()
                        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
                        display_df["home_win_prob"] = display_df["home_win_prob"].apply(lambda x: f"{x:.1%}")
                        display_df["draw_prob"] = display_df["draw_prob"].apply(lambda x: f"{x:.1%}")
                        display_df["away_win_prob"] = display_df["away_win_prob"].apply(lambda x: f"{x:.1%}")
                        display_df["prediction_date"] = pd.to_datetime(display_df["prediction_date"]).dt.strftime(
                            "%Y-%m-%d %H:%M"
                        )

                        st.dataframe(display_df, use_container_width=True)

                        # Confidence distribution chart
                        st.subheader("📊 Confidence Distribution")
                        fig = px.histogram(
                            df,
                            x="confidence",
                            bins=20,
                            title="Prediction Confidence Distribution",
                            labels={"confidence": "Confidence Score", "count": "Number of Predictions"},
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # Prediction distribution
                        st.subheader("⚽ Prediction Distribution")
                        prediction_counts = df["prediction"].value_counts()

                        fig = px.pie(
                            values=prediction_counts.values,
                            names=[
                                "Home Win" if x == "H" else "Draw" if x == "D" else "Away Win"
                                for x in prediction_counts.index
                            ],
                            title="Prediction Outcome Distribution",
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        self._show_mock_prediction_data()
                except Exception:
                    self._show_mock_prediction_data()

        except Exception as e:
            st.error(f"Error loading predictions data: {e}")
            self._show_mock_prediction_data()

    def _show_mock_prediction_data(self):
        """Show mock prediction data when real data is not available"""
        st.info("Using sample prediction data for demonstration")

        # Create mock data
        dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
        mock_data = {
            "home_team": [
                "Arsenal",
                "Liverpool",
                "Chelsea",
                "Man City",
                "Tottenham",
                "Leicester",
                "Everton",
                "West Ham",
                "Newcastle",
                "Aston Villa",
            ],
            "away_team": [
                "Man United",
                "Chelsea",
                "Tottenham",
                "Liverpool",
                "Arsenal",
                "Everton",
                "Leicester",
                "Newcastle",
                "Aston Villa",
                "West Ham",
            ],
            "prediction": ["H", "D", "H", "A", "H", "H", "A", "D", "H", "A"],
            "confidence": [0.72, 0.45, 0.68, 0.55, 0.62, 0.71, 0.58, 0.47, 0.65, 0.52],
            "home_win_prob": [0.72, 0.30, 0.68, 0.25, 0.62, 0.71, 0.22, 0.28, 0.65, 0.30],
            "draw_prob": [0.18, 0.45, 0.22, 0.20, 0.25, 0.19, 0.20, 0.47, 0.20, 0.18],
            "away_win_prob": [0.10, 0.25, 0.10, 0.55, 0.13, 0.10, 0.58, 0.25, 0.15, 0.52],
            "prediction_date": dates,
        }

        df = pd.DataFrame(mock_data)

        # Display recent predictions table
        st.subheader("📋 Latest Predictions (Sample Data)")

        # Format the dataframe for display
        display_df = df.copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
        display_df["home_win_prob"] = display_df["home_win_prob"].apply(lambda x: f"{x:.1%}")
        display_df["draw_prob"] = display_df["draw_prob"].apply(lambda x: f"{x:.1%}")
        display_df["away_win_prob"] = display_df["away_win_prob"].apply(lambda x: f"{x:.1%}")
        display_df["prediction_date"] = pd.to_datetime(display_df["prediction_date"]).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(display_df, use_container_width=True)

        # Confidence distribution chart
        st.subheader("📊 Confidence Distribution (Sample Data)")
        fig = px.histogram(
            df,
            x="confidence",
            bins=20,
            title="Prediction Confidence Distribution",
            labels={"confidence": "Confidence Score", "count": "Number of Predictions"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Prediction distribution
        st.subheader("⚽ Prediction Distribution (Sample Data)")
        prediction_counts = df["prediction"].value_counts()

        fig = px.pie(
            values=prediction_counts.values,
            names=["Home Win" if x == "H" else "Draw" if x == "D" else "Away Win" for x in prediction_counts.index],
            title="Prediction Outcome Distribution",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def render_betting_dashboard(self):
        """Render betting performance dashboard"""
        st.header("💰 Betting Performance")

        if self.betting_simulator:
            # Get betting statistics
            stats = self.betting_simulator.get_statistics()

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Balance", f"£{stats.get('current_balance', 0):.2f}")

            with col2:
                profit_loss = stats.get("profit_loss", 0)
                st.metric("Profit/Loss", f"£{profit_loss:.2f}", delta=f"{profit_loss:.2f}")

            with col3:
                win_rate = stats.get("win_rate", 0)
                st.metric("Win Rate", f"{win_rate:.1%}")

            with col4:
                roi = stats.get("overall_roi", 0)
                st.metric("Overall ROI", f"{roi:.1%}", delta=f"{roi:.1%}")

            # Betting history chart
            st.subheader("📈 Betting Performance Over Time")
            self._render_betting_chart()

    def _render_betting_chart(self):
        """Render betting performance chart"""
        try:
            with self.metrics_storage._get_connection() as conn:
                query = """
                    SELECT bet_date, bet_amount, result, payout, roi,
                           home_team, away_team, bet_type
                    FROM bets
                    ORDER BY bet_date ASC
                """
                try:
                    df = pd.read_sql(query, conn)

                    if not df.empty:
                        # Calculate cumulative profit/loss
                        df["profit_loss"] = df.apply(
                            lambda row: row["payout"] - row["bet_amount"]
                            if row["result"] == "W"
                            else -row["bet_amount"]
                            if row["result"] == "L"
                            else 0,
                            axis=1,
                        )
                        df["cumulative_pl"] = df["profit_loss"].cumsum()

                        # Cumulative P&L chart
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=df["bet_date"],
                                y=df["cumulative_pl"],
                                mode="lines+markers",
                                name="Cumulative P&L",
                                line=dict(color="green" if df["cumulative_pl"].iloc[-1] > 0 else "red"),
                            )
                        )

                        fig.update_layout(
                            title="Cumulative Profit/Loss Over Time",
                            xaxis_title="Date",
                            yaxis_title="Cumulative P&L (£)",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Recent bets table
                        st.subheader("📋 Recent Bets")
                        recent_bets = df.tail(10).copy()
                        recent_bets["bet_amount"] = recent_bets["bet_amount"].apply(lambda x: f"£{x:.2f}")
                        recent_bets["payout"] = recent_bets["payout"].apply(
                            lambda x: f"£{x:.2f}" if pd.notna(x) else "Pending"
                        )
                        recent_bets["roi"] = recent_bets["roi"].apply(
                            lambda x: f"{x:.1%}" if pd.notna(x) else "Pending"
                        )

                        st.dataframe(
                            recent_bets[
                                [
                                    "bet_date",
                                    "home_team",
                                    "away_team",
                                    "bet_type",
                                    "bet_amount",
                                    "result",
                                    "payout",
                                    "roi",
                                ]
                            ],
                            use_container_width=True,
                        )
                    else:
                        self._show_mock_betting_data()
                except Exception:
                    self._show_mock_betting_data()

        except Exception as e:
            st.error(f"Error loading betting data: {e}")
            self._show_mock_betting_data()

    def _show_mock_betting_data(self):
        """Show mock betting data when real data is not available"""
        st.info("Using sample betting data for demonstration")

        # Create mock data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=10, freq="D")
        mock_data = {
            "bet_date": dates,
            "bet_amount": [100, 50, 75, 100, 50, 75, 100, 50, 75, 100],
            "result": ["W", "L", "W", "L", "W", "W", "L", "W", "W", "L"],
            "payout": [210, 0, 187.5, 0, 125, 150, 0, 125, 187.5, 0],
            "roi": [1.1, -1.0, 1.5, -1.0, 1.5, 1.0, -1.0, 1.5, 1.5, -1.0],
            "home_team": [
                "Arsenal",
                "Liverpool",
                "Chelsea",
                "Man City",
                "Tottenham",
                "Leicester",
                "Everton",
                "West Ham",
                "Newcastle",
                "Aston Villa",
            ],
            "away_team": [
                "Man United",
                "Chelsea",
                "Tottenham",
                "Liverpool",
                "Arsenal",
                "Everton",
                "Leicester",
                "Newcastle",
                "Aston Villa",
                "West Ham",
            ],
            "bet_type": ["H", "D", "H", "A", "H", "H", "A", "D", "H", "A"],
        }

        df = pd.DataFrame(mock_data)
        df["profit_loss"] = df.apply(
            lambda row: row["payout"] - row["bet_amount"]
            if row["result"] == "W"
            else -row["bet_amount"]
            if row["result"] == "L"
            else 0,
            axis=1,
        )
        df["cumulative_pl"] = df["profit_loss"].cumsum()

        # Cumulative P&L chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["bet_date"],
                y=df["cumulative_pl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="green" if df["cumulative_pl"].iloc[-1] > 0 else "red"),
            )
        )

        fig.update_layout(
            title="Cumulative Profit/Loss Over Time (Sample Data)",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (£)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent bets table
        st.subheader("📋 Recent Bets (Sample Data)")
        recent_bets = df.copy()
        recent_bets["bet_amount"] = recent_bets["bet_amount"].apply(lambda x: f"£{x:.2f}")
        recent_bets["payout"] = recent_bets["payout"].apply(lambda x: f"£{x:.2f}" if pd.notna(x) and x > 0 else "£0.00")
        recent_bets["roi"] = recent_bets["roi"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0.0%")

        st.dataframe(
            recent_bets[["bet_date", "home_team", "away_team", "bet_type", "bet_amount", "result", "payout", "roi"]],
            use_container_width=True,
        )

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
        """Render Prefect workflows dashboard and controls"""
        st.header("🔄 Workflows & Orchestration")

        # Workflow status
        st.subheader("📊 Workflow Status")

        try:
            # Check if Prefect is running
            prefect_running = self._check_prefect_status() == "Healthy"

            if prefect_running:
                st.success("✅ Prefect server is running and accessible")

                # Display workflow cards
                col1, col2 = st.columns(2)

                with col1:
                    self._render_workflow_card(
                        "Hourly Monitoring",
                        "Monitors model performance and drift detection",
                        "hourly_monitoring",
                        "⏱️ Every hour",
                    )

                    self._render_workflow_card(
                        "Weekly Retraining",
                        "Automated model retraining evaluation",
                        "weekly_retraining",
                        "📅 Every week (Sunday)",
                    )

                with col2:
                    self._render_workflow_card(
                        "Daily Predictions",
                        "Generate predictions for upcoming matches",
                        "daily_predictions",
                        "🗓️ Every day (morning)",
                    )

                    self._render_workflow_card(
                        "Emergency Retraining", "Manual retraining trigger", "emergency_retraining", "🚨 On demand"
                    )
            else:
                st.error("❌ Prefect server is not running or not accessible")
                st.info("To start Prefect server, run: `uv run prefect server start --host 0.0.0.0 --port 4200`")

        except Exception as e:
            st.error(f"❌ Error accessing workflow information: {e}")

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
            # Add parameters based on flow type
            if flow_id == "daily_predictions":
                st.date_input(f"Date for {title}", datetime.now().date())
            elif flow_id == "weekly_retraining":
                st.selectbox(f"Dataset for {title}", ["Full Dataset", "Last 3 Seasons", "Last Season"])

        with col2:
            if st.button(f"Run {title}", key=f"run_{flow_id}"):
                try:
                    # Run the flow
                    st.info(f"🔄 Running {title} flow...")
                    # This would actually call the Prefect flow
                    run_flow(flow_id)
                    st.success(f"✅ {title} flow started successfully")
                except Exception as e:
                    st.error(f"❌ Failed to start {title} flow: {e}")

    def _check_prefect_status(self) -> str:
        """Check Prefect server status"""
        try:
            import requests

            response = requests.get("http://localhost:4200/api/health", timeout=5)
            return "Healthy" if response.status_code == 200 else "Unhealthy"
        except:
            return "Offline"

    def render_simulation_dashboard(self):
        """Render simulation controls and status"""
        st.header("🎮 Simulation Controls")

        # Get simulation status
        sim_status = get_simulation_status()
        is_simulation_running = sim_status.get("is_running", False)

        # Status indicator
        if is_simulation_running:
            st.success("✅ Simulation is currently running")

            # Show current simulation details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Date", sim_status.get("current_date", datetime.now()).strftime("%Y-%m-%d"))

            with col2:
                st.metric(
                    "Matches Simulated",
                    f"{sim_status.get('matches_simulated', 0)} / {sim_status.get('total_matches', 380)}",
                )

            with col3:
                current_balance = sim_status.get("current_balance", 1000.0)
                profit_loss = sim_status.get("profit_loss", 0.0)
                st.metric("Current Balance", f"£{current_balance:.2f}", delta=f"£{profit_loss:.2f}")

            # Progress bar
            progress = sim_status.get("progress", 0.0)
            st.progress(progress)

        else:
            st.info("ℹ️ No simulation is currently running")

        # Simulation controls
        with st.form("simulation_form"):
            st.subheader("⚙️ Simulation Parameters")

            col1, col2 = st.columns(2)

            with col1:
                season = st.selectbox("Season to simulate", ["2023/2024", "2022/2023", "2021/2022", "2020/2021"])
                start_date = st.date_input("Start date", datetime.now().date() - timedelta(days=30))

            with col2:
                simulation_speed = st.slider(
                    "Simulation speed", 1, 10, 5, help="Higher values = faster simulation (fewer days per step)"
                )
                end_date = st.date_input("End date", datetime.now().date())

            # Betting strategy
            st.subheader("💰 Betting Strategy")
            betting_strategy = st.selectbox(
                "Betting strategy", ["Confidence-based", "Kelly Criterion", "Fixed Amount", "No Betting"]
            )

            initial_balance = st.number_input("Initial balance (£)", min_value=100.0, value=1000.0, step=100.0)

            # Visualization options
            st.subheader("📊 Visualization Options")

            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Show metrics charts", value=True)
                st.checkbox("Show prediction results", value=True)

            with col2:
                st.checkbox("Show betting performance", value=True)
                st.checkbox("Show detailed logs", value=False)

            # Submit button
            submitted = st.form_submit_button("🚀 Start Simulation")

            if submitted:
                if is_simulation_running:
                    st.warning("⚠️ A simulation is already running. Please stop it before starting a new one.")
                else:
                    # Convert date inputs to datetime
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.min.time())

                    # Start the simulation
                    success = start_simulation(
                        season=season,
                        start_date=start_datetime,
                        end_date=end_datetime,
                        simulation_speed=simulation_speed,
                        betting_strategy=betting_strategy,
                        initial_balance=initial_balance,
                    )

                    if success:
                        st.success(f"✅ Starting simulation for {season} from {start_date} to {end_date}")
                    else:
                        st.error("❌ Failed to start simulation")

        # Simulation control buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⏸️ Pause Simulation", disabled=not is_simulation_running):
                if pause_simulation():
                    st.info("⏸️ Simulation paused")
                else:
                    st.error("❌ Failed to pause simulation")

        with col2:
            if st.button(
                "▶️ Resume Simulation", disabled=not is_simulation_running or not sim_status.get("is_paused", False)
            ):
                if resume_simulation():
                    st.info("▶️ Simulation resumed")
                else:
                    st.error("❌ Failed to resume simulation")

        with col3:
            if st.button("⏹️ Stop Simulation", disabled=not is_simulation_running):
                if stop_simulation():
                    st.info("⏹️ Simulation stopped")
                else:
                    st.error("❌ Failed to stop simulation")

        # Simulation results (if running)
        if is_simulation_running and sim_status.get("matches_simulated", 0) > 0:
            st.subheader("📈 Simulation Results")

            # This would show charts and tables based on simulation data
            # For now, just show a placeholder
            st.info("Simulation results will appear here as the simulation progresses")


def main():
    """Main Streamlit app"""
    # Add auto-refresh capability (every 60 seconds)
    refresh_interval = 60  # seconds
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")

    # Initialize session state for page navigation if it doesn't exist
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Overview"

    dashboard = StreamlitDashboard()

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")

    # Page selection buttons in sidebar
    if st.sidebar.button("🏠 Overview", use_container_width=True):
        st.session_state.page = "🏠 Overview"
        st.rerun()

    if st.sidebar.button("🤖 Model Performance", use_container_width=True):
        st.session_state.page = "🤖 Model Performance"
        st.rerun()

    if st.sidebar.button("🔮 Predictions", use_container_width=True):
        st.session_state.page = "🔮 Predictions"
        st.rerun()

    if st.sidebar.button("💰 Betting", use_container_width=True):
        st.session_state.page = "💰 Betting"
        st.rerun()

    if st.sidebar.button("🎯 Live Prediction", use_container_width=True):
        st.session_state.page = "🎯 Live Prediction"
        st.rerun()

    if st.sidebar.button("🔄 Workflows", use_container_width=True):
        st.session_state.page = "🔄 Workflows"
        st.rerun()

    if st.sidebar.button("🎮 Simulation", use_container_width=True):
        st.session_state.page = "🎮 Simulation"
        st.rerun()

    # Render header on all pages
    dashboard.render_header()

    # Route to different pages based on session state
    if st.session_state.page == "🏠 Overview":
        dashboard.render_landing_page()

    elif st.session_state.page == "🤖 Model Performance":
        dashboard.render_model_performance()

    elif st.session_state.page == "🔮 Predictions":
        dashboard.render_predictions_dashboard()

    elif st.session_state.page == "💰 Betting":
        dashboard.render_betting_dashboard()

    elif st.session_state.page == "🎯 Live Prediction":
        dashboard.render_live_prediction_tool()

    elif st.session_state.page == "🔄 Workflows":
        dashboard.render_workflows_dashboard()

    elif st.session_state.page == "🎮 Simulation":
        dashboard.render_simulation_dashboard()

    # Auto-refresh option
    st.sidebar.markdown("---")
    st.sidebar.info(f"🔄 Dashboard auto-refreshes every {refresh_interval} seconds")
    refresh_rate = st.sidebar.slider(
        "Refresh rate (seconds)", min_value=10, max_value=300, value=refresh_interval, step=10
    )
    if refresh_rate != refresh_interval:
        st.sidebar.success(f"✅ Refresh rate updated to {refresh_rate} seconds")
        # This will be applied on the next refresh

    # Troubleshooting link
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Troubleshooting")
    st.sidebar.info(
        "If you're experiencing issues, run the troubleshooting script: `python scripts/dashboard_troubleshoot.py`"
    )
    if st.sidebar.button("🔧 Run Troubleshooter", use_container_width=True):
        import subprocess

        subprocess.Popen([sys.executable, "scripts/dashboard_troubleshoot.py"])
        st.sidebar.success("Troubleshooter launched in a new window")


if __name__ == "__main__":
    main()
