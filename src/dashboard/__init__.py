"""
Dashboard module for the Premier League MLOps system.
"""

from .streamlit_app import StreamlitDashboard
from .simplified_app import SimplifiedDashboard

__all__ = ["StreamlitDashboard", "SimplifiedDashboard"]
