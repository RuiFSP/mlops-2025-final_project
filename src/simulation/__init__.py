"""
Season Simulation Engine for Premier League Match Predictor.

This module provides components for simulating a complete Premier League season
using historical data, enabling realistic MLOps testing without waiting for
actual season data.

Components:
- SeasonSimulator: Core simulation engine
- MatchScheduler: Realistic fixture management
- OddsGenerator: Betting odds based on historical patterns
- RetrainingOrchestrator: Automated model updates
"""

from .match_scheduler import MatchScheduler
from .odds_generator import OddsGenerator
from .retraining_orchestrator import RetrainingOrchestrator
from .season_simulator import SeasonSimulator

__all__ = [
    "SeasonSimulator",
    "MatchScheduler",
    "OddsGenerator",
    "RetrainingOrchestrator",
]
