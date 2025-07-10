#!/usr/bin/env python3
"""
Simple Test Suite for Simulation Engine Components
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ruff: noqa: E402
from src.simulation import (
    MatchScheduler,
    OddsGenerator,
    RetrainingOrchestrator,
    SeasonSimulator,
)


@pytest.mark.e2e
@pytest.mark.timeout(180)  # 3 minute timeout for simulation tests
def test_match_scheduler():
    """Test MatchScheduler basic functionality."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    match_calendar_path = base_dir / "data/simulation/match_calendar.parquet"

    if not match_calendar_path.exists():
        pytest.skip("Match calendar not found")

    scheduler = MatchScheduler(str(match_calendar_path))

    assert scheduler.get_total_weeks() > 0
    assert scheduler.get_current_week() == 1

    week_1_matches = scheduler.get_matches_for_week(1)
    assert len(week_1_matches) > 0


def test_odds_generator():
    """Test OddsGenerator basic functionality."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    simulation_data_path = base_dir / "data/simulation/simulation_data_2023_24.parquet"

    if not simulation_data_path.exists():
        pytest.skip("Simulation data not found")

    odds_gen = OddsGenerator(str(simulation_data_path))

    # Test odds generation
    home_odds, draw_odds, away_odds = odds_gen.generate_odds("Arsenal", "Chelsea")

    assert 1.0 < home_odds < 20.0
    assert 2.0 < draw_odds < 10.0
    assert 1.0 < away_odds < 20.0


def test_retraining_orchestrator():
    """Test RetrainingOrchestrator basic functionality."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    model_path = base_dir / "models/model.pkl"

    if not model_path.exists():
        pytest.skip("Model not found")

    orchestrator = RetrainingOrchestrator(model_path=str(model_path), threshold=0.1, frequency=5)

    # Test baseline setting
    orchestrator.set_baseline_performance(0.8)
    assert orchestrator.baseline_performance == 0.8

    # Test retraining count
    assert orchestrator.get_retraining_count() == 0


def test_season_simulator_integration():
    """Test SeasonSimulator integration."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root

    required_files = [
        base_dir / "data/simulation/simulation_data_2023_24.parquet",
        base_dir / "data/simulation/match_calendar.parquet",
        base_dir / "models/model.pkl",
    ]

    for file_path in required_files:
        if not file_path.exists():
            pytest.skip(f"Required file not found: {file_path}")

    simulator = SeasonSimulator(
        simulation_data_path=str(required_files[0]),
        match_calendar_path=str(required_files[1]),
        model_path=str(required_files[2]),
        output_dir="data/test_simulation",
        retraining_threshold=0.1,
        retraining_frequency=10,
    )

    # Test initial state
    state = simulator.get_simulation_state()
    assert state["current_week"] == 1
    assert state["max_week"] > 0

    # Test single week simulation
    week_data = simulator.simulate_week(1)
    assert "week" in week_data
    assert "predictions" in week_data
    assert "results" in week_data
    assert "performance" in week_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
