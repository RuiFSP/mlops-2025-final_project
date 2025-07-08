"""
Match Scheduler for Season Simulation Engine.

Handles realistic fixture management, including:
- Weekly match scheduling
- Match status tracking
- Realistic timing and sequencing
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class MatchScheduler:
    """
    Manages realistic Premier League match scheduling for simulation.

    Features:
    - Week-by-week match progression
    - Status tracking (pending, predicted, completed)
    - Realistic match timing
    - Batch processing for weekly predictions
    """

    def __init__(self, calendar_path: str):
        """Initialize the match scheduler with match calendar."""
        self.calendar_path = Path(calendar_path)
        self.match_calendar = self._load_calendar()
        self.current_week = 1
        self._initialize_status()

        logger.info("MatchScheduler initialized")
        logger.info(f"Total weeks: {self.get_total_weeks()}")
        logger.info(f"Total matches: {len(self.match_calendar)}")

    def _load_calendar(self) -> pd.DataFrame:
        """Load the match calendar from file."""
        if not self.calendar_path.exists():
            raise FileNotFoundError(f"Match calendar not found: {self.calendar_path}")

        calendar = pd.read_parquet(self.calendar_path)
        calendar["Date"] = pd.to_datetime(calendar["Date"])
        return calendar.sort_values("Date")

    def _initialize_status(self) -> None:
        """Initialize match status tracking."""
        if "prediction_made" not in self.match_calendar.columns:
            self.match_calendar["prediction_made"] = False
        if "actual_result_revealed" not in self.match_calendar.columns:
            self.match_calendar["actual_result_revealed"] = False
        if "simulation_status" not in self.match_calendar.columns:
            self.match_calendar["simulation_status"] = "pending"

    def get_total_weeks(self) -> int:
        """Get total number of simulation weeks."""
        return int(self.match_calendar["simulation_week"].max())

    def get_current_week(self) -> int:
        """Get current simulation week."""
        return self.current_week

    def get_upcoming_matches(self, week: int | None = None) -> pd.DataFrame:
        """
        Get upcoming matches for prediction.

        Args:
            week: Specific week to get matches for. If None, uses current week.

        Returns:
            DataFrame with matches ready for prediction.
        """
        target_week = week if week is not None else self.current_week

        # Get matches for the target week that haven't been predicted yet
        upcoming = self.match_calendar[
            (self.match_calendar["simulation_week"] == target_week)
            & (self.match_calendar["prediction_made"] is False)
        ].copy()

        logger.info(f"Found {len(upcoming)} upcoming matches for week {target_week}")
        return upcoming

    def get_matches_for_week(self, week: int) -> pd.DataFrame:
        """Get all matches for a specific week."""
        matches = self.match_calendar[self.match_calendar["simulation_week"] == week].copy()

        return matches

    def mark_predictions_made(self, match_ids: list[int]) -> None:
        """Mark matches as having predictions made."""
        mask = self.match_calendar.index.isin(match_ids)
        self.match_calendar.loc[mask, "prediction_made"] = True
        self.match_calendar.loc[mask, "simulation_status"] = "predicted"

        logger.info(f"Marked {len(match_ids)} matches as predicted")

    def reveal_results(self, match_ids: list[int]) -> pd.DataFrame:
        """
        Reveal actual results for completed matches.

        Args:
            match_ids: List of match indices to reveal results for.

        Returns:
            DataFrame with actual results revealed.
        """
        mask = self.match_calendar.index.isin(match_ids)
        completed_matches = self.match_calendar.loc[mask].copy()

        # Mark as completed
        self.match_calendar.loc[mask, "actual_result_revealed"] = True
        self.match_calendar.loc[mask, "simulation_status"] = "completed"

        logger.info(f"Revealed results for {len(match_ids)} matches")
        return completed_matches

    def advance_week(self) -> bool:
        """
        Advance to the next simulation week.

        Returns:
            True if advanced successfully, False if no more weeks.
        """
        if self.current_week >= self.get_total_weeks():
            logger.warning("Already at final week")
            return False

        self.current_week += 1
        logger.info(f"Advanced to week {self.current_week}")
        return True

    def get_simulation_progress(self) -> dict[str, Any]:
        """Get current simulation progress statistics."""
        total_matches = len(self.match_calendar)
        predicted_matches = (self.match_calendar["prediction_made"] is True).sum()
        completed_matches = (self.match_calendar["actual_result_revealed"] is True).sum()

        progress = {
            "current_week": self.current_week,
            "total_weeks": self.get_total_weeks(),
            "week_progress": f"{self.current_week}/{self.get_total_weeks()}",
            "total_matches": total_matches,
            "predicted_matches": int(predicted_matches),
            "completed_matches": int(completed_matches),
            "prediction_progress": f"{predicted_matches}/{total_matches}",
            "completion_progress": f"{completed_matches}/{total_matches}",
            "prediction_percentage": float(predicted_matches / total_matches * 100),
            "completion_percentage": float(completed_matches / total_matches * 100),
        }

        return progress

    def get_week_summary(self, week: int | None = None) -> dict[str, Any]:
        """Get summary for a specific week."""
        target_week = week if week is not None else self.current_week
        week_matches = self.get_matches_for_week(target_week)

        if len(week_matches) == 0:
            return {"week": target_week, "matches": 0, "error": "No matches found"}

        summary = {
            "week": target_week,
            "matches": len(week_matches),
            "date_range": {
                "start": week_matches["Date"].min().strftime("%Y-%m-%d"),
                "end": week_matches["Date"].max().strftime("%Y-%m-%d"),
            },
            "predictions_made": int((week_matches["prediction_made"] is True).sum()),
            "results_revealed": int((week_matches["actual_result_revealed"] is True).sum()),
            "status_counts": week_matches["simulation_status"].value_counts().to_dict(),
        }

        return summary

    def reset_simulation(self) -> None:
        """Reset simulation to beginning."""
        self.current_week = 1
        self.match_calendar["prediction_made"] = False
        self.match_calendar["actual_result_revealed"] = False
        self.match_calendar["simulation_status"] = "pending"

        logger.info("Simulation reset to week 1")

    def save_state(self, output_path: str | None = None) -> str:
        """Save current scheduler state."""
        if output_path is None:
            output_path = self.calendar_path

        self.match_calendar.to_parquet(output_path, index=True)
        logger.info(f"Scheduler state saved to {output_path}")
        return str(output_path)

    def is_simulation_complete(self) -> bool:
        """Check if simulation is complete."""
        return (self.match_calendar["actual_result_revealed"] is True).all()

    def get_completed_matches_since_week(self, since_week: int) -> pd.DataFrame:
        """Get all completed matches since a specific week."""
        completed = self.match_calendar[
            (self.match_calendar["simulation_week"] >= since_week)
            & (self.match_calendar["actual_result_revealed"] is True)
        ].copy()

        return completed
