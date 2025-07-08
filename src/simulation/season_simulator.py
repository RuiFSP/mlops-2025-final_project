"""
Season Simulator - Core simulation engine for Premier League matches.

Orchestrates the week-by-week simulation of matches, integrating with all
other components to provide a realistic testing environment for MLOps workflows.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .match_scheduler import MatchScheduler
from .odds_generator import OddsGenerator
from .retraining_orchestrator import RetrainingOrchestrator

logger = logging.getLogger(__name__)


class SeasonSimulator:
    """
    Core simulation engine that orchestrates a complete season simulation.

    Provides week-by-week match simulation with realistic timing, odds generation,
    predictions, and automated retraining triggers.
    """

    def __init__(
        self,
        simulation_data_path: str,
        match_calendar_path: str,
        model_path: str,
        output_dir: str = "data/simulation_output",
        retraining_threshold: float = 0.05,
        retraining_frequency: int = 5,  # weeks
    ):
        """
        Initialize the season simulator.

        Args:
            simulation_data_path: Path to the 2023-24 season data
            match_calendar_path: Path to the match calendar
            model_path: Path to the trained model
            output_dir: Directory to save simulation outputs
            retraining_threshold: Performance drop threshold for retraining
            retraining_frequency: Number of weeks between automatic retraining checks
        """
        self.simulation_data_path = simulation_data_path
        self.match_calendar_path = match_calendar_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.retraining_threshold = retraining_threshold
        self.retraining_frequency = retraining_frequency

        # Initialize components
        self.scheduler = MatchScheduler(match_calendar_path)
        self.odds_generator = OddsGenerator(simulation_data_path)
        self.retraining_orchestrator = RetrainingOrchestrator(
            model_path=model_path,
            threshold=retraining_threshold,
            frequency=retraining_frequency,
        )

        # Load simulation data
        self.simulation_data = pd.read_parquet(simulation_data_path)

        # Initialize state
        self.current_week = 1
        self.max_week = self.scheduler.get_total_weeks()
        self.simulation_history: list[dict] = []
        self.predictions_history: list[dict] = []
        self.performance_history: list[dict] = []

        # Create output directories
        self._setup_output_directories()

        logger.info(f"Season simulator initialized for {self.max_week} weeks")

    def _setup_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = [
            self.output_dir,
            self.output_dir / "predictions",
            self.output_dir / "results",
            self.output_dir / "performance",
            self.output_dir / "retraining",
            self.output_dir / "monitoring",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_simulation_state(self) -> dict:
        """Get current simulation state."""
        upcoming_matches = self.scheduler.get_matches_for_week(self.current_week)
        weeks_remaining = self.max_week - self.current_week + 1

        return {
            "current_week": self.current_week,
            "max_week": self.max_week,
            "weeks_remaining": weeks_remaining,
            "upcoming_matches": len(upcoming_matches) if upcoming_matches is not None else 0,
            "total_predictions": len(self.predictions_history),
            "total_matches_simulated": sum(
                len(week_data.get("results", [])) for week_data in self.simulation_history
            ),
            "retraining_count": self.retraining_orchestrator.get_retraining_count(),
            "last_performance": (
                self.performance_history[-1] if self.performance_history else None
            ),
        }

    def simulate_week(self, week_number: int | None = None) -> dict:
        """
        Simulate a single week of matches.

        Args:
            week_number: Specific week to simulate (defaults to current week)

        Returns:
            Dictionary containing week simulation results
        """
        if week_number is None:
            week_number = self.current_week

        if week_number > self.max_week:
            raise ValueError(f"Week {week_number} exceeds maximum week {self.max_week}")

        logger.info(f"Simulating week {week_number}")

        # Get matches for the week
        week_matches = self.scheduler.get_matches_for_week(week_number)
        if week_matches is None or len(week_matches) == 0:
            logger.warning(f"No matches found for week {week_number}")
            return {
                "week": week_number,
                "matches": [],
                "predictions": [],
                "results": [],
            }

        # Generate predictions for the week
        week_predictions = self._generate_week_predictions(week_matches)

        # Simulate match results (reveal actual outcomes)
        week_results = self._simulate_match_results(week_matches)

        # Calculate performance metrics
        week_performance = self._calculate_week_performance(week_predictions, week_results)

        # Check for retraining triggers
        retraining_triggered = self.retraining_orchestrator.check_retraining_trigger(
            week_number, week_performance
        )

        # Save week data
        week_data = {
            "week": week_number,
            "timestamp": datetime.now().isoformat(),
            "matches": week_matches.to_dict("records"),
            "predictions": week_predictions,
            "results": week_results,
            "performance": week_performance,
            "retraining_triggered": retraining_triggered,
        }

        # Update histories
        self.simulation_history.append(week_data)
        self.predictions_history.extend(week_predictions)
        self.performance_history.append(week_performance)

        # Save to files
        self._save_week_data(week_data)

        # Update current week if simulating sequentially
        if week_number == self.current_week:
            self.current_week += 1

        logger.info(f"Week {week_number} simulation completed")
        return week_data

    def _generate_week_predictions(self, week_matches: pd.DataFrame) -> list[dict]:
        """Generate predictions for matches in a week."""
        predictions = []

        for _, match in week_matches.iterrows():
            # Generate odds for the match
            home_odds, draw_odds, away_odds = self.odds_generator.generate_odds(
                match["HomeTeam"], match["AwayTeam"]
            )

            # Create odds dictionary
            odds = {"home_win": home_odds, "draw": draw_odds, "away_win": away_odds}

            # Create prediction based on odds (simple implementation)
            # In a real scenario, this would use the actual model
            home_prob = 1 / odds["home_win"]
            draw_prob = 1 / odds["draw"]
            away_prob = 1 / odds["away_win"]

            # Normalize probabilities
            total_prob = home_prob + draw_prob + away_prob
            home_prob /= total_prob
            draw_prob /= total_prob
            away_prob /= total_prob

            # Predict most likely outcome
            if home_prob > draw_prob and home_prob > away_prob:
                predicted_result = "H"
                confidence = home_prob
            elif away_prob > home_prob and away_prob > draw_prob:
                predicted_result = "A"
                confidence = away_prob
            else:
                predicted_result = "D"
                confidence = draw_prob

            prediction = {
                "match_id": f"{match['HomeTeam']}_vs_{match['AwayTeam']}_{match['Date']}",
                "home_team": match["HomeTeam"],
                "away_team": match["AwayTeam"],
                "date": match["Date"],
                "week": match["simulation_week"],
                "predicted_result": predicted_result,
                "confidence": confidence,
                "probabilities": {
                    "home_win": home_prob,
                    "draw": draw_prob,
                    "away_win": away_prob,
                },
                "odds": odds,
                "timestamp": datetime.now().isoformat(),
            }

            predictions.append(prediction)

        return predictions

    def _simulate_match_results(self, week_matches: pd.DataFrame) -> list[dict]:
        """Simulate match results by revealing actual outcomes from data."""
        results = []

        for _, match in week_matches.iterrows():
            # Find the actual match result in simulation data
            actual_match = self.simulation_data[
                (self.simulation_data["HomeTeam"] == match["HomeTeam"])
                & (self.simulation_data["AwayTeam"] == match["AwayTeam"])
                & (
                    pd.to_datetime(self.simulation_data["Date"]).dt.date
                    == pd.to_datetime(match["Date"]).date()
                )
            ]

            if len(actual_match) > 0:
                actual = actual_match.iloc[0]
                result = {
                    "match_id": f"{match['HomeTeam']}_vs_{match['AwayTeam']}_{match['Date']}",
                    "home_team": match["HomeTeam"],
                    "away_team": match["AwayTeam"],
                    "date": match["Date"],
                    "week": match["simulation_week"],
                    "home_goals": actual["FTHG"],
                    "away_goals": actual["FTAG"],
                    "actual_result": actual["FTR"],
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Fallback if match not found
                logger.warning(
                    f"Match not found in data: {match['HomeTeam']} vs {match['AwayTeam']}"
                )
                result = {
                    "match_id": f"{match['HomeTeam']}_vs_{match['AwayTeam']}_{match['Date']}",
                    "home_team": match["HomeTeam"],
                    "away_team": match["AwayTeam"],
                    "date": match["Date"],
                    "week": match["simulation_week"],
                    "home_goals": None,
                    "away_goals": None,
                    "actual_result": None,
                    "timestamp": datetime.now().isoformat(),
                }

            results.append(result)

        return results

    def _calculate_week_performance(self, predictions: list[dict], results: list[dict]) -> dict:
        """Calculate performance metrics for the week."""
        if not predictions or not results:
            return {"accuracy": 0.0, "total_matches": 0, "correct_predictions": 0}

        correct_predictions = 0
        total_matches = len(predictions)

        # Create lookup for results
        results_lookup = {r["match_id"]: r for r in results}

        for pred in predictions:
            result = results_lookup.get(pred["match_id"])
            if result and result["actual_result"] is not None:
                if pred["predicted_result"] == result["actual_result"]:
                    correct_predictions += 1

        accuracy = correct_predictions / total_matches if total_matches > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total_matches": total_matches,
            "correct_predictions": correct_predictions,
            "week": predictions[0]["week"] if predictions else None,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_week_data(self, week_data: dict) -> None:
        """Save week data to files."""
        week_num = week_data["week"]

        # Save predictions
        predictions_df = pd.DataFrame(week_data["predictions"])
        if not predictions_df.empty:
            predictions_path = (
                self.output_dir / "predictions" / f"week_{week_num:02d}_predictions.parquet"
            )
            predictions_df.to_parquet(predictions_path)

        # Save results
        results_df = pd.DataFrame(week_data["results"])
        if not results_df.empty:
            results_path = self.output_dir / "results" / f"week_{week_num:02d}_results.parquet"
            results_df.to_parquet(results_path)

        # Save performance
        performance_path = self.output_dir / "performance" / f"week_{week_num:02d}_performance.json"
        import json

        with open(performance_path, "w") as f:
            json.dump(week_data["performance"], f, indent=2)

    def simulate_season(self, start_week: int = 1, end_week: int | None = None) -> dict:
        """
        Simulate multiple weeks or the entire season.

        Args:
            start_week: First week to simulate
            end_week: Last week to simulate (defaults to max_week)

        Returns:
            Summary of season simulation
        """
        if end_week is None:
            end_week = self.max_week

        logger.info(f"Starting season simulation from week {start_week} to {end_week}")

        simulation_start = datetime.now()
        weeks_simulated = []

        for week in range(start_week, end_week + 1):
            try:
                week_data = self.simulate_week(week)
                weeks_simulated.append(week_data)
                logger.info(f"Completed week {week}/{end_week}")
            except Exception as e:
                logger.error(f"Error simulating week {week}: {str(e)}")
                break

        simulation_end = datetime.now()
        simulation_duration = (simulation_end - simulation_start).total_seconds()

        # Calculate overall performance
        overall_performance = self._calculate_overall_performance()

        # Create season summary
        season_summary = {
            "simulation_period": f"Weeks {start_week}-{end_week}",
            "weeks_simulated": len(weeks_simulated),
            "simulation_duration_seconds": simulation_duration,
            "overall_performance": overall_performance,
            "retraining_events": self.retraining_orchestrator.get_retraining_count(),
            "total_predictions": len(self.predictions_history),
            "simulation_completed": datetime.now().isoformat(),
        }

        # Save season summary
        summary_path = self.output_dir / "season_summary.json"
        import json

        with open(summary_path, "w") as f:
            json.dump(season_summary, f, indent=2)

        logger.info(f"Season simulation completed: {season_summary}")
        return season_summary

    def _calculate_overall_performance(self) -> dict:
        """Calculate overall performance across all simulated weeks."""
        if not self.performance_history:
            return {"overall_accuracy": 0.0, "total_matches": 0, "total_correct": 0}

        total_matches = sum(p["total_matches"] for p in self.performance_history)
        total_correct = sum(p["correct_predictions"] for p in self.performance_history)
        overall_accuracy = total_correct / total_matches if total_matches > 0 else 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "total_matches": total_matches,
            "total_correct": total_correct,
            "weeks_analyzed": len(self.performance_history),
        }

    def get_week_summary(self, week_number: int) -> dict | None:
        """Get summary of a specific week's simulation."""
        for week_data in self.simulation_history:
            if week_data["week"] == week_number:
                return week_data
        return None

    def export_simulation_data(self) -> dict[str, str]:
        """Export all simulation data to files."""
        export_paths = {}

        # Export predictions
        if self.predictions_history:
            predictions_df = pd.DataFrame(self.predictions_history)
            predictions_path = self.output_dir / "all_predictions.parquet"
            predictions_df.to_parquet(predictions_path)
            export_paths["predictions"] = str(predictions_path)

        # Export performance history
        if self.performance_history:
            performance_df = pd.DataFrame(self.performance_history)
            performance_path = self.output_dir / "performance_history.parquet"
            performance_df.to_parquet(performance_path)
            export_paths["performance"] = str(performance_path)

        # Export simulation history
        if self.simulation_history:
            import json

            history_path = self.output_dir / "simulation_history.json"
            with open(history_path, "w") as f:
                json.dump(self.simulation_history, f, indent=2)
            export_paths["history"] = str(history_path)

        logger.info(f"Simulation data exported to: {export_paths}")
        return export_paths
