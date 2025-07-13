"""
Weekly Batch Processor for Premier League Data
Processes historical Premier League matches in weekly batches for realistic MLOps simulation
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WeeklyBatch:
    """Represents a weekly batch of Premier League matches"""

    week_start: datetime
    week_end: datetime
    matches: list[dict[str, Any]]
    week_id: str
    season: str
    total_matches: int


class WeeklyBatchProcessor:
    """
    Processes Premier League data in weekly batches for realistic MLOps workflow simulation
    """

    def __init__(self, data_path: str = "data/real_data/premier_league_matches.parquet"):
        # Resolve path relative to project root
        if not Path(data_path).is_absolute():
            # Find project root (look for pyproject.toml or similar)
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / "pyproject.toml").exists():
                    self.data_path = str(current_dir / data_path)
                    break
                current_dir = current_dir.parent
            else:
                # Fallback to relative path
                self.data_path = str(Path(__file__).parent.parent.parent / data_path)
        else:
            self.data_path = data_path

        self.df = None
        self.load_data()

    def load_data(self) -> None:
        """Load and prepare the Premier League dataset"""
        try:
            self.df = pd.read_parquet(self.data_path)
            # Parse dates properly
            self.df["Date"] = pd.to_datetime(self.df["Date"], format="%d/%m/%Y", errors="coerce")
            # Add week periods
            self.df["Week"] = self.df.Date.dt.to_period("W")
            # Sort by date
            self.df = self.df.sort_values("Date")

            logger.info(f"✅ Loaded {len(self.df)} matches from {self.df.Date.min()} to {self.df.Date.max()}")
            logger.info(
                f"📅 Data spans {len(self.df.groupby('Week'))} weeks across {len(self.df.groupby('season'))} seasons"
            )

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def get_weekly_batches(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_weeks: int | None = None,
    ) -> list[WeeklyBatch]:
        """
        Get weekly batches of matches for processing

        Args:
            start_date: Start date for batch processing (default: first match)
            end_date: End date for batch processing (default: last match)
            max_weeks: Maximum number of weeks to process (default: all)

        Returns:
            List of WeeklyBatch objects
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        # Set default date range
        if start_date is None:
            start_date = self.df.Date.min()
        if end_date is None:
            end_date = self.df.Date.max()

        # Filter data by date range
        mask = (self.df.Date >= start_date) & (self.df.Date <= end_date)
        filtered_df = self.df[mask]

        # Group by week
        weekly_groups = filtered_df.groupby("Week")

        batches = []
        for week_period, week_data in weekly_groups:
            week_start = week_period.start_time
            week_end = week_period.end_time

            # Convert matches to dict format
            matches = []
            for _, match in week_data.iterrows():
                match_dict = {
                    "date": match["Date"].isoformat(),
                    "home_team": match["HomeTeam"],
                    "away_team": match["AwayTeam"],
                    "result": match["FTR"],  # H, D, A
                    "home_goals": int(match["FTHG"]),
                    "away_goals": int(match["FTAG"]),
                    "season": match.get("season", ""),
                    # Betting odds
                    "home_odds": float(match.get("B365H", 2.0)),
                    "draw_odds": float(match.get("B365D", 3.5)),
                    "away_odds": float(match.get("B365A", 3.0)),
                    # Match statistics for features
                    "home_shots": int(match.get("HS", 10)),
                    "away_shots": int(match.get("AS", 8)),
                    "home_shots_target": int(match.get("HST", 4)),
                    "away_shots_target": int(match.get("AST", 3)),
                    "home_corners": int(match.get("HC", 5)),
                    "away_corners": int(match.get("AC", 4)),
                    "home_fouls": int(match.get("HF", 12)),
                    "away_fouls": int(match.get("AF", 10)),
                    "home_yellows": int(match.get("HY", 2)),
                    "away_yellows": int(match.get("AY", 1)),
                    "home_reds": int(match.get("HR", 0)),
                    "away_reds": int(match.get("AR", 0)),
                }
                matches.append(match_dict)

            # Create batch
            batch = WeeklyBatch(
                week_start=week_start,
                week_end=week_end,
                matches=matches,
                week_id=week_period.strftime("%Y-W%U"),
                season=matches[0]["season"] if matches else "",
                total_matches=len(matches),
            )
            batches.append(batch)

            # Limit number of weeks if specified
            if max_weeks and len(batches) >= max_weeks:
                break

        logger.info(f"📦 Created {len(batches)} weekly batches")
        return batches

    def get_recent_weeks(self, num_weeks: int = 4) -> list[WeeklyBatch]:
        """Get the most recent N weeks of data"""
        end_date = self.df.Date.max()
        start_date = end_date - timedelta(weeks=num_weeks)
        return self.get_weekly_batches(start_date=start_date, end_date=end_date)

    def get_season_batches(self, season: str) -> list[WeeklyBatch]:
        """Get all weekly batches for a specific season"""
        season_data = self.df[self.df["season"] == season]
        if season_data.empty:
            logger.warning(f"No data found for season {season}")
            return []

        start_date = season_data.Date.min()
        end_date = season_data.Date.max()
        return self.get_weekly_batches(start_date=start_date, end_date=end_date)

    def simulate_live_processing(self, weeks_back: int = 12, weeks_ahead: int = 4) -> dict[str, list[WeeklyBatch]]:
        """
        Simulate live processing scenario with historical and upcoming data

        Args:
            weeks_back: Number of weeks of historical data to include
            weeks_ahead: Number of weeks to simulate as "upcoming"

        Returns:
            Dict with 'historical' and 'upcoming' batch lists
        """
        # Get a realistic "current" date (not the very latest to have some future data)
        latest_date = self.df.Date.max()
        current_date = latest_date - timedelta(weeks=weeks_ahead)

        # Historical data (for training/validation)
        historical_start = current_date - timedelta(weeks=weeks_back)
        historical_batches = self.get_weekly_batches(start_date=historical_start, end_date=current_date)

        # Upcoming data (simulate as "future" matches)
        upcoming_batches = self.get_weekly_batches(start_date=current_date, end_date=latest_date)

        return {
            "historical": historical_batches,
            "upcoming": upcoming_batches,
            "current_date": current_date.isoformat(),
            "simulation_summary": {
                "historical_weeks": len(historical_batches),
                "upcoming_weeks": len(upcoming_batches),
                "total_historical_matches": sum(b.total_matches for b in historical_batches),
                "total_upcoming_matches": sum(b.total_matches for b in upcoming_batches),
            },
        }

    def get_batch_statistics(self, batches: list[WeeklyBatch]) -> dict[str, Any]:
        """Get statistics about a list of batches"""
        if not batches:
            return {}

        total_matches = sum(b.total_matches for b in batches)
        matches_per_week = [b.total_matches for b in batches]

        # Result distribution across all batches
        all_results = []
        for batch in batches:
            all_results.extend([match["result"] for match in batch.matches])

        result_counts = pd.Series(all_results).value_counts()

        return {
            "total_batches": len(batches),
            "total_matches": total_matches,
            "avg_matches_per_week": sum(matches_per_week) / len(matches_per_week),
            "min_matches_per_week": min(matches_per_week),
            "max_matches_per_week": max(matches_per_week),
            "date_range": {
                "start": batches[0].week_start.isoformat(),
                "end": batches[-1].week_end.isoformat(),
            },
            "result_distribution": {
                "home_wins": int(result_counts.get("H", 0)),
                "draws": int(result_counts.get("D", 0)),
                "away_wins": int(result_counts.get("A", 0)),
                "home_win_rate": result_counts.get("H", 0) / len(all_results) if all_results else 0,
            },
        }

    def save_batch_to_file(self, batch: WeeklyBatch, output_dir: str = "data/batches") -> str:
        """Save a weekly batch to a JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"week_{batch.week_id}_{batch.total_matches}matches.json"
        filepath = output_path / filename

        # Convert batch to serializable format
        batch_data = {
            "week_id": batch.week_id,
            "week_start": batch.week_start.isoformat(),
            "week_end": batch.week_end.isoformat(),
            "season": batch.season,
            "total_matches": batch.total_matches,
            "matches": batch.matches,
        }

        with open(filepath, "w") as f:
            json.dump(batch_data, f, indent=2, default=str)

        logger.info(f"💾 Saved batch {batch.week_id} to {filepath}")
        return str(filepath)

    def load_batch_from_file(self, filepath: str) -> WeeklyBatch:
        """Load a weekly batch from a JSON file"""
        with open(filepath) as f:
            batch_data = json.load(f)

        return WeeklyBatch(
            week_start=datetime.fromisoformat(batch_data["week_start"]),
            week_end=datetime.fromisoformat(batch_data["week_end"]),
            matches=batch_data["matches"],
            week_id=batch_data["week_id"],
            season=batch_data["season"],
            total_matches=batch_data["total_matches"],
        )


def demo_weekly_processing():
    """Demonstrate the weekly batch processing system"""
    processor = WeeklyBatchProcessor()

    print("🏈 Premier League Weekly Batch Processing Demo")
    print("=" * 60)

    # Get recent weeks
    recent_batches = processor.get_recent_weeks(4)

    print("\n📅 Recent 4 weeks:")
    for batch in recent_batches:
        print(
            f"  Week {batch.week_id}: {batch.total_matches} matches ({batch.week_start.strftime('%Y-%m-%d')} to {batch.week_end.strftime('%Y-%m-%d')})"
        )

    # Get batch statistics
    stats = processor.get_batch_statistics(recent_batches)
    print("\n📊 Recent weeks statistics:")
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Avg matches/week: {stats['avg_matches_per_week']:.1f}")
    print(f"  Home win rate: {stats['result_distribution']['home_win_rate']:.2%}")

    # Simulate live processing
    simulation = processor.simulate_live_processing()
    print("\n🔄 Live processing simulation:")
    print(f"  Historical weeks: {simulation['simulation_summary']['historical_weeks']}")
    print(f"  Upcoming weeks: {simulation['simulation_summary']['upcoming_weeks']}")
    print(f"  Current date: {simulation['current_date']}")


if __name__ == "__main__":
    demo_weekly_processing()
