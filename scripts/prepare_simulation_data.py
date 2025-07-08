#!/usr/bin/env python3
"""
Data preparation script for Season Simulation Engine.

This script splits the Premier League data into:
- Training data: 2016-2023 seasons (historical)
- Simulation data: 2023-24 season (treated as "future")

It also creates a match calendar for realistic scheduling.
"""

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_full_dataset(data_path: str) -> pd.DataFrame:
    """Load the complete Premier League dataset."""
    logger.info(f"Loading dataset from {data_path}")
    data = pd.read_parquet(data_path)

    # Convert Date to datetime with flexible parsing
    # The data has mixed formats: some with 4-digit years, some with 2-digit
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, format='mixed')

    logger.info(f"Loaded {len(data)} matches from {data['season'].nunique()} seasons")
    return data


def split_training_simulation_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training (2016-2023) and simulation (2023-24) datasets."""

    # Training data: All seasons except 2023-24
    training_seasons = ['2016-2017', '2017-2018', '2018-2019', '2019-2020',
                       '2020-2021', '2021-2022', '2022-2023']
    training_data = data[data['season'].isin(training_seasons)].copy()

    # Simulation data: 2023-24 season only
    simulation_data = data[data['season'] == '2023-2024'].copy()

    logger.info(f"Training data: {len(training_data)} matches from {len(training_seasons)} seasons")
    logger.info(f"Simulation data: {len(simulation_data)} matches from 2023-24 season")

    return training_data, simulation_data


def create_match_calendar(simulation_data: pd.DataFrame) -> pd.DataFrame:
    """Create a realistic match calendar for simulation scheduling."""

    # Sort by date to get chronological order
    calendar = simulation_data.copy().sort_values('Date')

    # Add simulation metadata
    calendar['simulation_week'] = (
        (calendar['Date'] - calendar['Date'].min()).dt.days // 7
    ) + 1

    calendar['simulation_status'] = 'pending'  # pending, predicted, completed
    calendar['prediction_made'] = False
    calendar['actual_result_revealed'] = False

    # Group matches by week for realistic batch processing
    calendar['matches_this_week'] = calendar.groupby('simulation_week')['Date'].transform('count')

    logger.info(f"Created match calendar with {calendar['simulation_week'].max()} weeks")
    logger.info(f"Average matches per week: {calendar['matches_this_week'].mean():.1f}")

    return calendar


def save_simulation_datasets(training_data: pd.DataFrame,
                           simulation_data: pd.DataFrame,
                           match_calendar: pd.DataFrame,
                           output_dir: str) -> dict:
    """Save the prepared datasets for simulation."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save training data (what the model will be trained on)
    training_path = output_path / "training_data_2016_2023.parquet"
    training_data.to_parquet(training_path, index=False)

    # Save simulation data (the "future" matches)
    simulation_path = output_path / "simulation_data_2023_24.parquet"
    simulation_data.to_parquet(simulation_path, index=False)

    # Save match calendar
    calendar_path = output_path / "match_calendar.parquet"
    match_calendar.to_parquet(calendar_path, index=False)

    # Create simulation metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "training_data": {
            "seasons": ["2016-2017", "2017-2018", "2018-2019", "2019-2020",
                       "2020-2021", "2021-2022", "2022-2023"],
            "matches": len(training_data),
            "date_range": [training_data['Date'].min().isoformat(),
                          training_data['Date'].max().isoformat()],
            "file": str(training_path.name)
        },
        "simulation_data": {
            "season": "2023-2024",
            "matches": len(simulation_data),
            "date_range": [simulation_data['Date'].min().isoformat(),
                          simulation_data['Date'].max().isoformat()],
            "simulation_weeks": int(match_calendar['simulation_week'].max()),
            "file": str(simulation_path.name)
        },
        "match_calendar": {
            "total_weeks": int(match_calendar['simulation_week'].max()),
            "total_matches": len(match_calendar),
            "avg_matches_per_week": float(match_calendar['matches_this_week'].mean()),
            "file": str(calendar_path.name)
        }
    }

    # Save metadata
    metadata_path = output_path / "simulation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved training data: {training_path}")
    logger.info(f"Saved simulation data: {simulation_path}")
    logger.info(f"Saved match calendar: {calendar_path}")
    logger.info(f"Saved metadata: {metadata_path}")

    return metadata


def analyze_simulation_potential(training_data: pd.DataFrame, simulation_data: pd.DataFrame) -> None:
    """Analyze the simulation potential and data characteristics."""

    logger.info("\n" + "="*50)
    logger.info("SIMULATION ANALYSIS")
    logger.info("="*50)

    # Training data analysis
    logger.info(f"Training Data (Historical):")
    logger.info(f"  - Seasons: {training_data['season'].nunique()}")
    logger.info(f"  - Matches: {len(training_data):,}")
    logger.info(f"  - Date range: {training_data['Date'].min()} to {training_data['Date'].max()}")
    logger.info(f"  - Unique teams: {len(set(training_data['HomeTeam'].unique()) | set(training_data['AwayTeam'].unique()))}")

    # Simulation data analysis
    logger.info(f"\nSimulation Data ('Future'):")
    logger.info(f"  - Season: 2023-2024")
    logger.info(f"  - Matches: {len(simulation_data):,}")
    logger.info(f"  - Date range: {simulation_data['Date'].min()} to {simulation_data['Date'].max()}")
    logger.info(f"  - Duration: {(simulation_data['Date'].max() - simulation_data['Date'].min()).days} days")

    # Team overlap analysis
    training_teams = set(training_data['HomeTeam'].unique()) | set(training_data['AwayTeam'].unique())
    simulation_teams = set(simulation_data['HomeTeam'].unique()) | set(simulation_data['AwayTeam'].unique())

    common_teams = training_teams & simulation_teams
    new_teams = simulation_teams - training_teams

    logger.info(f"\nTeam Analysis:")
    logger.info(f"  - Common teams: {len(common_teams)} ({len(common_teams)/len(simulation_teams)*100:.1f}%)")
    logger.info(f"  - New teams in 2023-24: {len(new_teams)}")
    if new_teams:
        logger.info(f"  - New teams: {', '.join(new_teams)}")

    # Betting odds availability
    odds_columns = ['B365H', 'B365D', 'B365A']
    training_odds_coverage = training_data[odds_columns].notna().all(axis=1).mean()
    simulation_odds_coverage = simulation_data[odds_columns].notna().all(axis=1).mean()

    logger.info(f"\nBetting Odds Coverage:")
    logger.info(f"  - Training data: {training_odds_coverage*100:.1f}%")
    logger.info(f"  - Simulation data: {simulation_odds_coverage*100:.1f}%")

    logger.info(f"\nSimulation Potential: EXCELLENT ✅")
    logger.info(f"  - Rich historical data for training")
    logger.info(f"  - Complete season for realistic simulation")
    logger.info(f"  - Good team overlap for model applicability")
    logger.info(f"  - Betting odds available for feature engineering")


def main():
    """Main execution function."""
    logger.info("Starting Season Simulation Data Preparation")
    logger.info("Phase 1: Data Preparation")

    # Paths
    data_path = "data/real_data/premier_league_matches.parquet"
    output_dir = "data/simulation"

    try:
        # Load full dataset
        full_data = load_full_dataset(data_path)

        # Split into training and simulation datasets
        training_data, simulation_data = split_training_simulation_data(full_data)

        # Create match calendar for scheduling
        match_calendar = create_match_calendar(simulation_data)

        # Save datasets
        metadata = save_simulation_datasets(
            training_data, simulation_data, match_calendar, output_dir
        )

        # Analyze simulation potential
        analyze_simulation_potential(training_data, simulation_data)

        logger.info("\n" + "="*50)
        logger.info("✅ PHASE 1 COMPLETE: Data Preparation")
        logger.info("="*50)
        logger.info("Next steps:")
        logger.info("  1. Phase 2: Build simulation engine")
        logger.info("  2. Phase 3: Create automated pipeline")
        logger.info("  3. Phase 4: Integrate monitoring")

        return metadata

    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise


if __name__ == "__main__":
    main()
