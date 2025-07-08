#!/usr/bin/env python3
"""
Collect real Premier League data from football-data.co.uk.

This script collects historical Premier League match data from football-data.co.uk
for the past 8 seasons and saves it in a standardized format.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.data_collector import DataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to collect Premier League data."""
    logger.info("Premier League Data Collection")
    logger.info("=" * 40)
    logger.info("Collecting data from football-data.co.uk for the past 8 seasons...")

    try:
        # Create collector
        collector = DataCollector()

        # Collect data
        df = collector.collect_all_data()

        if df is not None and len(df) > 0:
            # Save data
            output_dir = Path("data/real_data")
            collector.save_data(df, output_dir)

            logger.info("Data Collection Complete!")
            logger.info(f"Successfully collected {len(df)} matches")
            logger.info(f"Data saved to {output_dir}")

            # Log detailed statistics
            logger.info("Data Statistics:")
            logger.info(f"  • Total matches: {len(df)}")
            if "date" in df.columns:
                logger.info(
                    f"  • Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                )
            if "season" in df.columns:
                logger.info(f"  • Seasons: {', '.join(sorted(df['season'].unique()))}")
            if "home_team" in df.columns and "away_team" in df.columns:
                logger.info(
                    f"  • Teams: {len(set(df['home_team'].unique()) | set(df['away_team'].unique()))}"
                )

            logger.info(f"  • Columns: {', '.join(df.columns)}")

            # Log sample data
            logger.info("Sample Data:")
            logger.info(df.head(3).to_string())

            return df
        else:
            logger.error("Failed to collect data")
            return None

    except Exception as e:
        logger.error(f"Failed to collect data: {e}")
        raise


if __name__ == "__main__":
    main()
