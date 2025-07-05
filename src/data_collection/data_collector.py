"""Data collection module for Premier League data.

This module provides data collection methods from football-data.co.uk
for Premier League match data.
"""

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects Premier League data from football-data.co.uk."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )

    def collect_from_football_data_co_uk(self) -> Optional[pd.DataFrame]:
        """Collect data from football-data.co.uk."""
        try:
            # Historical data from football-data.co.uk - 8 recent seasons
            seasons = ["2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617"]
            all_data = []

            for season in seasons:
                try:
                    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
                    logger.info(
                        f"Fetching data for season 20{season[:2]}-20{season[2:]}..."
                    )

                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()

                    # Clean response text - remove non-printable chars
                    clean_text = "".join(
                        char
                        for char in response.text
                        if char.isprintable() or char in "\r\n\t"
                    )
                    csv_data = StringIO(clean_text)

                    df = pd.read_csv(csv_data)
                    df["season"] = f"20{season[:2]}-20{season[2:]}"
                    all_data.append(df)

                    logger.info(
                        f"Successfully loaded {len(df)} matches from season "
                        f"20{season[:2]}-20{season[2:]}"
                    )
                    time.sleep(1)  # Be respectful to the server

                except Exception as e:
                    logger.warning(
                        f"Failed to load season 20{season[:2]}-20{season[2:]}: {e}"
                    )
                    continue

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(
                    f"Combined {len(combined_df)} matches from {len(all_data)} seasons"
                )
                return combined_df
            else:
                logger.error("No data could be loaded from football-data.co.uk")
                return None

        except Exception as e:
            logger.error(f"Error collecting from football-data.co.uk: {e}")
            return None

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data from football-data.co.uk to common format."""
        try:
            # Handle both 4-digit and 2-digit year formats
            try:
                # Try 4-digit year format first
                dates = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            except ValueError:
                try:
                    # Try 2-digit year format
                    dates = pd.to_datetime(df["Date"], format="%d/%m/%y")
                except ValueError:
                    # Fall back to automatic parsing
                    dates = pd.to_datetime(df["Date"], dayfirst=True)

            standardized = pd.DataFrame(
                {
                    "date": dates,
                    "home_team": df["HomeTeam"],
                    "away_team": df["AwayTeam"],
                    "home_score": df["FTHG"],  # Full Time Home Goals
                    "away_score": df["FTAG"],  # Full Time Away Goals
                    "season": df["season"],
                }
            )

            # Add additional features available in this dataset
            if "B365H" in df.columns:  # Bet365 home odds
                standardized["home_odds"] = df["B365H"]
            if "B365D" in df.columns:  # Bet365 draw odds
                standardized["draw_odds"] = df["B365D"]
            if "B365A" in df.columns:  # Bet365 away odds
                standardized["away_odds"] = df["B365A"]

            # Create result column
            standardized["result"] = standardized.apply(
                lambda row: (
                    "H"
                    if row["home_score"] > row["away_score"]
                    else "A" if row["away_score"] > row["home_score"] else "D"
                ),
                axis=1,
            )

            # Create goal difference
            standardized["goal_difference"] = (
                standardized["home_score"] - standardized["away_score"]
            )

            logger.info(
                f"Standardized {len(standardized)} matches from football-data.co.uk"
            )
            return standardized

        except Exception as e:
            logger.error(f"Error standardizing data: {e}")
            return df

    def collect_all_data(self) -> Optional[pd.DataFrame]:
        """Collect and standardize data from football-data.co.uk."""
        # Collect data from football-data.co.uk
        raw_data = self.collect_from_football_data_co_uk()
        if raw_data is not None:
            standardized_data = self.standardize_data(raw_data)

            # Remove duplicates based on available columns
            duplicate_columns = ["date", "home_team", "away_team"]
            standardized_data = standardized_data.drop_duplicates(
                subset=duplicate_columns, keep="first"
            )

            logger.info(
                f"Collected {len(standardized_data)} unique matches from "
                f"football-data.co.uk"
            )
            return standardized_data
        else:
            logger.error("No data could be collected from football-data.co.uk")
            return None

    def save_data(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Save collected data to files."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as CSV
            csv_path = output_dir / "premier_league_matches.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Data saved to {csv_path}")

            # Save as Parquet for better performance
            parquet_path = output_dir / "premier_league_matches.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Data saved to {parquet_path}")

            # Save summary statistics
            summary_path = output_dir / "data_summary.txt"
            with open(summary_path, "w") as f:
                f.write("Premier League Data Summary\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Total matches: {len(df)}\n")

                if "date" in df.columns:
                    f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")

                if "season" in df.columns:
                    f.write(f"Seasons: {', '.join(df['season'].unique())}\n")

                if "home_team" in df.columns and "away_team" in df.columns:
                    teams_count = len(
                        set(df["home_team"].unique()) | set(df["away_team"].unique())
                    )
                    f.write(f"Teams: {teams_count}\n")

                f.write(f"Columns: {', '.join(df.columns)}\n\n")

                f.write("Sample data:\n")
                f.write(df.head().to_string())

            logger.info(f"Summary saved to {summary_path}")

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise


def main():
    """Main function to collect data."""
    logging.basicConfig(level=logging.INFO)
    collector = DataCollector()

    # Collect data
    df = collector.collect_all_data()

    if df is not None:
        # Save data
        output_dir = Path("data/real_data")
        collector.save_data(df, output_dir)

        logger.info(f"Successfully collected {len(df)} matches!")
        logger.info(f"Data saved to {output_dir}")

        # Log statistics
        logger.info("Data Statistics:")
        logger.info(f"  - Total matches: {len(df)}")
        logger.info(
            f"  - Date range: {df['date'].min().strftime('%Y-%m-%d')} to "
            f"{df['date'].max().strftime('%Y-%m-%d')}"
        )
        logger.info(f"  - Seasons: {', '.join(sorted(df['season'].unique()))}")
        teams_count = len(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        logger.info(f"  - Teams: {teams_count}")

        return df
    else:
        logger.error("Failed to collect any data")
        return None


if __name__ == "__main__":
    main()
