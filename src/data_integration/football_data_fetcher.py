"""
Data fetcher for football-data.co.uk
Fetches Premier League historical data and current season data
"""

import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)


class FootballDataFetcher:
    """Fetches data from football-data.co.uk"""
    
    def __init__(self):
        self.base_url = "https://www.football-data.co.uk/mmz4281"
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Premier League division code
        self.division_code = "E0"  # Premier League
        
        # Column mapping for consistency
        self.column_mapping = {
            'Date': 'match_date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',  # H/D/A
            'HTHG': 'ht_home_goals',
            'HTAG': 'ht_away_goals',
            'HTR': 'ht_result',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_target',
            'AST': 'away_shots_target',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HF': 'home_fouls',
            'AF': 'away_fouls',
            'HY': 'home_yellow',
            'AY': 'away_yellow',
            'HR': 'home_red',
            'AR': 'away_red',
            # Betting odds
            'B365H': 'home_odds',
            'B365D': 'draw_odds',
            'B365A': 'away_odds',
        }
    
    def fetch_season_data(self, season_years: List[str]) -> pd.DataFrame:
        """
        Fetch data for multiple seasons
        
        Args:
            season_years: List of season strings like ['2324', '2223', '2122']
        
        Returns:
            Combined DataFrame with all seasons data
        """
        all_data = []
        
        for season in season_years:
            logger.info(f"Fetching data for season 20{season}")
            try:
                # URL format: https://www.football-data.co.uk/mmz4281/2324/E0.csv
                url = f"{self.base_url}/{season}/{self.division_code}.csv"
                
                # Fetch data with retry logic
                df = self._fetch_with_retry(url, season)
                if df is not None:
                    df['season'] = f"20{season[:2]}-{season[2:]}"
                    all_data.append(df)
                    
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching season {season}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully fetched {len(combined_df)} matches across {len(all_data)} seasons")
            return self._clean_data(combined_df)
        else:
            logger.warning("No data fetched")
            return pd.DataFrame()
    
    def _fetch_with_retry(self, url: str, season: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1})")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save raw data
                raw_file = self.data_dir / f"premier_league_{season}.csv"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Parse CSV
                df = pd.read_csv(url, encoding='latin-1')
                logger.info(f"Successfully fetched {len(df)} matches for season {season}")
                return df
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch data for season {season} after {max_retries} attempts")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching season {season}: {e}")
                return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        logger.info("Cleaning and standardizing data...")
        
        # Rename columns using our mapping
        available_columns = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        df = df.rename(columns=available_columns)
        
        # Convert date column
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'], format='%d/%m/%Y', errors='coerce')
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Clean team names
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].str.strip()
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].str.strip()
        
        # Remove rows with missing essential data
        essential_columns = ['match_date', 'home_team', 'away_team', 'result']
        df = df.dropna(subset=[col for col in essential_columns if col in df.columns])
        
        logger.info(f"Cleaned data: {len(df)} matches remaining")
        return df
    
    def get_current_season_data(self) -> pd.DataFrame:
        """Get current season data (2024-25)"""
        current_season = "2425"  # 2024-25 season
        return self.fetch_season_data([current_season])
    
    def get_historical_data(self, years_back: int = 5) -> pd.DataFrame:
        """Get historical data for the last N years"""
        current_year = datetime.now().year
        
        # Generate season codes for last N years
        seasons = []
        for i in range(years_back):
            start_year = current_year - 1 - i
            season_code = f"{str(start_year)[2:]}{str(start_year + 1)[2:]}"
            seasons.append(season_code)
        
        return self.fetch_season_data(seasons)
    
    def get_team_form(self, team: str, last_n_matches: int = 5) -> Dict:
        """Get recent form for a team"""
        # This would fetch recent matches for form analysis
        # For now, return mock data
        return {
            'team': team,
            'last_matches': last_n_matches,
            'wins': 3,
            'draws': 1,
            'losses': 1,
            'goals_for': 8,
            'goals_against': 4,
            'form_points': 10
        }
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "premier_league_processed.csv"):
        """Save processed data"""
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = processed_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
        return filepath 