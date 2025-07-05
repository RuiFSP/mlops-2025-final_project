"""Data loader for Premier League match data."""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of Premier League match data."""
    
    def __init__(self, data_path: str):
        """Initialize the data loader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw match data from CSV files.
        
        Returns:
            DataFrame containing raw match data
        """
        try:
            # Look for CSV files in the data directory and subdirectories
            csv_files = list(self.data_path.glob("*.csv"))
            csv_files.extend(list(self.data_path.glob("*/*.csv")))  # Include subdirectories
            
            if not csv_files:
                logger.warning(f"No CSV files found in {self.data_path}")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=[
                    'home_team', 'away_team', 'home_score', 'away_score',
                    'date', 'season', 'result'
                ])
            
            # Load and combine all CSV files
            dataframes = []
            for csv_file in csv_files:
                logger.info(f"Loading {csv_file}")
                df = pd.read_csv(csv_file)
                
                # Ensure we have the required columns
                required_cols = ['home_team', 'away_team', 'home_score', 'away_score', 'date']
                
                # Check for alternative column names (football-data.co.uk format)
                alt_mapping = {
                    'HomeTeam': 'home_team',
                    'AwayTeam': 'away_team', 
                    'FTHG': 'home_score',
                    'FTAG': 'away_score',
                    'Date': 'date',
                    'B365H': 'home_odds',
                    'B365D': 'draw_odds',
                    'B365A': 'away_odds'
                }
                
                # Rename columns if they exist in alternative format
                for old_col, new_col in alt_mapping.items():
                    if old_col in df.columns:
                        df = df.rename(columns={old_col: new_col})
                
                # Check if we have the required columns after renaming
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Skipping {csv_file} - missing required columns")
                    continue
                    
                dataframes.append(df)
            
            if not dataframes:
                logger.warning("No valid CSV files found")
                return pd.DataFrame(columns=[
                    'home_team', 'away_team', 'home_score', 'away_score',
                    'date', 'season', 'result'
                ])
            
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} matches from {len(dataframes)} files")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw match data.
        
        Args:
            df: Raw match data
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            logger.warning("DataFrame is empty, returning as-is")
            return df
            
        # Convert date column to datetime
        if 'date' in df.columns:
            try:
                # Handle mixed date formats using pandas automatic parsing
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            except Exception as e:
                logger.warning(f"Date parsing failed: {e}")
                # If all else fails, try to parse manually
                df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        
        # Create result column if it doesn't exist
        if 'result' not in df.columns and 'home_score' in df.columns and 'away_score' in df.columns:
            df['result'] = df.apply(self._determine_result, axis=1)
        
        # Add derived features
        df = self._add_features(df)
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def _determine_result(self, row) -> str:
        """Determine match result from scores.
        
        Args:
            row: DataFrame row containing home_score and away_score
            
        Returns:
            'H' for home win, 'A' for away win, 'D' for draw
        """
        home_score = row['home_score']
        away_score = row['away_score']
        
        if home_score > away_score:
            return 'H'
        elif away_score > home_score:
            return 'A'
        else:
            return 'D'
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Add goal difference
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['goal_difference'] = df['home_score'] - df['away_score']
        
        # Add total goals
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['total_goals'] = df['home_score'] + df['away_score']
        
        # Add month from date
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
        
        return df
    
    def load_and_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and split into train/validation sets.
        
        Args:
            test_size: Proportion of data to use for validation
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        # Load and preprocess data
        raw_data = self.load_raw_data()
        processed_data = self.preprocess_data(raw_data)
        
        if processed_data.empty:
            logger.warning("No data available for splitting")
            return processed_data, processed_data
        
        # Sort by date for time-based split
        if 'date' in processed_data.columns:
            processed_data = processed_data.sort_values('date')
        
        # Split data
        split_index = int(len(processed_data) * (1 - test_size))
        train_data = processed_data.iloc[:split_index]
        val_data = processed_data.iloc[split_index:]
        
        logger.info(f"Train set: {len(train_data)} samples")
        logger.info(f"Validation set: {len(val_data)} samples")
        
        return train_data, val_data
