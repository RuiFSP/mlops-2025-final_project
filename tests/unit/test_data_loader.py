"""Unit tests for DataLoader class."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.data_preprocessing.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("test_path")
        assert loader.data_path == Path("test_path")
    
    def test_determine_result_home_win(self):
        """Test result determination for home win."""
        loader = DataLoader("test_path")
        row = pd.Series({'home_score': 3, 'away_score': 1})
        result = loader._determine_result(row)
        assert result == 'H'
    
    def test_determine_result_away_win(self):
        """Test result determination for away win."""
        loader = DataLoader("test_path")
        row = pd.Series({'home_score': 1, 'away_score': 3})
        result = loader._determine_result(row)
        assert result == 'A'
    
    def test_determine_result_draw(self):
        """Test result determination for draw."""
        loader = DataLoader("test_path")
        row = pd.Series({'home_score': 2, 'away_score': 2})
        result = loader._determine_result(row)
        assert result == 'D'
    
    def test_add_features(self):
        """Test feature addition."""
        loader = DataLoader("test_path")
        df = pd.DataFrame({
            'home_score': [3, 1, 2],
            'away_score': [1, 3, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
        })
        
        result_df = loader._add_features(df)
        
        assert 'goal_difference' in result_df.columns
        assert 'total_goals' in result_df.columns
        assert 'month' in result_df.columns
        assert result_df['goal_difference'].tolist() == [2, -2, 0]
        assert result_df['total_goals'].tolist() == [4, 4, 4]
        assert result_df['month'].tolist() == [1, 2, 3]
    
    def test_preprocess_data_empty(self):
        """Test preprocessing with empty DataFrame."""
        loader = DataLoader("test_path")
        empty_df = pd.DataFrame()
        result = loader.preprocess_data(empty_df)
        assert result.empty
    
    def test_load_raw_data_no_files(self):
        """Test loading data when no CSV files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DataLoader(temp_dir)
            result = loader.load_raw_data()
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_load_raw_data_with_csv(self):
        """Test loading data with CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV file
            sample_data = pd.DataFrame({
                'home_team': ['Arsenal', 'Chelsea'],
                'away_team': ['Liverpool', 'Manchester United'],
                'home_score': [2, 1],
                'away_score': [1, 3],
                'date': ['2023-01-01', '2023-01-02']
            })
            
            csv_path = Path(temp_dir) / "sample.csv"
            sample_data.to_csv(csv_path, index=False)
            
            loader = DataLoader(temp_dir)
            result = loader.load_raw_data()
            
            assert len(result) == 2
            assert 'home_team' in result.columns
            assert 'away_team' in result.columns
