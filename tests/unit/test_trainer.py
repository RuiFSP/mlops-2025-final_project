"""Unit tests for ModelTrainer class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.model_training.trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer.model_type == "random_forest"
        assert trainer.model is None
        assert trainer.label_encoders == {}
    
    def test_init_with_model_type(self):
        """Test ModelTrainer initialization with specific model type."""
        trainer = ModelTrainer(model_type="xgboost")
        assert trainer.model_type == "xgboost"
    
    def test_prepare_features_empty_dataframe(self):
        """Test feature preparation with empty DataFrame."""
        trainer = ModelTrainer()
        empty_df = pd.DataFrame()
        result = trainer.prepare_features(empty_df)
        assert result.empty
    
    def test_prepare_features_with_data(self):
        """Test feature preparation with sample data."""
        trainer = ModelTrainer()
        df = pd.DataFrame({
            'home_team': ['Arsenal', 'Chelsea', 'Liverpool'],
            'away_team': ['Chelsea', 'Liverpool', 'Arsenal'],
            'month': [1, 2, 3],
            'goal_difference': [1, -1, 0],
            'total_goals': [3, 4, 2]
        })
        
        result = trainer.prepare_features(df)
        
        assert not result.empty
        assert len(result) == 3
        assert 'home_team' in result.columns
        assert 'away_team' in result.columns
        assert 'month' in result.columns
        assert 'goal_difference' in result.columns
        assert 'total_goals' in result.columns
    
    def test_encode_with_unknown(self):
        """Test encoding with unknown categories."""
        trainer = ModelTrainer()
        from sklearn.preprocessing import LabelEncoder
        
        # Create and fit encoder
        encoder = LabelEncoder()
        encoder.fit(['Arsenal', 'Chelsea', 'Liverpool'])
        
        # Test series with unknown category
        series = pd.Series(['Arsenal', 'ManCity', 'Chelsea'])
        result = trainer._encode_with_unknown(series, encoder)
        
        assert len(result) == 3
        assert isinstance(result, pd.Series)
    
    def test_train_with_empty_data(self):
        """Test training with empty data."""
        trainer = ModelTrainer()
        empty_df = pd.DataFrame()
        result = trainer.train(empty_df, empty_df)
        assert result is None
    
    def test_predict_without_model(self):
        """Test prediction without trained model."""
        trainer = ModelTrainer()
        df = pd.DataFrame({'home_team': ['Arsenal'], 'away_team': ['Chelsea']})
        
        with pytest.raises(ValueError, match="Model not trained or loaded"):
            trainer.predict(df)
