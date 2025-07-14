"""
Unit tests for PredictionPipeline class.
"""

from unittest.mock import Mock, patch

import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline


class TestPredictionPipeline:
    """Test cases for PredictionPipeline class."""

    def test_to_native_conversion(self):
        """Test conversion of numpy types to native Python types."""
        with (
            patch("src.pipelines.prediction_pipeline.create_engine"),
            patch("src.pipelines.prediction_pipeline.mlflow"),
        ):
            pipeline = PredictionPipeline()

            # Test basic conversion
            result = pipeline._to_native({"test": 1.0})
            assert isinstance(result, dict)
            assert result["test"] == 1.0

    def test_prepare_features_structure(self):
        """Test that prepare_features returns correct structure."""
        with (
            patch("src.pipelines.prediction_pipeline.create_engine"),
            patch("src.pipelines.prediction_pipeline.mlflow"),
        ):
            pipeline = PredictionPipeline()

            # Create sample match data
            matches = pd.DataFrame(
                [{"home_team": "Arsenal", "away_team": "Chelsea", "home_odds": 2.0, "draw_odds": 3.0, "away_odds": 4.0}]
            )

            features = pipeline.prepare_features(matches)

            # Check required columns exist
            expected_columns = [
                "B365H",
                "B365D",
                "B365A",
                "HS",
                "AS",
                "HST",
                "AST",
                "HC",
                "AC",
                "HF",
                "AF",
                "HY",
                "AY",
                "HR",
                "AR",
            ]

            assert isinstance(features, pd.DataFrame)
            assert len(features) == 1
            for col in expected_columns:
                assert col in features.columns

    def test_prepare_features_odds_mapping(self):
        """Test that odds are correctly mapped to features."""
        with (
            patch("src.pipelines.prediction_pipeline.create_engine"),
            patch("src.pipelines.prediction_pipeline.mlflow"),
        ):
            pipeline = PredictionPipeline()

            matches = pd.DataFrame(
                [{"home_team": "Arsenal", "away_team": "Chelsea", "home_odds": 1.5, "draw_odds": 4.0, "away_odds": 6.0}]
            )

            features = pipeline.prepare_features(matches)

            assert features["B365H"].iloc[0] == 1.5
            assert features["B365D"].iloc[0] == 4.0
            assert features["B365A"].iloc[0] == 6.0

    def test_get_model_metadata_structure(self):
        """Test model metadata structure."""
        with (
            patch("src.pipelines.prediction_pipeline.create_engine"),
            patch("src.pipelines.prediction_pipeline.mlflow"),
        ):
            pipeline = PredictionPipeline()

            # Mock model info
            with patch.object(pipeline, "_get_model_metadata") as mock_metadata:
                mock_metadata.return_value = {"model_name": "test_model", "version": "1", "accuracy": 0.65}

                metadata = pipeline._get_model_metadata()
                assert isinstance(metadata, dict)
                assert "model_name" in metadata

    def test_predict_single_match_structure(self):
        """Test single match prediction structure."""
        with (
            patch("src.pipelines.prediction_pipeline.create_engine"),
            patch("src.pipelines.prediction_pipeline.mlflow"),
        ):
            pipeline = PredictionPipeline()

            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = ["H"]
            mock_model.predict_proba.return_value = [[0.6, 0.3, 0.1]]
            mock_model.classes_ = ["H", "D", "A"]

            pipeline.model = mock_model

            match_data = {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_odds": 2.0,
                "draw_odds": 3.0,
                "away_odds": 4.0,
            }

            result = pipeline.predict_single_match(match_data)

            assert isinstance(result, dict)
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert isinstance(result["probabilities"], dict)
            assert "H" in result["probabilities"]
            assert "D" in result["probabilities"]
            assert "A" in result["probabilities"]
