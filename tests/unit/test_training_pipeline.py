"""
Unit tests for TrainingPipeline class.
"""

from unittest.mock import Mock, patch

from src.pipelines.training_pipeline import TrainingPipeline


class TestTrainingPipeline:
    """Test cases for TrainingPipeline class."""

    def test_init_with_default_path(self):
        """Test initialization with default data path."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            pipeline = TrainingPipeline()
            assert pipeline.data_path == "data/real_data/premier_league_matches.parquet"

    def test_init_with_custom_path(self):
        """Test initialization with custom data path."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            custom_path = "custom/path/data.parquet"
            pipeline = TrainingPipeline(data_path=custom_path)
            assert pipeline.data_path == custom_path

    def test_feature_columns_defined(self):
        """Test that feature columns are properly defined."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            pipeline = TrainingPipeline()

            expected_features = [
                "B365H",
                "B365D",
                "B365A",  # Bet365 odds
                "HS",
                "AS",  # Shots
                "HST",
                "AST",  # Shots on target
                "HC",
                "AC",  # Corners
                "HF",
                "AF",  # Fouls
                "HY",
                "AY",  # Yellow cards
                "HR",
                "AR",  # Red cards
            ]

            assert pipeline.feature_columns == expected_features
            assert pipeline.target_column == "FTR"

    def test_preprocess_data_structure(self):
        """Test data preprocessing structure."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            pipeline = TrainingPipeline()

            # Create a more realistic mock using pandas
            import pandas as pd

            # Create a real DataFrame with the expected structure
            mock_data = {
                "FTR": ["H", "D", "A"] * 34,  # 102 rows
                "B365H": [2.0] * 102,
                "B365D": [3.0] * 102,
                "B365A": [4.0] * 102,
            }

            # Add feature columns with some test data
            for col in pipeline.feature_columns:
                if col not in ["B365H", "B365D", "B365A"]:
                    mock_data[col] = [1.0] * 102

            mock_df = pd.DataFrame(mock_data)

            result = pipeline.preprocess_data(mock_df)

            # Verify the result is a DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    def test_train_model_parameters(self):
        """Test model training with correct parameters."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            pipeline = TrainingPipeline()

            # Mock training data
            mock_X_train = Mock()
            mock_X_train.shape = (1000, 15)
            mock_X_train.columns = pipeline.feature_columns

            mock_y_train = Mock()
            mock_y_train.value_counts.return_value.to_dict.return_value = {"H": 400, "D": 300, "A": 300}

            with patch("src.pipelines.training_pipeline.RandomForestClassifier") as mock_rf:
                mock_model = Mock()
                mock_rf.return_value = mock_model

                result = pipeline.train_model(mock_X_train, mock_y_train)

                # Verify RandomForest was created with correct parameters
                mock_rf.assert_called_once_with(n_estimators=100, max_depth=10, random_state=42)
                mock_model.fit.assert_called_once_with(mock_X_train, mock_y_train)
                assert result == mock_model

    def test_evaluate_model_structure(self):
        """Test model evaluation returns correct structure."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            pipeline = TrainingPipeline()

            # Mock model and data
            mock_model = Mock()
            mock_model.predict.return_value = ["H", "D", "A"]
            mock_model.feature_importances_ = [0.1] * len(pipeline.feature_columns)

            mock_X_test = Mock()
            mock_X_test.shape = (200, 15)

            mock_y_test = Mock()

            with (
                patch("src.pipelines.training_pipeline.accuracy_score") as mock_accuracy,
                patch("src.pipelines.training_pipeline.classification_report") as mock_report,
            ):
                mock_accuracy.return_value = 0.65
                mock_report.return_value = {
                    "accuracy": 0.65,
                    "macro avg": {"precision": 0.60, "recall": 0.58, "f1-score": 0.59},
                }

                result = pipeline.evaluate_model(mock_model, mock_X_test, mock_y_test)

                assert isinstance(result, dict)
                assert "accuracy" in result
                assert "feature_importance" in result
                assert "classification_report" in result
                assert result["accuracy"] == 0.65

    def test_setup_mlflow_configuration(self):
        """Test MLflow setup configuration."""
        with patch("src.pipelines.training_pipeline.mlflow") as mock_mlflow:
            TrainingPipeline()

            # Verify MLflow setup was called
            mock_mlflow.set_tracking_uri.assert_called()

    def test_model_attributes_initialization(self):
        """Test that model attributes are properly initialized."""
        with patch("src.pipelines.training_pipeline.mlflow"):
            pipeline = TrainingPipeline()

            assert pipeline.model is None
            assert isinstance(pipeline.feature_columns, list)
            assert len(pipeline.feature_columns) == 15
            assert pipeline.target_column == "FTR"
