"""
Integration tests for automated retraining flow.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.automation.retraining_flow import (
    automated_retraining_flow,
    backup_current_model,
    deploy_new_model,
    prepare_retraining_data,
    train_new_model,
    validate_new_model,
)


class TestRetrainingFlowTasks:
    """Test individual tasks in the retraining flow."""

    def test_backup_current_model(self):
        """Test model backup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy model file
            model_path = Path(temp_dir) / "model.pkl"
            model_path.write_text("dummy model content")

            # Create backup
            backup_dir = Path(temp_dir) / "backups"
            backup_path = backup_current_model(
                model_path=str(model_path), backup_dir=str(backup_dir), backup_reason="test_backup"
            )

            # Check backup was created
            assert backup_path != ""
            assert Path(backup_path).exists()
            assert "test_backup" in backup_path
            assert Path(backup_path).read_text() == "dummy model content"

    def test_backup_nonexistent_model(self):
        """Test backup when model doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "nonexistent_model.pkl"
            backup_dir = Path(temp_dir) / "backups"

            backup_path = backup_current_model(
                model_path=str(model_path), backup_dir=str(backup_dir), backup_reason="test_backup"
            )

            # Should return empty string for nonexistent model
            assert backup_path == ""

    def test_prepare_retraining_data(self):
        """Test data preparation for retraining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy training data
            training_data = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                    "home_team": ["Team1"] * 50 + ["Team2"] * 50,
                    "away_team": ["Team2"] * 50 + ["Team1"] * 50,
                    "result": ["H", "A", "D"] * 33 + ["H"],
                    "home_score": [1, 2, 1] * 33 + [1],
                    "away_score": [0, 1, 1] * 33 + [0],
                }
            )

            training_data_path = Path(temp_dir) / "training_data.parquet"
            training_data.to_parquet(training_data_path)

            # Mock DataLoader
            with patch("src.automation.retraining_flow.DataLoader") as mock_loader, patch(
                "src.automation.retraining_flow.pd.read_parquet"
            ) as mock_read_parquet:
                mock_instance = Mock()
                mock_instance.load_raw_data.return_value = training_data
                mock_loader.return_value = mock_instance
                mock_read_parquet.return_value = training_data

                train_data, val_data, data_stats = prepare_retraining_data(
                    original_data_path=str(training_data_path),
                )

                # Check data split
                assert len(train_data) + len(val_data) == len(training_data)
                assert len(train_data) == int(len(training_data) * 0.8)
                assert len(val_data) == len(training_data) - len(train_data)

                # Check statistics
                assert data_stats["total_samples"] == len(training_data)
                assert data_stats["training_samples"] == len(train_data)
                assert data_stats["validation_samples"] == len(val_data)

    def test_train_new_model(self):
        """Test new model training."""
        # Create dummy training data
        train_data = pd.DataFrame(
            {
                "home_team": ["Team1"] * 20,
                "away_team": ["Team2"] * 20,
                "result": ["H", "A", "D"] * 6 + ["H", "A"],
                "month": [1] * 20,
            }
        )

        val_data = pd.DataFrame(
            {
                "home_team": ["Team1"] * 5,
                "away_team": ["Team2"] * 5,
                "result": ["H", "A", "D", "H", "A"],
                "month": [1] * 5,
            }
        )

        with patch("src.automation.retraining_flow.ModelTrainer") as mock_trainer_class:
            # Mock trainer instance
            mock_trainer = Mock()
            mock_trainer.train.return_value = Mock()  # Mock model
            mock_trainer.training_history = {"accuracy": 0.85, "loss": 0.3}  # Mock training history
            mock_trainer_class.return_value = mock_trainer

            temp_model_path, metrics = train_new_model(
                train_data=train_data, val_data=val_data, model_type="random_forest"
            )

            # Check trainer was called correctly
            assert temp_model_path is not None  # Function returns temp_model_path, not trainer
            mock_trainer.train.assert_called_once_with(train_data, val_data)

            # Check metrics
            assert metrics["model_type"] == "random_forest"
            assert metrics["training_samples"] == len(train_data)
            assert metrics["validation_samples"] == len(val_data)

    def test_validate_new_model_improvement(self):
        """Test model validation when new model improves."""
        val_data = pd.DataFrame(
            {
                "home_team": ["Team1"] * 5,
                "away_team": ["Team2"] * 5,
                "result": ["H", "A", "D", "H", "A"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            current_model_path = Path(temp_dir) / "current_model.pkl"
            current_model_path.write_text("dummy current model")

            # Create temp model path
            temp_model_path = Path(temp_dir) / "new_model"
            temp_model_path.mkdir()

            with patch(
                "src.automation.retraining_flow.ModelEvaluator"
            ) as mock_evaluator_class, patch(
                "src.automation.retraining_flow.ModelTrainer"
            ) as mock_trainer_class:
                # Mock evaluator
                mock_evaluator = Mock()
                mock_evaluator_class.return_value = mock_evaluator

                # Mock new model evaluation (better performance)
                mock_evaluator.evaluate.side_effect = [
                    {"accuracy": 0.60},  # New model
                    {"accuracy": 0.55},  # Current model
                ]

                # Mock trainer loading
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer

                validation_results, should_deploy = validate_new_model(
                    temp_model_path=str(temp_model_path),
                    validation_data=val_data,
                    current_model_path=str(current_model_path),
                    min_accuracy_threshold=0.45,
                    improvement_threshold=0.01,
                )

                # Check validation passed
                assert should_deploy is True
                assert validation_results["should_deploy"] is True
                assert validation_results["new_accuracy"] == 0.60
                assert validation_results["current_accuracy"] == 0.55
                assert abs(validation_results["improvement"] - 0.05) < 1e-10
                assert validation_results["meets_min_threshold"] is True
                assert validation_results["improves_performance"] is True

    def test_validate_new_model_no_improvement(self):
        """Test model validation when new model doesn't improve."""
        val_data = pd.DataFrame(
            {
                "home_team": ["Team1"] * 5,
                "away_team": ["Team2"] * 5,
                "result": ["H", "A", "D", "H", "A"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            current_model_path = Path(temp_dir) / "current_model.pkl"
            current_model_path.write_text("dummy current model")

            # Create temp model path
            temp_model_path = Path(temp_dir) / "new_model"
            temp_model_path.mkdir()

            with patch(
                "src.automation.retraining_flow.ModelEvaluator"
            ) as mock_evaluator_class, patch(
                "src.automation.retraining_flow.ModelTrainer"
            ) as mock_trainer_class:
                # Mock evaluator
                mock_evaluator = Mock()
                mock_evaluator_class.return_value = mock_evaluator

                # Mock model evaluation (worse performance)
                mock_evaluator.evaluate.side_effect = [
                    {"accuracy": 0.52},  # New model
                    {"accuracy": 0.55},  # Current model
                ]

                # Mock trainer loading
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer

                validation_results, should_deploy = validate_new_model(
                    temp_model_path=str(temp_model_path),
                    validation_data=val_data,
                    current_model_path=str(current_model_path),
                    min_accuracy_threshold=0.45,
                    improvement_threshold=0.01,
                )

                # Check validation failed
                assert should_deploy is False
                assert validation_results["should_deploy"] is False
                assert validation_results["new_accuracy"] == 0.52
                assert validation_results["current_accuracy"] == 0.55
                assert abs(validation_results["improvement"] - (-0.03)) < 1e-10
                assert validation_results["meets_min_threshold"] is True
                assert validation_results["improves_performance"] is False

    def test_deploy_new_model_success(self):
        """Test successful model deployment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            # Create a model file path (source)
            model_file_path = Path(temp_dir) / "temp_model.pkl"
            model_file_path.write_text("dummy model")

            # Mock trainer
            mock_trainer = Mock()
            mock_trainer.save_model = Mock()

            validation_results = {
                "should_deploy": True,
                "new_accuracy": 0.60,
                "current_accuracy": 0.55,
            }

            deployment_results = deploy_new_model(
                model_file_path=str(model_file_path),
                model_path=str(model_path),
                validation_results=validation_results,
            )

            # Check deployment succeeded
            assert deployment_results["deployed"] is True
            assert "deployment_time" in deployment_results

    def test_deploy_new_model_validation_failed(self):
        """Test model deployment when validation failed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            # Create a model file path (source)
            model_file_path = Path(temp_dir) / "temp_model.pkl"
            model_file_path.write_text("dummy model")

            mock_trainer = Mock()

            validation_results = {
                "should_deploy": False,
                "new_accuracy": 0.40,
            }

            deployment_results = deploy_new_model(
                model_file_path=str(model_file_path),
                model_path=str(model_path),
                validation_results=validation_results,
            )

            # Check deployment was skipped
            assert deployment_results["deployed"] is False
            assert deployment_results["reason"] == "validation_failed"


class TestRetrainingFlowIntegration:
    """Test the complete retraining flow integration."""

    def test_automated_retraining_flow_success(self):
        """Test successful end-to-end retraining flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths
            model_path = Path(temp_dir) / "model.pkl"
            training_data_path = Path(temp_dir) / "training_data.parquet"
            backup_dir = Path(temp_dir) / "backups"

            # Create dummy model and data
            model_path.write_text("dummy model")

            training_data = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=50, freq="D"),
                    "home_team": ["Team1"] * 25 + ["Team2"] * 25,
                    "away_team": ["Team2"] * 25 + ["Team1"] * 25,
                    "result": ["H", "A", "D"] * 16 + ["H", "A"],
                    "month": [1] * 50,
                }
            )
            training_data.to_parquet(training_data_path)

            # Mock all dependencies
            with patch("src.automation.retraining_flow.DataLoader") as mock_loader_class, patch(
                "src.automation.retraining_flow.ModelTrainer"
            ) as mock_trainer_class, patch(
                "src.automation.retraining_flow.ModelEvaluator"
            ) as mock_evaluator_class, patch(
                "src.automation.retraining_flow.create_markdown_artifact"
            ):
                # Mock DataLoader
                mock_loader = Mock()
                mock_loader.data = training_data  # Fix attribute access
                mock_loader_class.return_value = mock_loader

                # Mock ModelTrainer
                mock_trainer = Mock()
                mock_trainer.train.return_value = Mock()
                mock_trainer.save_model = Mock()
                mock_trainer.training_history = {
                    "accuracy": 0.58,
                    "loss": 0.4,
                }  # Add training history
                mock_trainer_class.return_value = mock_trainer

                # Mock ModelEvaluator
                mock_evaluator = Mock()
                mock_evaluator.evaluate.side_effect = [
                    {"accuracy": 0.58},  # New model
                    {"accuracy": 0.55},  # Current model
                ]
                mock_evaluator_class.return_value = mock_evaluator

                # Run the flow
                result = automated_retraining_flow(
                    triggers=["test_trigger"],
                    model_path=str(model_path),
                    training_data_path=str(training_data_path),
                    backup_dir=str(backup_dir),
                    min_accuracy_threshold=0.45,
                    improvement_threshold=0.01,
                )

                # Check flow succeeded
                assert result["success"] is True
                assert result["deployed"] is True
                assert result["triggers"] == ["test_trigger"]
                assert "backup_path" in result
                assert "data_stats" in result
                assert "training_metrics" in result
                assert "validation_results" in result
                assert "deployment_results" in result
                assert "report" in result

    def test_automated_retraining_flow_deployment_failure(self):
        """Test retraining flow when model doesn't meet deployment criteria."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths
            model_path = Path(temp_dir) / "model.pkl"
            training_data_path = Path(temp_dir) / "training_data.parquet"
            backup_dir = Path(temp_dir) / "backups"

            # Create dummy model and data
            model_path.write_text("dummy model")

            training_data = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=30, freq="D"),
                    "home_team": ["Team1"] * 15 + ["Team2"] * 15,
                    "away_team": ["Team2"] * 15 + ["Team1"] * 15,
                    "result": ["H", "A", "D"] * 10,
                    "month": [1] * 30,
                }
            )
            training_data.to_parquet(training_data_path)

            # Mock all dependencies
            with patch("src.automation.retraining_flow.DataLoader") as mock_loader_class, patch(
                "src.automation.retraining_flow.ModelTrainer"
            ) as mock_trainer_class, patch(
                "src.automation.retraining_flow.ModelEvaluator"
            ) as mock_evaluator_class, patch(
                "src.automation.retraining_flow.create_markdown_artifact"
            ):
                # Mock DataLoader
                mock_loader = Mock()
                mock_loader.data = training_data  # Fix attribute access
                mock_loader_class.return_value = mock_loader

                # Mock ModelTrainer
                mock_trainer = Mock()
                mock_trainer.train.return_value = Mock()
                mock_trainer.training_history = {
                    "accuracy": 0.52,
                    "loss": 0.5,
                }  # Add training history
                mock_trainer_class.return_value = mock_trainer

                # Mock ModelEvaluator (new model performs worse)
                mock_evaluator = Mock()
                mock_evaluator.evaluate.side_effect = [
                    {"accuracy": 0.52},  # New model
                    {"accuracy": 0.55},  # Current model
                ]
                mock_evaluator_class.return_value = mock_evaluator

                # Run the flow
                result = automated_retraining_flow(
                    triggers=["performance_degradation"],
                    model_path=str(model_path),
                    training_data_path=str(training_data_path),
                    backup_dir=str(backup_dir),
                    min_accuracy_threshold=0.45,
                    improvement_threshold=0.01,
                )

                # Check flow succeeded but model wasn't deployed
                assert result["success"] is True
                assert result["deployed"] is False
                assert result["validation_results"]["should_deploy"] is False
                assert result["deployment_results"]["deployed"] is False

    def test_automated_retraining_flow_error_handling(self):
        """Test retraining flow error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup paths with nonexistent training data
            model_path = Path(temp_dir) / "model.pkl"
            training_data_path = Path(temp_dir) / "nonexistent_data.parquet"

            model_path.write_text("dummy model")

            # Run the flow (should fail gracefully)
            result = automated_retraining_flow(
                triggers=["test_trigger"],
                model_path=str(model_path),
                training_data_path=str(training_data_path),
            )

            # Check flow handled error gracefully
            assert result["success"] is False
            assert "error" in result
            assert result["triggers"] == ["test_trigger"]


if __name__ == "__main__":
    pytest.main([__file__])
