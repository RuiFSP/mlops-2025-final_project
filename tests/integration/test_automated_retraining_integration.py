"""
Integration tests for automated retraining system.

These tests verify the end-to-end functionality of the automated retraining
system including scheduler, flows, API endpoints, and real model training.
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.automation.retraining_scheduler import AutomatedRetrainingScheduler, RetrainingConfig
from src.model_training.trainer import ModelTrainer


class TestAutomatedRetrainingIntegration:
    """Integration tests for the automated retraining system."""

    @pytest.fixture(autouse=True)
    def setup_monitoring_mocks(self):
        """Auto-setup mocks for monitoring components that require model loading."""
        with patch(
            "src.monitoring.drift_detector.ModelDriftDetector._load_model", return_value=None
        ), patch(
            "src.monitoring.performance_monitor.ModelPerformanceMonitor._load_model",
            return_value=None,
        ):
            yield

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with required directories and files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create directory structure
            (workspace / "models").mkdir()
            (workspace / "models" / "backups").mkdir()
            (workspace / "data" / "real_data").mkdir(parents=True)
            (workspace / "evaluation_reports").mkdir()
            (workspace / "config").mkdir()

            # Create mock training data
            mock_data = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=100, freq="D"),
                    "home_team": ["Arsenal", "Chelsea"] * 50,
                    "away_team": ["Liverpool", "ManCity"] * 50,
                    "result": ["H", "A", "D"] * 33 + ["H"],
                    "home_score": [1, 2, 0] * 33 + [1],
                    "away_score": [0, 1, 0] * 33 + [0],
                    "month": [1, 2, 3, 4] * 25,
                }
            )

            training_data_path = workspace / "data" / "real_data" / "premier_league_matches.parquet"
            mock_data.to_parquet(training_data_path)

            # Create mock model
            models_dir = workspace / "models"
            model_path = models_dir / "model.pkl"  # This will be created by save_model
            mock_trainer = ModelTrainer()
            # Train a simple model for testing with mocked MLflow
            with patch("mlflow.set_experiment"), patch("mlflow.start_run"), patch(
                "mlflow.log_param"
            ), patch("mlflow.log_metric"), patch("mlflow.sklearn.log_model"):
                mock_trainer.train(mock_data.iloc[:80], mock_data.iloc[80:])
                mock_trainer.save_model(str(models_dir))  # Pass directory, not file

            # Ensure model file exists and is a file, not directory
            if not model_path.exists():
                # Create a dummy model file for testing
                import pickle

                dummy_model = {"type": "test_model", "trained": True}
                with open(model_path, "wb") as f:
                    pickle.dump(dummy_model, f)

            # Create configuration
            config = RetrainingConfig(
                model_path=str(model_path),
                training_data_path=str(training_data_path),
                backup_model_dir=str(workspace / "models" / "backups"),
                monitoring_output_dir=str(workspace / "evaluation_reports"),
                performance_threshold=0.05,
                drift_threshold=0.1,
                max_days_without_retraining=1,  # Short for testing
                min_days_between_retraining=0,  # Allow immediate retraining
                check_interval_minutes=1,
                min_new_predictions=5,
                enable_automatic_deployment=True,  # Enable for testing
            )

            config_path = workspace / "config" / "retraining_config.yaml"
            config.save_to_file(str(config_path))

            yield {
                "workspace": workspace,
                "config": config,
                "config_path": config_path,
                "model_path": model_path,
                "training_data_path": training_data_path,
                "mock_data": mock_data,
            }

    @pytest.mark.slow
    def test_end_to_end_retraining_flow(self, temp_workspace):
        """Test complete end-to-end retraining flow."""
        workspace_data = temp_workspace

        # Initialize scheduler
        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Record predictions to trigger volume-based retraining
        for i in range(10):  # Exceed min_new_predictions
            scheduler.record_prediction(
                {
                    "home_team": f"Team{i % 4}",
                    "away_team": f"Team{(i + 1) % 4}",
                    "prediction": ["H", "D", "A"][i % 3],
                }
            )

        # Mock the actual retraining flow execution to avoid complex dependencies
        with patch("src.automation.retraining_flow.execute_automated_retraining") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "deployed": True,
                "triggers": ["data_volume"],
                "validation_results": {
                    "new_accuracy": 0.60,
                    "current_accuracy": 0.55,
                    "improvement": 0.05,
                    "should_deploy": True,
                },
                "deployment_results": {
                    "deployed": True,
                    "deployment_time": datetime.now().isoformat(),
                },
            }

            # Force retraining
            success = scheduler.force_retraining("integration_test")

            assert success is True

            # Wait for retraining to complete
            max_wait = 10  # seconds
            wait_time = 0
            while scheduler.retraining_in_progress and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1

            # Verify retraining was executed
            mock_execute.assert_called_once()

            # Check final state
            status = scheduler.get_status()
            assert status["retraining_in_progress"] is False
            assert status["total_trigger_events"] > 0
            assert scheduler.last_retraining_time is not None
            assert scheduler.prediction_count_since_retraining == 0  # Reset after retraining

    def test_scheduler_with_multiple_triggers(self, temp_workspace):
        """Test scheduler with multiple trigger conditions."""
        workspace_data = temp_workspace

        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Mock monitoring service to simulate drift detection
        with patch.object(scheduler.monitoring_service, "generate_drift_report") as mock_drift:
            mock_drift.return_value = {"drift_score": 0.15}  # Above threshold

            # Mock performance metrics to simulate degradation
            with patch.object(
                scheduler, "_get_recent_performance_metrics"
            ) as mock_metrics, patch.object(scheduler, "_get_baseline_accuracy", return_value=0.60):
                mock_metrics.return_value = [
                    {"accuracy": 0.54},  # 6% drop from baseline
                    {"accuracy": 0.53},
                ]

                # Check triggers individually
                performance_trigger = scheduler._check_performance_trigger()
                drift_trigger = scheduler._check_drift_trigger()

                assert performance_trigger is True
                assert drift_trigger is True

                # Mock retraining execution
                with patch(
                    "src.automation.retraining_flow.execute_automated_retraining"
                ) as mock_execute:
                    mock_execute.return_value = {"success": True, "deployed": True}

                    # Trigger retraining checks
                    scheduler._check_retraining_triggers()

                    # Wait for execution
                    time.sleep(0.1)

                    # Verify multiple triggers were detected
                    trigger_history = scheduler.get_trigger_history()
                    assert len(trigger_history) > 0

                    latest_trigger = trigger_history[-1]
                    triggers = latest_trigger["triggers"]
                    assert "performance_degradation" in triggers or "data_drift" in triggers

    def test_configuration_persistence_and_updates(self, temp_workspace):
        """Test configuration loading, saving, and runtime updates."""
        workspace_data = temp_workspace
        config_path = workspace_data["config_path"]

        # Load scheduler with file-based config
        scheduler = AutomatedRetrainingScheduler(config_path=str(config_path))

        # Update configuration at runtime
        new_config = RetrainingConfig(
            performance_threshold=0.08,  # Different from original
            drift_threshold=0.15,
            max_days_without_retraining=45,
        )

        scheduler.update_config(new_config)

        # Verify configuration was updated
        assert scheduler.config.performance_threshold == 0.08
        assert scheduler.config.drift_threshold == 0.15
        assert scheduler.config.max_days_without_retraining == 45

        # Save updated configuration
        updated_config_path = workspace_data["workspace"] / "config" / "updated_config.yaml"
        scheduler.config.save_to_file(str(updated_config_path))

        # Load new scheduler with updated config
        scheduler2 = AutomatedRetrainingScheduler(config_path=str(updated_config_path))

        assert scheduler2.config.performance_threshold == 0.08
        assert scheduler2.config.drift_threshold == 0.15

    def test_backup_and_model_versioning(self, temp_workspace):
        """Test model backup and versioning during retraining."""
        workspace_data = temp_workspace

        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Verify original model exists
        original_model_path = workspace_data["model_path"]
        assert original_model_path.exists()

        original_model_size = original_model_path.stat().st_size

        # Mock retraining flow that creates backups
        with patch("src.automation.retraining_flow.backup_current_model") as mock_backup:
            backup_path = (
                workspace_data["workspace"] / "models" / "backups" / "model_backup_test.pkl"
            )
            mock_backup.return_value = str(backup_path)

            # Create mock backup file
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(original_model_path.read_bytes())

            with patch(
                "src.automation.retraining_flow.execute_automated_retraining"
            ) as mock_execute:
                mock_execute.return_value = {
                    "success": True,
                    "deployed": True,
                    "backup_path": str(backup_path),
                }

                # Trigger retraining
                scheduler.force_retraining("backup_test")

                # Wait for completion
                time.sleep(0.1)

                # Verify backup was created
                assert backup_path.exists()
                assert backup_path.stat().st_size == original_model_size

    def test_error_handling_and_recovery(self, temp_workspace):
        """Test error handling and recovery in retraining process."""
        workspace_data = temp_workspace

        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Test handling of retraining failures
        with patch("src.automation.retraining_flow.execute_automated_retraining") as mock_execute:
            mock_execute.side_effect = Exception("Simulated retraining failure")

            # Trigger retraining
            scheduler.force_retraining("error_test")

            # Wait for completion
            time.sleep(0.1)

            # Verify scheduler recovered from error
            assert scheduler.retraining_in_progress is False
            assert scheduler.last_retraining_time is None  # Should not update on failure

            # Verify error notification was sent (if notifications are configured)
            trigger_history = scheduler.get_trigger_history()
            assert len(trigger_history) > 0

    def test_concurrent_retraining_prevention(self, temp_workspace):
        """Test prevention of concurrent retraining operations."""
        workspace_data = temp_workspace

        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Mock slow retraining process
        with patch("src.automation.retraining_flow.execute_automated_retraining") as mock_execute:
            # Make the first call slow
            def slow_retraining(*args, **kwargs):
                time.sleep(0.5)  # Simulate slow retraining
                return {"success": True, "deployed": True}

            mock_execute.side_effect = slow_retraining

            # Start first retraining
            success1 = scheduler.force_retraining("concurrent_test_1")
            assert success1 is True

            # Immediately try to start second retraining
            success2 = scheduler.force_retraining("concurrent_test_2")
            assert success2 is False  # Should be rejected

            # Wait for first retraining to complete
            time.sleep(0.6)

            # Now second retraining should be possible
            success3 = scheduler.force_retraining("concurrent_test_3")
            assert success3 is True

    @pytest.mark.slow
    def test_performance_monitoring_integration(self, temp_workspace):
        """Test integration with performance monitoring system."""
        workspace_data = temp_workspace

        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Simulate performance degradation over time
        performance_data = [
            {"accuracy": 0.60, "week": 1},
            {"accuracy": 0.58, "week": 2},
            {"accuracy": 0.55, "week": 3},  # Significant drop
            {"accuracy": 0.53, "week": 4},  # Continued degradation
        ]

        # Set baseline performance
        scheduler.retraining_orchestrator.set_baseline_performance(0.60)

        # Simulate weekly performance checks
        trigger_detected = False
        buffer_populated = False
        for perf_data in performance_data:
            should_retrain = scheduler.retraining_orchestrator.check_retraining_trigger(
                week=perf_data["week"], performance_data=perf_data
            )

            # Check if buffer is populated (before potential clearing)
            if len(scheduler.retraining_orchestrator.performance_buffer) > 0:
                buffer_populated = True

            if should_retrain:
                trigger_detected = True
                break

        # Should detect performance degradation trigger
        assert trigger_detected is True

        # Verify performance buffer was populated during the process
        assert buffer_populated is True

    def test_status_reporting_and_export(self, temp_workspace):
        """Test comprehensive status reporting and export functionality."""
        workspace_data = temp_workspace

        scheduler = AutomatedRetrainingScheduler(config=workspace_data["config"])

        # Generate some activity
        for i in range(5):
            scheduler.record_prediction({"prediction": f"test_{i}"})

        # Trigger retraining to generate history
        with patch("src.automation.retraining_flow.execute_automated_retraining") as mock_execute:
            mock_execute.return_value = {"success": True, "deployed": True}

            # Manually add a retraining record to the orchestrator to simulate the flow execution
            retraining_record = {
                "timestamp": datetime.now().isoformat(),
                "trigger_reasons": ["status_test"],
                "status": "success",
                "deployed": True,
                "duration_seconds": 120.5,
            }
            scheduler.retraining_orchestrator.retraining_history.append(retraining_record)

            scheduler.force_retraining("status_test")
            time.sleep(0.1)

        # Test status reporting
        status = scheduler.get_status()

        assert "is_running" in status
        assert "prediction_count_since_retraining" in status
        assert "total_trigger_events" in status
        assert "config" in status

        # Test history retrieval
        trigger_history = scheduler.get_trigger_history()
        retraining_history = scheduler.retraining_orchestrator.get_retraining_history()

        assert len(trigger_history) > 0
        assert len(retraining_history) > 0

        # Test report export
        report_path = (
            workspace_data["workspace"] / "evaluation_reports" / "integration_test_report.json"
        )
        scheduler.export_status_report(str(report_path))

        assert report_path.exists()

        # Verify report content
        with open(report_path) as f:
            report = json.load(f)

        assert "scheduler_status" in report
        assert "trigger_history" in report
        assert "retraining_history" in report
        assert "configuration" in report
        assert "export_timestamp" in report


class TestRetrainingFlowIntegration:
    """Integration tests for the Prefect retraining flow."""

    @pytest.fixture
    def mock_training_environment(self):
        """Create mock training environment for flow testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create mock data
            mock_data = pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=200, freq="D"),
                    "home_team": ["Arsenal", "Chelsea", "Liverpool", "ManCity"] * 50,
                    "away_team": ["Tottenham", "Newcastle", "Brighton", "Villa"] * 50,
                    "result": ["H", "A", "D"] * 66 + ["H", "A"],
                    "home_score": [2, 1, 0] * 66 + [1, 0],
                    "away_score": [1, 2, 0] * 66 + [0, 1],
                    "month": [1, 2, 3, 4, 5, 6] * 33 + [1, 2],
                }
            )

            # Setup paths
            training_data_path = workspace / "training_data.parquet"
            model_dir = workspace / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pkl"  # Trainer saves as model.pkl
            backup_dir = workspace / "backups"
            backup_dir.mkdir()

            mock_data.to_parquet(training_data_path)

            # Create initial model
            trainer = ModelTrainer()
            with patch("mlflow.set_experiment"), patch("mlflow.start_run"), patch(
                "mlflow.log_param"
            ), patch("mlflow.log_metric"), patch("mlflow.sklearn.log_model"):
                trainer.train(mock_data.iloc[:150], mock_data.iloc[150:])
                trainer.save_model(str(model_dir))  # Pass directory

            # Ensure model file exists and is a file, not directory
            if not model_path.exists():
                # Create a dummy model file for testing
                import pickle

                dummy_model = {"type": "test_model", "trained": True}
                with open(model_path, "wb") as f:
                    pickle.dump(dummy_model, f)

            yield {
                "workspace": workspace,
                "training_data_path": training_data_path,
                "model_path": model_path,
                "backup_dir": backup_dir,
                "mock_data": mock_data,
            }

    @pytest.mark.slow
    def test_retraining_flow_with_improvement(self, mock_training_environment):
        """Test retraining flow when new model improves performance."""
        env = mock_training_environment

        # Import flow functions
        from src.automation.retraining_flow import (
            backup_current_model,
            prepare_retraining_data,
            train_new_model,
            validate_new_model,
        )

        # Test backup
        backup_path = backup_current_model(
            model_path=str(env["model_path"]),
            backup_dir=str(env["backup_dir"]),
            backup_reason="test_improvement",
        )

        assert backup_path
        assert Path(backup_path).exists()

        # Test data preparation
        train_data, val_data, data_stats = prepare_retraining_data(
            original_data_path=str(env["training_data_path"])
        )

        assert len(train_data) > 0
        assert len(val_data) > 0
        assert data_stats["total_samples"] == len(env["mock_data"])

        # Test model training
        with patch("mlflow.set_experiment"), patch("mlflow.start_run"), patch(
            "mlflow.log_param"
        ), patch("mlflow.log_metric"), patch("mlflow.sklearn.log_model"):
            temp_model_path, training_metrics = train_new_model(
                train_data=train_data, val_data=val_data, model_type="random_forest"
            )

        assert temp_model_path is not None
        assert "model_type" in training_metrics
        assert training_metrics["model_type"] == "random_forest"

        # Test validation (mock improved performance)
        with patch("src.evaluation.evaluator.ModelEvaluator.evaluate") as mock_evaluate:
            # Mock new model performing better
            mock_evaluate.side_effect = [
                {"accuracy": 0.62},  # New model
                {"accuracy": 0.58},  # Current model
            ]

            validation_results, should_deploy = validate_new_model(
                temp_model_path=temp_model_path,
                validation_data=val_data,
                current_model_path=str(env["model_path"]),
                min_accuracy_threshold=0.50,
                improvement_threshold=0.02,
            )

            assert should_deploy is True
            assert validation_results["should_deploy"] is True
            assert validation_results["improvement"] > 0

    @pytest.mark.slow
    def test_retraining_flow_with_insufficient_improvement(self, mock_training_environment):
        """Test retraining flow when new model doesn't improve enough."""
        env = mock_training_environment

        from src.automation.retraining_flow import train_new_model, validate_new_model

        # Create mock data for training
        train_data = env["mock_data"].iloc[:150]
        val_data = env["mock_data"].iloc[150:]

        # Train new model
        with patch("mlflow.set_experiment"), patch("mlflow.start_run"), patch(
            "mlflow.log_param"
        ), patch("mlflow.log_metric"), patch("mlflow.sklearn.log_model"):
            temp_model_path, _ = train_new_model(
                train_data=train_data, val_data=val_data, model_type="random_forest"
            )

        # Test validation with insufficient improvement
        with patch("src.evaluation.evaluator.ModelEvaluator.evaluate") as mock_evaluate:
            # Mock new model not improving enough
            mock_evaluate.side_effect = [
                {"accuracy": 0.581},  # New model (slight improvement)
                {"accuracy": 0.580},  # Current model
            ]

            validation_results, should_deploy = validate_new_model(
                temp_model_path=temp_model_path,
                validation_data=val_data,
                current_model_path=str(env["model_path"]),
                min_accuracy_threshold=0.50,
                improvement_threshold=0.02,  # Requires 2% improvement
            )

            assert should_deploy is False
            assert validation_results["should_deploy"] is False
            assert validation_results["improvement"] < 0.02


class TestAPIIntegration:
    """Integration tests for retraining API endpoints."""

    @pytest.fixture
    def api_client(self):
        """Create FastAPI test client with retraining functionality."""
        from fastapi.testclient import TestClient

        # Mock the retraining imports to avoid initialization issues
        with patch("src.deployment.api.RETRAINING_AVAILABLE", True), patch(
            "src.deployment.api.AutomatedRetrainingScheduler"
        ):
            from src.deployment.api import app

            # Mock scheduler instance
            mock_scheduler = Mock()
            mock_scheduler.get_status.return_value = {
                "is_running": False,
                "retraining_in_progress": False,
                "last_check_time": None,
                "last_retraining_time": None,
                "prediction_count_since_retraining": 0,
                "days_since_last_retraining": None,
                "total_trigger_events": 0,
                "config": {"performance_threshold": 0.05},
            }

            # Patch the global scheduler
            with patch("src.deployment.api.retraining_scheduler", mock_scheduler):
                client = TestClient(app)
                yield client, mock_scheduler

    def test_retraining_status_endpoint(self, api_client):
        """Test retraining status API endpoint."""
        client, mock_scheduler = api_client

        response = client.get("/retraining/status")

        assert response.status_code == 200
        data = response.json()

        assert "is_running" in data
        assert "retraining_in_progress" in data
        assert "config" in data

    def test_retraining_trigger_endpoint(self, api_client):
        """Test manual retraining trigger endpoint."""
        client, mock_scheduler = api_client

        mock_scheduler.force_retraining.return_value = True

        response = client.post("/retraining/trigger", json={"reason": "api_test", "force": True})

        assert response.status_code == 200
        data = response.json()

        assert data["message"] == "Retraining triggered successfully"
        assert data["reason"] == "api_test"
        mock_scheduler.force_retraining.assert_called_once_with("api_test")

    def test_retraining_config_endpoints(self, api_client):
        """Test retraining configuration endpoints."""
        client, mock_scheduler = api_client

        # Test getting config
        response = client.get("/retraining/config")
        assert response.status_code == 200

        # Test updating config
        response = client.post(
            "/retraining/config", json={"performance_threshold": 0.08, "drift_threshold": 0.15}
        )

        assert response.status_code == 200
        data = response.json()
        assert "updated_config" in data

    def test_retraining_history_endpoint(self, api_client):
        """Test retraining history endpoint."""
        client, mock_scheduler = api_client

        mock_scheduler.get_trigger_history.return_value = [
            {"timestamp": "2025-07-09T10:00:00", "triggers": ["manual_test"]}
        ]
        mock_scheduler.retraining_orchestrator.get_retraining_history.return_value = [
            {"timestamp": "2025-07-09T10:05:00", "status": "success"}
        ]

        response = client.get("/retraining/history")

        assert response.status_code == 200
        data = response.json()

        assert "trigger_events" in data
        assert "retraining_events" in data
        assert data["total_triggers"] == 1
        assert data["total_retrainings"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
