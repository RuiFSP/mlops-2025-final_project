"""
Tests for automated retraining scheduler.
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.automation.retraining_scheduler import AutomatedRetrainingScheduler, RetrainingConfig


class TestRetrainingConfig:
    """Test RetrainingConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetrainingConfig()

        assert config.performance_threshold == 0.05
        assert config.drift_threshold == 0.1
        assert config.max_days_without_retraining == 30
        assert config.min_days_between_retraining == 7
        assert config.check_interval_minutes == 60
        assert config.enable_automatic_deployment is False

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name

        try:
            # Create and save config
            original_config = RetrainingConfig(
                performance_threshold=0.03,
                drift_threshold=0.15,
                max_days_without_retraining=45,
            )
            original_config.save_to_file(config_path)

            # Load config
            loaded_config = RetrainingConfig.load_from_file(config_path)

            assert loaded_config.performance_threshold == 0.03
            assert loaded_config.drift_threshold == 0.15
            assert loaded_config.max_days_without_retraining == 45

        finally:
            Path(config_path).unlink(missing_ok=True)


class TestAutomatedRetrainingScheduler:
    """Test AutomatedRetrainingScheduler functionality."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        return RetrainingConfig(
            performance_threshold=0.05,
            drift_threshold=0.1,
            max_days_without_retraining=1,  # 1 day for testing
            min_days_between_retraining=0,  # Allow immediate retraining for testing
            check_interval_minutes=1,  # 1 minute for testing
            min_new_predictions=5,
            enable_automatic_deployment=False,
        )

    @pytest.fixture
    def scheduler(self, temp_config):
        """Create a scheduler instance for testing."""
        with patch("src.automation.retraining_scheduler.MLOpsMonitoringService"), patch(
            "src.simulation.retraining_orchestrator.RetrainingOrchestrator"
        ):
            scheduler = AutomatedRetrainingScheduler(config=temp_config)
            return scheduler

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.config.performance_threshold == 0.05
        assert scheduler.is_running is False
        assert scheduler.retraining_in_progress is False
        assert scheduler.prediction_count_since_retraining == 0

    def test_prediction_recording(self, scheduler):
        """Test recording predictions for volume tracking."""
        initial_count = scheduler.prediction_count_since_retraining

        prediction_data = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "prediction": "H",
        }

        scheduler.record_prediction(prediction_data)

        assert scheduler.prediction_count_since_retraining == initial_count + 1

    def test_force_retraining(self, scheduler):
        """Test forcing retraining."""
        with patch.object(scheduler, "_initiate_retraining") as mock_initiate:
            success = scheduler.force_retraining("test_reason")

            assert success is True
            mock_initiate.assert_called_once_with(["test_reason"])

    def test_force_retraining_when_in_progress(self, scheduler):
        """Test forcing retraining when already in progress."""
        scheduler.retraining_in_progress = True

        success = scheduler.force_retraining("test_reason")

        assert success is False

    def test_status_reporting(self, scheduler):
        """Test status reporting."""
        status = scheduler.get_status()

        assert isinstance(status, dict)
        assert "is_running" in status
        assert "retraining_in_progress" in status
        assert "prediction_count_since_retraining" in status
        assert "config" in status

        assert status["is_running"] is False
        assert status["retraining_in_progress"] is False
        assert status["prediction_count_since_retraining"] == 0

    def test_config_update(self, scheduler):
        """Test configuration update."""
        new_config = RetrainingConfig(performance_threshold=0.08)

        scheduler.update_config(new_config)

        assert scheduler.config.performance_threshold == 0.08

    def test_scheduler_lifecycle(self, scheduler):
        """Test starting and stopping scheduler."""
        # Test starting
        with patch.object(scheduler, "_scheduler_loop"):
            scheduler.start_scheduler()
            assert scheduler.is_running is True

            # Test stopping
            scheduler.stop_scheduler()
            assert scheduler.is_running is False

    def test_data_volume_trigger_check(self, scheduler):
        """Test data volume trigger logic."""
        # Set up scenario where data volume should trigger retraining
        scheduler.prediction_count_since_retraining = (
            scheduler.config.max_predictions_without_retraining
        )

        should_trigger = scheduler._check_data_volume_trigger()

        assert should_trigger is True

    def test_data_volume_trigger_check_below_threshold(self, scheduler):
        """Test data volume trigger when below threshold."""
        scheduler.prediction_count_since_retraining = (
            scheduler.config.max_predictions_without_retraining - 1
        )

        should_trigger = scheduler._check_data_volume_trigger()

        assert should_trigger is False

    def test_time_trigger_check_no_previous_retraining(self, scheduler):
        """Test time trigger when no previous retraining occurred."""
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat:
            # Mock file modification time to be old enough
            old_time = datetime.now() - timedelta(
                days=scheduler.config.max_days_without_retraining + 1
            )
            mock_stat.return_value.st_mtime = old_time.timestamp()

            should_trigger = scheduler._check_time_trigger()

            assert should_trigger is True

    def test_time_trigger_check_recent_retraining(self, scheduler):
        """Test time trigger when recent retraining occurred."""
        # Set recent retraining time
        scheduler.last_retraining_time = datetime.now() - timedelta(days=1)

        should_trigger = scheduler._check_time_trigger()

        assert should_trigger is False

    def test_too_soon_for_retraining(self, scheduler):
        """Test minimum time between retraining check."""
        # Set recent retraining time within minimum interval
        scheduler.last_retraining_time = datetime.now() - timedelta(hours=1)
        scheduler.config.min_days_between_retraining = 1  # 1 day minimum

        too_soon = scheduler._is_too_soon_for_retraining()

        assert too_soon is True

    def test_not_too_soon_for_retraining(self, scheduler):
        """Test when enough time has passed for retraining."""
        # Set retraining time beyond minimum interval
        scheduler.last_retraining_time = datetime.now() - timedelta(days=2)
        scheduler.config.min_days_between_retraining = 1  # 1 day minimum

        too_soon = scheduler._is_too_soon_for_retraining()

        assert too_soon is False

    @patch("src.automation.retraining_flow.execute_automated_retraining")
    def test_execute_retraining_success(self, mock_execute, scheduler):
        """Test successful retraining execution."""
        # Mock successful retraining
        mock_execute.return_value = {
            "success": True,
            "deployed": True,
        }

        scheduler._execute_retraining(["test_trigger"])

        # Check that state was updated
        assert scheduler.retraining_in_progress is False
        assert scheduler.prediction_count_since_retraining == 0
        assert scheduler.last_retraining_time is not None

    @patch("src.automation.retraining_flow.execute_automated_retraining")
    def test_execute_retraining_failure(self, mock_execute, scheduler):
        """Test failed retraining execution."""
        # Mock failed retraining
        mock_execute.return_value = {
            "success": False,
            "error": "Test error",
        }

        scheduler._execute_retraining(["test_trigger"])

        # Check that state was reset but other values not updated
        assert scheduler.retraining_in_progress is False
        assert scheduler.last_retraining_time is None  # Should not be updated on failure

    def test_trigger_history_tracking(self, scheduler):
        """Test trigger history tracking."""
        # Initially no history
        assert len(scheduler.get_trigger_history()) == 0

        # Simulate trigger events
        with patch.object(scheduler, "_execute_retraining"):
            scheduler._initiate_retraining(["test_trigger_1"])
            # Reset the retraining_in_progress flag for second call
            scheduler.retraining_in_progress = False
            scheduler._initiate_retraining(["test_trigger_2"])

        history = scheduler.get_trigger_history()
        assert len(history) == 2
        assert history[0]["triggers"] == ["test_trigger_1"]
        assert history[1]["triggers"] == ["test_trigger_2"]

    def test_export_status_report(self, scheduler):
        """Test status report export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            scheduler.export_status_report(output_path)

            # Check that file was created
            assert Path(output_path).exists()

            # Check file content
            import json

            with open(output_path) as f:
                report = json.load(f)

            assert "scheduler_status" in report
            assert "trigger_history" in report
            assert "retraining_history" in report
            assert "configuration" in report
            assert "export_timestamp" in report

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestRetrainingTriggers:
    """Test various retraining trigger scenarios."""

    @pytest.fixture
    def scheduler_with_mocks(self):
        """Create scheduler with mocked dependencies."""
        config = RetrainingConfig(
            performance_threshold=0.05,
            drift_threshold=0.1,
            max_days_without_retraining=30,
            min_days_between_retraining=7,
            check_interval_minutes=60,
        )

        with patch("src.automation.retraining_scheduler.MLOpsMonitoringService"), patch(
            "src.simulation.retraining_orchestrator.RetrainingOrchestrator"
        ):
            scheduler = AutomatedRetrainingScheduler(config=config)
            return scheduler

    def test_performance_trigger_with_degradation(self, scheduler_with_mocks):
        """Test performance trigger when accuracy drops."""
        scheduler = scheduler_with_mocks

        # Mock performance data showing degradation
        with patch.object(
            scheduler, "_get_recent_performance_metrics"
        ) as mock_metrics, patch.object(scheduler, "_get_baseline_accuracy", return_value=0.60):
            mock_metrics.return_value = [
                {"accuracy": 0.54},  # 6% drop from baseline
                {"accuracy": 0.53},
                {"accuracy": 0.52},
            ]

            should_trigger = scheduler._check_performance_trigger()

            assert should_trigger is True

    def test_performance_trigger_without_degradation(self, scheduler_with_mocks):
        """Test performance trigger when accuracy is stable."""
        scheduler = scheduler_with_mocks

        # Mock performance data showing stable performance
        with patch.object(
            scheduler, "_get_recent_performance_metrics"
        ) as mock_metrics, patch.object(scheduler, "_get_baseline_accuracy", return_value=0.60):
            mock_metrics.return_value = [
                {"accuracy": 0.59},  # Small drop, below threshold
                {"accuracy": 0.60},
                {"accuracy": 0.61},
            ]

            should_trigger = scheduler._check_performance_trigger()

            assert should_trigger is False

    def test_drift_trigger_with_high_drift(self, scheduler_with_mocks):
        """Test drift trigger when drift score is high."""
        scheduler = scheduler_with_mocks

        # Mock monitoring service to return high drift
        scheduler.monitoring_service.generate_drift_report.return_value = {
            "drift_score": 0.15  # Above threshold of 0.1
        }

        should_trigger = scheduler._check_drift_trigger()

        assert should_trigger is True

    def test_drift_trigger_with_low_drift(self, scheduler_with_mocks):
        """Test drift trigger when drift score is low."""
        scheduler = scheduler_with_mocks

        # Mock monitoring service to return low drift
        scheduler.monitoring_service.generate_drift_report.return_value = {
            "drift_score": 0.05  # Below threshold of 0.1
        }

        should_trigger = scheduler._check_drift_trigger()

        assert should_trigger is False


if __name__ == "__main__":
    pytest.main([__file__])
