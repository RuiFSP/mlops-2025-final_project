"""
Tests for the monitoring system components.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    MLOpsMonitoringService,
    ModelDriftDetector,
    ModelPerformanceMonitor,
)


class TestModelDriftDetector:
    """Test drift detection functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester United"]

        data = pd.DataFrame(
            {
                "home_team": np.random.choice(teams, 100),
                "away_team": np.random.choice(teams, 100),
                "month": np.random.randint(1, 13, 100),
                "home_odds": np.random.uniform(1.5, 4.0, 100),
                "draw_odds": np.random.uniform(2.5, 4.5, 100),
                "away_odds": np.random.uniform(1.5, 4.0, 100),
                "result": np.random.choice(["Home Win", "Draw", "Away Win"], 100),
            }
        )

        # Add margin-adjusted probabilities
        for outcome in ["home", "draw", "away"]:
            implied_prob = 1 / data[f"{outcome}_odds"]
            total_implied = (
                1 / data["home_odds"] + 1 / data["draw_odds"] + 1 / data["away_odds"]
            )
            data[f"{outcome}_prob_margin_adj"] = implied_prob / total_implied

        return data

    @pytest.fixture
    def temp_files(self, sample_data):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save reference data
            reference_path = temp_path / "reference.parquet"
            sample_data.to_parquet(reference_path)

            # Create dummy model file with actual content
            model_path = temp_path / "model.pkl"
            from sklearn.dummy import DummyClassifier

            dummy_model = DummyClassifier(strategy="constant", constant="Home Win")
            # Fit with minimal data
            X_dummy = np.array([[1, 2, 3]])
            y_dummy = np.array(["Home Win"])
            dummy_model.fit(X_dummy, y_dummy)

            import joblib

            joblib.dump(dummy_model, model_path)

            yield {
                "reference_path": str(reference_path),
                "model_path": str(model_path),
                "output_dir": str(temp_path / "output"),
            }

    def test_drift_detector_initialization(self, temp_files):
        """Test drift detector can be initialized."""
        detector = ModelDriftDetector(
            reference_data_path=temp_files["reference_path"],
            model_path=temp_files["model_path"],
            output_dir=temp_files["output_dir"],
        )

        assert detector.reference_data is not None
        assert len(detector.reference_data) > 0
        assert detector.drift_threshold == 0.05  # default value we use

    def test_drift_detector_with_missing_model(self, temp_files):
        """Test drift detector handles missing model gracefully."""
        detector = ModelDriftDetector(
            reference_data_path=temp_files["reference_path"],
            model_path="nonexistent_model.pkl",
            output_dir=temp_files["output_dir"],
        )

        assert detector.model is None
        assert detector.reference_data is not None


class TestModelPerformanceMonitor:
    """Test performance monitoring functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict.return_value = np.array(["Home Win", "Draw", "Away Win"] * 10)
        model.predict_proba.return_value = np.random.rand(30, 3)
        return model

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_performance_monitor_initialization(self, temp_output_dir):
        """Test performance monitor can be initialized."""
        with patch("joblib.load") as mock_load:
            mock_load.return_value = Mock()

            monitor = ModelPerformanceMonitor(
                model_path="dummy_model.pkl", output_dir=temp_output_dir
            )

            assert monitor.performance_threshold == 0.05  # default value
            assert monitor.baseline_accuracy == 0.55  # default from README

    def test_calculate_metrics(self, temp_output_dir, mock_model):
        """Test metrics calculation."""
        with patch("joblib.load", return_value=mock_model):
            monitor = ModelPerformanceMonitor(
                model_path="dummy_model.pkl", output_dir=temp_output_dir
            )

            # Create test data
            y_true = pd.Series(["Home Win", "Draw", "Away Win"] * 10)
            y_pred = np.array(["Home Win", "Draw", "Away Win"] * 10)

            # Create properly normalized probabilities that sum to 1
            np.random.seed(42)  # For reproducible results
            raw_probs = np.random.rand(30, 3)
            y_pred_proba = raw_probs / raw_probs.sum(
                axis=1, keepdims=True
            )  # Normalize to sum to 1

            metrics = monitor._calculate_metrics(y_true, y_pred, y_pred_proba)

            assert "accuracy" in metrics
            assert "precision_macro" in metrics
            assert "recall_macro" in metrics
            assert "f1_macro" in metrics
            assert "log_loss" in metrics
            assert "prediction_confidence" in metrics
            assert "class_metrics" in metrics

    def test_evaluate_batch_with_multiple_classes(self, temp_output_dir):
        """Test batch evaluation with multiple classes."""
        # Create a dummy model file
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="stratified", random_state=42)
        # Fit with multiple classes to avoid single-label warning
        model.fit([[1, 2, 3], [2, 3, 4], [3, 4, 5]], ["Home Win", "Draw", "Away Win"])

        import joblib

        model_path = Path(temp_output_dir) / "test_model.pkl"
        joblib.dump(model, model_path)

        monitor = ModelPerformanceMonitor(
            model_path=str(model_path), output_dir=temp_output_dir
        )

        # Create test data with multiple classes to avoid confusion matrix warning
        X = pd.DataFrame(
            {
                "home_team": ["Arsenal"] * 30,
                "away_team": ["Chelsea"] * 30,
                "month": [3] * 30,
            }
        )
        # Mix of different outcomes to avoid single-label warning
        y_true = pd.Series(["Home Win"] * 20 + ["Draw"] * 5 + ["Away Win"] * 5)

        results = monitor.evaluate_batch(X, y_true, batch_id="test_batch")

        # Since we're using stratified dummy classifier, we won't get perfect accuracy
        # but we can still test the structure
        assert "accuracy" in results
        assert results["batch_id"] == "test_batch"
        assert "performance_degraded" in results


class TestMLOpsMonitoringService:
    """Test unified monitoring service."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        teams = ["Arsenal", "Chelsea", "Liverpool"]

        return pd.DataFrame(
            {
                "home_team": np.random.choice(teams, 50),
                "away_team": np.random.choice(teams, 50),
                "month": np.random.randint(1, 13, 50),
                "home_odds": np.random.uniform(1.5, 4.0, 50),
                "draw_odds": np.random.uniform(2.5, 4.5, 50),
                "away_odds": np.random.uniform(1.5, 4.0, 50),
                "home_prob_margin_adj": np.random.rand(50),
                "draw_prob_margin_adj": np.random.rand(50),
                "away_prob_margin_adj": np.random.rand(50),
                "result": np.random.choice(["Home Win", "Draw", "Away Win"], 50),
            }
        )

    @pytest.fixture
    def temp_files(self, sample_data):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save reference data
            reference_path = temp_path / "reference.parquet"
            sample_data.to_parquet(reference_path)

            # Create dummy model file with actual content
            model_path = temp_path / "model.pkl"
            from sklearn.dummy import DummyClassifier

            dummy_model = DummyClassifier(strategy="constant", constant="Home Win")
            # Fit with minimal data
            X_dummy = np.array([[1, 2, 3]])
            y_dummy = np.array(["Home Win"])
            dummy_model.fit(X_dummy, y_dummy)

            import joblib

            joblib.dump(dummy_model, model_path)

            yield {
                "reference_path": str(reference_path),
                "model_path": str(model_path),
                "output_dir": str(temp_path / "output"),
            }

    def test_monitoring_service_initialization(self, temp_files):
        """Test monitoring service can be initialized."""
        service = MLOpsMonitoringService(
            model_path=temp_files["model_path"],
            reference_data_path=temp_files["reference_path"],
            output_dir=temp_files["output_dir"],
        )

        assert service.drift_detector is not None
        assert service.performance_monitor is not None
        assert Path(temp_files["output_dir"]).exists()

    def test_generate_alerts_and_recommendations(self, temp_files):
        """Test alert and recommendation generation."""
        service = MLOpsMonitoringService(
            model_path=temp_files["model_path"],
            reference_data_path=temp_files["reference_path"],
            output_dir=temp_files["output_dir"],
        )

        # Test with drift alert
        monitoring_results = {
            "drift_monitoring": {"alert_triggered": True, "drift_share": 0.15},
            "performance_monitoring": {"performance_degraded": False},
        }

        result = service._generate_alerts_and_recommendations(monitoring_results)

        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["type"] == "data_drift"
        assert result["overall_status"] == "warning"
        assert len(result["recommendations"]) > 0

    def test_health_check(self, temp_files):
        """Test monitoring system health check."""
        service = MLOpsMonitoringService(
            model_path=temp_files["model_path"],
            reference_data_path=temp_files["reference_path"],
            output_dir=temp_files["output_dir"],
        )

        health_check = service.run_health_check()

        assert "monitoring_service_status" in health_check
        assert "components" in health_check
        assert "drift_detector" in health_check["components"]
        assert "performance_monitor" in health_check["components"]
        assert "timestamp" in health_check


class TestMonitoringIntegration:
    """Test integration scenarios."""

    def test_monitoring_with_no_model_graceful_degradation(self):
        """Test monitoring handles missing model gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal reference data
            sample_data = pd.DataFrame(
                {
                    "home_team": ["Arsenal", "Chelsea"],
                    "away_team": ["Chelsea", "Arsenal"],
                    "month": [1, 2],
                    "result": ["Home Win", "Away Win"],
                }
            )

            reference_path = Path(temp_dir) / "reference.parquet"
            sample_data.to_parquet(reference_path)

            # Try to initialize with non-existent model
            service = MLOpsMonitoringService(
                model_path="nonexistent_model.pkl",
                reference_data_path=str(reference_path),
                output_dir=temp_dir,
            )

            # Should still be able to run health check
            health_check = service.run_health_check()
            assert health_check["monitoring_service_status"] in ["warning", "error"]

    def test_monitoring_dashboard_data_structure(self):
        """Test dashboard data structure is correct."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal setup
            sample_data = pd.DataFrame(
                {
                    "home_team": ["Arsenal"],
                    "away_team": ["Chelsea"],
                    "month": [1],
                    "result": ["Home Win"],
                }
            )

            reference_path = Path(temp_dir) / "reference.parquet"
            sample_data.to_parquet(reference_path)

            service = MLOpsMonitoringService(
                model_path="nonexistent_model.pkl",
                reference_data_path=str(reference_path),
                output_dir=temp_dir,
            )

            dashboard_data = service.get_monitoring_dashboard_data(days=7)

            assert "timestamp" in dashboard_data
            assert "period_days" in dashboard_data
            assert "model_status" in dashboard_data
            assert "drift_monitoring" in dashboard_data
            assert "performance_monitoring" in dashboard_data
            assert "recent_alerts" in dashboard_data
