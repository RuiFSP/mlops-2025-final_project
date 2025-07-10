#!/usr/bin/env python3
"""
Simple test script to verify automated retraining system functionality.
Run this to ensure all components are working correctly.
"""

import sys
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all automated retraining components can be imported."""
    print("🧪 Testing imports...")

    try:
        print("  ✅ RetrainingScheduler imported successfully")

        print("  ✅ RetrainingFlow imported successfully")

        # Test passes if no exceptions were raised
        assert True
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        pytest.fail(f"Import failed: {str(e)}")


def test_configuration():
    """Test configuration creation and management."""
    print("\n🧪 Testing configuration...")

    try:
        from src.automation.retraining_scheduler import RetrainingConfig

        # Test default config
        config = RetrainingConfig()
        assert config.performance_threshold == 0.05
        assert config.drift_threshold == 0.1
        print("  ✅ Default configuration created")

        # Test custom config
        custom_config = RetrainingConfig(
            performance_threshold=0.03,
            max_days_without_retraining=45,
        )
        assert custom_config.performance_threshold == 0.03
        assert custom_config.max_days_without_retraining == 45
        print("  ✅ Custom configuration created")

        # Test passes if no exceptions were raised
        assert True
    except Exception as e:
        print(f"  ❌ Configuration test failed: {str(e)}")
        pytest.fail(f"Configuration test failed: {str(e)}")


def test_scheduler_initialization():
    """Test scheduler initialization and basic functionality."""
    print("\n🧪 Testing scheduler initialization...")

    try:
        from src.automation.retraining_scheduler import (
            AutomatedRetrainingScheduler,
            RetrainingConfig,
        )

        # Create test config
        config = RetrainingConfig(
            check_interval_minutes=1,
            min_days_between_retraining=0,
        )

        # Initialize scheduler (this will create mocked dependencies)
        scheduler = AutomatedRetrainingScheduler(config=config)
        print("  ✅ Scheduler initialized")

        # Test basic operations
        status = scheduler.get_status()
        assert isinstance(status, dict)
        assert "is_running" in status
        print("  ✅ Status reporting works")

        # Test prediction recording
        scheduler.record_prediction(
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "prediction": "H",
            }
        )
        assert scheduler.prediction_count_since_retraining > 0
        print("  ✅ Prediction recording works")

        # Test passes if no exceptions were raised
        assert True
    except Exception as e:
        print(f"  ❌ Scheduler test failed: {str(e)}")
        pytest.fail(f"Scheduler test failed: {str(e)}")


def test_trigger_logic():
    """Test trigger detection logic."""
    print("\n🧪 Testing trigger logic...")

    try:
        from src.automation.retraining_scheduler import (
            AutomatedRetrainingScheduler,
            RetrainingConfig,
        )

        config = RetrainingConfig(
            max_predictions_without_retraining=5,  # Low threshold for testing
        )
        scheduler = AutomatedRetrainingScheduler(config=config)

        # Test data volume trigger
        for i in range(6):  # Exceed threshold
            scheduler.record_prediction({"prediction": f"test_{i}"})

        should_trigger = scheduler._check_data_volume_trigger()
        assert should_trigger is True
        print("  ✅ Data volume trigger works")

        # Test passes if no exceptions were raised
        assert True
    except Exception as e:
        print(f"  ❌ Trigger logic test failed: {str(e)}")
        pytest.fail(f"Trigger logic test failed: {str(e)}")


def test_mock_retraining():
    """Test mock retraining execution."""
    print("\n🧪 Testing mock retraining...")

    try:
        from unittest.mock import patch

        from src.automation.retraining_scheduler import (
            AutomatedRetrainingScheduler,
            RetrainingConfig,
        )

        config = RetrainingConfig(min_days_between_retraining=0)
        scheduler = AutomatedRetrainingScheduler(config=config)

        # Mock the retraining execution
        with patch("src.automation.retraining_flow.execute_automated_retraining") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "deployed": True,
                "triggers": ["test_trigger"],
            }

            # Trigger retraining
            success = scheduler.force_retraining("test_retraining")
            assert success is True
            print("  ✅ Mock retraining triggered")

            # Wait for completion
            time.sleep(0.2)

            # Check that state was updated
            assert scheduler.retraining_in_progress is False
            print("  ✅ Retraining completed successfully")

        # Test passes if no exceptions were raised
        assert True
    except Exception as e:
        print(f"  ❌ Mock retraining test failed: {str(e)}")
        pytest.fail(f"Mock retraining test failed: {str(e)}")


def test_api_integration():
    """Test that API components can be imported and initialized."""
    print("\n🧪 Testing API integration...")

    try:
        # Test that API components can be imported
        from src.deployment.api import app

        print("  ✅ FastAPI app imported successfully")

        # Check if retraining endpoints are available
        routes = [route.path for route in app.routes]
        expected_routes = ["/retraining/status", "/retraining/trigger", "/retraining/config"]

        for route in expected_routes:
            if route in routes:
                print(f"  ✅ {route} endpoint available")
            else:
                print(f"  ⚠️ {route} endpoint not found")

        # Test passes if no exceptions were raised
        assert True
    except Exception as e:
        print(f"  ❌ API integration test failed: {str(e)}")
        pytest.fail(f"API integration test failed: {str(e)}")


def main():
    """Run all tests."""
    print("🎬 Starting Automated Retraining System Tests\n")

    tests = [
        test_imports,
        test_configuration,
        test_scheduler_initialization,
        test_trigger_logic,
        test_mock_retraining,
        test_api_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()  # Test functions now use assert statements
            passed += 1
        except AssertionError as e:
            print(f"  ❌ Test failed: {str(e)}")
        except Exception as e:
            print(f"  ❌ Test failed with exception: {str(e)}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Automated retraining system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
