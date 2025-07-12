#!/usr/bin/env python3
"""
Simple Integration Test for Premier League MLOps System
Tests key components working together
"""

import logging
import os
import sys

import requests

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_endpoints():
    """Test main API endpoints"""
    api_base_url = "http://localhost:8000"

    logger.info("🚀 Testing API Integration")
    logger.info("=" * 50)

    # Test health endpoint
    logger.info("🔍 Testing API Health...")
    try:
        response = requests.get(f"{api_base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ API Health: {health_data['status']}")
            logger.info(f"   Components: {', '.join(health_data['components'].keys())}")
        else:
            logger.error(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API Health error: {e}")
        return False

    # Test prediction endpoint
    logger.info("🔮 Testing Prediction...")
    try:
        payload = {"home_team": "Arsenal", "away_team": "Chelsea"}
        response = requests.post(f"{api_base_url}/predict", json=payload)
        if response.status_code == 200:
            prediction_data = response.json()
            logger.info(
                f"✅ Prediction: {prediction_data['prediction']} (confidence: {prediction_data['confidence']:.2f})"
            )
        else:
            logger.error(f"❌ Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return False

    # Test model info
    logger.info("📊 Testing Model Info...")
    try:
        response = requests.get(f"{api_base_url}/model/info")
        if response.status_code == 200:
            model_info = response.json()
            logger.info(
                f"✅ Model: {model_info['model_name']} (accuracy: {model_info['accuracy']:.2f})"
            )
        else:
            logger.error(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Model info error: {e}")
        return False

    # Test upcoming matches
    logger.info("📅 Testing Upcoming Matches...")
    try:
        response = requests.get(f"{api_base_url}/matches/upcoming")
        if response.status_code == 200:
            matches_data = response.json()
            logger.info(f"✅ Found {len(matches_data['matches'])} upcoming matches")
        else:
            logger.error(f"❌ Upcoming matches failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Upcoming matches error: {e}")
        return False

    # Test betting simulation
    logger.info("💰 Testing Betting Simulation...")
    try:
        payload = {
            "predictions": [
                {
                    "prediction": "H",
                    "confidence": 0.7,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                }
            ]
        }
        response = requests.post(f"{api_base_url}/betting/simulate", json=payload)
        if response.status_code == 200:
            betting_data = response.json()
            logger.info(f"✅ Betting simulation: {betting_data['total_bets']} bets placed")
        else:
            logger.error(f"❌ Betting simulation failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Betting simulation error: {e}")
        return False

    logger.info("\n" + "=" * 50)
    logger.info("🎉 All API integration tests PASSED!")
    logger.info("🎯 System is working correctly!")
    return True


def test_orchestration_components():
    """Test orchestration components"""
    logger.info("🔧 Testing Orchestration Components...")

    try:
        # Test basic imports
        from monitoring.metrics_storage import MetricsStorage

        logger.info("✅ Orchestration imports successful")

        # Test metrics storage
        metrics_storage = MetricsStorage()
        logger.info("✅ Metrics storage initialized")

        return True
    except Exception as e:
        logger.error(f"❌ Orchestration components error: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Starting Simple Integration Test")
    logger.info("=" * 60)

    # Test API endpoints
    api_success = test_api_endpoints()

    # Test orchestration components
    orchestration_success = test_orchestration_components()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Integration Test Summary")
    logger.info("=" * 60)

    if api_success and orchestration_success:
        logger.info("🎉 ALL TESTS PASSED! System is fully integrated and functional.")
        sys.exit(0)
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs above.")
        sys.exit(1)
