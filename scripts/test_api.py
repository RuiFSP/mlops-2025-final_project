#!/usr/bin/env python3
"""
Test script for the Premier League Match Predictor API.
"""

import logging
import time

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


def test_api_endpoint(endpoint, method="GET", data=None):
    """Test a single API endpoint."""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)

        if response.status_code == 200:
            logger.info(f"✅ {method} {endpoint} - Success")
            return response.json()
        else:
            logger.error(f"❌ {method} {endpoint} - Failed: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ {method} {endpoint} - Error: {e}")
        return None


def wait_for_api():
    """Wait for the API to be available."""
    logger.info("Waiting for API to be available...")

    for _ in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is available")
                return True
        except:
            time.sleep(1)

    logger.error("❌ API is not available")
    return False


def test_all_endpoints():
    """Test all API endpoints."""
    logger.info("🚀 Testing Premier League Match Predictor API")

    # Wait for API to be available
    if not wait_for_api():
        return False

    # Test root endpoint
    logger.info("Testing root endpoint...")
    test_api_endpoint("/")

    # Test health check
    logger.info("Testing health check...")
    test_api_endpoint("/health")

    # Test model info
    logger.info("Testing model info...")
    test_api_endpoint("/model/info")

    # Test single prediction
    logger.info("Testing single prediction...")
    prediction_data = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_odds": 2.1,
        "away_odds": 3.2,
        "draw_odds": 3.0,
    }
    result = test_api_endpoint("/predict", method="POST", data=prediction_data)
    if result:
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")

    # Test batch predictions
    logger.info("Testing batch predictions...")
    batch_data = [
        {
            "home_team": "Manchester City",
            "away_team": "Liverpool",
            "home_odds": 1.8,
            "away_odds": 4.0,
            "draw_odds": 3.5,
        },
        {
            "home_team": "Manchester United",
            "away_team": "Tottenham",
            "home_odds": 2.5,
            "away_odds": 2.8,
            "draw_odds": 3.2,
        },
    ]
    test_api_endpoint("/predict/batch", method="POST", data=batch_data)

    # Test upcoming matches
    logger.info("Testing upcoming matches...")
    test_api_endpoint("/matches/upcoming")

    # Test upcoming predictions
    logger.info("Testing upcoming predictions...")
    test_api_endpoint("/predictions/upcoming")

    # Test betting statistics
    logger.info("Testing betting statistics...")
    test_api_endpoint("/betting/statistics")

    # Test betting simulation
    logger.info("Testing betting simulation...")
    betting_data = {
        "predictions": [
            {
                "match_id": "test_match_1",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "prediction": "H",
                "confidence": 0.65,
                "home_odds": 2.1,
                "draw_odds": 3.0,
                "away_odds": 3.2,
            }
        ]
    }
    test_api_endpoint("/betting/simulate", method="POST", data=betting_data)

    logger.info("✅ API testing completed")
    return True


if __name__ == "__main__":
    test_all_endpoints()
