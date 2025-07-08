"""
Test script for the enhanced API with probability outputs
"""

import json
import sys
from pathlib import Path

import requests

# Add project root to path
sys.path.append("/home/ruifspinto/projects/mlops-2025-final_project")


def test_enhanced_api():
    """Test the enhanced API with probability outputs"""
    base_url = "http://localhost:8000"

    print("🚀 Testing Enhanced Premier League Predictor API")
    print("=" * 60)

    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API is running: python -m src.deployment.api")
        return

    # Test teams endpoint
    print("\n2. Testing teams endpoint...")
    try:
        response = requests.get(f"{base_url}/teams")
        teams = response.json()
        print(f"Available teams: {teams['teams'][:5]}...")  # Show first 5
    except Exception as e:
        print(f"Error: {e}")

    # Test model info endpoint
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info")
        info = response.json()
        print(f"Model info: {info}")
    except Exception as e:
        print(f"Error: {e}")

    # Test prediction endpoint with probabilities
    print("\n4. Testing prediction endpoint with probabilities...")

    test_matches = [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_odds": 2.1,
            "draw_odds": 3.2,
            "away_odds": 3.5,
        },
        {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "home_odds": 2.8,
            "draw_odds": 3.0,
            "away_odds": 2.4,
        },
        {
            "home_team": "Manchester City",
            "away_team": "Tottenham",
            "home_odds": 1.5,
            "draw_odds": 4.0,
            "away_odds": 6.0,
        },
    ]

    for i, match in enumerate(test_matches, 1):
        print(f"\n🔍 Test Match {i}: {match['home_team']} vs {match['away_team']}")

        # Calculate odds probabilities for comparison
        odds_sum = (
            1 / match["home_odds"] + 1 / match["draw_odds"] + 1 / match["away_odds"]
        )
        margin = (odds_sum - 1) * 100
        home_odds_prob = (1 / match["home_odds"]) / odds_sum
        draw_odds_prob = (1 / match["draw_odds"]) / odds_sum
        away_odds_prob = (1 / match["away_odds"]) / odds_sum

        print(
            f"  Betting odds: H={match['home_odds']:.2f}, D={match['draw_odds']:.2f}, A={match['away_odds']:.2f}"
        )
        print(f"  Bookmaker margin: {margin:.2f}%")
        print(
            f"  Odds probabilities (adjusted): H={home_odds_prob:.3f}, D={draw_odds_prob:.3f}, A={away_odds_prob:.3f}"
        )

        try:
            response = requests.post(
                f"{base_url}/predict",
                json=match,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                prediction = response.json()
                print(f"  🎯 Prediction: {prediction['predicted_result']}")
                print(f"  📊 Model probabilities:")
                print(f"    Home Win: {prediction['home_win_probability']:.3f}")
                print(f"    Draw: {prediction['draw_probability']:.3f}")
                print(f"    Away Win: {prediction['away_win_probability']:.3f}")
                print(f"  🎚️ Confidence: {prediction['prediction_confidence']:.3f}")

                # Compare with odds
                model_home = prediction["home_win_probability"] or 0
                model_draw = prediction["draw_probability"] or 0
                model_away = prediction["away_win_probability"] or 0

                print(f"  📈 Model vs Odds comparison:")
                print(
                    f"    Home: Model={model_home:.3f} vs Odds={home_odds_prob:.3f} (diff: {model_home-home_odds_prob:+.3f})"
                )
                print(
                    f"    Draw: Model={model_draw:.3f} vs Odds={draw_odds_prob:.3f} (diff: {model_draw-draw_odds_prob:+.3f})"
                )
                print(
                    f"    Away: Model={model_away:.3f} vs Odds={away_odds_prob:.3f} (diff: {model_away-away_odds_prob:+.3f})"
                )

            else:
                print(f"  ❌ Error: {response.status_code}")
                print(f"  Response: {response.text}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("🎉 API testing completed!")
    print("Enhanced features:")
    print("1. ✅ Probability outputs for each outcome")
    print("2. ✅ Prediction confidence scores")
    print("3. ✅ Comparison with betting odds")
    print("4. ✅ Proper margin removal from odds")


if __name__ == "__main__":
    test_enhanced_api()
