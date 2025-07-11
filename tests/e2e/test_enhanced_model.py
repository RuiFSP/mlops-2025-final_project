"""
Enhanced Premier League Match Predictor Test Script

This script demonstrates the improved model that:
1. Outputs probabilities instead of just predictions
2. Removes bookmaker margin from odds
3. Evaluates using Brier score to compare against betting market
4. Uses better features and model architecture
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_preprocessing.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.model_training.trainer import ModelTrainer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


@pytest.mark.e2e
@pytest.mark.timeout(300)  # 5 minute timeout for model training
def main():
    print("🚀 Enhanced Premier League Match Predictor")
    print("=" * 60)

    # Load and prepare data
    print("📊 Loading data...")
    data_loader = DataLoader("data/real_data/")
    train_data, val_data = data_loader.load_and_split()

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Check if we have odds data
    odds_cols = ["home_odds", "draw_odds", "away_odds"]
    has_odds = all(col in train_data.columns for col in odds_cols)
    print(f"Has odds data: {has_odds}")

    if has_odds:
        print("\n📈 Odds data sample:")
        print(train_data[odds_cols].head())

        # Show margin calculation
        sample_odds = train_data[odds_cols].iloc[0]
        implied_probs = 1 / sample_odds
        total_prob = implied_probs.sum()
        margin = (total_prob - 1) * 100
        print("\n🔍 Example odds analysis:")
        print(f"Original odds: {sample_odds.values}")
        print(f"Implied probabilities: {implied_probs.values}")
        print(f"Total probability: {total_prob:.3f}")
        print(f"Bookmaker margin: {margin:.2f}%")

        # After removing margin
        adjusted_probs = implied_probs / total_prob
        print(f"Adjusted probabilities: {adjusted_probs.values}")
        print(f"Sum after adjustment: {adjusted_probs.sum():.3f}")

    # Train improved model
    print("\n" + "=" * 60)
    print("🧠 Training improved model...")
    trainer = ModelTrainer(model_type="random_forest")
    trainer.train(train_data, val_data)

    # Evaluate model
    print("\n" + "=" * 60)
    print("📈 Evaluating model...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(trainer, val_data)

    print("\n🏆 Model Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        if "brier" in metric:
            print(f"📊 {metric}: {value:.4f}")
        else:
            print(f"✅ {metric}: {value:.4f}")

    # Test probability predictions
    print("\n" + "=" * 60)
    print("🎯 Testing probability predictions...")

    # Create a sample prediction
    sample_match = pd.DataFrame(
        {
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_odds": [2.1],
            "draw_odds": [3.2],
            "away_odds": [3.5],
            "date": ["2024-01-01"],
            "home_score": [0],  # Placeholder
            "away_score": [0],  # Placeholder
            "season": ["2023-24"],
        }
    )

    # Preprocess the sample
    processed_sample = data_loader.preprocess_data(sample_match)

    # Make predictions
    prediction = trainer.predict(processed_sample)
    probabilities = trainer.predict_proba(processed_sample)
    class_order = trainer.get_class_order()

    print(
        f"\n⚽ Sample match: {sample_match['home_team'].iloc[0]} vs {sample_match['away_team'].iloc[0]}"
    )
    print(f"🎯 Predicted result: {prediction[0]}")
    print(f"📊 Class order: {class_order}")
    print(f"📈 Probabilities: {probabilities[0]}")

    # Create probability mapping
    prob_mapping = dict(zip(class_order, probabilities[0], strict=False))
    print("\n📊 Model probability breakdown:")
    for cls, prob in prob_mapping.items():
        result_name = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[cls]
        print(f"  {result_name}: {prob:.3f} ({prob*100:.1f}%)")

    # Compare with betting odds
    if has_odds:
        print("\n💰 Betting odds comparison:")
        sample_odds = sample_match[["home_odds", "draw_odds", "away_odds"]].iloc[0]
        implied_probs = 1 / sample_odds
        total_prob = implied_probs.sum()
        adjusted_probs = implied_probs / total_prob

        print(
            f"Original odds: H={sample_odds['home_odds']:.2f}, D={sample_odds['draw_odds']:.2f}, A={sample_odds['away_odds']:.2f}"
        )
        print(f"Bookmaker margin: {((total_prob - 1) * 100):.2f}%")
        print("\n📊 Adjusted odds probabilities:")
        print(
            f"  Home Win: {adjusted_probs['home_odds']:.3f} ({adjusted_probs['home_odds']*100:.1f}%)"
        )
        print(f"  Draw: {adjusted_probs['draw_odds']:.3f} ({adjusted_probs['draw_odds']*100:.1f}%)")
        print(
            f"  Away Win: {adjusted_probs['away_odds']:.3f} ({adjusted_probs['away_odds']*100:.1f}%)"
        )

        # Model vs Odds comparison
        model_probs = {
            "Home Win": prob_mapping.get("H", 0),
            "Draw": prob_mapping.get("D", 0),
            "Away Win": prob_mapping.get("A", 0),
        }

        odds_probs = {
            "Home Win": adjusted_probs["home_odds"],
            "Draw": adjusted_probs["draw_odds"],
            "Away Win": adjusted_probs["away_odds"],
        }

        print("\n🔍 Model vs Odds Comparison:")
        for outcome in ["Home Win", "Draw", "Away Win"]:
            model_prob = model_probs[outcome]
            odds_prob = odds_probs[outcome]
            diff = model_prob - odds_prob
            status = "🟢" if abs(diff) < 0.05 else "🔴"
            print(
                f"  {outcome}: Model={model_prob:.3f}, Odds={odds_prob:.3f}, Diff={diff:+.3f} {status}"
            )

    print("\n" + "=" * 60)
    print("🎉 Model improvements completed!")
    print("Key enhancements:")
    print("1. ✅ Probability outputs instead of just predictions")
    print("2. ✅ Brier score evaluation")
    print("3. ✅ Comparison with betting odds (margin removed)")
    print("4. ✅ Better model architecture with balanced classes")
    print("5. ✅ More sophisticated feature engineering")

    # Show if model beats the market
    if "brier_improvement" in metrics:
        improvement = metrics["brier_improvement"]
        improvement_pct = metrics["brier_improvement_pct"]
        if improvement > 0:
            print("\n🏆 Model BEATS the betting market!")
            print(f"   Brier score improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            print("\n📊 Model performs similar to betting market")
            print(f"   Brier score difference: {improvement:.4f} ({improvement_pct:.2f}%)")


if __name__ == "__main__":
    main()
