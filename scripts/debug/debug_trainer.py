#!/usr/bin/env python3
"""Debug script to test ModelTrainer directly."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
from model_training.trainer import ModelTrainer

def test_trainer():
    """Test ModelTrainer directly."""
    print("🧪 Testing ModelTrainer directly...")

    # Load training data
    data_path = "data/real_data/premier_league_matches.parquet"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    print(f"✅ Loaded data: {len(df)} rows")

    # Add the same preprocessing as the retraining flow
    if "FTR" in df.columns and "result" not in df.columns:
        df["result"] = df["FTR"]
        print("✅ Created 'result' column from 'FTR' column")

    # Split data
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    val_data = df[train_size:]

    print(f"📊 Train data: {len(train_data)} rows")
    print(f"📊 Val data: {len(val_data)} rows")

    # Test trainer
    trainer = ModelTrainer("random_forest")
    print(f"🔧 Created trainer, model_type: {trainer.model_type}")
    print(f"🔧 Initial model is None: {trainer.model is None}")

    try:
        print("🚀 Starting training...")
        model = trainer.train(train_data, val_data)

        print(f"✅ Training completed")
        print(f"🔧 Returned model: {model}")
        print(f"🔧 Trainer model is None: {trainer.model is None}")
        if trainer.model is not None:
            print(f"🔧 Trainer model type: {type(trainer.model)}")
            print(f"🔧 Model n_estimators: {getattr(trainer.model, 'n_estimators', 'N/A')}")

        # Test save/load
        print("💾 Testing save/load...")
        import tempfile
        temp_dir = tempfile.mkdtemp()
        trainer.save_model(temp_dir)
        print(f"💾 Saved to: {temp_dir}")

        # Check what was saved
        import os
        files = os.listdir(temp_dir)
        print(f"💾 Files saved: {files}")

        # Load into new trainer
        new_trainer = ModelTrainer("random_forest")
        new_trainer.load_model(temp_dir)
        print(f"📥 Loaded into new trainer")
        print(f"🔧 New trainer model is None: {new_trainer.model is None}")

        if new_trainer.model is not None:
            print("✅ SUCCESS: Model save/load works!")
        else:
            print("❌ FAILURE: Model not loaded properly")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trainer()
