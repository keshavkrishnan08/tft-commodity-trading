# train_meta_learner.py
"""
Train the Meta-Learning model that learns which trades to take and how to size them.

This script:
1. Loads the trained direction model
2. Generates training data from validation set
3. Trains the MetaLearner (XGBoost models)
4. Saves the trained meta-learner
"""
import torch
from pathlib import Path
import sys

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.models.meta_learner import MetaLearner, generate_meta_training_data


def main():
    """Train the meta-learning model."""
    config = Config()

    print("\n" + "="*80)
    print("META-LEARNER TRAINING PIPELINE")
    print("="*80)
    print("This trains a SECOND ML model that learns:")
    print("  1. Which trades to take (based on direction model + market state)")
    print("  2. How much to position (based on certainty + conditions)")
    print("  3. When to exit (based on learned patterns)")
    print("="*80 + "\n")

    # ====================================================================
    # STEP 1: LOAD TRAINED DIRECTION MODEL
    # ====================================================================
    print("STEP 1: Loading trained direction model...")
    print("-" * 80)

    model_path = config.MODEL_DIR / 'best_model.pth'
    if not model_path.exists():
        print(f"ERROR: No trained direction model found at {model_path}")
        print("Please train the direction model first with: python main_advanced.py")
        return

    direction_model = WeeklyPredictionModel(config)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    direction_model.load_state_dict(checkpoint['model_state_dict'])
    direction_model.to(config.DEVICE)
    direction_model.eval()

    print(f"✓ Direction model loaded from {model_path}")

    # ====================================================================
    # STEP 2: LOAD VALIDATION DATA
    # ====================================================================
    print("\nSTEP 2: Loading validation data...")
    print("-" * 80)

    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()

    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()

    # Split data
    n = len(df_features)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))

    # Create validation dataset (this is what we'll use to train meta-learner)
    val_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[train_end:val_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[train_end:val_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[train_end:val_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True
    )

    print(f"✓ Validation set: {len(val_dataset)} sequences")

    # ====================================================================
    # STEP 3: GENERATE META-LEARNING TRAINING DATA
    # ====================================================================
    print("\nSTEP 3: Generating meta-learning training data...")
    print("-" * 80)
    print("Running direction model on validation set to create training examples...")

    meta_training_data = generate_meta_training_data(
        direction_model,
        val_dataset,
        config
    )

    # ====================================================================
    # STEP 4: TRAIN META-LEARNER
    # ====================================================================
    print("\nSTEP 4: Training meta-learner...")
    print("-" * 80)

    meta_learner = MetaLearner(config)
    meta_learner.fit(meta_training_data)

    # ====================================================================
    # STEP 5: ANALYZE FEATURE IMPORTANCE
    # ====================================================================
    print("\nSTEP 5: Analyzing what the meta-learner learned...")
    print("-" * 80)

    importance = meta_learner.get_feature_importance()

    print("\nTrade Selection - Top Features:")
    trade_imp = sorted(importance['trade_selection'].items(),
                      key=lambda x: x[1], reverse=True)
    for feat, imp in trade_imp[:5]:
        print(f"  {feat:20s}: {imp:.3f}")

    print("\nPosition Sizing - Top Features:")
    pos_imp = sorted(importance['position_sizing'].items(),
                    key=lambda x: x[1], reverse=True)
    for feat, imp in pos_imp[:5]:
        print(f"  {feat:20s}: {imp:.3f}")

    # ====================================================================
    # STEP 6: SAVE META-LEARNER
    # ====================================================================
    print("\nSTEP 6: Saving meta-learner...")
    print("-" * 80)

    save_path = config.MODEL_DIR / 'meta_learner.pkl'
    meta_learner.save(save_path)

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print("\n" + "="*80)
    print("META-LEARNER TRAINING COMPLETE!")
    print("="*80)

    print(f"\n✓ Trained on {len(meta_training_data)} samples")
    print(f"✓ Models: XGBoost Classifier + XGBoost Regressors")
    print(f"✓ Saved to: {save_path}")

    print("\nThe meta-learner has learned:")
    print("  1. Which trades to take (not just thresholds)")
    print("  2. How much to position (based on certainty)")
    print("  3. Optimal exit timing (from patterns)")

    print("\nNext steps:")
    print("  1. Run: python main_meta_learning.py")
    print("  2. This will use BOTH models in the backtest")
    print("  3. Direction model predicts UP/DOWN")
    print("  4. Meta-learner decides: take trade? how much?")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
