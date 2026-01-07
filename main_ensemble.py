# main_ensemble.py
"""
Main script for ensemble training pipeline
Trains multiple models with different seeds and combines predictions
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.training.ensemble_trainer import EnsembleTrainer
from src.evaluation.backtest import SimpleBacktest


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Main ensemble training pipeline"""

    config = Config()
    set_seed(config.RANDOM_SEED)

    print("\n" + "="*80)
    print("ENSEMBLE TRAINING PIPELINE")
    print("="*80)
    print(f"Ensemble size: {config.ENSEMBLE_SIZE}")
    print(f"Seeds: {config.ENSEMBLE_SEEDS[:config.ENSEMBLE_SIZE]}")
    print(f"Optimizer: AdamW (weight_decay={config.WEIGHT_DECAY})")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T0={config.COSINE_T0}, T_mult={config.COSINE_T_MULT})")
    print(f"Device: {config.DEVICE}")
    print("="*80 + "\n")

    # ====================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ====================================================================
    print("STEP 1: Loading and preparing data...")
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

    # Create datasets
    train_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[:train_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[:train_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[:train_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True
    )

    val_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[train_end:val_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[train_end:val_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[train_end:val_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    test_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[val_end:],
        labels={asset: df_features[f'{asset}_Label'].iloc[val_end:]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val:   {len(val_dataset)} sequences")
    print(f"Test:  {len(test_dataset)} sequences")

    # ====================================================================
    # STEP 2: TRAIN ENSEMBLE
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 2: Training ensemble...")
    print("="*80)

    ensemble_trainer = EnsembleTrainer(config)
    histories = ensemble_trainer.train_ensemble(train_loader, val_loader)

    # ====================================================================
    # STEP 3: EVALUATE ENSEMBLE
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 3: Evaluating ensemble...")
    print("="*80)

    metrics = ensemble_trainer.evaluate_ensemble(test_loader)

    # ====================================================================
    # STEP 4: COMPARE WITH INDIVIDUAL MODELS
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 4: Comparing ensemble vs individual models...")
    print("="*80)

    comparison = ensemble_trainer.compare_with_individuals(test_loader)

    # ====================================================================
    # STEP 5: SUMMARY
    # ====================================================================
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*80)

    print("\nIndividual Model Performance:")
    for i, acc in enumerate(comparison['individual_accs']):
        print(f"  Model {i+1} (seed={config.ENSEMBLE_SEEDS[i]}): {acc:.4f}")

    print(f"\n  Individual Average: {comparison['individual_avg']:.4f}")
    print(f"  Ensemble Accuracy:  {comparison['ensemble_acc']:.4f}")
    print(f"  Improvement:        {comparison['improvement']:+.4f} ({comparison['improvement']*100:+.2f}%)")

    print(f"\nSaved Files:")
    print(f"  Ensemble models: {config.ENSEMBLE_DIR / 'model_*.pth'}")
    print(f"  Metadata:        {config.ENSEMBLE_DIR / 'ensemble_metadata.json'}")

    print("\n" + "="*80)
    print("Run 'python main.py' for single-model baseline comparison")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
