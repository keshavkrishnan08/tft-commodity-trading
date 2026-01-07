# train_hybrid_v6.py
"""
V6 CALIBRATED HYBRID TRAINING
- Restores 199 Features (V4 Architecture)
- Trains for 5 Epochs (User Request)
- Adds POST-TRAINING CALIBRATION (Isotonic Regression)
"""
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.training.trainer import Trainer

def train():
    print("="*80)
    print("HYBRID WISDOM V6: RESTORATION & CALIBRATION")
    print("="*80)
    print("Strategy: 199 Features + Isotonic Calibration")
    print("Goal: Fix Over-trading via Meaningful Probabilities")
    print("="*80 + "\n")
    
    config = Config()
    
    # 1. Setup Experiment Directory
    EXPERIMENT_DIR = Path('experiments/hybrid_wisdom_v6_calibrated')
    config.MODEL_DIR = EXPERIMENT_DIR / 'models'
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load and Engineer Features (Full 199 Set)
    print("Loading raw data...")
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    print("Engineering FULL Hybrid Features (199 total)...")
    engineer = CalibratedFeatureEngineer(config, d=0.4) 
    df = engineer.engineer_features(df_raw)
    
    feature_cols = engineer.get_feature_columns()
    print(f"Feature Vector Size: {len(feature_cols)} features (Restored V4).")
    
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    full_dataset = MultiAssetDataset(
        features=df[feature_cols],
        labels={asset: df[f'{asset}_Label'] for asset in config.TARGET_ASSETS},
        dates=df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(df)
    )
    
    # 3. Strict Temporal Split (Fixing Data Leakage)
    # Total data is ~5400 samples.
    # Train: First 70% (2001 - 2016 approx)
    # Calib: Next 15% (2016 - 2019 approx) - UNSEEN by weights, used for Calibration
    # Test: Final 15% (2020 - 2022) - UNSEEN by weights AND Calibrator
    
    total_len = len(full_dataset)
    train_end = int(total_len * 0.70)
    calib_end = int(total_len * 0.85) 
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_end))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_end, calib_end))
    test_dataset = torch.utils.data.Subset(full_dataset, range(calib_end, total_len))
    
    train_loader = TorchDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    # Important: Val loader shuffle=False to keep order for calibration
    val_loader = TorchDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Dataset Split:")
    print(f"  Training: {len(train_dataset)} samples (up to {df.iloc[train_end]['Date']})")
    print(f"  Calibration: {len(val_dataset)} samples (up to {df.iloc[calib_end]['Date']})")
    print(f"  Hidden Test: {len(test_dataset)} samples (up to {df.iloc[-1]['Date']})")
    
    # 4. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeeklyPredictionModel(config).to(device)
    
    # 5. Start Training (5 Epochs)
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Epochs: 5 (Fast Calibration Run)")
    print(f"Features: {len(feature_cols)}")
    print("="*80 + "\n")
    
    trainer = Trainer(model, train_loader, val_loader, config=config)
    trainer.fit(epochs=5)
    
    # 6. POST-TRAINING CALIBRATION
    print("\n" + "="*80)
    print("STARTING PROBABILITY CALIBRATION")
    print("="*80)
    
    model.eval()
    all_probs = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    
    print("Collecting validation predictions...")
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            predictions, _ = model(batch_features)
            
            for asset in config.TARGET_ASSETS:
                # Get raw logits -> sigmoid -> probability
                logits = predictions[asset]
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                labels = batch_labels[asset].cpu().numpy().flatten()
                
                all_probs[asset].extend(probs)
                all_labels[asset].extend(labels)
    
    # Train Isotonic Regression per Asset
    calibrators = {}
    for asset in config.TARGET_ASSETS:
        print(f"Calibrating {asset}...")
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        
        # Fit on VALIDATION data (unseen during backprop)
        X = np.array(all_probs[asset])
        y = np.array(all_labels[asset])
        
        iso_reg.fit(X, y)
        calibrators[asset] = iso_reg
        
        # Quick check
        cal_probs = iso_reg.predict(X)
        print(f"  Raw Mean: {X.mean():.4f} -> Calibrated Mean: {cal_probs.mean():.4f}")
        
    # Save Calibrators
    calibrator_path = config.MODEL_DIR / 'calibrators.pkl'
    joblib.dump(calibrators, calibrator_path)
    print(f"Calibrators saved to {calibrator_path}")
    
    print(f"\n{'='*80}")
    print("HYBRID V6 COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    train()
