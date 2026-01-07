# train_tft_expanded.py
"""
TFT Training Script for Expanded Scope (Energy + Metals + Crypto).
- Nature MI Generalizability Experiment
- STRICT NO-RETUNING: Uses identical hyperparameters to V8
- 7 Assets: WTI, Brent, NG, HO, Gold, Silver, BTC
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.isotonic import IsotonicRegression

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset, collate_tft_batch
from src.models.temporal_fusion_transformer import TemporalFusionTransformer


def calculate_class_weights(train_loader, config):
    """Calculate balanced class weights for BCE loss."""
    counts = {asset: {'pos': 0, 'neg': 0} for asset in config.TARGET_ASSETS}
    
    for _, _, labels in train_loader:
        for asset in config.TARGET_ASSETS:
            pos = labels[asset].sum().item()
            neg = len(labels[asset]) - pos
            counts[asset]['pos'] += pos
            counts[asset]['neg'] += neg
    
    pos_weights = {}
    for asset in config.TARGET_ASSETS:
        pos = counts[asset]['pos']
        neg = counts[asset]['neg']
        total = pos + neg
        ratio = pos / total if total > 0 else 0.5
        pos_weight = neg / pos if pos > 0 else 1.0
        pos_weights[asset] = pos_weight
        print(f"  {asset}: pos={int(pos)}, neg={int(neg)}, ratio={ratio:.3f}, pos_weight={pos_weight:.3f}")
    
    return pos_weights


def train():
    print("="*80)
    print("TFT EXPANDED SCOPE TRAINING (Nature MI)")
    print("="*80)
    
    config = Config()
    # Add TFT-specific config - IDENTICAL TO V8
    config.TFT_HIDDEN_DIM = 32
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.TFT_DROPOUT = 0.5
    config.LEARNING_RATE = 2e-4
    
    # NEW EXPERIMENT DIR
    EXPERIMENT_DIR = Path('experiments/tft_v8_expanded')
    MODEL_DIR = EXPERIMENT_DIR / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load & Engineer Features (Includes Gold, Silver, BTC now)
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()  # De-fragment
    
    # CRITICAL: Exclude all look-ahead columns (FutureReturn and Label)
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)
    
    print(f"\nDataset: {len(df)} rows")
    print(f"Features: {config.INPUT_DIM}")
    print(f"Assets: {config.TARGET_ASSETS}")
    print("="*80)
    
    # 2. Extract raw prices for backtest
    def extract_raw_prices(d):
        rp = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            rp[asset] = d[cols]
        return rp
    
    # 3. Strict Date-Based Split (Prevent Leakage)
    # Training: 2015-2017
    # Validation: Jan 2018 - May 2018
    # Testing: June 2018 onwards (per user request)
    
    train_df = df[df['Date'] <= '2017-12-31']
    calib_df = df[(df['Date'] >= '2018-01-01') & (df['Date'] <= '2018-05-31')]
    test_df = df[df['Date'] >= '2018-06-01']
    
    print(f"\nDataset Split (Date-Based):")
    print(f"  Training: {len(train_df)} samples (up to {train_df.iloc[-1]['Date']})")
    print(f"  Calibration: {len(calib_df)} samples (up to {calib_df.iloc[-1]['Date']})")
    print(f"  Hidden Test: {len(test_df)} samples (starts {test_df.iloc[0]['Date']})")
    
    # 4. Create TFT Datasets
    train_dataset = TFTDataset(
        features=train_df[feature_cols],
        labels={a: train_df[f'{a}_Label'] for a in config.TARGET_ASSETS},
        dates=train_df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        fit_scaler=True,
        raw_prices=extract_raw_prices(train_df)
    )
    
    calib_dataset = TFTDataset(
        features=calib_df[feature_cols],
        labels={a: calib_df[f'{a}_Label'] for a in config.TARGET_ASSETS},
        dates=calib_df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(calib_df)
    )
    
    test_dataset = TFTDataset(
        features=test_df[feature_cols],
        labels={a: test_df[f'{a}_Label'] for a in config.TARGET_ASSETS},
        dates=test_df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(test_df)
    )
    
    # 5. Data Loaders
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_tft_batch
    )
    calib_loader = TorchDataLoader(
        calib_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_tft_batch
    )
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_tft_batch
    )
    
    # 6. Calculate class weights
    print("\nCalculating class weights from training data...")
    print("\nClass distribution and pos_weights:")
    pos_weights = calculate_class_weights(train_loader, config)
    
    # 7. Initialize TFT Model
    model = TemporalFusionTransformer(config).to(device)
    print(f"\nTFT Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. Loss & Optimizer
    criteria = {}
    for asset in config.TARGET_ASSETS:
        weight = torch.tensor([pos_weights[asset]], device=device)
        criteria[asset] = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    
    # 9. Training Loop
    print("\n" + "="*80)
    print("TRAINING ON", str(device).upper())
    print("="*80)
    print(f"Epochs: 5")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(5):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = {a: 0 for a in config.TARGET_ASSETS}
        train_total = 0
        
        for features, time_feats, labels in train_loader:
            features = features.to(device)
            time_feats = time_feats.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(features, time_feats)
            
            loss = 0
            for asset in config.TARGET_ASSETS:
                asset_loss = criteria[asset](outputs[asset], labels[asset].to(device))
                loss += asset_loss
                
                # Accuracy
                preds = (torch.sigmoid(outputs[asset]) > 0.5).cpu()
                train_correct[asset] += (preds == labels[asset]).sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_total += len(labels[config.TARGET_ASSETS[0]])
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = np.mean([train_correct[a] / train_total for a in config.TARGET_ASSETS])
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = {a: 0 for a in config.TARGET_ASSETS}
        val_total = 0
        
        with torch.no_grad():
            for features, time_feats, labels in calib_loader:
                features = features.to(device)
                time_feats = time_feats.to(device)
                outputs, _ = model(features, time_feats)
                
                loss = 0
                for asset in config.TARGET_ASSETS:
                    asset_loss = criteria[asset](outputs[asset], labels[asset].to(device))
                    loss += asset_loss
                    
                    preds = (torch.sigmoid(outputs[asset]) > 0.5).cpu()
                    val_correct[asset] += (preds == labels[asset]).sum().item()
                
                val_loss += loss.item()
                val_total += len(labels[config.TARGET_ASSETS[0]])
        
        avg_val_loss = val_loss / len(calib_loader)
        avg_val_acc = np.mean([val_correct[a] / val_total for a in config.TARGET_ASSETS])
        
        print(f"Epoch {epoch+1:3d}/5 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.3f} | Val Acc: {avg_val_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_DIR / 'tft_best.pt')
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        
        scheduler.step()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)
    
    # 10. Load best model and calibrate
    print("\n" + "="*80)
    print("STARTING PROBABILITY CALIBRATION")
    print("="*80)
    
    model.load_state_dict(torch.load(MODEL_DIR / 'tft_best.pt'))
    model.eval()
    
    # Collect calibration predictions
    print(f"Collecting calibration predictions...")
    calib_preds = {a: [] for a in config.TARGET_ASSETS}
    calib_actuals = {a: [] for a in config.TARGET_ASSETS}
    
    with torch.no_grad():
        for features, time_feats, labels in calib_loader:
            features = features.to(device)
            time_feats = time_feats.to(device)
            outputs, _ = model(features, time_feats)
            
            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(outputs[asset]).cpu().numpy()
                calib_preds[asset].extend(probs)
                calib_actuals[asset].extend(labels[asset].numpy())
    
    # Fit Isotonic Calibrators
    calibrators = {}
    for asset in config.TARGET_ASSETS:
        print(f"Calibrating {asset}...")
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(calib_preds[asset], calib_actuals[asset])
        calibrators[asset] = iso
        
        raw_mean = np.mean(calib_preds[asset])
        cal_mean = np.mean(iso.predict(calib_preds[asset]))
        print(f"  Raw Mean: {raw_mean:.4f} -> Calibrated Mean: {cal_mean:.4f}")
    
    # Save calibrators
    joblib.dump(calibrators, MODEL_DIR / 'tft_calibrators.pkl')
    print(f"Calibrators saved to {MODEL_DIR / 'tft_calibrators.pkl'}")
    
    # Save scaler
    joblib.dump(train_dataset.scaler, MODEL_DIR / 'tft_scaler.pkl')
    print(f"Scaler saved to {MODEL_DIR / 'tft_scaler.pkl'}")
    
    print("\n" + "="*80)
    print("TFT EXPANDED SCOPE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    train()
