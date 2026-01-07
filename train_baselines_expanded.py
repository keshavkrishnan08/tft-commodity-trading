# train_baselines_expanded.py
"""
Train LSTM and TCN baselines on Expanded Scope (Gold, Silver, BTC).
- Uses identical meta-model and execution logic as TFT-VSN.
- Training: 2015-2017, Test: June 2018-2022.
- Ensures zero data leakage and proper temporal isolation.
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
from src.models.lstm_with_vsn import LSTMWithVSN
from src.models.tcn_with_vsn import TCNWithVSN
from src.evaluation.advanced_backtest import AdvancedBacktest


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
        pos_weight = neg / pos if pos > 0 else 1.0
        pos_weights[asset] = pos_weight
    
    return pos_weights


def train_model(model_name='LSTM'):
    print("="*80)
    print(f"{model_name} BASELINE TRAINING (Expanded Scope)")
    print("="*80)
    
    config = Config()
    # Unified architecture config (same as TFT)
    config.HIDDEN_DIM = 32
    config.NUM_HEADS = 4
    config.NUM_LAYERS = 2
    config.DROPOUT = 0.5
    config.LEARNING_RATE = 2e-4
    
    # Experiment directory
    EXPERIMENT_DIR = Path(f'experiments/{model_name.lower()}_expanded_baseline')
    MODEL_DIR = EXPERIMENT_DIR / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load & Engineer Features
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df, _ = engineer.engineer_features(df_raw)  # Unpack (df, thresholds)
    df = df.copy()
    
    # Exclude look-ahead columns
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)
    
    print(f"\nDataset: {len(df)} rows")
    print(f"Features: {config.INPUT_DIM}")
    print(f"Assets: {config.TARGET_ASSETS}")
    
    # 2. Extract raw prices for backtest
    def extract_raw_prices(d):
        rp = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            rp[asset] = d[cols]
        return rp
    
    # 3. Strict Date-Based Split (NO LEAKAGE)
    train_df = df[df['Date'] <= '2017-12-31'].copy()
    calib_df = df[(df['Date'] >= '2018-01-01') & (df['Date'] <= '2018-05-31')].copy()
    test_df = df[df['Date'] >= '2018-06-01'].copy()
    
    print(f"\nDataset Split (Date-Based):")
    print(f"  Training: {len(train_df)} samples (up to {train_df.iloc[-1]['Date']})")
    print(f"  Calibration: {len(calib_df)} samples (up to {calib_df.iloc[-1]['Date']})")
    print(f"  Test: {len(test_df)} samples (starts {test_df.iloc[0]['Date']})")
    
    # 4. Create Datasets
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
    
    # 6. Calculate class weights
    print("\nCalculating class weights...")
    pos_weights = calculate_class_weights(train_loader, config)
    
    # 7. Initialize Model
    if model_name == 'LSTM':
        model = LSTMWithVSN(config).to(device)
    elif model_name == 'TCN':
        model = TCNWithVSN(config).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"\n{model_name} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. Loss & Optimizer
    criteria = {}
    for asset in config.TARGET_ASSETS:
        weight = torch.tensor([pos_weights[asset]], device=device)
        criteria[asset] = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    
    # 9. Training Loop (5 epochs - same as TFT)
    print("\n" + "="*80)
    print(f"TRAINING {model_name} ON {str(device).upper()}")
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
            outputs = model(features)  # LSTM/TCN only take features
            
            loss = 0
            for asset in config.TARGET_ASSETS:
                asset_loss = criteria[asset](outputs[asset].squeeze(-1), labels[asset].to(device))
                loss += asset_loss
                
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
                outputs = model(features)  # LSTM/TCN only take features
                
                loss = 0
                for asset in config.TARGET_ASSETS:
                    asset_loss = criteria[asset](outputs[asset].squeeze(-1), labels[asset].to(device))
                    loss += asset_loss
                    
                    preds = (torch.sigmoid(outputs[asset]) > 0.5).cpu()
                    val_correct[asset] += (preds == labels[asset]).sum().item()
                
                val_loss += loss.item()
                val_total += len(labels[config.TARGET_ASSETS[0]])
        
        avg_val_loss = val_loss / len(calib_loader)
        avg_val_acc = np.mean([val_correct[a] / val_total for a in config.TARGET_ASSETS])
        
        print(f"Epoch {epoch+1}/5 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.3f} | Val Acc: {avg_val_acc:.3f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_DIR / f'{model_name.lower()}_best.pt')
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        
        scheduler.step()
    
    # 10. Calibration
    print("\n" + "="*80)
    print("PROBABILITY CALIBRATION")
    print("="*80)
    
    model.load_state_dict(torch.load(MODEL_DIR / f'{model_name.lower()}_best.pt'))
    model.eval()
    
    calib_preds = {a: [] for a in config.TARGET_ASSETS}
    calib_actuals = {a: [] for a in config.TARGET_ASSETS}
    
    with torch.no_grad():
        for features, time_feats, labels in calib_loader:
            features = features.to(device)
            time_feats = time_feats.to(device)
            outputs = model(features)  # LSTM/TCN only take features
            
            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(outputs[asset]).cpu().numpy()
                calib_preds[asset].extend(probs)
                calib_actuals[asset].extend(labels[asset].numpy())
    
    # Fit Isotonic Calibrators
    calibrators = {}
    for asset in config.TARGET_ASSETS:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(calib_preds[asset], calib_actuals[asset])
        calibrators[asset] = iso
    
    # Save artifacts
    joblib.dump(calibrators, MODEL_DIR / f'{model_name.lower()}_calibrators.pkl')
    joblib.dump(train_dataset.scaler, MODEL_DIR / f'{model_name.lower()}_scaler.pkl')
    
    print(f"✓ Model, scaler, and calibrators saved to {MODEL_DIR}/")
    
    # 11. Backtest on Test Set (June 2018 - 2022)
    print("\n" + "="*80)
    print(f"{model_name} BACKTEST (June 2018 - 2022)")
    print("="*80)
    
    predictions = {a: [] for a in config.TARGET_ASSETS}
    actuals = {a: [] for a in config.TARGET_ASSETS}
    dates = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            features, time_feats, labels = test_dataset[i]
            
            x = features.unsqueeze(0).to(device)
            t = time_feats.unsqueeze(0).to(device)
            outputs = model(x)  # LSTM/TCN only take features
            
            date = test_dataset.get_date(i)
            dates.append(date)
            
            for asset in config.TARGET_ASSETS:
                raw_prob = torch.sigmoid(outputs[asset]).item()
                cal_prob = calibrators[asset].predict([raw_prob])[0]
                
                predictions[asset].append(cal_prob)
                actuals[asset].append(labels[asset].item())
    
    # Run backtest per asset
    print("\nRunning Per-Asset Backtests...")
    asset_results = {}
    
    for asset in config.TARGET_ASSETS:
        from copy import deepcopy
        asset_config = deepcopy(config)
        asset_config.TARGET_ASSETS = [asset]
        asset_config.INITIAL_CAPITAL = 10000
        
        bt = AdvancedBacktest(asset_config)
        res = bt.run_backtest(
            predictions={asset: predictions[asset]},
            labels={asset: actuals[asset]},
            dates=dates,
            dataset=test_dataset,
            price_data=df_raw,
            calibrate=False
        )
        asset_results[asset] = res
    
    # Display results
    print("\n" + "="*80)
    print(f"{model_name} PER-ASSET RESULTS (June 2018 - 2022)")
    print("="*80)
    print(f"{'Asset':12s} | {'Return':\u003e10s} | {'Win Rate':\u003e8s} | {'Sharpe':\u003e6s} | {'Trades':\u003e6s}")
    print("-" * 55)
    
    for asset, res in asset_results.items():
        wr = res['win_rate']
        ret = res['total_return_pct']
        sharpe = res['sharpe_ratio']
        trades = res['total_trades']
        print(f"{asset:12s} | {ret:+10.2f}% | {wr:8.1%} | {sharpe:6.2f} | {trades:6d}")
    
    print("="*80)
    
    # Save results
    summary_data = []
    for asset, res in asset_results.items():
        summary_data.append({
            'Model': model_name,
            'Asset': asset,
            'Return_Pct': res['total_return_pct'],
            'Win_Rate': res['win_rate'],
            'Sharpe': res['sharpe_ratio'],
            'Trades': res['total_trades']
        })
    pd.DataFrame(summary_data).to_csv(EXPERIMENT_DIR / 'results.csv', index=False)
    
    print(f"\n✓ Results saved to {EXPERIMENT_DIR}/results.csv")
    return asset_results


if __name__ == "__main__":
    # Train both baselines
    print("\n" + "="*80)
    print("BASELINE COMPARISON EXPERIMENT")
    print("="*80)
    
    lstm_results = train_model('LSTM')
    print("\n\n")
    tcn_results = train_model('TCN')
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON COMPLETE")
    print("="*80)
