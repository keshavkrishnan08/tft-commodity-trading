# main_tft_v8_sliding_proper.py
"""
V8 TFT Baseline: Properly Trained with Early Stopping
- Same as V8 but with early stopping and more epochs
- Use this as the baseline for ablation comparisons
- Saves results to: experiments/tft_v8_sliding_proper/
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset, collate_tft_batch
from src.models.temporal_fusion_transformer import TemporalFusionTransformer
from src.evaluation.advanced_backtest import AdvancedBacktest

def calculate_class_weights(train_df, config):
    counts = {asset: {'pos': 0, 'neg': 0} for asset in config.TARGET_ASSETS}
    for asset in config.TARGET_ASSETS:
        pos = train_df[f'{asset}_Label'].sum()
        neg = len(train_df) - pos
        counts[asset]['pos'] = pos
        counts[asset]['neg'] = neg

    pos_weights = {}
    for asset in config.TARGET_ASSETS:
        pos = counts[asset]['pos']
        neg = counts[asset]['neg']
        pos_weight = neg / pos if pos > 0 else 1.0
        pos_weights[asset] = pos_weight
    return pos_weights

def run_v8_proper():
    print("="*80)
    print("V8 TFT BASELINE: PROPERLY TRAINED (Early Stopping)")
    print("="*80)

    config = Config()
    config.TFT_HIDDEN_DIM = 32
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.TFT_DROPOUT = 0.5
    config.LEARNING_RATE = 2e-4

    EXPERIMENT_DIR = Path('experiments/tft_v8_sliding_proper')
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()

    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()

    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)

    # 2. ATR Rank for filtering
    print("Calculating Volatility Ranks...")
    for asset in config.TARGET_ASSETS:
        tr = pd.concat([
            df[f'{asset}_High'] - df[f'{asset}_Low'],
            abs(df[f'{asset}_High'] - df[f'{asset}_Close'].shift(1)),
            abs(df[f'{asset}_Low'] - df[f'{asset}_Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df[f'{asset}_ATR_Rank'] = atr.rolling(60).rank(pct=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # 3. Sliding Window Setup
    test_years = [2018, 2019, 2020, 2021, 2022]
    all_predictions = {a: [] for a in config.TARGET_ASSETS}
    all_actuals = {a: [] for a in config.TARGET_ASSETS}
    all_dates = []

    for year in test_years:
        print(f"\n>>> PROCESSING YEAR: {year}")
        train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(year-5, year)]))]
        test_df = df[df['Date'].str.contains(str(year))]

        if len(test_df) == 0: continue

        # Split train into train/calib
        split_idx = int(len(train_df) * 0.85)
        t_df = train_df.iloc[:split_idx]
        c_df = train_df.iloc[split_idx:]

        # Datasets
        train_ds = TFTDataset(t_df[feature_cols], {a: t_df[f'{a}_Label'] for a in config.TARGET_ASSETS}, t_df['Date'], config.SEQUENCE_LENGTH, fit_scaler=True)
        calib_ds = TFTDataset(c_df[feature_cols], {a: c_df[f'{a}_Label'] for a in config.TARGET_ASSETS}, c_df['Date'], config.SEQUENCE_LENGTH, scaler=train_ds.scaler)
        test_ds = TFTDataset(test_df[feature_cols], {a: test_df[f'{a}_Label'] for a in config.TARGET_ASSETS}, test_df['Date'], config.SEQUENCE_LENGTH, scaler=train_ds.scaler)

        train_loader = TorchDataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_tft_batch)
        calib_loader = TorchDataLoader(calib_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_tft_batch)

        # Weights & Model
        pos_weights = calculate_class_weights(t_df, config)
        model = TemporalFusionTransformer(config).to(device)

        criteria = {a: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights[a]], device=device)) for a in config.TARGET_ASSETS}
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

        # Train with Early Stopping
        print(f"  Training on {len(train_ds)} samples, {len(train_loader)} batches")
        print(f"  Validation on {len(calib_ds)} samples")

        best_val_loss = float('inf')
        patience = 5
        no_improve = 0
        max_epochs = 30

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0
            batch_count = 0
            for feat, time, label in train_loader:
                feat, time = feat.to(device), time.to(device)
                optimizer.zero_grad()
                out, _ = model(feat, time)
                loss = sum(criteria[a](out[a], label[a].to(device)) for a in config.TARGET_ASSETS)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            avg_train_loss = train_loss / batch_count

            # Validation
            model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for feat, time, label in calib_loader:
                    feat, time = feat.to(device), time.to(device)
                    out, _ = model(feat, time)
                    loss = sum(criteria[a](out[a], label[a].to(device)) for a in config.TARGET_ASSETS)
                    val_loss += loss.item()
                    val_batch_count += 1
            avg_val_loss = val_loss / val_batch_count

            scheduler.step(avg_val_loss)

            print(f"    Epoch {epoch+1:2d}/{max_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}", end='')

            # Early stopping check
            if avg_val_loss < best_val_loss - 0.001:
                best_val_loss = avg_val_loss
                no_improve = 0
                print(" ✓")
            else:
                no_improve += 1
                print(f" (no improve: {no_improve}/{patience})")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

        # Calibrate
        model.eval()
        calibrators = {}
        with torch.no_grad():
            for a in config.TARGET_ASSETS:
                p_list, y_list = [], []
                for feat, time, label in calib_loader:
                    out, _ = model(feat.to(device), time.to(device))
                    p_list.extend(torch.sigmoid(out[a]).cpu().numpy())
                    y_list.extend(label[a].numpy())
                iso = IsotonicRegression(out_of_bounds='clip').fit(p_list, y_list)
                calibrators[a] = iso

        # Predict Test Set
        CONVICTION_THRESHOLD = 0.08
        with torch.no_grad():
            for i in range(len(test_ds)):
                feat, time, label = test_ds[i]
                out, _ = model(feat.unsqueeze(0).to(device), time.unsqueeze(0).to(device))
                date = test_ds.get_date(i)
                all_dates.append(date)

                date_row = test_df[test_df['Date'] == date]

                for a in config.TARGET_ASSETS:
                    raw_prob = torch.sigmoid(out[a]).item()
                    cal_prob = calibrators[a].predict([raw_prob])[0]

                    atr_rank = date_row[f'{a}_ATR_Rank'].values[0]
                    is_tradable = (atr_rank >= 0.20) and (atr_rank <= 0.90)

                    final_p = 0.5
                    if is_tradable and abs(cal_prob - 0.5) >= CONVICTION_THRESHOLD:
                        final_p = cal_prob

                    all_predictions[a].append(final_p)
                    all_actuals[a].append(label[a].item())

    # Run Backtest
    print("\nRunning Advanced Backtest (V8 Proper)...")
    backtester = AdvancedBacktest(config)
    results = backtester.run_backtest(
        predictions=all_predictions,
        labels=all_actuals,
        dates=all_dates,
        dataset=TFTDataset(df[feature_cols], {a: df[f'{a}_Label'] for a in config.TARGET_ASSETS}, df['Date'], config.SEQUENCE_LENGTH),
        price_data=df_raw,
        calibrate=False
    )

    # Display & Save
    print("\n" + "="*80)
    print("V8 TFT BASELINE (PROPERLY TRAINED) - RESULTS")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']:.1%}")
    print(f"Total Trades:    {results['total_trades']}")
    print("="*80)
    print("\nComparison to Original V8 (5 epochs, no early stopping):")
    print("Original V8 Return:  +245.23%")
    print("Original V8 Sharpe:  3.47")
    print(f"\nΔ Return:   {results['total_return_pct'] - 245.23:+.2f}%")
    print(f"Δ Sharpe:   {results['sharpe_ratio'] - 3.47:+.2f}")
    print("="*80)

    pd.DataFrame([results]).to_csv(EXPERIMENT_DIR / 'metrics.csv', index=False)
    results['trades_df'].to_csv(EXPERIMENT_DIR / 'trades.csv', index=False)

if __name__ == "__main__":
    run_v8_proper()
