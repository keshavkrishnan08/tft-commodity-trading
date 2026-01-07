#!/usr/bin/env python3
"""
Train models with Variable Selection Network - Sliding Window

Models with VSN should achieve positive returns by:
1. Filtering noise from 219 features
2. Focusing on signal
3. Preventing overfitting

Usage:
    python train_vsn_models_sliding.py lstm  # LSTM with VSN
    python train_vsn_models_sliding.py tcn   # TCN with VSN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset
from src.models.lstm_with_vsn import LSTMWithVSN
from src.models.tcn_with_vsn import TCNWithVSN
from src.models.trading_models import ProbabilityCalibrator
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.utils.config import Config


def train_sliding(model_type='lstm'):
    """
    Train VSN model with sliding window (same as TFT V8).

    Args:
        model_type: 'lstm' or 'tcn'
    """
    print("="*80)
    print(f"{model_type.upper()} WITH VSN - SLIDING WINDOW (2018-2022)")
    print("="*80)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data with calibrated features (d=0.4, same as TFT)
    print("\n>>> LOADING DATA WITH CALIBRATED FEATURES")
    loader = DataLoader(config)
    df_raw = loader.get_data()
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()

    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)

    print(f"Total features: {config.INPUT_DIM}")

    # Calculate volatility ranks
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

    # Sliding window: 5 years training
    test_years = [2018, 2019, 2020, 2021, 2022]

    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}

    for year in test_years:
        print(f"\n{'='*80}")
        print(f">>> PROCESSING YEAR: {year}")
        print(f"{'='*80}")

        # 5-year sliding window
        train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(year-5, year)]))]
        test_df = df[df['Date'].str.contains(str(year))]

        print(f"  Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} rows)")
        print(f"  Test:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} rows)")

        # Split train into train/val (85/15)
        split_idx = int(len(train_df) * 0.85)
        actual_train_df = train_df.iloc[:split_idx]
        calib_df = train_df.iloc[split_idx:]

        # Raw prices for backtest
        raw_prices = {
            asset: test_df[[f'{asset}_Open', f'{asset}_High', f'{asset}_Low', f'{asset}_Close']]
            for asset in config.TARGET_ASSETS
        }

        # Create datasets
        train_ds = TFTDataset(
            features=actual_train_df[feature_cols],
            labels={asset: actual_train_df[f'{asset}_Label'].values
                   for asset in config.TARGET_ASSETS},
            dates=actual_train_df['Date'].values,
            sequence_length=config.SEQUENCE_LENGTH,
            fit_scaler=True
        )

        calib_ds = TFTDataset(
            features=calib_df[feature_cols],
            labels={asset: calib_df[f'{asset}_Label'].values
                   for asset in config.TARGET_ASSETS},
            dates=calib_df['Date'].values,
            sequence_length=config.SEQUENCE_LENGTH,
            scaler=train_ds.scaler,
            fit_scaler=False
        )

        test_ds = TFTDataset(
            features=test_df[feature_cols],
            labels={asset: test_df[f'{asset}_Label'].values
                   for asset in config.TARGET_ASSETS},
            dates=test_df['Date'].values,
            sequence_length=config.SEQUENCE_LENGTH,
            scaler=train_ds.scaler,
            fit_scaler=False,
            raw_prices=raw_prices
        )

        # Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True
        )
        calib_loader = torch.utils.data.DataLoader(
            calib_ds, batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.BATCH_SIZE, shuffle=False
        )

        # Initialize model with VSN
        print(f"\n  Initializing {model_type.upper()} with VSN...")
        config.LSTM_HIDDEN_SIZE = 128
        config.LSTM_LAYERS = 2
        config.DROPOUT = 0.3  # Lower dropout since VSN handles regularization

        if model_type == 'lstm':
            model = LSTMWithVSN(config).to(device)
        elif model_type == 'tcn':
            model = TCNWithVSN(config).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Class weights
        class_weights = {}
        for asset in config.TARGET_ASSETS:
            labels = actual_train_df[f'{asset}_Label'].values
            pos_count = labels.sum()
            neg_count = len(labels) - pos_count
            class_weights[asset] = neg_count / pos_count if pos_count > 0 else 1.0

        # Loss and optimizer
        criteria = {
            asset: nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([class_weights[asset]]).to(device)
            )
            for asset in config.TARGET_ASSETS
        }

        # AdamW with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

        # Training with early stopping
        print(f"  Training on {len(train_ds)} samples")
        best_val_loss = float('inf')
        patience = 10
        no_improve = 0
        max_epochs = 50

        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0
            for feat, time, label in train_loader:
                feat = feat.to(device)
                optimizer.zero_grad()
                out = model(feat)
                loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                          for a in config.TARGET_ASSETS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for feat, time, label in calib_loader:
                    feat = feat.to(device)
                    out = model(feat)
                    loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                              for a in config.TARGET_ASSETS)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(calib_loader)

            scheduler.step(avg_val_loss)

            print(f"    Epoch {epoch+1:2d}/{max_epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}", end='')

            if avg_val_loss < best_val_loss - 0.001:
                best_val_loss = avg_val_loss
                no_improve = 0
                print(" ✓")
            else:
                no_improve += 1
                print(f" ({no_improve}/{patience})")

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        # Calibrate
        print(f"\n  Calibrating probabilities...")
        calibrator = ProbabilityCalibrator()

        model.eval()
        calib_probs = {asset: [] for asset in config.TARGET_ASSETS}
        calib_labels_data = {asset: [] for asset in config.TARGET_ASSETS}

        with torch.no_grad():
            for feat, time, label in calib_loader:
                feat = feat.to(device)
                out = model(feat)
                for asset in config.TARGET_ASSETS:
                    probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                    calib_probs[asset].extend(probs)
                    calib_labels_data[asset].extend(label[asset].numpy())

        calib_probs = {a: np.array(calib_probs[a]) for a in config.TARGET_ASSETS}
        calib_labels_data = {a: np.array(calib_labels_data[a]) for a in config.TARGET_ASSETS}
        calibrator.fit(calib_probs, calib_labels_data, config.TARGET_ASSETS)

        # Predict
        print(f"  Predicting...")
        model.eval()
        with torch.no_grad():
            for feat, time, label in test_loader:
                feat = feat.to(device)
                out = model(feat)

                batch_probs = {}
                for asset in config.TARGET_ASSETS:
                    probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                    batch_probs[asset] = probs
                    all_labels[asset].extend(label[asset].numpy())

                for asset in config.TARGET_ASSETS:
                    calibrated = calibrator.transform(batch_probs[asset], asset)
                    all_predictions[asset].extend(calibrated)

        valid_dates = [test_ds.get_date(i) for i in range(len(test_ds))]
        all_dates.extend(valid_dates)

        print(f"  ✓ Year {year} complete")

    # Backtest
    print(f"\n{'='*80}")
    print("Running Backtest...")
    print(f"{'='*80}\n")

    test_full_df = df[df['Date'].str.contains('|'.join([str(y) for y in test_years]))]

    full_raw_prices = {
        asset: test_full_df[[f'{asset}_Open', f'{asset}_High', f'{asset}_Low', f'{asset}_Close']]
        for asset in config.TARGET_ASSETS
    }

    full_test_ds = TFTDataset(
        features=test_full_df[feature_cols],
        labels={asset: test_full_df[f'{asset}_Label'].values
               for asset in config.TARGET_ASSETS},
        dates=test_full_df['Date'].values,
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True,
        raw_prices=full_raw_prices
    )

    backtest = AdvancedBacktest(config)
    results = backtest.run_backtest(
        predictions={asset: np.array(all_predictions[asset])
                    for asset in config.TARGET_ASSETS},
        labels={asset: np.array(all_labels[asset])
               for asset in config.TARGET_ASSETS},
        dates=all_dates,
        dataset=full_test_ds,
        calibrate=False
    )

    # Save
    output_dir = Path(f'experiments/{model_type}_vsn_sliding')
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([{k: v for k, v in results.items()
                               if k not in ['equity_curve', 'daily_returns', 'trades',
                                           'trades_df', 'asset_stats', 'runners', 'dates']}])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(output_dir / 'trades.csv', index=False)

    # Print results
    print("\n" + "="*80)
    print(f"{model_type.upper()} WITH VSN - RESULTS")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.1f}%")
    print("="*80)
    print(f"\n✅ Results saved to {output_dir}/")


if __name__ == '__main__':
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'lstm'

    if model_type not in ['lstm', 'tcn']:
        print(f"Usage: python {sys.argv[0]} [lstm|tcn]")
        sys.exit(1)

    train_sliding(model_type=model_type)
