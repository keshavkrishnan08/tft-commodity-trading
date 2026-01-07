#!/usr/bin/env python3
"""
Enhanced LSTM with Sliding Window Training

Same execution model as TFT V8 Sliding:
- 5-year sliding window training
- Walk-forward validation (2018-2022)
- Early stopping with validation monitoring
- Same meta-algorithm (Kelly, ATR exits, volatility filtering)
- Same advanced backtest
- Same 199 calibrated features as TFT
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
from src.models.enhanced_lstm import EnhancedLSTM
from src.models.trading_models import ProbabilityCalibrator
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.utils.config import Config


def main():
    print("="*80)
    print("ENHANCED LSTM - SLIDING WINDOW (2018-2022)")
    print("="*80)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load and engineer features with SAME calibration as TFT (d=0.4)
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

    # Calculate volatility ranks for backtest
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

    # SLIDING WINDOW: 5 years training, test on next year
    test_years = [2018, 2019, 2020, 2021, 2022]

    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []
    all_probs_raw = {asset: [] for asset in config.TARGET_ASSETS}
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

        # Split train into train/calibration (85/15)
        split_idx = int(len(train_df) * 0.85)
        actual_train_df = train_df.iloc[:split_idx]
        calib_df = train_df.iloc[split_idx:]

        # Create raw prices for backtest
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

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True
        )
        calib_loader = torch.utils.data.DataLoader(
            calib_ds, batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.BATCH_SIZE, shuffle=False
        )

        # Initialize Enhanced LSTM model
        print(f"\n  Initializing Enhanced LSTM model...")
        config.LSTM_HIDDEN_SIZE = 128
        config.LSTM_LAYERS = 2
        config.DROPOUT = 0.4

        model = EnhancedLSTM(config).to(device)

        # Class weights for balanced loss
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

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )

        # Train with early stopping
        print(f"  Training on {len(train_ds)} samples, {len(train_loader)} batches")
        print(f"  Validation on {len(calib_ds)} samples")

        best_val_loss = float('inf')
        patience = 10
        no_improve = 0
        max_epochs = 50

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0
            batch_count = 0
            for feat, time, label in train_loader:
                feat, time = feat.to(device), time.to(device)
                optimizer.zero_grad()
                out = model(feat)
                loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                          for a in config.TARGET_ASSETS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    out = model(feat)
                    loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                              for a in config.TARGET_ASSETS)
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

        # Calibrate probabilities
        print(f"\n  Calibrating probabilities on validation set...")
        calibrator = ProbabilityCalibrator()

        model.eval()
        calib_probs = {asset: [] for asset in config.TARGET_ASSETS}
        calib_labels = {asset: [] for asset in config.TARGET_ASSETS}

        with torch.no_grad():
            for feat, time, label in calib_loader:
                feat = feat.to(device)
                out = model(feat)
                for asset in config.TARGET_ASSETS:
                    probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                    calib_probs[asset].extend(probs)
                    calib_labels[asset].extend(label[asset].numpy())

        # Convert to numpy arrays
        calib_probs = {asset: np.array(calib_probs[asset]) for asset in config.TARGET_ASSETS}
        calib_labels = {asset: np.array(calib_labels[asset]) for asset in config.TARGET_ASSETS}

        # Fit calibrator
        calibrator.fit(calib_probs, calib_labels, config.TARGET_ASSETS)

        # Predict on test set
        print(f"  Predicting on test set...")
        model.eval()
        with torch.no_grad():
            for feat, time, label in test_loader:
                feat = feat.to(device)
                out = model(feat)

                # Collect raw probabilities for this batch
                batch_raw_probs = {}
                for asset in config.TARGET_ASSETS:
                    raw_probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                    batch_raw_probs[asset] = raw_probs
                    all_probs_raw[asset].extend(raw_probs)
                    all_labels[asset].extend(label[asset].numpy())

                # Calibrate each asset
                for asset in config.TARGET_ASSETS:
                    calibrated = calibrator.transform(batch_raw_probs[asset], asset)
                    all_predictions[asset].extend(calibrated)

        # Collect valid dates (after sequence offset)
        valid_dates = [test_ds.get_date(i) for i in range(len(test_ds))]
        all_dates.extend(valid_dates)

        print(f"  ✓ Year {year} complete")

    # Backtest
    print(f"\n{'='*80}")
    print("Running Advanced Backtest...")
    print(f"{'='*80}\n")

    # Reconstruct test dataset for backtest
    test_full_df = df[df['Date'].str.contains('|'.join([str(y) for y in test_years]))]

    # Create raw_prices dict for full backtest
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

    # Run backtest
    backtest = AdvancedBacktest(config)

    results = backtest.run_backtest(
        predictions={asset: np.array(all_predictions[asset])
                    for asset in config.TARGET_ASSETS},
        labels={asset: np.array(all_labels[asset])
               for asset in config.TARGET_ASSETS},
        dates=all_dates,
        dataset=full_test_ds,
        calibrate=False  # Already calibrated
    )

    # Save results
    output_dir = Path('experiments/enhanced_lstm_sliding')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame([{k: v for k, v in results.items()
                               if k not in ['equity_curve', 'daily_returns', 'trades',
                                           'trades_df', 'asset_stats', 'runners', 'dates']}])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    # Save trades
    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(output_dir / 'trades.csv', index=False)

    # Print summary
    print("\n" + "="*80)
    print("ENHANCED LSTM SLIDING - RESULTS")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.1f}%")
    print("="*80)

    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   - metrics.csv")
    print(f"   - trades.csv")


if __name__ == '__main__':
    main()
