#!/usr/bin/env python3
"""
Train final LSTM with best config from grid search.

Best config: threshold=0.45, Focal Loss
This gave +1.19% on 2022 test set with 72 trades.

Will train on full 2018-2022 sliding window.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset
from src.models.lstm_with_vsn import LSTMWithVSN
from src.models.trading_models import ProbabilityCalibrator
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.utils.config import Config


class FocalLoss(nn.Module):
    """Focal Loss for better hard example handling."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class SelectiveBacktest(AdvancedBacktest):
    """Backtest with configurable threshold."""
    def __init__(self, config, probability_threshold=0.45):
        super().__init__(config)
        self.probability_threshold = probability_threshold

    def run_backtest(self, predictions, labels, dates, dataset, price_data=None,
                     intraday_data=None, calibrate=True):
        original_make_decision = self.decision_engine.make_decision

        def custom_make_decision(raw_probability, asset, volatility, account_balance,
                                recent_wins=0, recent_losses=0, max_drawdown=0.0):
            decision = original_make_decision(
                raw_probability, asset, volatility, account_balance,
                recent_wins, recent_losses, max_drawdown
            )

            if decision.get('take_trade', False):
                confidence = decision.get('confidence', 0)
                if confidence < self.probability_threshold:
                    decision['take_trade'] = False
                    decision['direction'] = 'hold'

            return decision

        self.decision_engine.make_decision = custom_make_decision
        result = super().run_backtest(predictions, labels, dates, dataset,
                                     price_data, intraday_data, calibrate)
        self.decision_engine.make_decision = original_make_decision

        return result


def train_lstm_final(threshold=0.45):
    """Train LSTM with best config on full 2018-2022 period."""
    print("="*80)
    print(f"LSTM WITH VSN - FINAL MODEL (Threshold={threshold}, Focal Loss)")
    print("="*80)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n>>> LOADING DATA")
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
    for asset in config.TARGET_ASSETS:
        tr = pd.concat([
            df[f'{asset}_High'] - df[f'{asset}_Low'],
            abs(df[f'{asset}_High'] - df[f'{asset}_Close'].shift(1)),
            abs(df[f'{asset}_Low'] - df[f'{asset}_Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df[f'{asset}_ATR_Rank'] = atr.rolling(60).rank(pct=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Sliding window
    test_years = [2018, 2019, 2020, 2021, 2022]

    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}

    for year in test_years:
        print(f"\n{'='*80}")
        print(f">>> PROCESSING YEAR: {year}")
        print(f"{'='*80}")

        train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(year-5, year)]))]
        test_df = df[df['Date'].str.contains(str(year))]

        print(f"  Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} rows)")
        print(f"  Test:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} rows)")

        split_idx = int(len(train_df) * 0.85)
        actual_train_df = train_df.iloc[:split_idx]
        calib_df = train_df.iloc[split_idx:]

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

        # Initialize LSTM
        print(f"\n  Initializing LSTM with VSN...")
        config.LSTM_HIDDEN_SIZE = 128
        config.LSTM_LAYERS = 2
        config.DROPOUT = 0.3
        model = LSTMWithVSN(config).to(device)

        # Use Focal Loss
        criteria = {
            asset: FocalLoss(alpha=0.25, gamma=2.0)
            for asset in config.TARGET_ASSETS
        }

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

        # Training
        print(f"  Training on {len(train_ds)} samples")
        best_val_loss = float('inf')
        patience = 3
        no_improve = 0
        max_epochs = 5

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
    print(f"Running Backtest (Threshold={threshold})...")
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

    backtest = SelectiveBacktest(config, probability_threshold=threshold)
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
    output_dir = Path(f'experiments/lstm_vsn_opt_t{int(threshold*100)}')
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
    print(f"LSTM WITH VSN - FINAL RESULTS (Threshold={threshold}, Focal Loss)")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.1f}%")
    print(f"Profit Factor:   {results.get('profit_factor', 0):.2f}")
    print("="*80)
    print(f"\n✅ Results saved to {output_dir}/")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.45,
                       help='Probability threshold (default: 0.45 from grid search)')
    args = parser.parse_args()

    train_lstm_final(threshold=args.threshold)
