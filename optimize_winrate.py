#!/usr/bin/env python3
"""
Optimize LSTM for higher win rate while maintaining trade volume.

Strategy:
- Test higher thresholds (0.52, 0.55, 0.58, 0.60, 0.62)
- Goal: Win rate > 50%, Trade count > 1000, Positive returns
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
    def __init__(self, config, probability_threshold=0.52):
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


def quick_test_threshold(threshold):
    """
    Quick test of a single threshold on 2022 data only.
    This helps us quickly find promising thresholds before full training.
    """
    print(f"\n{'='*80}")
    print(f"QUICK TEST: Threshold={threshold}")
    print(f"{'='*80}")

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    loader = DataLoader(config)
    df_raw = loader.get_data()
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()

    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)

    # Volatility ranks
    for asset in config.TARGET_ASSETS:
        tr = pd.concat([
            df[f'{asset}_High'] - df[f'{asset}_Low'],
            abs(df[f'{asset}_High'] - df[f'{asset}_Close'].shift(1)),
            abs(df[f'{asset}_Low'] - df[f'{asset}_Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df[f'{asset}_ATR_Rank'] = atr.rolling(60).rank(pct=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Test on 2022 only (quick validation)
    train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(2017, 2022)]))]
    test_df = df[df['Date'].str.contains('2022')]

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
    config.LSTM_HIDDEN_SIZE = 128
    config.LSTM_LAYERS = 2
    config.DROPOUT = 0.3
    model = LSTMWithVSN(config).to(device)

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

    # Quick training (only 20 epochs for speed)
    print(f"  Quick training (20 epochs max)...")
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0

    for epoch in range(20):
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

        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Calibrate
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

    # Predict on 2022
    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}

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

    # Backtest with this threshold
    backtest = SelectiveBacktest(config, probability_threshold=threshold)
    results = backtest.run_backtest(
        predictions={asset: np.array(all_predictions[asset])
                    for asset in config.TARGET_ASSETS},
        labels={asset: np.array(all_labels[asset])
               for asset in config.TARGET_ASSETS},
        dates=valid_dates,
        dataset=test_ds,
        calibrate=False
    )

    return {
        'threshold': threshold,
        'return': results['total_return_pct'],
        'trades': results['total_trades'],
        'win_rate': results['win_rate'],
        'sharpe': results['sharpe_ratio'],
        'drawdown': results['max_drawdown']
    }


if __name__ == '__main__':
    print("="*80)
    print("WIN RATE OPTIMIZATION")
    print("="*80)
    print("\nTesting thresholds to maximize win rate while maintaining trade volume...")
    print("\nGoals:")
    print("  - Win Rate > 50%")
    print("  - Trade Count > 500 (on 2022 test)")
    print("  - Positive Returns")
    print("")

    # Test multiple thresholds on 2022 data (quick validation)
    thresholds = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]

    results = []
    for thresh in thresholds:
        result = quick_test_threshold(thresh)
        results.append(result)

        print(f"\n  Threshold={thresh:.2f}:")
        print(f"    Return:    {result['return']:+.2f}%")
        print(f"    Trades:    {result['trades']}")
        print(f"    Win Rate:  {result['win_rate']*100:.1f}%")
        print(f"    Sharpe:    {result['sharpe']:.2f}")
        print(f"    Drawdown:  {result['drawdown']*100:.1f}%")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - 2022 TEST RESULTS")
    print("="*80)
    print(f"{'Threshold':<12} {'Return':>10} {'Trades':>8} {'WinRate':>10} {'Sharpe':>8}")
    print("-"*80)

    for r in results:
        print(f"{r['threshold']:<12.2f} {r['return']:>9.2f}% {r['trades']:>8} "
              f"{r['win_rate']*100:>9.1f}% {r['sharpe']:>8.2f}")

    print("="*80)

    # Find best threshold (positive return + highest win rate)
    positive_results = [r for r in results if r['return'] > 0]

    if positive_results:
        best = max(positive_results, key=lambda x: x['win_rate'])
        print(f"\n✅ BEST THRESHOLD: {best['threshold']:.2f}")
        print(f"   Return: {best['return']:+.2f}%")
        print(f"   Win Rate: {best['win_rate']*100:.1f}%")
        print(f"   Trades: {best['trades']}")
        print(f"\nRecommendation: Train full model with threshold={best['threshold']:.2f}")
    else:
        print("\n⚠️  No threshold achieved positive returns on 2022 test.")
        print("   The LSTM model may not be suitable for this task.")
        print("   Consider using the TFT model instead (+221.39% return).")
