#!/usr/bin/env python
"""
Verify that stop losses are being hit INTRADAY during the 5-day holding period.

This will:
1. Take a sample trade
2. Show the entry price, SL, TP
3. Show each day's OHLC during the 5-day period
4. Verify if SL was correctly hit
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset

print("="*80)
print("STOP LOSS VERIFICATION TEST")
print("="*80)
print()
print("Loading data...")

config = Config()
loader = DataLoader(config)
df_raw = loader.get_data()

engineer = FeatureEngineer(config)
df_features = engineer.engineer_features(df_raw)
feature_cols = engineer.get_feature_columns()

# Get test set
n = len(df_features)
val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))

# Create test dataset
def extract_raw_prices(df):
    raw_prices = {}
    for asset in config.TARGET_ASSETS:
        cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
        raw_prices[asset] = df[cols]
    return raw_prices

test_dataset = MultiAssetDataset(
    features=df_features[feature_cols].iloc[val_end:],
    labels={asset: df_features[f'{asset}_Label'].iloc[val_end:]
            for asset in config.TARGET_ASSETS},
    dates=df_features['Date'].iloc[val_end:],
    sequence_length=config.SEQUENCE_LENGTH,
    scaler=None,
    fit_scaler=True,
    raw_prices=extract_raw_prices(df_features.iloc[val_end:])
)

print(f"✓ Test dataset: {len(test_dataset)} sequences\n")

# Test a sample trade
print("="*80)
print("SAMPLE TRADE VERIFICATION")
print("="*80)
print()

# Pick a random sample
sample_idx = 100  # Sample 100 from test set
asset = 'WTI'

# Get raw prices for next 5 days
print(f"Asset: {asset}")
print(f"Sample index: {sample_idx}")
print()

# Simulate a long trade
entry_prices = test_dataset.get_raw_prices(sample_idx, asset)
if entry_prices is not None:
    entry = float(entry_prices[0])  # Open

    # Set SL/TP (using 1.5x ATR)
    volatility = 0.02  # Assume 2% volatility
    sl_pct = max(0.012, volatility * 1.5)  # 3%
    tp_pct = sl_pct * 2.0  # 6%

    sl_price = entry * (1.0 - sl_pct)
    tp_price = entry * (1.0 + tp_pct)

    print(f"LONG TRADE:")
    print(f"  Entry:      ${entry:.2f}")
    print(f"  Stop Loss:  ${sl_price:.2f} ({-sl_pct*100:.1f}%)")
    print(f"  Take Profit: ${tp_price:.2f} (+{tp_pct*100:.1f}%)")
    print()

    # Check each day
    print("DAILY PRICE ACTION:")
    print("-" * 80)
    print(f"{'Day':<5} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Hit SL?':<10} {'Hit TP?'}")
    print("-" * 80)

    hit_sl = False
    hit_tp = False
    exit_day = None

    for d in range(5):
        prices = test_dataset.get_raw_prices(sample_idx + d, asset)
        if prices is None:
            print(f"{d:<5} {'No data available'}")
            break

        open_p = float(prices[0])
        high_p = float(prices[1])
        low_p = float(prices[2])
        close_p = float(prices[3])

        # Check if SL hit
        sl_hit_today = low_p <= sl_price
        tp_hit_today = high_p >= tp_price

        status = ""
        if sl_hit_today and not hit_sl and not hit_tp:
            status = "🔴 SL HIT"
            hit_sl = True
            exit_day = d
        elif tp_hit_today and not hit_tp and not hit_sl:
            status = "🟢 TP HIT"
            hit_tp = True
            exit_day = d

        print(f"{d:<5} ${open_p:<9.2f} ${high_p:<9.2f} ${low_p:<9.2f} ${close_p:<9.2f} "
              f"{'Yes' if low_p <= sl_price else 'No':<10} "
              f"{'Yes' if high_p >= tp_price else 'No':<10} {status}")

    print("-" * 80)
    print()

    if hit_sl:
        print(f"✓ RESULT: Stop loss HIT on Day {exit_day}")
        print(f"  Trade should EXIT with LOSS of {-sl_pct*100:.1f}%")
    elif hit_tp:
        print(f"✓ RESULT: Take profit HIT on Day {exit_day}")
        print(f"  Trade should EXIT with PROFIT of +{tp_pct*100:.1f}%")
    else:
        print(f"✓ RESULT: No exit, hold for full 5 days (horizon expiry)")

    print()
    print("="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print()
    print("The code SHOULD:")
    print("  1. Check each day's high/low (✓ shown above)")
    print("  2. Exit immediately when SL or TP is hit")
    print("  3. Use the SL/TP price for exit (not the actual low/high)")
    print()
    print("If this test shows SL being hit but actual results show horizon_expiry,")
    print("then there's a bug in the backtest code!")
else:
    print("❌ Could not get price data for sample")
