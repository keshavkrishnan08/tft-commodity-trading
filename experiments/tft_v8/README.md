# TFT V8 - Temporal Fusion Transformer

## Overview
This is the **V8 Temporal Fusion Transformer (TFT)** implementation for multi-asset energy trading. It replaces the LSTM architecture with a state-of-the-art attention-based model that can dynamically "prioritize" which features matter in the current market regime.

## Architecture Highlights
- **Causal Multi-Head Attention**: 4 heads with lower-triangular mask (prevents future data leakage)
- **Gated Residual Networks (GRN)**: Skip connections with learnable gates
- **Variable Selection Network (VSN)**: Learns feature importance per sample
- **Positional Encoding**: Injects temporal awareness (day-of-week, month)
- **Multi-Asset Heads**: Separate output heads for WTI, Brent, NaturalGas, HeatingOil

## Data Leakage Prevention
✅ **Strict 70/15/15 Temporal Split** (Train/Calib/Hidden Test)  
✅ **Causal Attention Mask** (each day can only attend to past days)  
✅ **Scaler Re-use** (fitted on train, applied to calib/test without re-fitting)  
✅ **Isotonic Calibration** (fitted on unseen calibration set)  

## 5-Day Intra-Period Fidelity
The backtest uses the existing `AdvancedBacktest` engine which:
- Simulates stop-loss and take-profit using daily High/Low data
- Handles gaps correctly (if market opens below stop, exit at open price)
- Enforces 5-day maximum holding period
- Applies ATR Volatility Filter (20th-90th percentile)

## Files Created
```
src/models/temporal_fusion_transformer.py  # Core TFT architecture
src/data/tft_dataset.py                    # TFT-compatible dataset
train_tft_v8.py                            # Training script
main_tft_v8.py                             # Backtest script
experiments/tft_v8/                        # Output directory
```

┌─────────────────────────────────────────────────────────────────┐
│              ULTIMATE V8: SLIDING WINDOW TFT                      │
├─────────────────────────────────────────────────────────────────┤
│  REGIME ADAPTATION: Annual retraining (Rolling 5yr Window)        │
│  PRIORITIZATION: Variable Selection Network (VSN)                │
│  GENERALIZATION: 32 Hidden Dim + 0.5 Dropout + Weight Decay      │
│  HONESTY: Causal Attention Mask (Zero Future Leakage)            │
│  VERIFICATION: 5-day Intra-period High/Low Fidelity              │
└─────────────────────────────────────────────────────────────────┘
Run

### Step 1: Train the TFT Model
```bash
python train_tft_v8.py
```
Expected output:
- Model saved to `experiments/tft_v8/models/tft_best.pt`
- Calibrators saved to `experiments/tft_v8/models/tft_calibrators.pkl`
- Scaler saved to `experiments/tft_v8/models/tft_scaler.pkl`

### Step 2: Run Backtest
```bash
python main_tft_v8.py
```
Expected output:
- Metrics saved to `experiments/tft_v8/metrics.csv`
- Trades saved to `experiments/tft_v8/trades.csv`
- Equity curve saved to `experiments/tft_v8/equity_curve.csv`

## Model Parameters
- Hidden Dimension: 128
- Attention Heads: 4
- Transformer Layers: 2
- Dropout: 0.1
- Sequence Length: 60 days
- Input Features: 199 (same as V6/V7)

## Expected Performance
The TFT should outperform the LSTM (V7) by:
1. Better handling of long-range dependencies
2. More interpretable attention (can visualize which days matter)
3. Dynamic feature selection (VSN learns what to ignore)

Target: **Sharpe Ratio > 1.0** on the 2020-2022 hidden test set.

## Verification Checklist
- [ ] Training completes without errors
- [ ] Calibrators saved successfully
- [ ] Backtest runs on 15% hidden test set
- [ ] `trades.csv` contains `exit_type` column (stop_loss, take_profit, etc.)
- [ ] All trades have ≤ 5 day holding period
- [ ] ATR filter skips trades correctly

## Next Steps (After Training)
1. Compare TFT vs LSTM attention patterns
2. Visualize which features VSN prioritizes during volatile periods
3. Consider adding macro data (Gold, VIX, Renewables) as additional features
