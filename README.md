# TFT-VSN for Commodity Trading - Reproduction Guide

This repository contains code to reproduce all experiments from the paper:

**"Temporal Fusion Transformers with Variable Selection Networks for Commodity Trading: When Prediction Accuracy Diverges from Trading Profitability"**

*Keshav Krishnan, Olmsted Capital LLC*

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib seaborn scipy tqdm xgboost
   ```

2. **Download data:**
   - Energy commodities: Download from EIA (Energy Information Administration)
   - Precious metals & Bitcoin: Download from Yahoo Finance using `yfinance` library
   - Place files in `data/` directory:
     - `data/oil and gas.csv` - WTI, Brent, Natural Gas, Heating Oil
     - `data/metals_crypto.csv` - Gold, Silver, Bitcoin
     - `data/dxy.csv` - US Dollar Index

3. **Run main experiment (TFT-VSN):**
   ```bash
   python main_tft_v8_sliding.py
   ```
   This trains TFT-VSN on 2013-2017, tests on 2018-2022, saves results to `experiments/tft_v8_sliding/`

4. **Run LSTM-VSN baseline:**
   ```bash
   python main_advanced.py
   ```
   This runs the LSTM-VSN baseline with identical execution policies

5. **Run meta-learner training:**
   ```bash
   python main_meta_learning.py
   ```
   Trains meta-learner on 2013-2017 training period (no data leakage)

6. **Run Monte Carlo bootstrap:**
   ```bash
   python monte_carlo_comprehensive.py
   ```
   Performs 50,000 bootstrap samples for statistical validation

7. **Run statistical tests:**
   ```bash
   python run_monte_carlo_baselines.py
   ```
   Runs Jobson-Korkie, Diebold-Mariano, and White's Reality Check tests

## Repository Structure

```
├── src/                          # Core source code
│   ├── models/                   # Model architectures
│   │   ├── temporal_fusion_transformer.py  # TFT-VSN
│   │   ├── meta_learner.py       # Meta-learner
│   │   ├── lstm_with_vsn.py       # LSTM-VSN baseline
│   │   └── tcn_with_vsn.py        # TCN baseline
│   ├── data/                      # Data loading and feature engineering
│   │   ├── loader.py              # Multi-asset data loader
│   │   ├── features.py            # Feature engineering (199 features)
│   │   └── calibrated_features.py # Fractional differencing
│   ├── evaluation/                # Backtesting and metrics
│   │   ├── advanced_backtest.py   # Realistic backtesting engine
│   │   ├── high_fidelity_simulator.py  # Intraday simulation
│   │   └── metrics.py             # Performance metrics
│   ├── training/                  # Training utilities
│   └── utils/                     # Configuration
├── main_tft_v8_sliding.py        # Main TFT-VSN experiment
├── main_advanced.py              # LSTM-VSN baseline
├── main_meta_learning.py          # Meta-learner training
├── monte_carlo_comprehensive.py  # Monte Carlo bootstrap (50K samples)
├── monte_carlo_simulation.py     # Core Monte Carlo implementation
├── run_monte_carlo_baselines.py   # Statistical tests
└── README.md                      # This file
```

## Expected Results

### Main Results (WTI Crude Oil, 2018-2022)

| Model | Return | Sharpe | MDD | Accuracy |
|-------|--------|--------|-----|----------|
| TFT-VSN | +245% | 4.67 | 8.2% | 54% |
| LSTM-VSN | +8% | 0.35 | 32.1% | 79% |
| Buy-Hold | +45% | 1.12 | 42.7% | 50% |

### Cross-Asset Validation (2018-2022)

| Asset | TFT-VSN Return | Sharpe | Buy-Hold Return |
|-------|----------------|--------|-----------------|
| Gold | +18% | 0.9 | +8% |
| Silver | +22% | 1.1 | +12% |
| Bitcoin | +28% | 1.3 | +15% |

### Monte Carlo Bootstrap (WTI)

- Mean Return: 245% (95% CI: [198%, 292%])
- Sharpe Ratio: 4.67 (95% CI: [3.89, 5.45])
- P(Return > 0): 100% (p < 0.0001)

## Key Implementation Details

### Random Seeds
All experiments use random seed 42 for reproducibility:
- `torch.manual_seed(42)`
- `np.random.seed(42)`
- `random.seed(42)`

### Data Preprocessing
- **Fractional Differencing:** Parameter `d=0.38` (selected from 2013-2017 training period only)
- **Winsorization:** 1st and 99th percentiles clipped
- **Normalization:** StandardScaler fit on training data only
- **Missing Values:** Forward-filled

### Training Configuration
- **Optimizer:** Adam (lr=2e-4, weight_decay=1e-3)
- **Batch Size:** 64
- **Dropout:** 0.5
- **Epochs:** 5 per sliding window
- **Loss:** BCEWithLogitsLoss with class weights

### Backtesting Configuration
- **Initial Capital:** $10,000
- **Transaction Costs:** 0.6% round-trip (0.3% slippage + 0.3% commission)
- **Position Sizing:** Kelly Criterion with volatility adjustments
- **Stop Loss:** ATR-adaptive (max(0.02, 1.5×ATR₂₀))
- **Take Profit:** 2.5×Stop Loss
- **Long-Only:** No short positions

## Reproducing Specific Experiments

### TFT-VSN Main Experiment
```bash
python main_tft_v8_sliding.py
```
- Trains on 2013-2017
- Tests on 2018-2022 (sliding window, retrains each year)
- Saves predictions, trades, and metrics to `experiments/tft_v8_sliding/`

### LSTM-VSN Baseline
```bash
python main_advanced.py
```
- Uses LSTM architecture with Variable Selection Network
- Same execution policies as TFT-VSN for fair comparison
- Results show prediction-trading gap (79% accuracy but only 8% return)

### Cross-Asset Validation (Gold, Silver, Bitcoin)
The main script processes all assets in `config.TARGET_ASSETS`. Results for Gold, Silver, and Bitcoin are automatically included.

### Monte Carlo Bootstrap
```bash
python monte_carlo_comprehensive.py
```
- Performs 50,000 bootstrap samples
- Computes 95% confidence intervals
- Generates probability distributions
- Results saved to `experiments/monte_carlo_comprehensive/`

### Statistical Tests
```bash
python run_monte_carlo_baselines.py
```
- Jobson-Korkie test (Sharpe ratio comparison)
- Diebold-Mariano test (forecast errors)
- White's Reality Check (multiple testing correction)

## Troubleshooting

**Out of Memory:**
- Reduce batch size in `src/utils/config.py`
- Use CPU instead of GPU (slower but uses less memory)

**Missing Data Files:**
- Ensure all CSV files are in `data/` directory
- Check file names match expected format (see data loading code)

**Different Results:**
- Verify random seed is set to 42
- Check data preprocessing matches paper (fractional differencing d=0.38)
- Ensure training/test split is correct (2013-2017 train, 2018-2022 test)

## Citation

```bibtex
@article{krishnan2026tftvsn,
  title={Temporal Fusion Transformers with Variable Selection Networks for Commodity Trading: When Prediction Accuracy Diverges from Trading Profitability},
  author={Krishnan, Keshav},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026}
}
```

## Contact

- **Author:** Keshav Krishnan
- **Email:** keshav-krishnan@outlook.com
- **Institution:** Olmsted Capital LLC
