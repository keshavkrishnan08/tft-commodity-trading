# Experimental Setup and Reproduction Guide

This document provides complete instructions for reproducing all experiments, statistical tests, and results reported in the IEEE TNNLS paper "Temporal Fusion Transformers with Variable Selection Networks for Commodity Trading: When Prediction Accuracy Diverges from Trading Profitability."

## Overview

This repository contains the code and data necessary to reproduce:
- Main trading experiments (TFT-VSN, LSTM-VSN, baselines)
- Monte Carlo bootstrap simulations (50,000 samples)
- Statistical validation tests (Jobson-Korkie, Diebold-Mariano, White's Reality Check)
- Ablation studies
- Cross-asset validation (energy commodities, precious metals, cryptocurrency)

## Repository Structure

```
.
├── src/                          # Core source code
│   ├── models/                   # Model architectures
│   │   ├── temporal_fusion_transformer.py  # TFT-VSN implementation
│   │   ├── meta_learner.py       # Meta-learner for trade selection
│   │   └── ...
│   ├── data/                     # Data loading and feature engineering
│   ├── evaluation/               # Backtesting and metrics
│   ├── training/                 # Training utilities
│   └── utils/                    # Configuration and utilities
├── experiments/                  # Experiment results
│   ├── tft_v8_sliding/          # Main TFT-VSN results
│   ├── sliding_window_v7/       # LSTM-VSN baseline results
│   └── baselines_comprehensive/  # Baseline comparisons
├── main_advanced.py              # Main experiment script (TFT-VSN)
├── main_meta_learning.py         # Meta-learner training script
├── main_tft_v8_sliding.py        # Sliding window TFT-VSN
├── monte_carlo_comprehensive.py  # Monte Carlo bootstrap simulations
├── monte_carlo_simulation.py      # Core Monte Carlo implementation
├── run_monte_carlo_baselines.py  # Statistical tests on baselines
├── references.bib                # Bibliography
├── ieee_tnnls_paper_enhanced_v3.tex  # Main paper
└── EXPERIMENTAL_SETUP.md        # This file
```

## Requirements

### Software Dependencies

```bash
# Python 3.9+
pip install torch>=1.13.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install scipy>=1.7.0
pip install tqdm>=4.62.0
```

### Hardware Requirements

- **Minimum:** CPU with 8GB RAM (training will be slow)
- **Recommended:** GPU with 8GB+ VRAM (NVIDIA RTX 3090 used in paper)
- **Storage:** ~5GB for data and results

### Data Requirements

The experiments use publicly available commodity price data:
- **Energy Commodities:** WTI Crude Oil, Brent Crude, Natural Gas, Heating Oil
- **Precious Metals:** Gold, Silver
- **Cryptocurrency:** Bitcoin (BTC-USD)

**Data Sources:**
- Energy commodities: EIA (Energy Information Administration) public data
- Precious metals: Yahoo Finance historical data
- Bitcoin: Yahoo Finance historical data

**Data Format:**
All data files should be CSV format with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume` (if available).

**Required Files:**
- `data/oil and gas.csv` - Energy commodities (WTI, Brent, Natural Gas, Heating Oil)
- `data/metals_crypto.csv` - Gold, Silver, Bitcoin
- `data/dxy.csv` - US Dollar Index (confluence factor)

**Note on Data Sharing:**
The raw data files are publicly available from the sources above. However, we do not include them in this repository due to:
1. File size constraints (data files can be large)
2. Terms of use from data providers (some require attribution or restrict redistribution)
3. Data freshness (users should download current data from original sources)

To obtain the data:
1. Energy commodities: Download from EIA website (free, public data)
2. Precious metals and Bitcoin: Download from Yahoo Finance using `yfinance` Python library
3. DXY: Download from Federal Reserve Economic Data (FRED) or Yahoo Finance

A sample script for downloading data is provided in `scripts/fetch_robust.py` (if available).

## Reproducing Main Experiments

### 1. Training TFT-VSN Model

The main TFT-VSN model with sliding window retraining:

```bash
python main_tft_v8_sliding.py
```

This script:
- Trains TFT-VSN on 2013-2017 data
- Performs sliding window retraining for each test year (2018-2022)
- Generates predictions and runs backtests
- Saves results to `experiments/tft_v8_sliding/`

**Expected Runtime:** ~8-10 hours on GPU, ~40-50 hours on CPU

**Key Parameters:**
- Training period: 2013-2017 (1,258 days)
- Test period: 2018-2022 (1,262 days, 5 years)
- Model dimension: 160
- Attention heads: 4
- Lookback window: 30 days
- Random seed: 42

### 2. Training LSTM-VSN Baseline

```bash
python main_advanced.py --model lstm_vsn
```

This trains the LSTM-VSN baseline with identical execution policies for fair comparison.

**Expected Runtime:** ~2-3 hours on GPU

### 3. Meta-Learner Training

The meta-learner is trained on 2013-2017 training period simulations:

```bash
python main_meta_learning.py
```

This script:
- Generates training data by simulating trades on 2013-2017 period
- Trains Random Forest ensemble for trade selection, position sizing, and exit timing
- Ensures strict out-of-sample evaluation (no data leakage)

**Expected Runtime:** ~1-2 hours

## Reproducing Statistical Tests

### Monte Carlo Bootstrap Simulation

To reproduce the Monte Carlo bootstrap validation (50,000 samples):

```bash
python monte_carlo_comprehensive.py
```

This script:
- Performs 10,000 bootstrap samples per asset (50,000 total)
- Computes 95% confidence intervals for returns, Sharpe ratios, and drawdowns
- Generates probability distributions
- Creates visualization figures

**Expected Runtime:** ~2-3 hours

**Output:** Results saved to `experiments/monte_carlo_comprehensive/`

### Statistical Significance Tests

To run Jobson-Korkie, Diebold-Mariano, and White's Reality Check tests:

```bash
python run_monte_carlo_baselines.py
```

This script:
- Compares TFT-VSN Sharpe ratios vs. all baselines (Jobson-Korkie)
- Tests forecast error differences (Diebold-Mariano)
- Corrects for multiple testing (White's Reality Check)

**Expected Runtime:** ~1 hour

## Reproducing Ablation Studies

Ablation studies quantify the contribution of each component. Results are in Table~\ref{tab:ablation} in the paper.

### 1. Without Variable Selection Network

The VSN reduces dimensionality from 199 to ~20 effective features. To test its impact:

1. Modify `src/models/temporal_fusion_transformer.py`:
   - Comment out VSN layer
   - Pass raw features directly to LSTM/attention layers

2. Run experiment:
   ```bash
   python main_tft_v8_sliding.py
   ```

3. Expected Impact:
   - Return reduction: -68% (from 245% to ~78%)
   - Shows VSN is critical for filtering noise

### 2. Without Multi-Head Attention

Attention enables regime-adaptive temporal focus (e.g., 87% weight on recent 7 days during COVID). To test:

1. Set `config.TFT_NUM_HEADS = 0` or use uniform attention weights

2. Run experiment:
   ```bash
   python main_tft_v8_sliding.py
   ```

3. Expected Impact:
   - Return reduction: -64% (from 245% to ~88%)
   - Shows attention is critical for regime adaptation

### 3. Without Probability Calibration

Isotonic regression calibration improves probability estimates. To test:

1. Set `calibrate=False` in `advanced_backtest.py` `run_backtest()` call

2. Run experiment:
   ```bash
   python main_tft_v8_sliding.py
   ```

3. Expected Impact:
   - Return reduction: -45% (from 245% to ~135%)
   - Shows calibration improves trade selection

### 4. Without Meta-Learner

The meta-learner optimizes trade selection, position sizing, and exit timing. To test:

1. Use `AdvancedBacktest` instead of `MetaLearningBacktest` in `main_meta_learning.py`

2. Run experiment:
   ```bash
   python main_advanced.py  # Uses heuristics instead of meta-learner
   ```

3. Expected Impact:
   - Return reduction: -30% (from 245% to ~172%)
   - Shows meta-learner adds significant value

### 5. Architecture vs. Execution Alone

To show that neither architecture nor execution alone suffices:

1. **LSTM predictions with full execution:**
   ```bash
   python main_advanced.py  # LSTM-VSN with full execution pipeline
   ```
   - Expected: +8% return (architecture matters!)

2. **TFT predictions with naive execution (buy when prob > 0.5):**
   - Modify backtest to use simple threshold instead of meta-learner
   - Expected: +67% return (execution matters!)

3. **TFT-VSN with full execution:**
   ```bash
   python main_tft_v8_sliding.py
   ```
   - Expected: +245% return (synergy: architecture + execution)

This demonstrates emergent synergy - the combination outperforms either component alone.

## Reproducing Cross-Asset Validation

The paper validates TFT-VSN on precious metals (Gold, Silver) and cryptocurrency (Bitcoin) to show generalization beyond energy commodities. Results are reported in Table~\ref{tab:cross_asset} in the paper.

### Gold Experiment

1. Ensure `data/metals_crypto.csv` contains Gold data with columns: `Date`, `Open`, `High`, `Low`, `Close`

2. Run TFT-VSN on Gold:
   ```bash
   python main_tft_v8_sliding.py
   ```
   The script automatically processes all assets in `config.TARGET_ASSETS`, including Gold.

3. Expected Results (2018-2022):
   - TFT-VSN Return: +18%
   - Sharpe Ratio: 0.9
   - Maximum Drawdown: 12.3%
   - Buy-Hold Return: +8%

### Silver Experiment

1. Same data file as Gold (`data/metals_crypto.csv`)

2. Run TFT-VSN on Silver:
   ```bash
   python main_tft_v8_sliding.py
   ```

3. Expected Results (2018-2022):
   - TFT-VSN Return: +22%
   - Sharpe Ratio: 1.1
   - Maximum Drawdown: 15.1%
   - Buy-Hold Return: +12%

### Bitcoin Experiment

1. Ensure `data/metals_crypto.csv` contains Bitcoin (BTC) data

2. Run TFT-VSN on Bitcoin:
   ```bash
   python main_tft_v8_sliding.py
   ```

3. Expected Results (2018-2022):
   - TFT-VSN Return: +28%
   - Sharpe Ratio: 1.3
   - Maximum Drawdown: 18.7%
   - Buy-Hold Return: +15%

**Note:** The paper does NOT report LSTM-VSN or TCN-VSN results for Gold, Silver, or Bitcoin in the main text (only in appendix for completeness). The main comparison is TFT-VSN vs. Buy-Hold for these assets.

### LSTM-VSN Baseline (Energy Commodities Only)

The LSTM-VSN baseline is used for comparison on energy commodities (WTI, Brent, Natural Gas, Heating Oil) only:

```bash
python main_advanced.py
```

This runs the LSTM-VSN model with:
- Variable Selection Network (same as TFT-VSN)
- LSTM architecture (128 hidden units, 2 layers)
- Multi-head attention (8 heads)
- Identical execution policies (Kelly Criterion, ATR stops, etc.)

Expected Results on WTI (2018-2022):
- LSTM-VSN Return: +8%
- Sharpe Ratio: 0.35
- Maximum Drawdown: 32.1%
- Directional Accuracy: 79% (much higher than TFT-VSN's 54%, but lower returns!)

This demonstrates the prediction-trading gap: higher accuracy does not guarantee higher returns.

## Key Implementation Details

### Random Seeds

All experiments use random seed 42 for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python: `random.seed(42)`

### Data Preprocessing

1. **Fractional Differencing:** Parameter `d=0.38` selected from training period (2013-2017) only
2. **Winsorization:** 1st and 99th percentiles clipped
3. **Normalization:** StandardScaler fit on training data only
4. **Missing Values:** Forward-filled (1.0% gaps)

### Training Configuration

- **Optimizer:** Adam (lr=1e-3, β₁=0.9, β₂=0.999)
- **Batch Size:** 64
- **Dropout:** 0.1
- **Gradient Clipping:** Max norm 1.0
- **Early Stopping:** Patience 30 epochs
- **Loss:** Quantile regression (q ∈ {0.1, 0.5, 0.9}) + L₂ regularization (λ=10⁻⁵)

### Backtesting Configuration

- **Initial Capital:** $10,000
- **Transaction Costs:** 0.6% round-trip (0.3% slippage + 0.3% commission)
- **Position Sizing:** Kelly Criterion with volatility and drawdown multipliers
- **Stop Loss:** ATR-adaptive (max(0.02, 1.5×ATR₂₀))
- **Take Profit:** 2.5×Stop Loss
- **Long-Only:** No short positions

## Expected Results

### Main Results (WTI Crude Oil, 2018-2022)

| Model | Return | Sharpe | MDD | Accuracy |
|-------|--------|--------|-----|----------|
| TFT-VSN | +245% | 4.67 | 8.2% | 54% |
| LSTM-VSN | +8% | 0.35 | 32.1% | 79% |
| Informer | +78% | 1.87 | 15.4% | 56% |
| Buy-Hold | +45% | 1.12 | 42.7% | 50% |

### Monte Carlo Bootstrap (WTI)

- Mean Return: 245% (95% CI: [198%, 292%])
- Sharpe Ratio: 4.67 (95% CI: [3.89, 5.45])
- Maximum Drawdown: 8.2% (95% CI: [5.8%, 11.1%])
- P(Return > 0): 100% (p < 0.0001)

### Statistical Tests

- **Jobson-Korkie:** TFT-VSN vs. LSTM-VSN: z = 42.3, p < 0.0001
- **Diebold-Mariano:** TFT-VSN vs. LSTM-VSN: DM = -6.73, p < 0.0001
- **White's Reality Check:** Bootstrap p-value = 0.028 (significant at α = 0.05)

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Reduce batch size in `src/utils/config.py`
   - Use CPU instead of GPU (slower but uses less memory)

2. **Missing Data Files:**
   - Ensure all CSV files are in `data/` directory
   - Check file names match expected format

3. **CUDA Errors:**
   - Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage
   - Or install correct PyTorch version for your CUDA version

4. **Different Results:**
   - Verify random seed is set to 42
   - Check data preprocessing matches paper (fractional differencing d=0.38)
   - Ensure training/test split is correct (2013-2017 train, 2018-2022 test)

## Code Organization

### Core Components

- **TFT-VSN Architecture:** `src/models/temporal_fusion_transformer.py`
  - Gated Residual Networks (GRN)
  - Variable Selection Networks (VSN)
  - Multi-head attention with causal masking

- **Meta-Learner:** `src/models/meta_learner.py`
  - Random Forest ensemble (200 trees, depth 10)
  - Trade selection, position sizing, exit timing

- **Backtesting Engine:** `src/evaluation/advanced_backtest.py`
  - Realistic transaction costs
  - ATR-adaptive stops
  - Kelly Criterion position sizing

- **Feature Engineering:** `src/data/features.py`
  - 199 features from OHLCV data
  - Technical indicators, volatility, microstructure, seasonal patterns

## Citation

If you use this code, please cite:

```bibtex
@article{krishnan2026tftvsn,
  title={Temporal Fusion Transformers with Variable Selection Networks for Commodity Trading: When Prediction Accuracy Diverges from Trading Profitability},
  author={Krishnan, Keshav},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2026}
}
```

## Contact

For questions or issues, please contact:
- **Author:** Keshav Krishnan
- **Email:** keshav-krishnan@outlook.com
- **Institution:** Olmsted Capital LLC

## License

This code is provided for research purposes. See LICENSE file for details.

