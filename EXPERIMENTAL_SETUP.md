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

Data sources are cited in the paper. Preprocessed data files should be placed in `data/` directory.

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

Ablation studies are performed by modifying the model architecture:

1. **Without Variable Selection Network:**
   - Modify `src/models/temporal_fusion_transformer.py` to bypass VSN
   - Run `main_tft_v8_sliding.py`

2. **Without Attention:**
   - Set attention heads to 0 or use uniform attention
   - Run `main_tft_v8_sliding.py`

3. **Without Calibration:**
   - Set `calibrate=False` in backtest call
   - Run `main_tft_v8_sliding.py`

Results are compared in Table~\ref{tab:ablation} in the paper.

## Reproducing Cross-Asset Validation

To validate on precious metals and cryptocurrency:

1. Ensure data files are in `data/` directory:
   - `gold prices.csv`
   - `silver prices.csv`
   - `BTC-USD.csv`

2. Run main experiment with cross-asset flag:
   ```bash
   python main_tft_v8_sliding.py --assets gold silver bitcoin
   ```

Results are reported in Table~\ref{tab:cross_asset} in the paper.

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

