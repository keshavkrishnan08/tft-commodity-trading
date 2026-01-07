# Temporal Fusion Transformers for Commodity Trading

This repository contains the code and data for reproducing the experiments in:

**"Temporal Fusion Transformers with Variable Selection Networks for Commodity Trading: When Prediction Accuracy Diverges from Trading Profitability"**

*Keshav Krishnan, Olmsted Capital LLC*

Published in IEEE Transactions on Neural Networks and Learning Systems (2026)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run main experiment:**
   ```bash
   python main_tft_v8_sliding.py
   ```

3. **Reproduce statistical tests:**
   ```bash
   python monte_carlo_comprehensive.py
   python run_monte_carlo_baselines.py
   ```

For detailed instructions, see [EXPERIMENTAL_SETUP.md](EXPERIMENTAL_SETUP.md).

## What This Code Does

This code implements a complete trading system that demonstrates the **prediction-trading gap**: the phenomenon where high prediction accuracy doesn't guarantee profitable trading.

### Key Findings

- **TFT-VSN** achieves 245% return (Sharpe 4.67) on WTI crude oil despite only 54% directional accuracy
- **LSTM-VSN** achieves 79% accuracy yet only 8% return (Sharpe 0.35)
- The gap emerges from transaction costs, asymmetric P&L distributions, and regime shift vulnerability
- During COVID crisis, TFT-VSN's adaptive attention reduced trades by 79%, enabling profitable trading while LSTM lost 28.4%

### Architecture

1. **TFT-VSN Model:** Temporal Fusion Transformer with Variable Selection Networks
   - Learns which of 199 features matter most (reduces to ~20 effective features)
   - Multi-head attention adapts to market regimes (87% weight on recent 7 days during COVID)
   - Trained on 2013-2017, tested on 2018-2022 (5-year out-of-sample)

2. **Meta-Learner:** Random Forest ensemble that decides:
   - Which trades to take (trade selection)
   - How much to risk (position sizing)
   - When to exit (stop loss/take profit)

3. **Backtesting Engine:** Realistic simulation with:
   - 0.6% transaction costs
   - ATR-adaptive stops
   - Kelly Criterion position sizing
   - High-fidelity intraday simulation

## Repository Structure

```
├── src/                          # Core source code
│   ├── models/                   # Model architectures (TFT, LSTM, etc.)
│   ├── data/                     # Data loading and feature engineering
│   ├── evaluation/               # Backtesting and metrics
│   ├── training/                 # Training utilities
│   └── utils/                    # Configuration
├── experiments/                  # Experiment results
│   ├── tft_v8_sliding/          # Main TFT-VSN results
│   ├── sliding_window_v7/       # LSTM-VSN baseline
│   └── baselines_comprehensive/  # Baseline comparisons
├── main_tft_v8_sliding.py        # Main experiment (TFT-VSN)
├── main_advanced.py              # LSTM-VSN baseline
├── main_meta_learning.py         # Meta-learner training
├── monte_carlo_comprehensive.py  # Monte Carlo bootstrap (50K samples)
├── run_monte_carlo_baselines.py  # Statistical tests
├── references.bib                # Bibliography
├── ieee_tnnls_paper_enhanced_v3.tex  # Main paper
└── EXPERIMENTAL_SETUP.md        # Detailed reproduction guide
```

## Results Summary

### Main Results (WTI Crude Oil, 2018-2022)

| Model | Return | Sharpe | MDD | Accuracy |
|-------|--------|--------|-----|----------|
| **TFT-VSN** | **+245%** | **4.67** | **8.2%** | 54% |
| LSTM-VSN | +8% | 0.35 | 32.1% | 79% |
| Informer | +78% | 1.87 | 15.4% | 56% |
| Buy-Hold | +45% | 1.12 | 42.7% | 50% |

### Statistical Validation

- **Monte Carlo Bootstrap:** 50,000 samples, P(Return > 0) = 100%, p < 0.0001
- **Jobson-Korkie Test:** TFT-VSN vs. LSTM-VSN: z = 42.3, p < 0.0001
- **White's Reality Check:** Bootstrap p-value = 0.028 (survives multiple testing correction)

### Cross-Asset Validation

| Asset | TFT-VSN Return | Sharpe | Buy-Hold Return |
|-------|----------------|--------|-----------------|
| Gold | +18% | 0.9 | +8% |
| Silver | +22% | 1.1 | +12% |
| Bitcoin | +28% | 1.3 | +15% |

## Requirements

- Python 3.9+
- PyTorch 1.13+
- NumPy, Pandas, Scikit-learn
- GPU recommended (RTX 3090 used in paper)

See `requirements.txt` for complete list.

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

- **Author:** Keshav Krishnan
- **Email:** keshav-krishnan@outlook.com
- **Institution:** Olmsted Capital LLC

## License

This code is provided for research purposes. See LICENSE file for details.

## Acknowledgments

The author thanks the anonymous reviewers for constructive feedback that strengthened the manuscript.

**Conflict of Interest:** The author declares no conflicts of interest.
