# TFT-VSN

**Research code for trading seven energy, metals, and crypto markets with a Temporal Fusion
Transformer and a variable selection network. It is built to test whether a model that predicts
direction better also trades better.**

A direction classifier is scored on labels. A trading system is scored on money, after position
sizing, stops, and costs, and the two can disagree badly. This project wraps the forecaster in a
full execution layer (probability calibration, a conviction filter, Kelly-style sizing, and
volatility-scaled stops) and retrains it every year on a sliding window, so forecast quality and
trading outcome can be measured separately on the same predictions.

```
daily OHLC: WTI, Brent, natural gas, heating oil, gold, silver, bitcoin, plus the dollar index
    ->  technical indicators + fractionally differenced closes (d = 0.4), 90-day sequences
    ->  label per asset: 5-day forward return above the training window's median
    ->  TFT with a variable selection network, retrained each year on the prior five years
    ->  isotonic calibration, conviction filter, Kelly-style sizing, ATR-scaled stops
    ->  one $10,000 backtest across all seven assets
```

Accuracy is not profit. Measure both.

## The core idea

1. **Select variables before attending.** The feature stack is large (returns, volatility, moving
   averages, RSI, MACD, Bollinger bands, stochastic RSI, rate of change, regime, divergence, and
   cross-asset features for eight series). A variable selection network learns per-feature
   weights first, so attention works on what survives.
2. **Retrain every year, never look ahead.** For each test year from 2018 to 2022 the model
   trains on the five years before it, drops the last 5 days of that window so no label overlaps
   the test year, and recomputes each asset's label threshold from the training window alone.
3. **Trade only on conviction.** Probabilities are calibrated with isotonic regression on the
   last 15% of each training window, and a trade needs a calibrated probability at least 0.08
   away from 0.5.

## What it does (as configured in the code)

The repository ships code only. Experiment outputs, trade logs, figures, and the paper are
excluded by `.gitignore`, so this README quotes no performance numbers. The setup:

| Piece | Setting |
|---|---|
| Targets | WTI, Brent, natural gas, heating oil, gold, silver, bitcoin; the dollar index is an input only |
| Label | 5-day forward return above the training-window median, per asset |
| Lookback | 90 trading days |
| Model | VSN, positional encoding, causal 4-head attention (2 layers, hidden size 32), gated residual networks, one head per asset |
| Training | Adam (learning rate 2e-4, weight decay 1e-3), dropout 0.5, batch 64, 5 epochs per window, class-weighted BCE |
| Test protocol | yearly retraining for 2018 to 2022 on the prior five years, 5-day purge |
| Sizing | edge-proportional Kelly fraction, cut for volatility, losing streaks, and drawdown, clipped to 1% to 10% of capital |
| Exits | stop at max(1.2%, 1.5 x ATR) capped at 4%, target at 2 x stop capped at 12%, trailing stop, long and short |
| Costs | 0.02% slippage plus 0.01% commission, each side |
| Comparisons in code | an LSTM with temporal and cross-asset attention (`main_advanced.py`), a trade-selection meta-learner; LSTM-VSN, TCN-VSN, and TFT ablation modules |
| Statistics in code | trade-level bootstrap, block bootstrap, permutation test, Sharpe-ratio bootstrap |

## How it works

**Data.** `src/data/loader.py` reads three files that are not committed: `data/oil and gas.csv`
(EIA energy prices with a `Symbol` column), `data/metals_crypto.csv` (gold, silver, bitcoin from
Yahoo Finance), and `data/dxy.csv`. `scripts/fetch_robust.py` pulls futures bars from Yahoo
Finance, and `scripts/standardize_metals_crypto.py` reshapes the metals and crypto file.

**Features.** `src/data/features.py` builds the technical layer for every series. The hybrid and
calibrated engineers add a fractionally differenced close per target (d = 0.4, 100-step weights),
which keeps long memory while staying close to stationary. Features are standardized with a
scaler fit on the training window only.

**Model.** `src/models/temporal_fusion_transformer.py`: a variable selection network over all
features, positional encoding, multi-head self-attention with a causal mask, gated residual
networks, and a separate logit per asset.

**Execution.** `src/evaluation/advanced_backtest.py` and `src/models/trading_models.py` turn
each calibrated probability into a decision: direction, size from the edge over 0.5, a stop and
target from recent ATR, and a trailing stop that tightens with time and profit. When raw OHLC is
supplied, trades are walked day by day through highs and lows with gap fills.

**Statistics.** `monte_carlo_comprehensive.py` resamples per-trade P&L from saved trade logs
(250,000 paths). `run_monte_carlo_baselines.py` runs a bootstrap, a 20-day block bootstrap, a
permutation test, and a Sharpe-ratio bootstrap over the saved LSTM and TCN baseline runs.

## Run it

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy tqdm xgboost
# put data/oil and gas.csv, data/metals_crypto.csv, data/dxy.csv in place first
python main_tft_v8_sliding.py        # TFT-VSN, yearly retraining, writes experiments/tft_v8_sliding/
python main_advanced.py              # attention LSTM with the same execution layer
python main_meta_learning.py         # trade-selection meta-learner
python monte_carlo_comprehensive.py  # bootstrap over saved trade logs
python run_monte_carlo_baselines.py  # bootstrap and permutation tests over saved baseline runs
```

As committed, the `main_*.py` scripts stop at the feature step; see Honesty for the fix.

## Layout

```
main_tft_v8_sliding.py          TFT-VSN with yearly sliding-window retraining
main_advanced.py                attention-LSTM pipeline with the same backtest
main_meta_learning.py           meta-learner over direction-model outputs
monte_carlo_comprehensive.py    trade-level bootstrap
monte_carlo_simulation.py       parametric return simulation (see Honesty)
run_monte_carlo_baselines.py    bootstrap and permutation tests for baseline runs
src/
  data/                         loader, technical and fractional-difference features, datasets
  models/                       TFT and ablations, LSTM, TCN, attention and hybrid variants, VSN, meta-learners
  training/                     trainer, sliding window, ensemble trainer
  evaluation/                   backtests, intraday simulator, metrics, plots
  meta/                         policy, uncertainty, and manifold helpers for the meta-learner
  utils/                        config, losses, diagnostics
  visualization/                trade analytics
scripts/                        data fetch and reshaping
```

## Honesty

* No results are committed. Experiment outputs, trade logs, and the paper are git-ignored, so the
  performance figures in earlier versions of this README cannot be checked here and are not
  repeated.
* The entry points need a fix before they run. The feature engineers return a
  `(DataFrame, thresholds)` tuple that the `main_*.py` scripts use as a DataFrame, and
  `main_tft_v8_sliding.py` reads an `{asset}_ATR_Rank` column that no feature class creates.
* The sliding-window backtest needs real prices. Without raw OHLC, every trade falls back to
  `_fallback_sim`, which pays the target when the label is right and the stop when it is wrong.
  That measures hit rate, not trading, so no P&L from that path should be quoted.
* `monte_carlo_simulation.py` is a parametric illustration rescaled to fixed totals, not a
  resampling of trades. `monte_carlo_comprehensive.py` is the real trade-level bootstrap, and it
  needs the trade logs.
* The Jobson-Korkie, Diebold-Mariano, and White's Reality Check tests named in
  `run_monte_carlo_baselines.py` are not implemented yet.
* `main_advanced.py` is an attention LSTM without variable selection. The LSTM-VSN, TCN-VSN, and
  TFT ablation models exist as modules, but no committed script trains them.

MIT licensed.
