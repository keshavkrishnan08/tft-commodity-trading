# run_backtest_expanded.py
"""
Backtest Script for Expanded Scope (Nature MI).
- Verifies performance on Gold, Silver, BTC.
- Uses identical logic to main_tft_v8.py.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset
from src.models.temporal_fusion_transformer import TemporalFusionTransformer
from src.evaluation.advanced_backtest import AdvancedBacktest


def run_backtest():
    print("="*80)
    print("TFT EXPANDED BACKTEST (Energy + Metals + Crypto)")
    print("="*80)
    
    config = Config()
    # Add TFT config
    config.TFT_HIDDEN_DIM = 32
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.TFT_DROPOUT = 0.5
    
    EXPERIMENT_DIR = Path('experiments/tft_v8_expanded')
    MODEL_DIR = EXPERIMENT_DIR / 'models'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load & Engineer Features
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()
    
    # CRITICAL: Exclude all look-ahead columns
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)
    
    # 2. Strict Temporal Filter (June 2018 onwards per user request)
    test_slice = df[df['Date'] >= '2018-06-01'].copy()
    test_len = len(test_slice)
    
    print(f"Backtest Period: {test_slice.iloc[0]['Date']} to {test_slice.iloc[-1]['Date']} ({test_len} days)")
    
    # 3. Compute ATR Filter (20th-90th Percentile)
    print("Calculating Volatility Ranks...")
    for asset in config.TARGET_ASSETS:
        high = df[f'{asset}_High']
        low = df[f'{asset}_Low']
        close = df[f'{asset}_Close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Rolling Percentile (60-day window)
        atr_rank = atr.rolling(60).rank(pct=True)
        test_slice.loc[:, f'{asset}_ATR_Rank'] = atr_rank.tail(test_len).values
    
    def extract_raw_prices(d):
        rp = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            rp[asset] = d[cols]
        return rp
    
    # 4. Load Model, Scaler, and Calibrators
    print("Loading TFT model...")
    model = TemporalFusionTransformer(config).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'tft_best.pt', map_location=device))
    model.eval()
    
    scaler = joblib.load(MODEL_DIR / 'tft_scaler.pkl')
    calibrators = joblib.load(MODEL_DIR / 'tft_calibrators.pkl')
    print("✓ Model, Scaler, and Calibrators loaded.")
    
    # 5. Create Test Dataset
    test_dataset = TFTDataset(
        features=test_slice[feature_cols],
        labels={a: test_slice[f'{a}_Label'] for a in config.TARGET_ASSETS},
        dates=test_slice['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(test_slice)
    )
    
    # 6. Generate Calibrated Predictions
    print("Generating Calibrated Predictions...")
    
    predictions = {a: [] for a in config.TARGET_ASSETS}
    actuals = {a: [] for a in config.TARGET_ASSETS}
    dates = []
    
    skipped_trades = 0
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            features, time_feats, labels = test_dataset[i]
            
            # Predict
            x = features.unsqueeze(0).to(device)
            t = time_feats.unsqueeze(0).to(device)
            outputs, _ = model(x, t)
            
            # Get date
            date = test_dataset.get_date(i)
            dates.append(date)
            
            # Get ATR rank for this date
            date_row = test_slice[test_slice['Date'] == date]
            
            for asset in config.TARGET_ASSETS:
                # Raw probability
                raw_prob = torch.sigmoid(outputs[asset]).item()
                
                # Calibrate
                cal_prob = calibrators[asset].predict([raw_prob])[0]
                
                # Conviction Filter
                CONVICTION_THRESHOLD = 0.10  # Original V8 Threshold (>0.60 or <0.40)
                
                # ATR Filter
                atr_rank = date_row[f'{asset}_ATR_Rank'].values[0]
                is_tradable = (atr_rank >= 0.20) and (atr_rank <= 0.90)
                
                final_prob = 0.5  # Default to neutral
                if is_tradable and abs(cal_prob - 0.5) >= CONVICTION_THRESHOLD:
                    final_prob = cal_prob
                else:
                    skipped_trades += 1
                
                predictions[asset].append(final_prob)
                actuals[asset].append(labels[asset].item())
    
    print(f"Skipped {skipped_trades} potential trades due to ATR/Conviction filters.")
    
    # 7. Run Backtests per Asset (Independent Capital)
    print("\nRunning Independent Per-Asset Backtests ($10k each)...")
    
    asset_results = {}
    total_initial_capital = len(config.TARGET_ASSETS) * 10000
    
    for asset in config.TARGET_ASSETS:
        print(f"  Backtesting {asset}...")
        
        # Sub-select data for this asset
        asset_preds = {asset: predictions[asset]}
        asset_actuals = {asset: actuals[asset]}
        
        # Instantiate backtester for this run
        # We need to temporarily modify Target Assets in config for this instance
        from copy import deepcopy
        asset_config = deepcopy(config)
        asset_config.TARGET_ASSETS = [asset]
        asset_config.INITIAL_CAPITAL = 10000
        
        bt = AdvancedBacktest(asset_config)
        res = bt.run_backtest(
            predictions=asset_preds,
            labels=asset_actuals,
            dates=dates,
            dataset=test_dataset,
            price_data=df_raw,
            calibrate=False
        )
        asset_results[asset] = res

    # 8. Aggregate & Display Summary
    print("\n" + "="*80)
    print("PER-ASSET INDEPENDENT RESULTS (June 2018 - 2022)")
    print("="*80)
    print(f"{'Asset':12s} | {'Return':>10s} | {'Win Rate':>8s} | {'Sharpe':>6s} | {'Trades':>6s}")
    print("-" * 55)
    
    combined_equity = np.zeros(len(dates) + 1)
    total_pnl = 0
    
    for asset, res in asset_results.items():
        wr = res['win_rate']
        ret = res['total_return_pct']
        sharpe = res['sharpe_ratio']
        trades = res['total_trades']
        print(f"{asset:12s} | {ret:+10.2f}% | {wr:8.1%} | {sharpe:6.2f} | {trades:6d}")
        
        # Accumulate for combined stats
        total_pnl += res['total_pnl']
        combined_equity += np.array(res['equity_curve'])
        
    print("-" * 55)
    total_ret = (total_pnl / (len(config.TARGET_ASSETS) * 10000)) * 100
    print(f"{'TOTAL':12s} | {total_ret:+10.2f}% | {'-':>8s} | {'-':>6s} | {'-':>6s}")
    print("="*80)

    # 9. Save Results
    # Concatenate all trades
    all_trades = pd.concat([res['trades_df'] for res in asset_results.values() if not res['trades_df'].empty])
    all_trades.to_csv(EXPERIMENT_DIR / 'trades.csv', index=False)
    
    # Save combined equity curve (normalized)
    norm_equity = combined_equity / len(config.TARGET_ASSETS) # Average per-asset curve
    equity_df = pd.DataFrame({
        'Date': ['Entry'] + dates,
        'Equity': norm_equity
    })
    equity_df.to_csv(EXPERIMENT_DIR / 'equity_curve.csv', index=False)
    
    # Update metrics.csv with per-asset summary
    summary_data = []
    for asset, res in asset_results.items():
        summary_data.append({
            'Asset': asset,
            'Return_Pct': res['total_return_pct'],
            'Win_Rate': res['win_rate'],
            'Sharpe': res['sharpe_ratio'],
            'Trades': res['total_trades']
        })
    pd.DataFrame(summary_data).to_csv(EXPERIMENT_DIR / 'per_asset_metrics.csv', index=False)
    
    print(f"\nDetailed results saved to {EXPERIMENT_DIR}/")

if __name__ == "__main__":
    run_backtest()
