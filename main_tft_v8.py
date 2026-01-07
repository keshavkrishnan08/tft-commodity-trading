# main_tft_v8.py
"""
V8 TFT BACKTEST
- Loads trained TFT model
- Uses Isotonic Calibration
- Integrates with AdvancedBacktest (5-day intra-period fidelity)
- Applies ATR Filter (20th-90th percentile)
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
    print("TFT V8 BACKTEST")
    print("="*80)
    
    config = Config()
    # Add TFT config
    config.TFT_HIDDEN_DIM = 32
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.TFT_DROPOUT = 0.5
    
    EXPERIMENT_DIR = Path('experiments/tft_v8')
    MODEL_DIR = EXPERIMENT_DIR / 'models'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load & Engineer Features
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy() # De-fragment
    
    # CRITICAL: Exclude all look-ahead columns
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)
    
    # 2. Strict Temporal Filter (Tail 15% only - Hidden Test Set)
    test_len = int(len(df) * 0.15)
    test_slice = df.tail(test_len).copy()
    
    print(f"Backtest Period: {test_slice.iloc[0]['Date']} to {test_slice.iloc[-1]['Date']} ({len(test_slice)} days)")
    
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
                
                # Conviction Filter: Only trade if prob is significantly different from 0.5
                CONVICTION_THRESHOLD = 0.10  # Must be > 0.60 or < 0.40
                
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
    
    print(f"Skipped {skipped_trades} predicted trades due to ATR Filter.")
    
    # 7. Run Backtest
    print("\nRunning Advanced Backtest...")
    
    backtester = AdvancedBacktest(config)
    results = backtester.run_backtest(
        predictions=predictions,
        labels=actuals,
        dates=dates,
        dataset=test_dataset,
        price_data=df_raw,
        calibrate=False
    )
    
    # 8. Display Results
    print("\n" + "="*80)
    print("V8 TFT RESULTS (Hidden Test Set 2020-2022)")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']:.1%}")
    print(f"Max Drawdown:    {results['max_drawdown']:.1%}")
    print(f"Total Trades:    {results['total_trades']}")
    print("="*80)
    
    # 9. Save Results
    pd.DataFrame([results]).to_csv(EXPERIMENT_DIR / 'metrics.csv', index=False)
    
    if not results['trades_df'].empty:
        results['trades_df'].to_csv(EXPERIMENT_DIR / 'trades.csv', index=False)
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'Date': results['dates'],
        'Equity': results['equity_curve'][1:],
        'Daily_Return': results['daily_returns'],
        'Drawdown': (np.maximum.accumulate(results['equity_curve'][1:]) - results['equity_curve'][1:]) / np.maximum.accumulate(results['equity_curve'][1:])
    })
    equity_df.to_csv(EXPERIMENT_DIR / 'equity_curve.csv', index=False)
    
    print(f"\nResults saved to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    run_backtest()
