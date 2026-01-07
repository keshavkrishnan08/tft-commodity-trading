# main_hybrid_v6.py
"""
V6 CALIBRATED BACKTEST
- Uses V6 Model (199 Features)
- Applies Isotonic Calibration to probabilities
- Applies ATR Volatility Filter
"""
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.evaluation.advanced_backtest import AdvancedBacktest

def run_backtest():
    print("="*80)
    print("HYBRID WISDOM V6: CALIBRATED BACKTEST")
    print("="*80)
    
    config = Config()
    EXPERIMENT_DIR = Path('experiments/hybrid_wisdom_v6_calibrated')
    MODEL_PATH = EXPERIMENT_DIR / 'models' / 'best_model.pth'
    CALIBRATOR_PATH = EXPERIMENT_DIR / 'models' / 'calibrators.pkl'
    
    # 1. Load Data & Engineer Features
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()
    
    # 2. Strict Temporal Filter (Tail 15% only)
    # This matches the 'Hidden Test' set from train_hybrid_v6.py
    # This data was NEVER seen by the neural network weights OR the Isotonic Calibrator.
    test_len = int(len(df) * 0.15)
    test_slice = df.tail(test_len).copy() # Use copy to avoid SettingWithCopyWarning
    
    print(f"Backtest Period: {test_slice.iloc[0]['Date']} to {test_slice.iloc[-1]['Date']} ({len(test_slice)} days)")
    
    # 3. Compute ATR Filter (20th-90th Percentile)
    atr_cols = {}
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
        
        # Calculate Rolling Percentile (60-day window)
        atr_rank = atr.rolling(60).rank(pct=True)
        # We must align this rank with our test_slice
        test_slice[f'{asset}_ATR_Rank'] = atr_rank.tail(test_len)
        atr_cols[asset] = f'{asset}_ATR_Rank'
    
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    dataset = MultiAssetDataset(
        features=test_slice[feature_cols],
        labels={asset: test_slice[f'{asset}_Label'] for asset in config.TARGET_ASSETS},
        dates=test_slice['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(test_slice)
    )
    
    # 3. Load Model & Calibrators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeeklyPredictionModel(config).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    model.eval()
    
    calibrators = joblib.load(CALIBRATOR_PATH)
    print("Loaded Calibrators.")

    # 4. Generate Predictions with Filters
    pred_dict = {asset: [] for asset in config.TARGET_ASSETS}
    label_dict = {asset: [] for asset in config.TARGET_ASSETS}
    dates_list = []
    
    CONFIDENCE_THRESHOLD = 0.55 # Standard threshold, but now strictly calibrated
    
    skipped_trades = 0
    total_opportunities = 0
    
    print("Generating Calibrated Predictions...")
    with torch.no_grad():
        for i in range(len(dataset)):
            features, labels = dataset[i]
            x = features.unsqueeze(0).to(device)
            predictions, _ = model(x)
            
            # Get Context info (from DataFrame) using index
            # dataset[i] maps to test_slice.iloc[i + sequence_length] roughly
            # Safer to access via date map if needed, but for sequential it matches
            current_date = dataset.get_date(i)
            dates_list.append(current_date)
            
            # Get row from dataframe for ATR check
            # We need the row corresponding to this Prediction Time
            row = test_slice[test_slice['Date'] == current_date]
            
            for asset in config.TARGET_ASSETS:
                logit = predictions[asset][0].item()
                raw_prob = 1 / (1 + np.exp(-logit))
                
                # 1. APPLY CALIBRATION
                cal_prob = calibrators[asset].predict([raw_prob])[0]
                
                # 2. APPLY ATR FILTER
                atr_rank = row[atr_cols[asset]].values[0] if not row.empty else 0.5
                
                # Check Filter Condition
                is_tradable = (atr_rank >= 0.20) and (atr_rank <= 0.90)
                
                final_prob = cal_prob
                if not is_tradable:
                    # Force probability to neutral 0.5 to prevent trading
                    final_prob = 0.5 
                    skipped_trades += 1
                
                pred_dict[asset].append(final_prob)
                label_dict[asset].append(labels[asset].item())
                total_opportunities += 1

    print(f"Skipped {skipped_trades} predicted trades due to ATR Filter (Dead/Panic Markets).")

    # 5. execute Backtest
    backtester = AdvancedBacktest(config=config)
    results = backtester.run_backtest(
        predictions=pred_dict,
        labels=label_dict,
        dates=dates_list,
        dataset=dataset,
        price_data=df_raw
    )
    
    # 6. Save Data for Visualization
    print("Saving detailed data for visualization...")
    
    # Save Equity Curve
    equity_df = pd.DataFrame({
        'Date': results['dates'],
        'Equity': results['equity_curve'][1:], # Skip initial capital index 0 match
        'Daily_Return': results['daily_returns'],
        'Drawdown': (np.maximum.accumulate(results['equity_curve'][1:]) - results['equity_curve'][1:]) / np.maximum.accumulate(results['equity_curve'][1:])
    })
    equity_df.to_csv(EXPERIMENT_DIR / 'equity_curve.csv', index=False)
    
    # Save Trade Log
    if not results['trades_df'].empty:
        results['trades_df'].to_csv(EXPERIMENT_DIR / 'trades.csv', index=False)
    
    # Save Calibration Data (Transformation Function)
    # We can reconstruct the calibration curve by generating a range of probabilities 0-1
    cal_curves = []
    x_range = np.linspace(0, 1, 100)
    for asset, calibrator in calibrators.items():
        y_calibrated = calibrator.predict(x_range)
        for x, y in zip(x_range, y_calibrated):
            cal_curves.append({'Asset': asset, 'Raw_Prob': x, 'Calibrated_Prob': y})
    pd.DataFrame(cal_curves).to_csv(EXPERIMENT_DIR / 'calibration_curve.csv', index=False)
    
    # Save Metrics
    save_path = EXPERIMENT_DIR / 'metrics.csv'
    pd.DataFrame([results]).to_csv(save_path, index=False)
    
    print("\n" + "="*80)
    print("V6 CALIBRATED RESULTS (Saved to experiments/hybrid_wisdom_v6_calibrated)")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']:.1%}")
    print(f"Max Drawdown:    {results['max_drawdown']:.1%}")
    print(f"Total Trades:    {results['total_trades']}")
    print("="*80)

if __name__ == "__main__":
    run_backtest()
