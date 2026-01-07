# main_sliding_v7.py
"""
V7 SLIDING WINDOW BACKTEST (The "Honest" Walk-Forward)
- Expands training window annually (2001-2017, 2001-2018, ...)
- Retrains model every year to adapt to regime shifts.
- Uses Isotonic Calibration on the most recent known data.
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.training.sliding_window import SlidingWindowTrainer
from src.evaluation.advanced_backtest import AdvancedBacktest

def run_experiment():
    print("="*80)
    print("HYBRID WISDOM V7: SLIDING WINDOW ADAPTATION")
    print("="*80)
    
    config = Config()
    EXPERIMENT_DIR = Path('experiments/sliding_window_v7')
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load & Engineer Features (199 Features)
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    
    # 2. Pre-Calculate ATR Rank (Global calculation is valid for past data)
    # We must be careful not to leak future volatility, but rolling rank is causal.
    print("Calculating Volatility Ranks...")
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
        
        # 60-day Rolling Percentile (Casual)
        atr_rank = atr.rolling(60).rank(pct=True)
        df[f'{asset}_ATR_Rank'] = atr_rank
        atr_cols[asset] = f'{asset}_ATR_Rank'
    
    # 3. Run Sliding Window Loop for 2018-2022
    # This will train 5 separate models (one for each year) and stitch predictions.
    RAW_PREDS_PATH = EXPERIMENT_DIR / 'raw_predictions.csv'
    
    if RAW_PREDS_PATH.exists():
        print(f"Found existing predictions at {RAW_PREDS_PATH}. Skipping training phase...")
        predictions_df = pd.read_csv(RAW_PREDS_PATH)
    else:
        trainer = SlidingWindowTrainer(config)
        
        # Check bounds
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        print(f"Data Range: {df['Year'].min()} - {df['Year'].max()}")
        
        start_year = 2018
        predictions_df = trainer.run_sliding_window(df, start_year=start_year)
        predictions_df.to_csv(RAW_PREDS_PATH, index=False)
    
    # 4. Integrate into Backtest Format
    # predictions_df has columns like 'WTI_CalProb', 'WTI_Label', 'WTI_ATR_Rank', 'Date'
    
    # Reconstruct the dictionaries expected by AdvancedBacktest
    assets = config.TARGET_ASSETS
    dates = predictions_df['Date'].tolist()
    
    pred_dict = {a: [] for a in assets}
    label_dict = {a: [] for a in assets}
    
    skipped_trades = 0
    
    for i, row in predictions_df.iterrows():
        for asset in assets:
            cal_prob = row[f'{asset}_CalProb']
            label = row[f'{asset}_Label']
            atr_rank = row[f'{asset}_ATR_Rank']
            
            # Apply V6 Logic: ATR Filter
            # 20th - 90th percentile is the "Golden Zone"
            is_tradable = (atr_rank >= 0.20) and (atr_rank <= 0.90)
            
            final_prob = cal_prob
            if not is_tradable:
                final_prob = 0.5  # Neutralize
                if abs(cal_prob - 0.5) > 0.05: # Only count if it was a signal
                    skipped_trades += 1
            
            pred_dict[asset].append(final_prob)
            label_dict[asset].append(label)
            
    print(f"Skipped {skipped_trades} predicted trades due to ATR Filter.")
    
    # 5. Run Backtest
    # We need a Dataset object for the backtester (mostly for volatility calc)
    # We can create a dummy dataset or reuse the last test one, 
    # but simplest is to pass price_data explicitly to backtester.
    
    # Actually AdvancedBacktest needs 'dataset' for _estimate_volatility
    # Let's create a wrapper dataset for the Prediction Period
    pred_start_date = predictions_df['Date'].min()
    pred_end_date = predictions_df['Date'].max()
    
    mask = (df['Date'] >= pred_start_date) & (df['Date'] <= pred_end_date)
    # AdvancedBacktest expects dataset to align with dates list
    # The 'dataset' argument is primarily used for `get_raw_prices`
    # We can pass the full original dataset wrapper
    
    # Use full dataset wrapper but beware of indexing
    # AdvancedBacktest uses index 'i' from the dates loop
    # dates[i] corresponds to the i-th date in the backtest loop
    # We need to ensure dataset[i] aligns or adapt AdvancedBacktest
    
    # Hack: We will create a dataset that ALIGNS with the prediction dates 1:1
    # MultiAssetDataset logic: dates[idx + seq_len]
    # So we need to feed it data such that the output matches.
    
    # Actually, easiest way is to let the backtester handle price lookups via date map?
    # No, existing code builds int-based index.
    
    # Let's match the slice exactly.
    cols = engineer.get_feature_columns()
    
    def extract_raw_prices(d):
        rp = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            rp[asset] = d[cols]
        return rp
        
    # We need to provide enough history for the FIRST date in predictions
    # to have a valid sequence.
    # Prediction[0] is at T. Dataset needs [T-seq : T].
    # So we find the index of the first prediction in df, and subtract seq_len
    first_pred_idx = df[df['Date'] == pred_start_date].index[0]
    start_slice_idx = first_pred_idx - config.SEQUENCE_LENGTH
    end_slice_idx = df[df['Date'] == pred_end_date].index[0]
    
    ds_slice = df.iloc[start_slice_idx : end_slice_idx + 1]
    
    backtest_dataset = pd.DataFrame() # Dummy, we'll implement a Mock Dataset
    
    # Creating a Dataset aligned with predictions
    # If we pass features=ds_slice, dates=ds_slice...
    # Dataset[0] will be the sequence ending at start_slice_idx + seq_len = first_pred_idx
    # Which is exactly what we want.
    
    from src.data.dataset import MultiAssetDataset
    aligned_dataset = MultiAssetDataset(
        features=df[cols].iloc[start_slice_idx : end_slice_idx + 1],
        labels={a: df[f'{a}_Label'].iloc[start_slice_idx : end_slice_idx + 1] for a in assets},
        dates=df['Date'].iloc[start_slice_idx : end_slice_idx + 1],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(df.iloc[start_slice_idx : end_slice_idx + 1])
    )
    
    print(f"Backtest Dataset Length: {len(aligned_dataset)} (dates: {len(dates)})")
    
    backtester = AdvancedBacktest(config)
    # Override calibration inside backtester (we already did it annually)
    # We strip the calibration logic from backtester or just pass calibrated probs as raw
    
    # Important: AdvancedBacktest tries to calibrate again by default.
    # We should bypass that or feed it dummy labels for calibration phase?
    # Or better: Subclass or Modify AdvancedBacktest to skip calibration?
    # Actually, we can just pass the already calibrated probs. 
    # But AdvancedBacktest splits data into 20% calib / 80% test.
    # WE DO NOT WANT THAT. We want to test on 100% of the predictions.
    
    # Quick Monkey Patch on the instance
    def no_op_calibrate(self, predictions, labels, dates, dataset, price_data=None, intraday_data=None):
        self.reset()
        self.hf_simulator = None
        # SKIP CALIBRATION SPLIT
        n_samples = len(dates)
        start_idx = 0 
        
        # Populate decision engine with dummy fits (so is_fitted=True)
        # But we pass calibrated probs as "raw", so transform should be identity
        # Or we can just set is_fitted=False and let it pass through
        self.decision_engine.is_fitted = False 
        
        # Run loop
        for i in range(start_idx, n_samples):
            date = dates[i]
            daily_pnl = 0.0
            for asset in self.config.TARGET_ASSETS:
                # We are passing pre-calibrated probabilities
                prob = predictions[asset][i] 
                actual = labels[asset][i]
                
                volatility = self._estimate_volatility(dataset, asset, i)
                current_drawdown = (self.max_equity - self.capital) / self.max_equity
                
                # Make Decision (Pass prob as both raw and calib)
                # We need to hack make_decision to accept pre-calibrated
                
                # Actually, easier:
                # decision_engine.make_decision calls calibrator if is_fitted.
                # If is_fitted=False, it uses raw as calibrated.
                # So we pass our sliding-window calibrated probs as "raw".
                
                decision = self.decision_engine.make_decision(
                    raw_probability=prob, # This is actually V7 calibrated
                    asset=asset,
                    volatility=volatility,
                    account_balance=self.capital,
                    recent_wins=self.recent_wins[asset],
                    recent_losses=self.recent_losses[asset],
                    max_drawdown=current_drawdown
                )
                
                if decision['take_trade']:
                    res = self._simulate_trade(decision, actual, asset, date, volatility, i, dataset)
                    if res:
                        daily_pnl += res['pnl']
                        if res['won']:
                            self.recent_wins[asset] = min(5, self.recent_wins[asset] + 1)
                            self.recent_losses[asset] = 0
                        else:
                            self.recent_losses[asset] = min(5, self.recent_losses[asset] + 1)
                            self.recent_wins[asset] = 0
                        self.trades.append(res)
            
            self.capital += daily_pnl
            self.equity_curve.append(self.capital)
            self.daily_returns.append(daily_pnl / self.equity_curve[-2] if self.equity_curve[-2] > 0 else 0)
            if self.capital > self.max_equity: self.max_equity = self.capital
            self.max_drawdown = max(self.max_drawdown, (self.max_equity - self.capital)/self.max_equity)
            
        return self._generate_results(dates)
    
    # Attach patch
    import types
    backtester.run_backtest = types.MethodType(no_op_calibrate, backtester)
    
    results = backtester.run_backtest(
        predictions=pred_dict,
        labels=label_dict,
        dates=dates,
        dataset=aligned_dataset,
        price_data=df_raw
    )
    
    # 6. Save Results & Logs
    print("\n" + "="*80)
    print("V7 SLIDING WINDOW RESULTS (Honest 2018-2022)")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']:.1%}")
    print(f"Max Drawdown:    {results['max_drawdown']:.1%}")
    print(f"Total Trades:    {results['total_trades']}")
    print("="*80)
    
    pd.DataFrame([results]).to_csv(EXPERIMENT_DIR / 'metrics.csv', index=False)
    if not results['trades_df'].empty:
        results['trades_df'].to_csv(EXPERIMENT_DIR / 'trades.csv', index=False)
    
    # Save Detailed Equity Curve
    equity_df = pd.DataFrame({
        'Date': results['dates'],
        'Equity': results['equity_curve'][1:],
        'Daily_Return': results['daily_returns'],
        'Drawdown': (np.maximum.accumulate(results['equity_curve'][1:]) - results['equity_curve'][1:]) / np.maximum.accumulate(results['equity_curve'][1:])
    })
    equity_df.to_csv(EXPERIMENT_DIR / 'equity_curve.csv', index=False)

if __name__ == "__main__":
    run_experiment()
