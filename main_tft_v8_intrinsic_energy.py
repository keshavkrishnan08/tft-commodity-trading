
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader as TorchDataLoader

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset, collate_tft_batch
from src.models.temporal_fusion_transformer import TemporalFusionTransformer
from src.evaluation.advanced_backtest import AdvancedBacktest

def run_intrinsic_energy_experiment():
    print("="*80)
    print("INTRINSIC VALUE EXPERIMENT: ENERGY MARKETS")
    print("="*80)
    
    config = Config()
    # PURE ENERGY UNIVERSE
    config.TARGET_ASSETS = ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']
    config.TFT_HIDDEN_DIM = 32
    config.TFT_NUM_HEADS = 4
    config.TFT_NUM_LAYERS = 2
    config.TFT_DROPOUT = 0.5
    config.LEARNING_RATE = 2e-4
    
    EXPERIMENT_DIR = Path('experiments/tft_v8_intrinsic_energy')
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    # 2. Engineer Features (Scale-Invariant Stationary Protocol)
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    # Global engineering for technicals, but we will override labels inside the loop
    df, global_thresholds = engineer.engineer_features(df_raw)
    
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)
    
    # 3. ATR Rank for Filtering
    for asset in config.TARGET_ASSETS:
        tr = pd.concat([
            df[f'{asset}_High'] - df[f'{asset}_Low'],
            abs(df[f'{asset}_High'] - df[f'{asset}_Close'].shift(1)),
            abs(df[f'{asset}_Low'] - df[f'{asset}_Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df[f'{asset}_ATR_Rank'] = atr.rolling(60).rank(pct=True)
    
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    horizon_days = config.PREDICTION_HORIZONS[config.PREDICTION_HORIZON]

    # 4. Sliding Window: June 2018 Start
    test_years = [2018, 2019, 2020, 2021, 2022]
    all_predictions = {a: [] for a in config.TARGET_ASSETS}
    all_actuals = {a: [] for a in config.TARGET_ASSETS}
    all_dates = []
    entropy_records = []
    
    for year in test_years:
        print(f"\nProcessing Year {year}...")
        train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(year-5, year)]))].copy()
        # ANTI-LEAKAGE: Purge last horizon_days from training to prevent label peeking into test year
        train_df = train_df.iloc[:-horizon_days]
        test_df = df[df['Date'].str.contains(str(year))].copy()
        
        if len(test_df) <= config.SEQUENCE_LENGTH: continue

        # ANTI-LEAKAGE: Re-calculate labels based on training set's median return ONLY
        current_thresholds = {}
        for a in config.TARGET_ASSETS:
            # Use training set to find median future return
            median_ret = train_df[f'{a}_FutureReturn_{horizon_days}d'].quantile(0.5)
            current_thresholds[a] = median_ret
            # Re-label train and test based on this localized threshold
            train_df[f'{a}_Label'] = (train_df[f'{a}_FutureReturn_{horizon_days}d'] > median_ret).astype(int)
            test_df[f'{a}_Label'] = (test_df[f'{a}_FutureReturn_{horizon_days}d'] > median_ret).astype(int)
            
        split_idx = int(len(train_df) * 0.85)
        t_df = train_df.iloc[:split_idx]
        c_df = train_df.iloc[split_idx:]
        
        train_ds = TFTDataset(t_df[feature_cols], {a: t_df[f'{a}_Label'] for a in config.TARGET_ASSETS}, t_df['Date'], config.SEQUENCE_LENGTH, fit_scaler=True)
        calib_ds = TFTDataset(c_df[feature_cols], {a: c_df[f'{a}_Label'] for a in config.TARGET_ASSETS}, c_df['Date'], config.SEQUENCE_LENGTH, scaler=train_ds.scaler)
        test_ds = TFTDataset(test_df[feature_cols], {a: test_df[f'{a}_Label'] for a in config.TARGET_ASSETS}, test_df['Date'], config.SEQUENCE_LENGTH, scaler=train_ds.scaler)
        
        train_loader = TorchDataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_tft_batch)
        calib_loader = TorchDataLoader(calib_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_tft_batch)
        
        # 5. Train Model (5 Epochs)
        model = TemporalFusionTransformer(config).to(device)
        criteria = {a: nn.BCEWithLogitsLoss() for a in config.TARGET_ASSETS}
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        for epoch in range(5):
            model.train()
            for feat, time, label in train_loader:
                feat, time = feat.to(device), time.to(device)
                optimizer.zero_grad()
                out, _ = model(feat, time)
                loss = sum(criteria[a](out[a], label[a].to(device)) for a in config.TARGET_ASSETS)
                loss.backward()
                optimizer.step()
        
        # 6. Calibrate (Isotonic)
        model.eval()
        calibrators = {}
        with torch.no_grad():
            for a in config.TARGET_ASSETS:
                p_list, y_list = [], []
                for feat, time, label in calib_loader:
                    out, _ = model(feat.to(device), time.to(device))
                    p_list.extend(torch.sigmoid(out[a]).cpu().numpy())
                    y_list.extend(label[a].numpy())
                iso = IsotonicRegression(out_of_bounds='clip').fit(p_list, y_list)
                calibrators[a] = iso
        
        # 7. Predict & Capture VSN Entropy
        CONVICTION_THRESHOLD = 0.10
        with torch.no_grad():
            for i in range(len(test_ds)):
                feat, time, label = test_ds[i]
                feat_input = feat.unsqueeze(0).to(device)
                time_input = time.unsqueeze(0).to(device)
                out, _ = model(feat_input, time_input)
                date = test_ds.get_date(i)
                all_dates.append(date)
                date_row = test_df[test_df['Date'] == date]

                # Capture Entropy (VSN weights: [Batch, Seq, Inputs])
                vsn_weights = model.get_vsn_weights(feat_input)
                # Take the last time step, mean over batch (if any)
                vsn_weights = vsn_weights[:, -1, :] # [Batch, Inputs]
                vsn_weights = vsn_weights.mean(dim=0) # [Inputs]
                entropy = -(vsn_weights * torch.log(vsn_weights + 1e-10)).sum().item()
                entropy_records.append({'Date': date, 'Entropy': entropy})
                
                for a in config.TARGET_ASSETS:
                    raw_prob = torch.sigmoid(out[a]).item()
                    cal_prob = calibrators[a].predict([raw_prob])[0]
                    atr_rank = date_row[f'{a}_ATR_Rank'].values[0]
                    is_tradable = (atr_rank >= 0.20) and (atr_rank <= 0.90)
                    final_p = 0.5
                    if is_tradable and abs(cal_prob - 0.5) >= CONVICTION_THRESHOLD:
                        final_p = cal_prob
                    all_predictions[a].append(final_p)
                    all_actuals[a].append(label[a].item())

    # 8. Run 1:2 R/R Backtest
    print("\nRunning Advanced 1:2 R/R Backtest...")
    bt = AdvancedBacktest(config)
    # Note: AdvancedBacktest uses 1.5x ATR stops and 2:1 R/R by default (target = price + 2 * stop_dist)
    res = bt.run_backtest(all_predictions, all_actuals, all_dates, TFTDataset(df[feature_cols], {a: df[f'{a}_Label'] for a in config.TARGET_ASSETS}, df['Date'], config.SEQUENCE_LENGTH), df_raw, calibrate=False)
    
    # 9. Results Summary
    print("\n" + "="*80)
    print("ENERGY INTRINSIC VALUE RESULTS")
    print("="*80)
    print(f"Total Return:    {res['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {res['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {res['win_rate']:.1%}")
    print(f"Total Trades:    {res['total_trades']}")
    print("="*80)
    
    # Save Artifacts
    pd.DataFrame([res]).to_csv(EXPERIMENT_DIR / 'metrics.csv', index=False)
    res['trades_df'].to_csv(EXPERIMENT_DIR / 'trades.csv', index=False)
    pd.DataFrame(entropy_records).to_csv(EXPERIMENT_DIR / 'vsn_entropy.csv', index=False)

if __name__ == "__main__":
    run_intrinsic_energy_experiment()
