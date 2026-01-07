# src/training/sliding_window.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import joblib

from src.utils.config import Config
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.training.trainer import Trainer

class SlidingWindowTrainer:
    """
    Orchestrates Walk-Forward Validation (Expanding Window).
    
    Logic:
    1. Start at a given test year (e.g., 2018).
    2. Train on ALL history before that year (2001-2017).
    3. Calibrate on the recent history (e.g., 2017).
    4. Predict the test year (2018).
    5. Expand window: Add 2018 to train, repeat for 2019.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_sliding_window(self, df_full, start_year=2018):
        """
        Run the full sliding window loop.
        
        Args:
            df_full: DataFrame containing full history (2001-2022) with 'Date' column.
            start_year: The first year to predict (test year).
            
        Returns:
            DataFrame: Stitched predictions for the test period (start_year to end).
        """
        print("="*80)
        print(f"STARTING SLIDING WINDOW BACKTEST (Start Year: {start_year})")
        print("="*80)
        
        # Ensure Date is datetime
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        df_full['Year'] = df_full['Date'].dt.year
        
        end_year = df_full['Year'].max()
        years_to_predict = range(start_year, end_year + 1)
        
        all_predictions = []
        
        for year in years_to_predict:
            print(f"\n>>> PROCESSING YEAR: {year} (Training on 2001-{year-1})")
            
            # 1. Temporal Splitting
            # Train: All data BEFORE this year
            train_df = df_full[df_full['Year'] < year].copy()
            # Test: This specific year
            test_df = df_full[df_full['Year'] == year].copy()
            
            if test_df.empty:
                print(f"Skipping {year} (No data)")
                continue
                
            # 2. Prepare Datasets
            # We need to extract raw prices for the dataset wrapper
            def extract_raw_prices(d):
                rp = {}
                for asset in self.config.TARGET_ASSETS:
                    cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
                    rp[asset] = d[cols]
                return rp

            # Use last 20% of TRAIN for Calibration/Validation
            train_len = len(train_df)
            calib_start_idx = int(train_len * 0.85) # Last 15% for calibration
            
            # Create Datasets
            # Note: We pass the feature columns explicitly to ensure alignment
            feature_cols = [c for c in train_df.columns if c not in ['Date', 'Year'] and 'Label' not in c and 'ATR_Rank' not in c]
            
            # Main Dataset (Train + Calib)
            full_train_ds = MultiAssetDataset(
                features=train_df[feature_cols], 
                labels={a: train_df[f'{a}_Label'] for a in self.config.TARGET_ASSETS},
                dates=train_df['Date'],
                sequence_length=self.config.SEQUENCE_LENGTH,
                raw_prices=extract_raw_prices(train_df),
                fit_scaler=True
            )
            
            # Split for Training loop
            train_ds = torch.utils.data.Subset(full_train_ds, range(0, calib_start_idx))
            calib_ds = torch.utils.data.Subset(full_train_ds, range(calib_start_idx, len(full_train_ds)))
            
            # Test Dataset
            # CRITICAL: Re-use the scaler from the training set to avoid leakage
            test_ds = MultiAssetDataset(
                features=test_df[feature_cols],
                labels={a: test_df[f'{a}_Label'] for a in self.config.TARGET_ASSETS},
                dates=test_df['Date'],
                sequence_length=self.config.SEQUENCE_LENGTH,
                raw_prices=extract_raw_prices(test_df),
                scaler=full_train_ds.scaler,
                fit_scaler=False
            )
            
            train_loader = TorchDataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
            calib_loader = TorchDataLoader(calib_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
            test_loader = TorchDataLoader(test_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
            
            # 3. Train Model (Fresh start every year)
            model = WeeklyPredictionModel(self.config).to(self.device)
            trainer = Trainer(model, train_loader, calib_loader, config=self.config)
            # Train for limited epochs to simulate "fine-tuning" or fresh learning
            trainer.fit(epochs=5) 
            
            # 4. Fit Isotonic Calibrator (on Calib set)
            print(f"  Calibrating on {len(calib_ds)} samples...")
            model.eval()
            calibrators = {}
            
            # Collect calib predictions
            calib_preds = {a: [] for a in self.config.TARGET_ASSETS}
            calib_actuals = {a: [] for a in self.config.TARGET_ASSETS}
            
            with torch.no_grad():
                for batch_x, batch_y in calib_loader:
                    batch_x = batch_x.to(self.device)
                    out, _ = model(batch_x)
                    for a in self.config.TARGET_ASSETS:
                        p = torch.sigmoid(out[a]).cpu().numpy().flatten()
                        calib_preds[a].extend(p)
                        calib_actuals[a].extend(batch_y[a].cpu().numpy().flatten())
            
            for a in self.config.TARGET_ASSETS:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(calib_preds[a], calib_actuals[a])
                calibrators[a] = iso
            
            # 5. Predict Test Year
            print(f"  Predicting Year {year} ({len(test_ds)} samples)...")
            year_preds = []
            
            # We iterate differently here to keep track of dates/indices
            with torch.no_grad():
                for i in range(len(test_ds)):
                    seq, labels = test_ds[i]
                    x = seq.unsqueeze(0).to(self.device)
                    out, _ = model(x)
                    
                    # Get Date/ATR info
                    # Note: index i in dataset corresponds to index i+seq_len in dataframe
                    # Careful with indexing. test_ds[i] uses test_df.iloc[i:i+seq]
                    # The date returned by ds.get_date(i) is the PREDICTION DATE (T+seq)
                    
                    # Retrieve the row corresponding to this prediction date
                    # to get ATR info
                    pred_date = test_ds.get_date(i)
                    row = test_df[test_df['Date'] == pred_date]
                    
                    metrics = {
                        'Date': pred_date, 
                        'Year': year
                    }
                    
                    for a in self.config.TARGET_ASSETS:
                        raw_logit = out[a].item()
                        raw_prob = 1 / (1 + np.exp(-raw_logit))
                        cal_prob = calibrators[a].predict([raw_prob])[0]
                        label = labels[a].item()
                        
                        # ATR Filter Check
                        atr_rank = row[f'{a}_ATR_Rank'].values[0] if f'{a}_ATR_Rank' in row.columns else 0.5
                        
                        metrics[f'{a}_RawProb'] = raw_prob
                        metrics[f'{a}_CalProb'] = cal_prob
                        metrics[f'{a}_Label'] = label
                        metrics[f'{a}_ATR_Rank'] = atr_rank
                    
                    year_preds.append(metrics)
            
            all_predictions.extend(year_preds)
            
        return pd.DataFrame(all_predictions)

