
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import Config
from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer

config = Config()
config.TARGET_ASSETS = ['Gold', 'Silver', 'BTC']
loader = DataLoader(config)
df_raw = loader.get_data()

engineer = CalibratedFeatureEngineer(config, d=0.4)
df = engineer.engineer_features(df_raw)

print("BTC Data Summary:")
for year in [2018, 2019, 2020, 2021, 2022]:
    year_df = df[df['Date'].astype(str).str.contains(str(year))]
    print(f"  Year {year}: {len(year_df)} samples")
    if len(year_df) > 0:
        # Check ATR
        high = year_df['BTC_High']
        low = year_df['BTC_Low']
        close = year_df['BTC_Close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        # Note: Rank needs history
        atr_rank = atr.rolling(60).rank(pct=True)
        print(f"    Avg ATR: {atr.mean():.4f}")
        
    # Check if BTC data is all NaNs?
    print(f"    BTC Close NaN count: {year_df['BTC_Close'].isna().sum()}")

# Check all_predictions for BTC if possible? No, need to rerun.
