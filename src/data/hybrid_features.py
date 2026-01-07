import pandas as pd
import numpy as np
from src.data.features import FeatureEngineer
from src.utils.config import Config

class HybridFeatureEngineer(FeatureEngineer):
    """
    STAMP OF HYBRID WISDOM (V4):
    Stacks Proven Technicals (V1) + FracDiff d=0.4 (Long-term memory).
    """
    def __init__(self, config=None, d=0.4, floor=1e-3):
        super().__init__(config)
        self.d = d
        self.floor = floor

    def get_weights(self, d, size):
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_fixed(self, series, d, floor=1e-3):
        # 1) Get weights
        w = self.get_weights(d, size=100) # Window size 100
        w_idx = np.where(np.abs(w) > floor)[0]
        w = w[w_idx]
        width = len(w)
        
        # 2) Apply weights
        res = {}
        for i in range(width, len(series)):
            res[series.index[i]] = np.dot(w.T, series.iloc[i-width:i].values.reshape(-1, 1))[0,0]
        return pd.Series(res)

    def engineer_features(self, df_raw):
        df = df_raw.copy()
        
        # --- LAYER 1: Prove V1 Technicals (Short-Term Reflexes) ---
        # Match complete list from base FeatureEngineer
        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_sma(df)
        df = self.compute_ema(df)
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_bollinger_bands(df)
        df = self.compute_stochastic_rsi(df)
        df = self.compute_rate_of_change(df)
        df = self.compute_market_regime(df)
        df = self.compute_divergences(df)
        df = self.compute_cross_asset_features(df)
        
        # --- LAYER 2: FracDiff (Long-Term Wisdom) ---
        for asset in self.config.TARGET_ASSETS:
            close_col = f'{asset}_Close'
            if close_col in df.columns:
                # Calculate d=0.4 differentiated series
                fd_series = self.frac_diff_fixed(df[close_col], d=self.d, floor=self.floor)
                # Align and add to dataframe
                df[f'{asset}_FracDiff'] = fd_series
        
        # --- LAYER 3: Labels and Cleanup ---
        df, thresholds = self.create_labels(df)
        
        # Drop rows with NaNs from the windowing
        df = df.dropna()
        
        return df, thresholds

    def get_feature_columns(self):
        # Dynamically get all V1 base features
        base_features = super().get_feature_columns()
        
        # Add the new FracDiff columns
        frac_diff_features = [f'{asset}_FracDiff' for asset in self.config.TARGET_ASSETS]
        
        return base_features + frac_diff_features
