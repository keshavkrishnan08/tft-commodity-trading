
# src/data/fractional_features.py
import pandas as pd
import numpy as np
from src.data.features import FeatureEngineer

class FractionalFeatureEngineer(FeatureEngineer):
    """
    Extends FeatureEngineer to use Fractional Differentiation 
    instead of standard differencing for Returns/Volatility.
    
    Reference: Marcos Lopez de Prado, 'Advances in Financial Machine Learning'
    """
    def __init__(self, config):
        super().__init__(config)
        self.d = 0.4 # Fractional differentiation order (tunable, usually 0.3-0.5 for commods)
        
    def get_weights_floored(self, d, size, floor=1e-5):
        """Calculate weights for fractional diff: (1-L)^d"""
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < floor and k > size:
                break
            w.append(w_k)
            k += 1
            if k >= size: # Cap at window size
                break
        return np.array(w[::-1])

    def frac_diff_fixed(self, series, d, window=25):
        """
        Apply fractional differentiation with a fixed window to preserve memory.
        """
        # Determine weights
        w = self.get_weights_floored(d, window)
        weights = w.reshape(-1, 1) # (window, 1)
        
        output = pd.Series(index=series.index, dtype=float)
        series_vals = series.values
        
        for i in range(window, len(series)):
            window_vals = series_vals[i-window+1 : i+1]
            if len(window_vals) == len(weights):
                val = np.dot(window_vals, weights)
                output.iloc[i] = val[0]
                
        return output

    def engineer_features(self, df_raw):
        """Override to use FracDiff for price-derived features"""
        print(f"Feature Engineering with FRACTIONAL DIFFERENTIATION (d={self.d})...")
        df = df_raw.copy().sort_values('Date').reset_index(drop=True)
        
        # 1. Calculate Standard Returns (Used for Labeling and reference)
        df = self.compute_returns(df)
        
        # 2. Apply Custom Fractional Differentiation to Prices
        for asset in self.config.TARGET_ASSETS:
            close_col = f'{asset}_Close'
            if close_col in df.columns:
                # Calculate FRACTIONAL DIFF Feature (The "Novelty")
                # We use log prices for better stationarity
                log_prices = np.log(df[close_col])
                df[f'{asset}_FracDiff'] = self.frac_diff_fixed(log_prices, d=self.d, window=30)
                # Fill NaNs from window
                df[f'{asset}_FracDiff'] = df[f'{asset}_FracDiff'].fillna(0)
                print(f"  ✓ Computed FracDiff for {asset}")

        # 3. Compute all other technicals using base class methods
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

        # 4. Create Labels (Prediction targets remain 5-day absolute returns)
        df = self.create_labels(df)
        
        # CLEANUP: Drop NaNs
        original_len = len(df)
        df.dropna(inplace=True)
        print(f"Dropped {original_len - len(df)} rows with NaN")
        
        return df
        
    def get_feature_columns(self):
        """Use super's features and append FracDiff"""
        features = super().get_feature_columns()
        
        # Add FracDiff for each asset
        for asset in self.config.TARGET_ASSETS:
            features.append(f'{asset}_FracDiff')
            
        return features
