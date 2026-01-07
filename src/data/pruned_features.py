import pandas as pd
import numpy as np
from src.data.features import FeatureEngineer
from src.utils.config import Config

class PrunedHybridFeatureEngineer(FeatureEngineer):
    """
    V5 PRUNED HYBRID WISDOM:
    Top 25 Features per Asset (5 FracDiff + 20 Technicals)
    
    Philosophy: "Less is More" - Remove noise, keep signal.
    """
    def __init__(self, config=None, d=0.4, floor=1e-3):
        super().__init__(config)
        self.d = d
        self.floor = floor

    def get_weights(self, d, size):
        """Fractional differentiation weights"""
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_fixed(self, series, d, floor=1e-3):
        """Apply fractional differentiation to a series"""
        w = self.get_weights(d, size=100)
        w_idx = np.where(np.abs(w) > floor)[0]
        w = w[w_idx]
        width = len(w)
        
        res = {}
        for i in range(width, len(series)):
            res[series.index[i]] = np.dot(w.T, series.iloc[i-width:i].values.reshape(-1, 1))[0,0]
        return pd.Series(res)

    def engineer_features(self, df_raw):
        df = df_raw.copy()
        
        # === LAYER 1: Core Technicals (20 per asset) ===
        
        # 1-2: Returns
        df = self.compute_returns(df)
        for asset in self.config.TARGET_ASSETS:
            df[f'{asset}_LogReturn'] = np.log(df[f'{asset}_Close'] / df[f'{asset}_Close'].shift(1))
        
        # 3-4: Volatility (2 windows)
        for asset in self.config.TARGET_ASSETS:
            return_col = f'{asset}_Return'
            df[f'{asset}_Vol_5d'] = df[return_col].rolling(5).std()
            df[f'{asset}_Vol_20d'] = df[return_col].rolling(20).std()
        
        # 5: RSI
        df = self.compute_rsi(df)
        
        # 6-8: MACD
        df = self.compute_macd(df)
        
        # 9-11: Bollinger Bands
        df = self.compute_bollinger_bands(df)
        
        # 12: Stochastic RSI (single column)
        df = self.compute_stochastic_rsi(df)
        
        # 14: ROC
        for asset in self.config.TARGET_ASSETS:
            df[f'{asset}_ROC_10'] = ((df[f'{asset}_Close'] / df[f'{asset}_Close'].shift(10)) - 1) * 100
        
        # 15-16: SMA Distance (not raw price)
        for asset in self.config.TARGET_ASSETS:
            df[f'{asset}_SMA_10'] = df[f'{asset}_Close'].rolling(10).mean()
            df[f'{asset}_SMA_50'] = df[f'{asset}_Close'].rolling(50).mean()
            df[f'{asset}_SMA_10_Dist'] = (df[f'{asset}_Close'] / df[f'{asset}_SMA_10'] - 1) * 100
            df[f'{asset}_SMA_50_Dist'] = (df[f'{asset}_Close'] / df[f'{asset}_SMA_50'] - 1) * 100
        
        # 17-18: EMA Distance
        for asset in self.config.TARGET_ASSETS:
            df[f'{asset}_EMA_12'] = df[f'{asset}_Close'].ewm(span=12).mean()
            df[f'{asset}_EMA_26'] = df[f'{asset}_Close'].ewm(span=26).mean()
            df[f'{asset}_EMA_12_Dist'] = (df[f'{asset}_Close'] / df[f'{asset}_EMA_12'] - 1) * 100
            df[f'{asset}_EMA_26_Dist'] = (df[f'{asset}_Close'] / df[f'{asset}_EMA_26'] - 1) * 100
        
        # 19: ATR (Average True Range)
        for asset in self.config.TARGET_ASSETS:
            high = df[f'{asset}_High']
            low = df[f'{asset}_Low']
            close = df[f'{asset}_Close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'{asset}_ATR_14'] = tr.rolling(14).mean()
        
        # 20: MFI (Money Flow Index) - Volume-weighted RSI
        # Note: If volume not available, use High-Low range as proxy
        for asset in self.config.TARGET_ASSETS:
            typical_price = (df[f'{asset}_High'] + df[f'{asset}_Low'] + df[f'{asset}_Close']) / 3
            # Use High-Low range as volume proxy
            volume_proxy = df[f'{asset}_High'] - df[f'{asset}_Low']
            money_flow = typical_price * volume_proxy
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
            df[f'{asset}_MFI_14'] = mfi
        
        # === LAYER 2: FracDiff Features (5 per asset) ===
        
        for asset in self.config.TARGET_ASSETS:
            # 1. FracDiff on Close Price
            close_col = f'{asset}_Close'
            if close_col in df.columns:
                fd_close = self.frac_diff_fixed(df[close_col], d=self.d, floor=self.floor)
                df[f'{asset}_FracDiff_Close'] = fd_close
            
            # 2. FracDiff on Volatility
            vol_col = f'{asset}_Vol_20d'
            if vol_col in df.columns:
                fd_vol = self.frac_diff_fixed(df[vol_col].fillna(0), d=self.d, floor=self.floor)
                df[f'{asset}_FracDiff_Vol'] = fd_vol
            
            # 3. FracDiff on MACD
            macd_col = f'{asset}_MACD'
            if macd_col in df.columns:
                fd_macd = self.frac_diff_fixed(df[macd_col].fillna(0), d=self.d, floor=self.floor)
                df[f'{asset}_FracDiff_MACD'] = fd_macd
            
            # 4. FracDiff on RSI
            rsi_col = f'{asset}_RSI'
            if rsi_col in df.columns:
                fd_rsi = self.frac_diff_fixed(df[rsi_col].fillna(50), d=self.d, floor=self.floor)
                df[f'{asset}_FracDiff_RSI'] = fd_rsi
            
            # 5. FracDiff on High-Low Range (Volume proxy)
            df[f'{asset}_HL_Range'] = df[f'{asset}_High'] - df[f'{asset}_Low']
            hl_col = f'{asset}_HL_Range'
            if hl_col in df.columns:
                fd_hl = self.frac_diff_fixed(df[hl_col].fillna(0), d=self.d, floor=self.floor)
                df[f'{asset}_FracDiff_Volume'] = fd_hl
        
        # === LAYER 3: Labels and Cleanup ===
        df = self.create_labels(df)
        df = df.dropna()
        
        return df

    def get_feature_columns(self):
        """Return the pruned feature set (24 per asset) - TARGET_ASSETS only"""
        features = []
        
        # Only include features for TARGET_ASSETS (not DXY)
        for asset in self.config.TARGET_ASSETS:
            # Technicals (19 - reduced by 1 since StochRSI is single column)
            features.extend([
                f'{asset}_Return',
                f'{asset}_LogReturn',
                f'{asset}_Vol_5d',
                f'{asset}_Vol_20d',
                f'{asset}_RSI',
                f'{asset}_MACD',
                f'{asset}_MACD_Signal',
                f'{asset}_MACD_Hist',
                f'{asset}_BB_Upper',
                f'{asset}_BB_Lower',
                f'{asset}_BB_Width',
                f'{asset}_StochRSI',  # Single column
                f'{asset}_ROC_10',
                f'{asset}_SMA_10_Dist',
                f'{asset}_SMA_50_Dist',
                f'{asset}_EMA_12_Dist',
                f'{asset}_EMA_26_Dist',
                f'{asset}_ATR_14',
                f'{asset}_MFI_14',
            ])
            
            # FracDiff (5)
            features.extend([
                f'{asset}_FracDiff_Close',
                f'{asset}_FracDiff_Vol',
                f'{asset}_FracDiff_MACD',
                f'{asset}_FracDiff_RSI',
                f'{asset}_FracDiff_Volume',
            ])
        
        return features
