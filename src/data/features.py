# src/data/features.py
"""
Feature engineering for multi-asset directional prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class FeatureEngineer:
    """Compute technical indicators and create labels"""

    def __init__(self, config=None):
        self.config = config if config else Config()

    def compute_returns(self, df):
        """Compute returns and log-returns"""
        print("Computing returns...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'
            # Simple returns
            df[f'{asset}_Return'] = df[close_col].pct_change()
            # Log returns
            df[f'{asset}_LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))

        return df

    def compute_volatility(self, df):
        """Compute rolling volatility"""
        print("Computing volatility...")
        for asset in self.config.ALL_ASSETS:
            return_col = f'{asset}_Return'
            for window in self.config.VOLATILITY_WINDOWS:
                df[f'{asset}_Vol_{window}d'] = df[return_col].rolling(window).std()

        return df

    def compute_sma(self, df):
        """Compute Simple Moving Averages as Ratio to Close"""
        print("Computing SMA Ratios...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'
            for window in self.config.SMA_WINDOWS:
                sma = df[close_col].rolling(window).mean()
                # Stationary: Price relative to SMA with clipping for stability
                ratio = (df[close_col] - sma) / (sma.abs().clip(lower=1e-6))
                df[f'{asset}_SMA_{window}'] = ratio.clip(-10, 10)

        return df

    def compute_ema(self, df):
        """Compute Exponential Moving Averages as Ratio to Close"""
        print("Computing EMA Ratios...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'
            for window in self.config.EMA_WINDOWS:
                ema = df[close_col].ewm(span=window, adjust=False).mean()
                # Stationary: Price relative to EMA with clipping
                ratio = (df[close_col] - ema) / (ema.abs().clip(lower=1e-6))
                df[f'{asset}_EMA_{window}'] = ratio.clip(-10, 10)

        return df

    def compute_rsi(self, df):
        """Compute Relative Strength Index"""
        print("Computing RSI...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'
            delta = df[close_col].diff()

            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=self.config.RSI_PERIOD).mean()
            avg_loss = loss.rolling(window=self.config.RSI_PERIOD).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            df[f'{asset}_RSI'] = 100 - (100 / (1 + rs))

        return df

    def compute_macd(self, df):
        """Compute MACD as % of Price"""
        print("Computing MACD Ratios...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'

            ema_fast = df[close_col].ewm(span=self.config.MACD_FAST, adjust=False).mean()
            ema_slow = df[close_col].ewm(span=self.config.MACD_SLOW, adjust=False).mean()

            macd = ema_fast - ema_slow
            # Stationary: MACD relative to current price (%)
            ratio = macd / (df[close_col].abs().clip(lower=1e-6)) * 100
            df[f'{asset}_MACD'] = ratio.clip(-100, 100) # MACD signal is typically small
            
            sig = df[f'{asset}_MACD'].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
            df[f'{asset}_MACD_Signal'] = sig
            df[f'{asset}_MACD_Hist'] = df[f'{asset}_MACD'] - df[f'{asset}_MACD_Signal']

        return df

    def compute_bollinger_bands(self, df):
        """Compute Bollinger Bands Components (Position and Width)"""
        print("Computing Bollinger Band Analytics...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'

            # 20-day Bollinger Bands
            sma_20 = df[close_col].rolling(20).mean()
            std_20 = df[close_col].rolling(20).std()

            upper = sma_20 + (2 * std_20)
            lower = sma_20 - (2 * std_20)
            middle = sma_20

            # BB Position: where price is relative to bands (0-1 scale) - STATIONARY
            bb_range = upper - lower
            df[f'{asset}_BB_Position'] = (df[close_col] - lower) / (bb_range + 1e-10)

            # BB Width: volatility indicator - STATIONARY
            df[f'{asset}_BB_Width'] = bb_range / (middle + 1e-10)

        return df

    def compute_stochastic_rsi(self, df):
        """Compute Stochastic RSI for better overbought/oversold signals"""
        print("Computing Stochastic RSI...")
        for asset in self.config.ALL_ASSETS:
            rsi_col = f'{asset}_RSI'

            # Stochastic of RSI over 14 periods
            rsi_min = df[rsi_col].rolling(14).min()
            rsi_max = df[rsi_col].rolling(14).max()

            df[f'{asset}_StochRSI'] = (df[rsi_col] - rsi_min) / (rsi_max - rsi_min + 1e-10)

        return df

    def compute_rate_of_change(self, df):
        """Compute Rate of Change (ROC) indicators"""
        print("Computing Rate of Change...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'

            # ROC over different periods
            for period in [5, 10, 20]:
                df[f'{asset}_ROC_{period}d'] = (df[close_col] / df[close_col].shift(period) - 1) * 100

        return df

    def compute_market_regime(self, df):
        """
        Compute market regime indicators (trending vs ranging).
        Uses ADX-like calculation without True Range.
        """
        print("Computing market regime indicators...")
        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'
            high_col = f'{asset}_High'
            low_col = f'{asset}_Low'

            # Trend strength: ratio of 50-day SMA slope to volatility
            sma_50 = df[f'{asset}_SMA_50']
            sma_slope = sma_50.diff(10)  # 10-day slope
            volatility = df[f'{asset}_Vol_20d']

            # Normalize by price to make comparable across assets
            df[f'{asset}_TrendStrength'] = (sma_slope / (df[close_col] * volatility + 1e-10))

            # Ranging indicator: price oscillation around SMA
            # Low values = trending, high values = ranging
            deviation_from_sma = abs(df[close_col] - sma_50) / (df[close_col] + 1e-10)
            df[f'{asset}_RangingScore'] = 1.0 - deviation_from_sma.rolling(20).mean()

        return df

    def compute_cross_asset_features(self, df):
        """
        Compute cross-asset features (correlations, spreads).
        """
        print("Computing cross-asset features...")

        # WTI-Brent spread (crucial for oil trading)
        if 'WTI' in self.config.TARGET_ASSETS and 'Brent' in self.config.TARGET_ASSETS:
            df['WTI_Brent_Spread'] = df['WTI_Close'] / (df['Brent_Close'] + 1e-10) - 1.0
            df['WTI_Brent_Corr'] = df['WTI_Return'].rolling(20).corr(df['Brent_Return'])

        # DXY impact on oil (inverse relationship typically)
        if 'DXY' in self.config.ALL_ASSETS:
            for asset in self.config.TARGET_ASSETS:
                # Rolling correlation
                df[f'{asset}_DXY_Corr'] = df[f'{asset}_Return'].rolling(20).corr(df['DXY_Return'])

                # Relative strength
                df[f'{asset}_DXY_RelativeStrength'] = df[f'{asset}_RSI'] - df['DXY_RSI']

        return df

    def compute_divergences(self, df):
        """
        Compute price divergences from moving averages.
        These indicate when price is above/below its historical trend.
        """
        print("Computing divergences from moving averages...")

        for asset in self.config.ALL_ASSETS:
            close_col = f'{asset}_Close'

            # Divergence from each SMA (% above/below)
            for window in self.config.SMA_WINDOWS:
                sma_col = f'{asset}_SMA_{window}' # Note: We already changed this to be the ratio!
                # Wait, if f'{asset}_SMA_{window}' IS the ratio, we don't need Divergence_SMA twice.
                # But for compatibility, let's keep the name Divergence_SMA but point it to the ratio.
                df[f'{asset}_Divergence_SMA_{window}'] = df[sma_col]

            # Divergence from each EMA
            for window in self.config.EMA_WINDOWS:
                ema_col = f'{asset}_EMA_{window}'
                df[f'{asset}_Divergence_EMA_{window}'] = df[ema_col]

            # Price momentum (current vs N days ago)
            df[f'{asset}_Momentum_20d'] = df[close_col].pct_change(20)
            df[f'{asset}_Momentum_60d'] = df[close_col].pct_change(60)

        return df

    def create_labels(self, df):
        """
        Create binary directional labels for WEEKLY/MONTHLY predictions.
        Uses forward-looking returns over the specified horizon.
        """
        print(f"Creating {self.config.PREDICTION_HORIZON} directional labels...")

        # Get prediction horizon in days
        horizon_days = self.config.PREDICTION_HORIZONS[self.config.PREDICTION_HORIZON]
        
        thresholds = {}

        for asset in self.config.TARGET_ASSETS:
            close_col = f'{asset}_Close'

            # Forward return over the horizon period
            future_return = (
                df[close_col].shift(-horizon_days) / df[close_col] - 1
            )

            # Binary label: 1 if return > median, 0 otherwise
            threshold = future_return.quantile(0.5)
            thresholds[asset] = threshold

            df[f'{asset}_Label'] = (future_return > threshold).astype(int)
            df[f'{asset}_FutureReturn_{horizon_days}d'] = future_return
            df[f'{asset}_LabelThreshold'] = threshold

            # Count labels
            up_count = df[f'{asset}_Label'].sum()
            down_count = (df[f'{asset}_Label'] == 0).sum()
            total = up_count + down_count

            print(f"  {asset}: median_threshold = {threshold:.4f}, "
                  f"up = {up_count}/{total} ({up_count/total*100:.1f}%), "
                  f"down = {down_count}/{total} ({down_count/total*100:.1f}%)")

        return df, thresholds

    def engineer_features(self, df):
        """Run full feature engineering pipeline"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        df = df.copy()

        # Compute all features
        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_sma(df)
        df = self.compute_ema(df)
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_bollinger_bands(df)  # NEW: Bollinger Bands
        df = self.compute_stochastic_rsi(df)   # NEW: Stochastic RSI
        df = self.compute_rate_of_change(df)   # NEW: ROC indicators
        df = self.compute_market_regime(df)    # NEW: Market regime
        df = self.compute_divergences(df)      # Divergence features
        df = self.compute_cross_asset_features(df)  # NEW: Cross-asset features

        # Create labels
        df, thresholds = self.create_labels(df)

        # Drop NaN created by rolling/shifting
        original_len = len(df)
        df.dropna(inplace=True)
        print(f"\nDropped {original_len - len(df)} rows with NaN")
        print(f"Final dataset: {len(df)} rows")

        return df, thresholds

    def get_feature_columns(self):
        """Return list of all feature column names"""
        features = []
        
        for asset in self.config.ALL_ASSETS:
            # Price-based
            features.extend([
                f'{asset}_Return',
                f'{asset}_LogReturn',
            ])

            # Volatility
            for window in self.config.VOLATILITY_WINDOWS:
                features.append(f'{asset}_Vol_{window}d')

            # SMA
            for window in self.config.SMA_WINDOWS:
                features.append(f'{asset}_SMA_{window}')

            # EMA
            for window in self.config.EMA_WINDOWS:
                features.append(f'{asset}_EMA_{window}')

            # RSI
            features.append(f'{asset}_RSI')

            # MACD
            features.extend([
                f'{asset}_MACD',
                f'{asset}_MACD_Signal',
                f'{asset}_MACD_Hist'
            ])

            # Divergences from SMA
            for window in self.config.SMA_WINDOWS:
                features.append(f'{asset}_Divergence_SMA_{window}')

            # Divergences from EMA
            for window in self.config.EMA_WINDOWS:
                features.append(f'{asset}_Divergence_EMA_{window}')

            # Momentum
            features.extend([
                f'{asset}_Momentum_20d',
                f'{asset}_Momentum_60d'
            ])

            # Bollinger Bands (Only Position and Width are stationary)
            features.extend([
                f'{asset}_BB_Position',
                f'{asset}_BB_Width'
            ])

            # Stochastic RSI
            features.append(f'{asset}_StochRSI')

            # Rate of Change
            for period in [5, 10, 20]:
                features.append(f'{asset}_ROC_{period}d')

            # Market Regime
            features.extend([
                f'{asset}_TrendStrength',
                f'{asset}_RangingScore'
            ])

            # Cross-asset features (DXY correlation)
            if asset in self.config.TARGET_ASSETS and 'DXY' in self.config.ALL_ASSETS:
                features.extend([
                    f'{asset}_DXY_Corr',
                    f'{asset}_DXY_RelativeStrength'
                ])

        # Global cross-asset features
        if 'WTI' in self.config.TARGET_ASSETS and 'Brent' in self.config.TARGET_ASSETS:
            features.extend([
                'WTI_Brent_Spread',
                'WTI_Brent_Corr'
            ])

        return features


if __name__ == "__main__":
    # Test feature engineering
    from src.data.loader import DataLoader

    loader = DataLoader()
    df_raw = loader.get_data()

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_raw)

    print("\nFeature columns:")
    feature_cols = engineer.get_feature_columns()
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Features per asset: {len(feature_cols) / len(Config.ALL_ASSETS):.0f}")

    print("\nLabel distribution:")
    for asset in Config.TARGET_ASSETS:
        label_col = f'{asset}_Label'
        counts = df_features[label_col].value_counts()
        up_pct = counts.get(1, 0) / len(df_features) * 100
        print(f"  {asset:12s}: {up_pct:.1f}% up, {100-up_pct:.1f}% down")

    print("\nSample features:")
    print(df_features[feature_cols[:10]].head())
