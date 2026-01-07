# src/data/tft_dataset.py
"""
TFT-compatible Dataset for Multi-Asset Energy Trading.
Wraps the base MultiAssetDataset with TFT-specific preprocessing.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TFTDataset(Dataset):
    """
    Dataset for Temporal Fusion Transformer.
    
    Features:
    - Standard scaling with STRICT train/test separation
    - Positional time features (day-of-week, month)
    - Compatible with TFT input format
    """
    
    def __init__(self, features, labels, dates, sequence_length=60, 
                 scaler=None, fit_scaler=True, raw_prices=None):
        """
        Args:
            features: DataFrame or ndarray of input features
            labels: Dict of {asset: Series/ndarray} labels
            dates: Series/ndarray of dates
            sequence_length: Lookback window size
            scaler: Pre-fitted StandardScaler (for test set)
            fit_scaler: Whether to fit the scaler (True for train, False for test)
            raw_prices: Dict of {asset: DataFrame} with OHLC prices for backtest
        """
        self.sequence_length = sequence_length
        
        # Convert to numpy
        self.features = features.values if hasattr(features, 'values') else features
        self.dates = dates.values if hasattr(dates, 'values') else dates
        self.raw_prices = raw_prices
        
        # Process labels
        self.labels = {}
        for asset, label_series in labels.items():
            l_values = label_series.values if hasattr(label_series, 'values') else label_series
            self.labels[asset] = l_values.astype(np.float32)
        
        # STRICT SCALING: Only fit on training data
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
            else:
                # This case shouldn't happen - scaler should be provided for test
                self.features = self.features.astype(np.float32)
        else:
            self.scaler = scaler
            # Apply pre-fitted scaler (NO FITTING - prevents leakage)
            self.features = self.scaler.transform(self.features)
        
        # Convert to float32
        self.features = self.features.astype(np.float32)
        
        # Pre-compute time features for positional encoding
        self._compute_time_features()
        
        # Convert raw prices to numpy if provided
        self.raw_prices_np = {}
        if self.raw_prices:
            for asset, price_df in self.raw_prices.items():
                self.raw_prices_np[asset] = price_df.values.astype(np.float32)
    
    def _compute_time_features(self):
        """Compute day-of-week and month features for positional encoding."""
        dates_dt = pd.to_datetime(self.dates)
        
        # Day of week (0-4 for Mon-Fri, normalized to 0-1)
        self.day_of_week = (dates_dt.dayofweek.values / 4.0).astype(np.float32)
        
        # Month (1-12, normalized to 0-1)
        self.month = ((dates_dt.month.values - 1) / 11.0).astype(np.float32)
        
        # Week of year (1-52, normalized to 0-1)
        self.week_of_year = ((dates_dt.isocalendar().week.values - 1) / 51.0).astype(np.float32)
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        Returns:
            features: (seq_len, input_dim) tensor
            time_features: (seq_len, 3) tensor [day_of_week, month, week_of_year]
            labels: dict of {asset: scalar} labels
        """
        # Sequence of features
        seq_features = self.features[idx : idx + self.sequence_length]
        
        # Time features for the sequence
        seq_time = np.stack([
            self.day_of_week[idx : idx + self.sequence_length],
            self.month[idx : idx + self.sequence_length],
            self.week_of_year[idx : idx + self.sequence_length]
        ], axis=-1)
        
        # Labels at end of sequence (prediction target)
        target_idx = idx + self.sequence_length
        item_labels = {
            asset: self.labels[asset][target_idx]
            for asset in self.labels.keys()
        }
        
        return (
            torch.from_numpy(seq_features),
            torch.from_numpy(seq_time),
            item_labels
        )
    
    def get_date(self, idx):
        """Return the date for a given index (end of sequence)."""
        return self.dates[idx + self.sequence_length]
    
    def get_raw_prices(self, idx, asset):
        """Return raw OHLC prices for backtest engine."""
        if asset not in self.raw_prices_np:
            return None
        
        target_idx = idx + self.sequence_length
        
        if target_idx >= len(self.raw_prices_np[asset]):
            return None
        
        return self.raw_prices_np[asset][target_idx]


def collate_tft_batch(batch):
    """Custom collate function for TFT batches."""
    features = torch.stack([item[0] for item in batch])
    time_features = torch.stack([item[1] for item in batch])
    
    # Combine labels
    labels = {}
    for asset in batch[0][2].keys():
        labels[asset] = torch.tensor([item[2][asset] for item in batch])
    
    return features, time_features, labels


if __name__ == "__main__":
    # Test the TFT Dataset
    import pandas as pd
    
    n_rows = 200
    n_feats = 199
    
    features = pd.DataFrame(np.random.randn(n_rows, n_feats))
    labels = {
        'WTI': pd.Series(np.random.randint(0, 2, n_rows)),
        'Brent': pd.Series(np.random.randint(0, 2, n_rows))
    }
    dates = pd.Series(pd.date_range('2020-01-01', periods=n_rows, freq='B'))
    
    ds = TFTDataset(features, labels, dates, sequence_length=60)
    print(f"Dataset length: {len(ds)}")
    
    features, time_feats, lbls = ds[0]
    print(f"Features shape: {features.shape}")
    print(f"Time features shape: {time_feats.shape}")
    print(f"Labels: {lbls}")
    print("✓ TFT Dataset Test Passed!")
