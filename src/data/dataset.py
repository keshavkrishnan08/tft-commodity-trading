# src/data/dataset.py
"""
PyTorch Dataset for Multi-Asset Time Series.
Handles sequence creation and scaling for multiple target assets.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class MultiAssetDataset(Dataset):
    """
    Dataset for multi-asset predictive modeling.
    Creates sliding window sequences from multiple assets.
    """
    def __init__(self, features, labels, dates, sequence_length=60, scaler=None, fit_scaler=True, raw_prices=None):
        self.features = features.values if hasattr(features, 'values') else features
        self.labels = labels
        self.dates = dates.values if hasattr(dates, 'values') else dates
        self.sequence_length = sequence_length
        self.raw_prices = raw_prices
        
        # Scaling
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
            
        # Convert to float32 tensors
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        
        # Convert labels to dict of tensors
        self.labels_tensors = {}
        for asset, label_series in self.labels.items():
            l_values = label_series.values if hasattr(label_series, 'values') else label_series
            self.labels_tensors[asset] = torch.tensor(l_values, dtype=torch.float32)

        # Convert raw prices to tensors if provided
        self.raw_prices_tensors = {}
        if self.raw_prices:
            for asset, price_df in self.raw_prices.items():
                self.raw_prices_tensors[asset] = torch.tensor(price_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features_tensor) - self.sequence_length

    def __getitem__(self, idx):
        # Sliding window sequence
        sequence = self.features_tensor[idx : idx + self.sequence_length]
        
        # Labels for all target assets at the end of the window
        # Note: labels[idx + sequence_length] matches the next-period prediction target
        target_idx = idx + self.sequence_length
        
        item_labels = {}
        for asset in self.labels_tensors.keys():
            item_labels[asset] = self.labels_tensors[asset][target_idx]
            
        return sequence, item_labels

    def get_date(self, idx):
        """Return the date for a given index (end of sequence)"""
        return self.dates[idx + self.sequence_length]

    def get_raw_prices(self, idx, asset):
        """Return raw prices for a given index and asset."""
        if asset not in self.raw_prices_tensors:
            return None
        
        # We need the price at the target_idx (the day of prediction/entry)
        target_idx = idx + self.sequence_length
        
        # Bounds check
        if target_idx >= len(self.raw_prices_tensors[asset]):
            return None
            
        return self.raw_prices_tensors[asset][target_idx]

if __name__ == "__main__":
    # Test dataset
    import pandas as pd
    n_rows = 200
    n_feats = 10
    features = pd.DataFrame(np.random.randn(n_rows, n_feats))
    labels = {
        'WTI': pd.Series(np.random.randint(0, 2, n_rows)),
        'Brent': pd.Series(np.random.randint(0, 2, n_rows))
    }
    dates = pd.Series(pd.date_range('2020-01-01', periods=n_rows))
    
    ds = MultiAssetDataset(features, labels, dates, sequence_length=60)
    print(f"Dataset length: {len(ds)}")
    
    seq, lbls = ds[0]
    print(f"Sequence shape: {seq.shape}")
    print(f"Labels: {lbls}")
    print("✓ src/data/dataset.py re-implementation passed test!")
