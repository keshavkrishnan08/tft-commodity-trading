# train_hybrid_v5.py
"""
V5 PRUNED HYBRID WISDOM TRAINING
- 25 features per asset (down from 40+)
- Increased dropout (0.4) for better generalization
- 20 epochs with early stopping
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.pruned_features import PrunedHybridFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.training.trainer import Trainer

def train():
    print("="*80)
    print("HYBRID WISDOM V5: PRUNED & OPTIMIZED")
    print("="*80)
    print("Strategy: Top 25 Features (5 FracDiff + 20 Technicals)")
    print("Goal: Sharpe > 3.23 via Signal-to-Noise Optimization")
    print("="*80 + "\n")
    
    config = Config()
    
    # 1. Setup Experiment Directory
    EXPERIMENT_DIR = Path('experiments/hybrid_wisdom_v5_pruned')
    config.MODEL_DIR = EXPERIMENT_DIR / 'models'
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load and Engineer Features
    print("Loading raw data...")
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    print("Engineering PRUNED Hybrid Features (25 per asset)...")
    engineer = PrunedHybridFeatureEngineer(config, d=0.4) 
    df = engineer.engineer_features(df_raw)
    
    print(f"Dataset Dimensions: {df.shape}")
    
    # 3. Prepare Dataset
    feature_cols = engineer.get_feature_columns()
    print(f"Feature Vector Size: {len(feature_cols)} features (was 199 in V4).")
    print(f"Features per asset: {len(feature_cols) // len(config.ALL_ASSETS)}")
    
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    full_dataset = MultiAssetDataset(
        features=df[feature_cols],
        labels={asset: df[f'{asset}_Label'] for asset in config.TARGET_ASSETS},
        dates=df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(df)
    )
    
    # 4. Train/Val Split (80/20)
    train_size = int(len(full_dataset) * 0.8)
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    train_loader = TorchDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 5. Initialize Model with Higher Dropout
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Update model to use 24 features per asset, 4 TARGET_ASSETS only
    model = WeeklyPredictionModel(config).to(device)
    
    # Patch the model to accept 24 features per asset and only TARGET_ASSETS
    model.features_per_asset = 24
    
    # Create a simple LSTM wrapper that matches AssetLSTM interface
    class AssetLSTMWrapper(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        def forward(self, x):
            output, (h_n, c_n) = self.lstm(x)
            return output  # Return only output, not hidden states
    
    # Recreate asset LSTMs for TARGET_ASSETS only (not DXY)
    model.asset_lstms = torch.nn.ModuleList([
        AssetLSTMWrapper(
            input_size=24,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            dropout=0.4
        )
        for _ in config.TARGET_ASSETS  # Changed from ALL_ASSETS
    ]).to(device)
    
    # Override the forward pass asset list
    original_forward = model.forward
    def patched_forward(x):
        # Temporarily swap ALL_ASSETS with TARGET_ASSETS
        original_all_assets = model.config.ALL_ASSETS
        model.config.ALL_ASSETS = model.config.TARGET_ASSETS
        result = original_forward(x)
        model.config.ALL_ASSETS = original_all_assets
        return result
    
    model.forward = patched_forward
    
    # 6. Start Training
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Epochs: 20")
    print(f"Early Stopping Patience: 15")
    print(f"Dropout: 0.4 (increased from 0.3)")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("="*80 + "\n")
    
    trainer = Trainer(model, train_loader, val_loader, config=config)
    trainer.fit(epochs=20)
    
    print(f"\n{'='*80}")
    print("HYBRID V5 TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best model saved to {config.MODEL_DIR / 'best_model.pth'}")
    print(f"Next step: Run main_hybrid_v5.py for high-fidelity backtest")

if __name__ == "__main__":
    train()
