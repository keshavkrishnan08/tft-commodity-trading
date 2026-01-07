
# train_frac_diff.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
# NEW: Import Fractional Engineer
from src.data.fractional_features import FractionalFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.training.trainer import Trainer

def train():
    print("INITIALIZING FRACTIONAL DIFF TRAINING (V3)...")
    config = Config()
    
    # OVERRIDE SAVE PATHS in Config
    EXPERIMENT_DIR = Path('experiments/frac_diff_experiment_v3')
    config.MODEL_DIR = EXPERIMENT_DIR / 'models'
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    # 2. Engineer Features (FRACTIONAL)
    print("Calculating Fractional Features (d=0.4)...")
    engineer = FractionalFeatureEngineer(config) 
    df = engineer.engineer_features(df_raw)
    
    print(f"Data Shape: {df.shape}")
    
    # 3. Create Dataset
    feature_cols = engineer.get_feature_columns()
    print(f"Using {len(feature_cols)} features (includes FracDiff).")
    
    # Helper for raw prices
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
    
    # 4. Train/Test Split
    train_size = int(len(full_dataset) * 0.8)
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    train_loader = TorchDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 5. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeeklyPredictionModel(config).to(device)
    
    # 6. Training with standardized Trainer
    trainer = Trainer(model, train_loader, val_loader, config=config)
    trainer.fit(epochs=config.EPOCHS)
    
    print(f"\nTraining Complete. Best model saved to {config.MODEL_DIR / 'best_model.pth'}")

if __name__ == "__main__":
    train()
