# train_hybrid_v4.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.hybrid_features import HybridFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.training.trainer import Trainer

def train():
    print("INITIALIZING HYBRID WISDOM TRAINING (V4)...")
    config = Config()
    
    # 1. Setup Experiment Directory
    EXPERIMENT_DIR = Path('experiments/hybrid_wisdom_v4')
    config.MODEL_DIR = EXPERIMENT_DIR / 'models'
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load and Scale Data
    print("Loading raw data...")
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    print("Calculating Hybrid Features (V1 Technicals + FracDiff d=0.4)...")
    engineer = HybridFeatureEngineer(config, d=0.4) 
    df = engineer.engineer_features(df_raw)
    
    print(f"Dataset Dimensions: {df.shape}")
    
    # 3. Prepare Dataset
    feature_cols = engineer.get_feature_columns()
    print(f"Feature Vector Size: {len(feature_cols)} features.")
    
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
    
    # 5. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # We use the standard WeeklyPredictionModel (LSTM+Attention)
    model = WeeklyPredictionModel(config).to(device)
    
    # 6. Start Training
    # Hybrid signals are clearer, so we run for more epochs (15 instead of 5) 
    # but with early stopping to prevent the 'Memory Trap'.
    trainer = Trainer(model, train_loader, val_loader, config=config)
    trainer.fit(epochs=20)
    
    print(f"\nHYBRID V4 TRAINING COMPLETE.")
    print(f"Best model saved to {config.MODEL_DIR / 'best_model.pth'}")

if __name__ == "__main__":
    train()
