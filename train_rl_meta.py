
# train_rl_meta.py
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from src.utils.config import Config
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.models.meta_learner import generate_meta_training_data
from src.models.rl_meta_learner import RLMetaLearner

def train():
    print("INITIALIZING RL TRAINING...")
    config = Config()
    
    # 1. Load Data
    print("Loading validated dataset...")
    # Use standard loader and feature engineer
    from src.data.loader import DataLoader as MultiAssetLoader
    from src.data.features import FeatureEngineer
    
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = FeatureEngineer(config)
    df = engineer.engineer_features(df_raw)
    
    # 2. Setup Dataset & Model
    feature_cols = engineer.get_feature_columns()
    
    # Extract raw prices helper
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    dataset = MultiAssetDataset(
        features=df[feature_cols],
        labels={asset: df[f'{asset}_Label'] for asset in config.TARGET_ASSETS},
        dates=df['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(df)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeeklyPredictionModel(config).to(device)
    
    # Load constraints
    model_path = 'experiments/validated_experiment_v1/models/best_model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        print(f"Loaded pre-trained direction model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load direction model from {model_path}: {e}")
        print("Using random init (Training will be noisy).")

    # 3. Generate Training Data (Contexts)
    # We need the "Potential Trades" dataframe
    print("Generating training contexts...")
    meta_train_df = generate_meta_training_data(model, dataset, config)
    
    # 4. Initialize & Train RL Agent
    rl_learner = RLMetaLearner(config)
    
    print(f"\nSTARTING PPO TRAINING on {len(meta_train_df)} contexts...")
    
    # Use FULL data and 200 episodes for production training
    rl_learner.fit(meta_train_df, episodes=200)
    
    # 5. Save
    save_path = 'experiments/rl_sharpe_optimized_v2/models/rl_meta_agent.pth'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    rl_learner.save(save_path)
    print(f"\nSAVED RL AGENT to {save_path}")

if __name__ == "__main__":
    train()
