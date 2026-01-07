# main_hybrid_v5.py
"""
V5 PRUNED HYBRID WISDOM BACKTEST
- Uses trained V5 model (25 features)
- CONFIDENCE_THRESHOLD = 0.65 (increased from 0.55)
- Goal: Sharpe > 3.23
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.pruned_features import PrunedHybridFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.evaluation.advanced_backtest import AdvancedBacktest

def run_backtest():
    print("="*80)
    print("HYBRID WISDOM V5: HIGH-FIDELITY BACKTEST")
    print("="*80)
    print("Confidence Threshold: 0.65 (Golden Setups Only)")
    print("="*80 + "\n")
    
    config = Config()
    
    # 1. Experiment Directory Setup
    EXPERIMENT_DIR = Path('experiments/hybrid_wisdom_v5_pruned')
    MODEL_PATH = EXPERIMENT_DIR / 'models' / 'best_model.pth'
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Please run train_hybrid_v5.py first.")
        return

    # 2. Data Pipeline (Must match Training)
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = PrunedHybridFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()
    
    print(f"Feature count: {len(feature_cols)} (25 per asset)")
    
    # helper
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    # Use the last 500 days for testing (roughly 2 years)
    test_slice = df.tail(500)
    
    dataset = MultiAssetDataset(
        features=test_slice[feature_cols],
        labels={asset: test_slice[f'{asset}_Label'] for asset in config.TARGET_ASSETS},
        dates=test_slice['Date'],
        sequence_length=config.SEQUENCE_LENGTH,
        raw_prices=extract_raw_prices(test_slice)
    )

    # 3. Model Loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeeklyPredictionModel(config).to(device)
    
    # Patch model for 24 features, TARGET_ASSETS only
    model.features_per_asset = 24
    
    # Create LSTM wrapper
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
            return output
    
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
        original_all_assets = model.config.ALL_ASSETS
        model.config.ALL_ASSETS = model.config.TARGET_ASSETS
        result = original_forward(x)
        model.config.ALL_ASSETS = original_all_assets
        return result
    
    model.forward = patched_forward
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Generate Strategy Decisions with HIGHER THRESHOLD
    CONFIDENCE_THRESHOLD = 0.65  # Increased from 0.55
    
    # AdvancedBacktest expects predictions and labels as dicts
    pred_dict = {asset: [] for asset in config.TARGET_ASSETS}
    label_dict = {asset: [] for asset in config.TARGET_ASSETS}
    dates_list = []

    print(f"Generating predictions for {len(dataset)} timestamps...")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD} (Only Golden Setups)\n")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features, labels = dataset[i]
            dates_list.append(dataset.get_date(i))
            
            x = features.unsqueeze(0).to(device)
            predictions, _ = model(x)
            
            for asset in config.TARGET_ASSETS:
                logit = predictions[asset][0].item()
                prob = 1 / (1 + np.exp(-logit))
                pred_dict[asset].append(prob)
                label_dict[asset].append(labels[asset].item())

    # 5. Execute Advanced Backtest
    backtester = AdvancedBacktest(config=config)
    results = backtester.run_backtest(
        predictions=pred_dict,
        labels=label_dict,
        dates=dates_list,
        dataset=dataset,
        price_data=df_raw
    )
    
    # 6. Save Results
    save_path = EXPERIMENT_DIR / 'metrics.csv'
    pd.DataFrame([results]).to_csv(save_path, index=False)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print(f"Results saved to {EXPERIMENT_DIR}")
    print("\n" + "="*80)
    print("V5 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']:.1%}")
    print(f"Max Drawdown:    {results['max_drawdown']:.1%}")
    print(f"Total Trades:    {results['total_trades']}")
    print("="*80)
    
    # Compare with baselines
    print("\nCOMPARISON WITH BASELINES:")
    print(f"  V1 Baseline:  Sharpe 3.23, Return +104%")
    print(f"  V4 Hybrid:    Sharpe 1.58, Return +30%")
    print(f"  V5 Pruned:    Sharpe {results['sharpe_ratio']:.2f}, Return {results['total_return_pct']:+.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_backtest()
