# main_hybrid_v4.py
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.hybrid_features import HybridFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.evaluation.advanced_backtest import AdvancedBacktest

def run_backtest():
    print("RUNNING HYBRID WISDOM (V4) HIGH-FIDELITY BACKTEST...")
    config = Config()
    
    # 1. Experiment Directory Setup
    EXPERIMENT_DIR = Path('experiments/hybrid_wisdom_v4')
    MODEL_PATH = EXPERIMENT_DIR / 'models' / 'best_model.pth'
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Please run train_hybrid_v4.py first.")
        return

    # 2. Data Pipeline (Must match Training)
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = HybridFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()
    
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
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Generate Strategy Decisions
    # Threshold 0.55 for higher accuracy trades
    CONFIDENCE_THRESHOLD = 0.55
    all_decisions = []

    print(f"Generating predictions for {len(dataset)} timestamps...")
    with torch.no_grad():
        for i in range(len(dataset)):
            features, labels = dataset[i]
            date = dataset.get_date(i)
            
            # Batch dimension
            x = features.unsqueeze(0).to(device)
            predictions, _ = model(x)
            
            for asset in config.TARGET_ASSETS:
                logit = predictions[asset][0].item()
                prob = 1 / (1 + np.exp(-logit))  # Sigmoid
                
                if prob > CONFIDENCE_THRESHOLD:
                    all_decisions.append({
                        'Date': date, 'Asset': asset,
                        'Action': 'long', 'Confidence': prob
                    })
                elif prob < (1 - CONFIDENCE_THRESHOLD):
                    all_decisions.append({
                        'Date': date, 'Asset': asset,
                        'Action': 'short', 'Confidence': 1-prob
                    })

    # 5. Execute Advanced Backtest
    backtester = AdvancedBacktest(config=config)
    
    # AdvancedBacktest.run_backtest(self, predictions, labels, dates, dataset, price_data=None, intraday_data=None)
    # We need to wrap our local predictions/labels into dicts for the backtester
    pred_dict = {asset: [] for asset in config.TARGET_ASSETS}
    label_dict = {asset: [] for asset in config.TARGET_ASSETS}
    dates_list = []

    print(f"Generating batch predictions for backtester...")
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

    results = backtester.run_backtest(
        predictions=pred_dict,
        labels=label_dict,
        dates=dates_list,
        dataset=dataset,
        price_data=df_raw
    )
    
    # 6. Save Results
    # The backtester saves internal results, we need to pass the directory
    save_path = EXPERIMENT_DIR / 'metrics.csv'
    pd.DataFrame([results]).to_csv(save_path, index=False)
    print(f"\nBacktest Complete. Results saved to {EXPERIMENT_DIR}")

if __name__ == "__main__":
    run_backtest()
