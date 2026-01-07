import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.visualization.trade_analytics import TradeAnalytics, create_summary_dashboard

def main():
    config = Config()
    print("\n" + "="*80)
    print("BACKTEST INFERENCE ENGINE (No Training)")
    print("="*80)

    # 1. Load Data
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()
    
    # Define splits (Same as training)
    n = len(df_features)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))

    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    # 2. Create Test Dataset
    test_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[val_end:],
        labels={asset: df_features[f'{asset}_Label'].iloc[val_end:]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        fit_scaler=True, # In a production system, you'd load the scaler, but for backtest consistency we can re-fit on the same test window
        raw_prices=extract_raw_prices(df_features.iloc[val_end:])
    )

    # 3. Load Model
    model = WeeklyPredictionModel(config)
    checkpoint_path = config.MODEL_DIR / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"✗ Model not found at {checkpoint_path}. Please run main_advanced.py once to train.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model_state_dict from {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {checkpoint_path}")
    
    model.eval()

    # 4. Generate Predictions
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []
    
    print("Generating predictions on test set...")
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = features.to(config.DEVICE)
            preds, _ = model(features)
            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(preds[asset]).cpu().numpy().flatten()
                all_predictions[asset].extend(probs)
                all_labels[asset].extend(labels[asset].numpy().flatten())
            
            for j in range(len(features)):
                all_dates.append(test_dataset.get_date(i * config.BATCH_SIZE + j))

    # 5. Run Backtest
    print("\nRunning Backtest with High-Fidelity Verification...")
    backtest = AdvancedBacktest(config)
    results = backtest.run_backtest(
        predictions=all_predictions,
        labels=all_labels,
        dates=all_dates,
        dataset=test_dataset
    )
    
    backtest.print_results(results)
    
    # 6. Save Updated Plots
    plot_path = config.PLOT_DIR / 'latest_backtest_results.png'
    create_summary_dashboard(results, plot_path)
    print(f"\n✓ Dashboard saved to {plot_path}")

if __name__ == "__main__":
    main()
