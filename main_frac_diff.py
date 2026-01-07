
# main_frac_diff.py
import pandas as pd
import torch
from pathlib import Path
from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
# NEW: Import Fractional Engineer
from src.data.fractional_features import FractionalFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.evaluation.advanced_backtest import AdvancedBacktest

def main():
    print("Running FRACTIONAL DIFF Backtest (V3)...")
    config = Config()
    
    # OVERRIDE PATHS
    EXPERIMENT_DIR = Path('experiments/frac_diff_experiment_v3')
    MODEL_PATH = EXPERIMENT_DIR / 'models' / 'best_model.pth'
    PLOT_DIR = EXPERIMENT_DIR / 'plots'
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Update Config for plotting
    config.PLOT_DIR = PLOT_DIR
    
    # 1. Load & Engineer Data (FRACTIONAL)
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    
    engineer = FractionalFeatureEngineer(config) # <--- CRITICAL: Use FracDiff
    df = engineer.engineer_features(df_raw)
    
    # 2. Prepare Dataset
    feature_cols = engineer.get_feature_columns()
    
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
    
    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeeklyPredictionModel(config).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded FracDiff model from {MODEL_PATH}")
    except FileNotFoundError:
        print("Model not found! Run train_frac_diff.py first.")
        return

    # 4. Generate Predictions
    print("Generating predictions...")
    model.eval()
    all_preds = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    dates = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features, labels = dataset[i]
            date = dataset.get_date(i)
            features = features.unsqueeze(0).to(device)
            
            predictions, _ = model(features)
            
            for asset in config.TARGET_ASSETS:
                prob = predictions[asset].item()
                all_preds[asset].append(prob)
                all_labels[asset].append(labels[asset].item())
                
            dates.append(date) # Already string
            
            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}/{len(dataset)} samples")

    # 5. Run Backtest (High Fidelity)
    print("\nRunning Backtest...")
    backtest = AdvancedBacktest(config)
    
    # Load Intraday Data if available (same as v1)
    intraday_data = None 
    # (Leaving separate loading of intraday out for simplicity unless requested, 
    # backtest logic handles missing intraday by using daily Low/High from Raw)
    
    results = backtest.run_backtest(all_preds, all_labels, dates, dataset)
    
    # Save Metrics
    metrics = pd.DataFrame([results])
    metrics.to_csv(EXPERIMENT_DIR / 'metrics.csv', index=False)
    print(f"Results saved to {EXPERIMENT_DIR}")

if __name__ == "__main__":
    main()
