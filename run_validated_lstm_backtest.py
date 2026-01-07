"""
Run backtest for the validated LSTM model (validated_experiment_v1)
with the same execution model as TFT (Kelly, ATR exits, volatility filtering)
"""
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

def main():
    print("=" * 80)
    print("VALIDATED LSTM (v1) - BACKTEST WITH TFT EXECUTION MODEL")
    print("=" * 80)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load and engineer features
    print("\n>>> LOADING DATA")
    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()

    # Extract raw prices
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    # Use same test split as other experiments (85% train/val, 15% test)
    n = len(df_features)
    train_val_end = int(n * 0.85)

    # Create test dataset
    test_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[train_val_end:],
        labels={asset: df_features[f'{asset}_Label'].iloc[train_val_end:]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[train_val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        fit_scaler=True,
        raw_prices=extract_raw_prices(df_features.iloc[train_val_end:])
    )

    print(f"Test period: {test_dataset.dates[0]} to {test_dataset.dates[-1]}")
    print(f"Test samples: {len(test_dataset)}")

    # Load the validated LSTM model
    print("\n>>> LOADING VALIDATED LSTM MODEL")
    model = WeeklyPredictionModel(config).to(device)
    model_path = 'experiments/validated_experiment_v1/models/best_model.pth'

    if not Path(model_path).exists():
        print(f"✗ Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✓ Loaded model from {model_path}")

    # Generate predictions
    print("\n>>> GENERATING PREDICTIONS")
    model.eval()
    predictions = {asset: [] for asset in config.TARGET_ASSETS}
    labels = {asset: [] for asset in config.TARGET_ASSETS}
    dates = []

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    with torch.no_grad():
        for feat, label in test_loader:
            feat = feat.to(device)
            out, _ = model(feat)  # out is predictions dict, _ is attention_weights

            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                predictions[asset].extend(probs)
                labels[asset].extend(label[asset].numpy())

    # Get valid dates (after sequence offset)
    for i in range(len(test_dataset)):
        dates.append(test_dataset.get_date(i))

    # Convert to numpy arrays
    predictions = {asset: np.array(predictions[asset]) for asset in config.TARGET_ASSETS}
    labels = {asset: np.array(labels[asset]) for asset in config.TARGET_ASSETS}

    print(f"Generated {len(predictions[config.TARGET_ASSETS[0]])} predictions")

    # Run advanced backtest with TFT execution model
    print("\n>>> RUNNING ADVANCED BACKTEST")
    backtest = AdvancedBacktest(config)

    results = backtest.run_backtest(
        predictions=predictions,
        labels=labels,
        dates=dates,
        dataset=test_dataset,
        calibrate=True  # Use isotonic calibration
    )

    # Save results
    output_dir = Path('experiments/validated_lstm_v1_backtest')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame([{k: v for k, v in results.items()
                               if k not in ['equity_curve', 'daily_returns', 'trades',
                                           'trades_df', 'asset_stats', 'runners', 'dates']}])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    # Save trades
    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(output_dir / 'trades.csv', index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATED LSTM V1 - BACKTEST RESULTS")
    print("=" * 80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.1f}%")
    print("=" * 80)

    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   - metrics.csv")
    print(f"   - trades.csv")

if __name__ == '__main__':
    main()
