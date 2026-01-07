"""
Train optimized baseline models with same features as TFT.

Models:
1. LSTM V2 (WeeklyPredictionModel) - Fixed validated model
2. TCN - New temporal convolutional network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.models.tcn_model import MultiAssetTCN
from src.models.enhanced_lstm import EnhancedLSTM
from src.evaluation.advanced_backtest import AdvancedBacktest


def train_model(model_type='lstm', device='cpu'):
    """
    Train a model with proper features and optimization.

    Args:
        model_type: 'lstm' or 'tcn'
        device: 'cpu' or 'cuda'
    """
    print("=" * 80)
    print(f"TRAINING OPTIMIZED {model_type.upper()} MODEL")
    print("=" * 80)

    config = Config()
    device = torch.device(device)

    # Load data with CALIBRATED features (same as TFT)
    print("\n>>> LOADING DATA WITH CALIBRATED FEATURES")
    loader = DataLoader(config)
    df_raw = loader.get_data()

    # Use CalibratedFeatureEngineer with d=0.4 (SAME AS TFT)
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()

    # Get feature columns
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)

    print(f"Total features: {config.INPUT_DIM}")

    # Extract raw prices
    def extract_raw_prices(df_subset):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df_subset[cols]
        return raw_prices

    # Split: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    print(f"\nData splits:")
    print(f"  Train: {df['Date'].iloc[0]} to {df['Date'].iloc[train_end-1]} ({train_end} rows)")
    print(f"  Val:   {df['Date'].iloc[train_end]} to {df['Date'].iloc[val_end-1]} ({val_end-train_end} rows)")
    print(f"  Test:  {df['Date'].iloc[val_end]} to {df['Date'].iloc[-1]} ({n-val_end} rows)")

    # Create datasets
    train_ds = MultiAssetDataset(
        features=df[feature_cols].iloc[:train_end],
        labels={asset: df[f'{asset}_Label'].iloc[:train_end]
               for asset in config.TARGET_ASSETS},
        dates=df['Date'].iloc[:train_end],
        sequence_length=config.SEQUENCE_LENGTH,
        fit_scaler=True,
        raw_prices=extract_raw_prices(df.iloc[:train_end])
    )

    val_ds = MultiAssetDataset(
        features=df[feature_cols].iloc[train_end:val_end],
        labels={asset: df[f'{asset}_Label'].iloc[train_end:val_end]
               for asset in config.TARGET_ASSETS},
        dates=df['Date'].iloc[train_end:val_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_ds.scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(df.iloc[train_end:val_end])
    )

    test_ds = MultiAssetDataset(
        features=df[feature_cols].iloc[val_end:],
        labels={asset: df[f'{asset}_Label'].iloc[val_end:]
               for asset in config.TARGET_ASSETS},
        dates=df['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_ds.scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(df.iloc[val_end:])
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False
    )

    # Initialize model
    print(f"\n>>> INITIALIZING {model_type.upper()} MODEL")
    config.LSTM_HIDDEN_SIZE = 128
    config.LSTM_LAYERS = 2
    config.DROPOUT = 0.4  # Slightly higher dropout for regularization

    if model_type == 'lstm':
        model = WeeklyPredictionModel(config).to(device)
    elif model_type == 'enhanced_lstm':
        model = EnhancedLSTM(config).to(device)
    elif model_type == 'tcn':
        model = MultiAssetTCN(config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Class weights for balanced loss
    class_weights = {}
    for asset in config.TARGET_ASSETS:
        labels = df[f'{asset}_Label'].iloc[:train_end].values
        pos_count = labels.sum()
        neg_count = len(labels) - pos_count
        class_weights[asset] = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"  {asset}: weight = {class_weights[asset]:.2f}")

    # Loss and optimizer
    criteria = {
        asset: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([class_weights[asset]]).to(device)
        )
        for asset in config.TARGET_ASSETS
    }

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Higher initial LR
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop with early stopping
    print(f"\n>>> TRAINING")
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        batch_count = 0

        for feat, label in train_loader:
            feat = feat.to(device)
            optimizer.zero_grad()

            # Forward pass
            if model_type == 'lstm':
                out, _ = model(feat)
            else:  # enhanced_lstm or tcn
                out = model(feat)

            # Compute loss
            loss = sum(
                criteria[asset](out[asset], label[asset].to(device).float().unsqueeze(1))
                for asset in config.TARGET_ASSETS
            )

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count

        # Validation
        model.eval()
        val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for feat, label in val_loader:
                feat = feat.to(device)

                if model_type == 'lstm':
                    out, _ = model(feat)
                else:  # enhanced_lstm or tcn
                    out = model(feat)

                loss = sum(
                    criteria[asset](out[asset], label[asset].to(device).float().unsqueeze(1))
                    for asset in config.TARGET_ASSETS
                )
                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:2d}/{max_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}", end='')

        # Early stopping
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            no_improve = 0

            # Save best model
            output_dir = Path(f'experiments/{model_type}_optimized_v2')
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, output_dir / 'models' / 'best_model.pth')
            print(" ✓ (saved)")
        else:
            no_improve += 1
            print(f" (no improve: {no_improve}/{patience})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Generate predictions on test set
    print(f"\n>>> GENERATING TEST PREDICTIONS")
    model.eval()
    predictions = {asset: [] for asset in config.TARGET_ASSETS}
    labels_test = {asset: [] for asset in config.TARGET_ASSETS}
    dates = []

    with torch.no_grad():
        for feat, label in test_loader:
            feat = feat.to(device)

            if model_type == 'lstm':
                out, _ = model(feat)
            else:  # enhanced_lstm or tcn
                out = model(feat)

            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                predictions[asset].extend(probs)
                labels_test[asset].extend(label[asset].numpy())

    # Get test dates
    for i in range(len(test_ds)):
        dates.append(test_ds.get_date(i))

    # Convert to numpy
    predictions = {asset: np.array(predictions[asset]) for asset in config.TARGET_ASSETS}
    labels_test = {asset: np.array(labels_test[asset]) for asset in config.TARGET_ASSETS}

    # Run backtest
    print(f"\n>>> RUNNING BACKTEST")
    backtest = AdvancedBacktest(config)
    results = backtest.run_backtest(
        predictions=predictions,
        labels=labels_test,
        dates=dates,
        dataset=test_ds,
        calibrate=True
    )

    # Save results
    output_dir = Path(f'experiments/{model_type}_optimized_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([{k: v for k, v in results.items()
                               if k not in ['equity_curve', 'daily_returns', 'trades',
                                           'trades_df', 'asset_stats', 'runners', 'dates']}])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(output_dir / 'trades.csv', index=False)

    # Print results
    print("\n" + "=" * 80)
    print(f"{model_type.upper()} OPTIMIZED V2 - RESULTS")
    print("=" * 80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.1f}%")
    print("=" * 80)
    print(f"\n✅ Results saved to {output_dir}/")

    return results


if __name__ == '__main__':
    import sys

    # Allow command-line argument for model type
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'lstm'

    if model_type not in ['lstm', 'enhanced_lstm', 'tcn']:
        print(f"Usage: python {sys.argv[0]} [lstm|enhanced_lstm|tcn]")
        sys.exit(1)

    train_model(model_type=model_type, device='cpu')
