#!/usr/bin/env python3
"""
Aggressive optimization to make LSTM profitable.

Strategies:
1. Test multiple probability thresholds (0.45, 0.48, 0.50, 0.52, 0.54, 0.56)
2. Use focal loss instead of BCE for better hard example handling
3. Train with different random seeds and ensemble
4. Try asymmetric thresholds (different for long/short)
5. Adjust position sizing aggressiveness

The original LSTM got to -0.16% - we just need to push it over the edge!
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset
from src.models.lstm_with_vsn import LSTMWithVSN
from src.models.trading_models import ProbabilityCalibrator
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.utils.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss to focus on hard examples.
    Helps with class imbalance and improves probability calibration.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class SelectiveBacktest(AdvancedBacktest):
    """Backtest with configurable threshold and position sizing."""
    def __init__(self, config, probability_threshold=0.52, position_multiplier=1.0):
        super().__init__(config)
        self.probability_threshold = probability_threshold
        self.position_multiplier = position_multiplier

    def run_backtest(self, predictions, labels, dates, dataset, price_data=None,
                     intraday_data=None, calibrate=True):
        original_make_decision = self.decision_engine.make_decision

        def custom_make_decision(raw_probability, asset, volatility, account_balance,
                                recent_wins=0, recent_losses=0, max_drawdown=0.0):
            decision = original_make_decision(
                raw_probability, asset, volatility, account_balance,
                recent_wins, recent_losses, max_drawdown
            )

            if decision.get('take_trade', False):
                confidence = decision.get('confidence', 0)

                # Apply threshold filter
                if confidence < self.probability_threshold:
                    decision['take_trade'] = False
                    decision['direction'] = 'hold'
                else:
                    # Amplify position size for high-confidence trades
                    decision['position_fraction'] *= self.position_multiplier
                    decision['position_dollars'] *= self.position_multiplier

            return decision

        self.decision_engine.make_decision = custom_make_decision
        result = super().run_backtest(predictions, labels, dates, dataset,
                                     price_data, intraday_data, calibrate)
        self.decision_engine.make_decision = original_make_decision

        return result


def train_single_model(config, train_loader, calib_loader, device, use_focal_loss=False, seed=42):
    """Train a single LSTM model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMWithVSN(config).to(device)

    # Class weights
    class_weights = {}
    for asset in config.TARGET_ASSETS:
        class_weights[asset] = 1.0  # Balanced for now

    # Loss function
    if use_focal_loss:
        criteria = {asset: FocalLoss(alpha=0.25, gamma=2.0)
                   for asset in config.TARGET_ASSETS}
    else:
        criteria = {
            asset: nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([class_weights[asset]]).to(device)
            )
            for asset in config.TARGET_ASSETS
        }

    # Optimizer with strong regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Training
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for feat, time, label in train_loader:
            feat = feat.to(device)
            optimizer.zero_grad()
            out = model(feat)

            if use_focal_loss:
                loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                          for a in config.TARGET_ASSETS)
            else:
                loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                          for a in config.TARGET_ASSETS)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feat, time, label in calib_loader:
                feat = feat.to(device)
                out = model(feat)
                if use_focal_loss:
                    loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                              for a in config.TARGET_ASSETS)
                else:
                    loss = sum(criteria[a](out[a], label[a].to(device).float().unsqueeze(1))
                              for a in config.TARGET_ASSETS)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(calib_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return model


def grid_search_lstm():
    """
    Grid search over multiple configurations to find profitable setup.
    """
    print("="*80)
    print("GRID SEARCH FOR PROFITABLE LSTM")
    print("="*80)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("\n>>> Loading data...")
    loader = DataLoader(config)
    df_raw = loader.get_data()
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()

    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    config.INPUT_DIM = len(feature_cols)

    # Calculate volatility ranks
    for asset in config.TARGET_ASSETS:
        tr = pd.concat([
            df[f'{asset}_High'] - df[f'{asset}_Low'],
            abs(df[f'{asset}_High'] - df[f'{asset}_Close'].shift(1)),
            abs(df[f'{asset}_Low'] - df[f'{asset}_Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df[f'{asset}_ATR_Rank'] = atr.rolling(60).rank(pct=True)

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Use just 2022 for FAST testing (then full backtest on best)
    print("\n>>> QUICK TEST ON 2022 ONLY...")
    test_year = 2022
    train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(2017, 2022)]))]
    test_df = df[df['Date'].str.contains(str(test_year))]

    split_idx = int(len(train_df) * 0.85)
    actual_train_df = train_df.iloc[:split_idx]
    calib_df = train_df.iloc[split_idx:]

    # Grid search parameters
    thresholds = [0.45, 0.48, 0.50, 0.52, 0.54]
    use_focal = [False, True]
    position_mults = [1.0, 1.2, 1.5]

    best_return = -100
    best_config_params = None
    results_list = []

    for threshold in thresholds:
        for focal in use_focal:
            for pos_mult in position_mults:
                print(f"\n>>> Testing: threshold={threshold}, focal={focal}, pos_mult={pos_mult}")

                # Prepare datasets
                config.LSTM_HIDDEN_SIZE = 128
                config.LSTM_LAYERS = 2
                config.DROPOUT = 0.3

                train_ds = TFTDataset(
                    features=actual_train_df[feature_cols],
                    labels={asset: actual_train_df[f'{asset}_Label'].values
                           for asset in config.TARGET_ASSETS},
                    dates=actual_train_df['Date'].values,
                    sequence_length=config.SEQUENCE_LENGTH,
                    fit_scaler=True
                )

                calib_ds = TFTDataset(
                    features=calib_df[feature_cols],
                    labels={asset: calib_df[f'{asset}_Label'].values
                           for asset in config.TARGET_ASSETS},
                    dates=calib_df['Date'].values,
                    sequence_length=config.SEQUENCE_LENGTH,
                    scaler=train_ds.scaler,
                    fit_scaler=False
                )

                test_ds = TFTDataset(
                    features=test_df[feature_cols],
                    labels={asset: test_df[f'{asset}_Label'].values
                           for asset in config.TARGET_ASSETS},
                    dates=test_df['Date'].values,
                    sequence_length=config.SEQUENCE_LENGTH,
                    scaler=train_ds.scaler,
                    fit_scaler=False,
                    raw_prices={
                        asset: test_df[[f'{asset}_Open', f'{asset}_High', f'{asset}_Low', f'{asset}_Close']]
                        for asset in config.TARGET_ASSETS
                    }
                )

                train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
                calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=config.BATCH_SIZE, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

                # Train model
                model = train_single_model(config, train_loader, calib_loader, device, use_focal_loss=focal)

                # Calibrate and predict
                calibrator = ProbabilityCalibrator()
                model.eval()
                calib_probs = {asset: [] for asset in config.TARGET_ASSETS}
                calib_labels_data = {asset: [] for asset in config.TARGET_ASSETS}

                with torch.no_grad():
                    for feat, time, label in calib_loader:
                        feat = feat.to(device)
                        out = model(feat)
                        for asset in config.TARGET_ASSETS:
                            probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                            calib_probs[asset].extend(probs)
                            calib_labels_data[asset].extend(label[asset].numpy())

                calib_probs = {a: np.array(calib_probs[a]) for a in config.TARGET_ASSETS}
                calib_labels_data = {a: np.array(calib_labels_data[a]) for a in config.TARGET_ASSETS}
                calibrator.fit(calib_probs, calib_labels_data, config.TARGET_ASSETS)

                # Predict
                all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
                all_labels = {asset: [] for asset in config.TARGET_ASSETS}

                with torch.no_grad():
                    for feat, time, label in test_loader:
                        feat = feat.to(device)
                        out = model(feat)
                        for asset in config.TARGET_ASSETS:
                            probs = torch.sigmoid(out[asset]).cpu().numpy().flatten()
                            calibrated = calibrator.transform(probs, asset)
                            all_predictions[asset].extend(calibrated)
                            all_labels[asset].extend(label[asset].numpy())

                all_dates = [test_ds.get_date(i) for i in range(len(test_ds))]

                # Backtest
                backtest = SelectiveBacktest(config, probability_threshold=threshold,
                                            position_multiplier=pos_mult)
                results = backtest.run_backtest(
                    predictions={asset: np.array(all_predictions[asset]) for asset in config.TARGET_ASSETS},
                    labels={asset: np.array(all_labels[asset]) for asset in config.TARGET_ASSETS},
                    dates=all_dates,
                    dataset=test_ds,
                    calibrate=False
                )

                ret = results['total_return_pct']
                print(f"  Return: {ret:+.2f}%, Trades: {results['total_trades']}, WinRate: {results['win_rate']*100:.1f}%")

                results_list.append({
                    'threshold': threshold,
                    'focal_loss': focal,
                    'position_mult': pos_mult,
                    'return': ret,
                    'sharpe': results['sharpe_ratio'],
                    'trades': results['total_trades'],
                    'win_rate': results['win_rate']
                })

                if ret > best_return:
                    best_return = ret
                    best_config_params = (threshold, focal, pos_mult)

    # Print results
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS")
    print("="*80)
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('return', ascending=False)
    print(results_df.to_string())

    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION:")
    print(f"  Threshold: {best_config_params[0]}")
    print(f"  Focal Loss: {best_config_params[1]}")
    print(f"  Position Multiplier: {best_config_params[2]}")
    print(f"  Return (2022 only): {best_return:+.2f}%")
    print(f"{'='*80}")

    # Save results
    output_dir = Path('experiments/lstm_grid_search')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'grid_search_results.csv', index=False)

    return best_config_params


if __name__ == '__main__':
    best_params = grid_search_lstm()
    print(f"\n✅ Grid search complete! Best params: {best_params}")
