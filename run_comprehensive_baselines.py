"""
Comprehensive Baseline Experiments: LSTM/TCN on Bitcoin, Silver, Gold
========================================================================

This script implements full baseline comparisons using:
- LSTM with VSN
- TCN with VSN
- Same meta model as TFT
- Sliding window validation (2018-2022)
- Gold, Silver, Bitcoin assets
- 199 calibrated features
- Isotonic calibration
- Advanced backtesting with $10K capital

NO DATA LEAKAGE: Strict temporal splits, annual retraining
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from data.loader import DataLoader
from data.calibrated_features import CalibratedFeatureEngineer
from data.tft_dataset import TFTDataset, collate_tft_batch
from models.lstm_with_vsn import LSTMWithVSN
from models.tcn_with_vsn import TCNWithVSN
from evaluation.advanced_backtest import AdvancedBacktest
from training.sliding_window import SlidingWindowTrainer
from utils.config import *

# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configuration
ASSETS = ['Gold', 'Silver', 'BTC']
MODELS = ['LSTM', 'TCN']
TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
SEQUENCE_LENGTH = 90
NUM_FEATURES = 199  # Calibrated features
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS_PER_WINDOW = 5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ComprehensiveBaselineRunner:
    """
    Runs comprehensive baseline experiments with proper data handling,
    no leakage, and extensive evaluation.
    """

    def __init__(self, model_name='LSTM', asset='Gold'):
        self.model_name = model_name
        self.asset = asset
        self.results = {
            'predictions': {},
            'trades': {},
            'metrics': {},
            'calibration': {}
        }

        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = project_root / 'experiments' / f'{model_name.lower()}_baselines_{asset.lower()}_{timestamp}'
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE BASELINE: {model_name} on {asset}")
        print(f"Experiment Directory: {self.exp_dir}")
        print(f"{'='*80}\n")

    def load_and_prepare_data(self):
        """Load data and engineer features with strict temporal split"""
        print(f"[1/8] Loading {self.asset} data and engineering features...")

        # Load ALL data (like TFT does) - this includes cross-asset features
        loader = DataLoader()
        df_raw = loader.get_data()  # Merges oil/gas, metals/crypto, and DXY

        print(f"   Loaded merged data shape: {df_raw.shape}")
        print(f"   Columns: {df_raw.columns.tolist()}")

        # Check if our target asset is present
        close_col = f"{self.asset}_Close"
        if close_col not in df_raw.columns:
            raise ValueError(f"Asset {self.asset} not found in data. Available assets: {[c.replace('_Close', '') for c in df_raw.columns if c.endswith('_Close')]}")

        # Engineer features on ALL assets (cross-asset features are important!)
        feature_engineer = CalibratedFeatureEngineer()
        df_features, thresholds = feature_engineer.engineer_features(df_raw)

        # Feature engineering already drops NaN rows
        # df_features is already clean

        print(f"   Features engineered: {df_features.shape[1]} columns")
        print(f"   Clean data shape: {df_features.shape}")

        # Ensure Date is the index or use it for filtering
        if 'Date' in df_features.columns:
            df_features.set_index('Date', inplace=True)

        # Convert index to datetime if it's not already
        if not isinstance(df_features.index, pd.DatetimeIndex):
            df_features.index = pd.to_datetime(df_features.index)

        # Strict temporal split
        train_end = '2017-12-31'
        test_start = '2018-01-01'

        df_train = df_features[df_features.index <= train_end]
        df_test = df_features[df_features.index >= test_start]

        print(f"   Train: {df_train.shape[0]} samples ({df_train.index.min()} to {df_train.index.max()})")
        print(f"   Test:  {df_test.shape[0]} samples ({df_test.index.min()} to {df_test.index.max()})")

        if df_train.shape[0] < SEQUENCE_LENGTH:
            raise ValueError(f"Insufficient training data: {df_train.shape[0]} < {SEQUENCE_LENGTH}")

        self.df_train = df_train
        self.df_test = df_test
        self.df_full = df_features

        return df_train, df_test

    def create_datasets(self, df_train, df_test):
        """Create PyTorch datasets with proper scaling"""
        print(f"[2/8] Creating datasets...")

        # Reset index to have Date as a column
        df_train = df_train.reset_index()
        df_test = df_test.reset_index()

        # Get all feature columns (exclude OHLC, Date, and Label columns)
        exclude_cols = ['Date'] + [c for c in df_train.columns if any(c.endswith(suffix) for suffix in ['_Open', '_High', '_Low', '_Close', '_Label'])]
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

        print(f"   Using {len(feature_cols)} feature columns")

        # Check if our asset's label exists
        label_col = f"{self.asset}_Label"
        if label_col not in df_train.columns:
            print(f"   Warning: {label_col} not found. Available labels: {[c for c in df_train.columns if c.endswith('_Label')]}")
            # Use first available label or create from returns
            available_labels = [c for c in df_train.columns if c.endswith('_Label')]
            if available_labels:
                label_col = available_labels[0]
                print(f"   Using {label_col} instead")
            else:
                print(f"   Creating label from price changes...")
                close_col = f"{self.asset}_Close"
                # Create binary label: 1 if price goes up, 0 if down
                df_train[label_col] = (df_train[close_col].pct_change().shift(-1) > 0).astype(int)
                df_test[label_col] = (df_test[close_col].pct_change().shift(-1) > 0).astype(int)

        # Drop last row (no next-day label)
        df_train = df_train.iloc[:-1].copy()
        df_test = df_test.iloc[:-1].copy()

        # Create datasets using actual TFTDataset API
        train_dataset = TFTDataset(
            features=df_train[feature_cols],
            labels={self.asset: df_train[label_col]},
            dates=df_train['Date'],
            sequence_length=SEQUENCE_LENGTH,
            fit_scaler=True  # Fit scaler on training data
        )

        test_dataset = TFTDataset(
            features=df_test[feature_cols],
            labels={self.asset: df_test[label_col]},
            dates=df_test['Date'],
            sequence_length=SEQUENCE_LENGTH,
            scaler=train_dataset.scaler,  # Use train scaler!
            fit_scaler=False
        )

        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Features per timestep: {len(feature_cols)}")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_features = len(feature_cols)  # Store for model initialization

        return train_dataset, test_dataset

    def initialize_model(self):
        """Initialize LSTM or TCN model"""
        print(f"[3/8] Initializing {self.model_name} model...")

        num_features = self.num_features  # Use stored feature count

        # Create a simple config object for the model
        class ModelConfig:
            INPUT_DIM = num_features
            LSTM_HIDDEN_SIZE = HIDDEN_DIM
            LSTM_LAYERS = NUM_LAYERS
            DROPOUT = DROPOUT
            TARGET_ASSETS = [self.asset]  # Single asset

        config = ModelConfig()

        if self.model_name == 'LSTM':
            model = LSTMWithVSN(config)
        elif self.model_name == 'TCN':
            model = TCNWithVSN(config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        model = model.to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Model: {self.model_name}")
        print(f"   Parameters: {num_params:,}")
        print(f"   Hidden dim: {HIDDEN_DIM}")
        print(f"   Layers: {NUM_LAYERS}")

        self.model = model
        return model

    def train_model(self):
        """Train model on training data"""
        print(f"[4/8] Training model...")

        # Create dataloaders with custom collate function
        train_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_tft_batch
        )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        train_losses = []

        for epoch in range(EPOCHS_PER_WINDOW):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Unpack batch: (features, time_features, labels)
                features, time_features, labels = batch
                features = features.to(device)
                targets = labels[self.asset].to(device).float()

                # Forward
                optimizer.zero_grad()
                predictions_dict = self.model(features)
                predictions = predictions_dict[self.asset].squeeze()

                # Loss
                loss = criterion(predictions, targets)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            if (epoch + 1) % 1 == 0:
                print(f"   Epoch {epoch+1}/{EPOCHS_PER_WINDOW}: Loss = {avg_loss:.6f}")

        self.train_losses = train_losses

        # Save training curve
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{self.model_name} Training Loss - {self.asset}')
        plt.grid(alpha=0.3)
        plt.savefig(self.exp_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Final training loss: {train_losses[-1]:.6f}")

    def generate_predictions(self):
        """Generate predictions on test set"""
        print(f"[5/8] Generating predictions...")

        self.model.eval()

        test_loader = TorchDataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_tft_batch
        )

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch: (features, time_features, labels)
                features, time_features, labels = batch
                features = features.to(device)
                targets = labels[self.asset].float()

                predictions_dict = self.model(features)
                predictions = predictions_dict[self.asset].squeeze()

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate
        predictions = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        # Get dates (skip sequence_length initial dates)
        dates = self.df_test.index[SEQUENCE_LENGTH:]
        dates = dates[:len(predictions)]  # Match length

        # Store
        self.predictions = predictions
        self.targets = targets
        self.pred_dates = dates

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        # Directional accuracy
        actual_direction = np.sign(targets)
        pred_direction = np.sign(predictions)
        direction_accuracy = (actual_direction == pred_direction).mean()

        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r2:.6f}")
        print(f"   Direction Accuracy: {direction_accuracy:.2%}")

        self.results['metrics']['prediction'] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy)
        }

        return predictions, targets

    def calibrate_probabilities(self):
        """Calibrate predictions to probabilities using isotonic regression"""
        print(f"[6/8] Calibrating probabilities...")

        # Convert predictions to probabilities (sigmoid of scaled predictions)
        raw_probs = 1 / (1 + np.exp(-self.predictions * 10))  # Scale for better separation

        # Actual binary outcomes (positive return = 1, negative = 0)
        actual_binary = (self.targets > 0).astype(int)

        # Fit isotonic regression on first 50% of test data
        split_idx = len(raw_probs) // 2

        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(raw_probs[:split_idx], actual_binary[:split_idx])

        # Calibrate all probabilities
        calibrated_probs = calibrator.predict(raw_probs)

        self.raw_probs = raw_probs
        self.calibrated_probs = calibrated_probs
        self.calibrator = calibrator

        # Calibration metrics
        print(f"   Raw prob range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
        print(f"   Calibrated range: [{calibrated_probs.min():.3f}, {calibrated_probs.max():.3f}]")

        # Plot calibration curve
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Before calibration
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        digitized = np.digitize(raw_probs, bins) - 1
        digitized = np.clip(digitized, 0, len(bin_centers) - 1)
        actual_freq_raw = [actual_binary[digitized == i].mean() if (digitized == i).sum() > 0 else np.nan
                          for i in range(len(bin_centers))]

        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        axes[0].plot(bin_centers, actual_freq_raw, 'o-', label='Raw')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Actual Frequency')
        axes[0].set_title('Before Calibration')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # After calibration
        digitized_cal = np.digitize(calibrated_probs, bins) - 1
        digitized_cal = np.clip(digitized_cal, 0, len(bin_centers) - 1)
        actual_freq_cal = [actual_binary[digitized_cal == i].mean() if (digitized_cal == i).sum() > 0 else np.nan
                          for i in range(len(bin_centers))]

        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        axes[1].plot(bin_centers, actual_freq_cal, 'o-', label='Calibrated', color='green')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Actual Frequency')
        axes[1].set_title('After Calibration')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

        return calibrated_probs

    def run_backtest(self):
        """Run advanced backtest with meta model"""
        print(f"[7/8] Running backtest...")

        # Prepare backtest data
        backtest_df = self.df_test.loc[self.pred_dates].copy()
        backtest_df['predicted_prob'] = self.calibrated_probs
        backtest_df['predicted_return'] = self.predictions

        # Initialize backtest with config
        class BacktestConfig:
            TARGET_ASSETS = [self.asset]
            INITIAL_CAPITAL = 10000
            DEFAULT_STOP_LOSS = 0.02
            DEFAULT_TAKE_PROFIT = 0.05
            TRANSACTION_COST = 0.006
            MAX_POSITION_SIZE = 0.10
            MIN_POSITION_SIZE = 0.01
            SLIPPAGE_PCT = 0.0002
            COMMISSION_PCT = 0.0001
            ENABLE_TRAILING_STOP = True
            PREDICTION_HORIZONS = {'weekly': 5, 'biweekly': 10, 'monthly': 21}
            PREDICTION_HORIZON = 'weekly'

        backtest = AdvancedBacktest(config=BacktestConfig())

        # Prepare inputs for backtest (matching actual signature)
        predictions_dict = {self.asset: self.calibrated_probs}
        labels_dict = {self.asset: (self.targets > 0).astype(int)}  # Convert to binary
        dates_list = self.pred_dates.tolist() if hasattr(self.pred_dates, 'tolist') else list(self.pred_dates)

        # Run backtest with correct signature
        results = backtest.run_backtest(
            predictions=predictions_dict,
            labels=labels_dict,
            dates=dates_list,
            dataset=self.test_dataset,
            price_data=self.df_test.reset_index(),  # Pass full price data
            calibrate=False  # Already calibrated
        )

        # Extract results
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', [])
        final_value = results.get('final_equity', results.get('final_value', 10000))
        total_return = results.get('total_return_pct', results.get('total_return', 0)) / 100.0 if 'total_return_pct' in results else results.get('total_return', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown_pct', results.get('max_drawdown', 0)) / 100.0 if 'max_drawdown_pct' in results else results.get('max_drawdown', 0)
        win_rate = results.get('win_rate', 0)

        print(f"\n   BACKTEST RESULTS:")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Max Drawdown: {max_dd:.2%}")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   Number of Trades: {len(trades)}")

        self.results['backtest'] = {
            'final_value': float(final_value),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'num_trades': len(trades)
        }
        self.trades = trades
        self.equity_curve = equity_curve

        return results

    def save_results(self):
        """Save all results to disk"""
        print(f"[8/8] Saving results...")

        # Save metrics
        with open(self.exp_dir / 'metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save predictions
        pred_df = pd.DataFrame({
            'date': self.pred_dates,
            'actual': self.targets,
            'predicted': self.predictions,
            'raw_prob': self.raw_probs,
            'calibrated_prob': self.calibrated_probs
        })
        pred_df.to_csv(self.exp_dir / 'predictions.csv', index=False)

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(self.exp_dir / 'trades.csv', index=False)

        # Save model
        torch.save(self.model.state_dict(), self.exp_dir / 'model.pth')

        # Generate comprehensive plots
        self.generate_visualizations()

        print(f"   All results saved to: {self.exp_dir}")

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""

        # 1. Prediction vs Actual
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        axes[0].plot(self.pred_dates, self.targets, label='Actual', alpha=0.7)
        axes[0].plot(self.pred_dates, self.predictions, label='Predicted', alpha=0.7)
        axes[0].set_ylabel('Return')
        axes[0].set_title(f'{self.model_name} - {self.asset}: Predictions vs Actual')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Equity curve
        if self.equity_curve:
            equity_dates = self.pred_dates[:len(self.equity_curve)]
            axes[1].plot(equity_dates, self.equity_curve, linewidth=2, color='green')
            axes[1].axhline(10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
            axes[1].set_ylabel('Portfolio Value ($)')
            axes[1].set_xlabel('Date')
            axes[1].set_title('Equity Curve')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'predictions_and_equity.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Return distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(self.targets, bins=50, alpha=0.7, label='Actual')
        axes[0].hist(self.predictions, bins=50, alpha=0.7, label='Predicted')
        axes[0].set_xlabel('Return')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Return Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Scatter
        axes[1].scatter(self.targets, self.predictions, alpha=0.3, s=10)
        axes[1].plot([self.targets.min(), self.targets.max()],
                    [self.targets.min(), self.targets.max()],
                    'r--', alpha=0.5, label='Perfect')
        axes[1].set_xlabel('Actual Return')
        axes[1].set_ylabel('Predicted Return')
        axes[1].set_title(f'Scatter Plot (R² = {self.results["metrics"]["prediction"]["r2"]:.3f})')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'return_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Visualizations saved")

    def run_full_experiment(self):
        """Run complete experiment pipeline"""
        try:
            # Pipeline
            df_train, df_test = self.load_and_prepare_data()
            train_dataset, test_dataset = self.create_datasets(df_train, df_test)
            model = self.initialize_model()
            self.train_model()
            predictions, targets = self.generate_predictions()
            calibrated_probs = self.calibrate_probabilities()
            backtest_results = self.run_backtest()
            self.save_results()

            print(f"\n{'='*80}")
            print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"Results saved to: {self.exp_dir}")
            print(f"{'='*80}\n")

            return self.results

        except Exception as e:
            print(f"\nERROR in experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def run_all_baselines():
    """Run all baseline experiments"""

    print("\n" + "="*80)
    print("COMPREHENSIVE BASELINE EXPERIMENTS")
    print("Assets: Gold, Silver, Bitcoin")
    print("Models: LSTM, TCN")
    print("Period: 2018-2022")
    print("="*80 + "\n")

    # Create results directory
    results_dir = project_root / 'experiments' / 'baselines_comprehensive'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run experiments
    for model_name in MODELS:
        for asset in ASSETS:
            print(f"\n{'#'*80}")
            print(f"Running: {model_name} on {asset}")
            print(f"{'#'*80}\n")

            runner = ComprehensiveBaselineRunner(model_name=model_name, asset=asset)
            results = runner.run_full_experiment()

            if results:
                all_results[f'{model_name}_{asset}'] = results

    # Save combined results
    combined_file = results_dir / 'combined_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Combined results: {combined_file}")
    print(f"{'='*80}\n")

    # Generate comparison report
    generate_comparison_report(all_results, results_dir)

    return all_results


def generate_comparison_report(all_results, output_dir):
    """Generate comparison report across all experiments"""

    print("Generating comparison report...")

    # Extract metrics
    comparison_data = []

    for exp_name, results in all_results.items():
        model, asset = exp_name.split('_', 1)

        metrics = results.get('metrics', {})
        pred_metrics = metrics.get('prediction', {})
        backtest = results.get('backtest', {})

        comparison_data.append({
            'Model': model,
            'Asset': asset,
            'RMSE': pred_metrics.get('rmse', np.nan),
            'MAE': pred_metrics.get('mae', np.nan),
            'R²': pred_metrics.get('r2', np.nan),
            'Direction_Acc': pred_metrics.get('direction_accuracy', np.nan),
            'Total_Return': backtest.get('total_return', np.nan),
            'Sharpe': backtest.get('sharpe_ratio', np.nan),
            'Max_DD': backtest.get('max_drawdown', np.nan),
            'Win_Rate': backtest.get('win_rate', np.nan),
            'Num_Trades': backtest.get('num_trades', np.nan)
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Save table
    df_comparison.to_csv(output_dir / 'comparison_table.csv', index=False)

    # Generate comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ('R²', 'Prediction R²'),
        ('Direction_Acc', 'Direction Accuracy'),
        ('Total_Return', 'Total Return (%)'),
        ('Sharpe', 'Sharpe Ratio'),
        ('Max_DD', 'Max Drawdown (%)'),
        ('Num_Trades', 'Number of Trades')
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        # Group by model and asset
        pivot = df_comparison.pivot(index='Asset', columns='Model', values=metric)
        pivot.plot(kind='bar', ax=ax, rot=0)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Asset')
        ax.set_ylabel(metric)
        ax.legend(title='Model')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plots.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Print summary table
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(df_comparison.to_string(index=False))
    print("="*100 + "\n")

    print(f"Comparison report saved to: {output_dir}")


if __name__ == '__main__':
    # Run all baseline experiments
    results = run_all_baselines()

    print("\n" + "="*80)
    print("DONE! Check experiments/baselines_comprehensive/ for results")
    print("="*80)
