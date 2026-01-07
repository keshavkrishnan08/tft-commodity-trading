#!/usr/bin/env python3
"""
Train simple ML models (Random Forest, XGBoost, LightGBM).

These models:
- Use the same 219 calibrated features
- Use the same SelectiveBacktest execution framework
- Are simpler than neural networks but can be very effective

Per user request: "idealy they should be simple like random forests etc., keep
the same execution and trade taking ml model"
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.data.tft_dataset import TFTDataset
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.utils.config import Config


class SelectiveBacktest(AdvancedBacktest):
    """Backtest with configurable threshold."""
    def __init__(self, config, probability_threshold=0.50):
        super().__init__(config)
        self.probability_threshold = probability_threshold

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
                if confidence < self.probability_threshold:
                    decision['take_trade'] = False
                    decision['direction'] = 'hold'

            return decision

        self.decision_engine.make_decision = custom_make_decision
        result = super().run_backtest(predictions, labels, dates, dataset,
                                     price_data, intraday_data, calibrate)
        self.decision_engine.make_decision = original_make_decision

        return result


def train_simple_ml_model(model_name, model_params, threshold=0.50):
    """
    Train a simple ML model using the same framework as neural networks.

    Args:
        model_name: Name of the model (e.g., 'RandomForest', 'XGBoost')
        model_params: Parameters for the model
        threshold: Probability threshold for taking trades
    """
    print("="*80)
    print(f"{model_name.upper()} MODEL (Threshold={threshold})")
    print("="*80)

    config = Config()

    # Load data
    print("\n>>> LOADING DATA")
    loader = DataLoader(config)
    df_raw = loader.get_data()
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)
    df = df.copy()

    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Total features: {len(feature_cols)}")

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

    # Sliding window - same as neural networks
    test_years = [2018, 2019, 2020, 2021, 2022]

    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    all_models = {asset: [] for asset in config.TARGET_ASSETS}

    for year in test_years:
        print(f"\n{'='*80}")
        print(f">>> PROCESSING YEAR: {year}")
        print(f"{'='*80}")

        train_df = df[df['Date'].str.contains('|'.join([str(y) for y in range(year-5, year)]))]
        test_df = df[df['Date'].str.contains(str(year))]

        print(f"  Train: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} rows)")
        print(f"  Test:  {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} rows)")

        # Prepare data - use rolling window for features (like neural networks)
        # But for simple ML, we can use the current features directly
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        # Remove NaN rows
        train_mask = ~np.isnan(X_train).any(axis=1)
        test_mask = ~np.isnan(X_test).any(axis=1)

        X_train_clean = X_train[train_mask]
        X_test_clean = X_test[test_mask]

        raw_prices = {
            asset: test_df[[f'{asset}_Open', f'{asset}_High', f'{asset}_Low', f'{asset}_Close']]
            for asset in config.TARGET_ASSETS
        }

        # Train per-asset models
        for asset in config.TARGET_ASSETS:
            print(f"\n  Training {model_name} for {asset}...")

            y_train = train_df[f'{asset}_Label'].values[train_mask]
            y_test = test_df[f'{asset}_Label'].values[test_mask]

            # Create model
            if model_name == 'RandomForest':
                base_model = RandomForestClassifier(**model_params)
            elif model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(**model_params)
            elif model_name == 'LightGBM':
                base_model = lgb.LGBMClassifier(**model_params)
            elif model_name == 'GradientBoosting':
                base_model = GradientBoostingClassifier(**model_params)
            elif model_name == 'ExtraTrees':
                base_model = ExtraTreesClassifier(**model_params)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Train with probability calibration
            print(f"    Training on {len(X_train_clean)} samples...")
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_clean, y_train)

            # Predict probabilities
            probs = calibrated_model.predict_proba(X_test_clean)[:, 1]

            # Store predictions (expand to full test set with NaN handling)
            full_probs = np.full(len(test_df), 0.5)  # Default to 0.5 for NaN rows
            full_probs[test_mask] = probs
            all_predictions[asset].extend(full_probs)
            all_labels[asset].extend(test_df[f'{asset}_Label'].values)

            all_models[asset].append(calibrated_model)

            print(f"    ✓ {asset} complete")

        # Store dates
        all_dates.extend(test_df['Date'].values)
        print(f"  ✓ Year {year} complete")

    # Backtest
    print(f"\n{'='*80}")
    print(f"Running Backtest (Threshold={threshold})...")
    print(f"{'='*80}\n")

    test_full_df = df[df['Date'].str.contains('|'.join([str(y) for y in test_years]))]

    full_raw_prices = {
        asset: test_full_df[[f'{asset}_Open', f'{asset}_High', f'{asset}_Low', f'{asset}_Close']]
        for asset in config.TARGET_ASSETS
    }

    # Create dummy dataset for backtest (just needs dates and prices)
    full_test_ds = TFTDataset(
        features=test_full_df[feature_cols],
        labels={asset: test_full_df[f'{asset}_Label'].values
               for asset in config.TARGET_ASSETS},
        dates=test_full_df['Date'].values,
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True,
        raw_prices=full_raw_prices
    )

    backtest = SelectiveBacktest(config, probability_threshold=threshold)
    results = backtest.run_backtest(
        predictions={asset: np.array(all_predictions[asset])
                    for asset in config.TARGET_ASSETS},
        labels={asset: np.array(all_labels[asset])
               for asset in config.TARGET_ASSETS},
        dates=all_dates,
        dataset=full_test_ds,
        calibrate=False  # Already calibrated
    )

    # Save
    output_dir = Path(f'experiments/{model_name.lower()}_t{int(threshold*100)}')
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([{k: v for k, v in results.items()
                               if k not in ['equity_curve', 'daily_returns', 'trades',
                                           'trades_df', 'asset_stats', 'runners', 'dates']}])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(output_dir / 'trades.csv', index=False)

    # Save models
    for asset in config.TARGET_ASSETS:
        joblib.dump(all_models[asset], output_dir / f'{asset}_models.pkl')

    # Print results
    print("\n" + "="*80)
    print(f"{model_name.upper()} - FINAL RESULTS (Threshold={threshold})")
    print("="*80)
    print(f"Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Win Rate:        {results['win_rate']*100:.1f}%")
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Max Drawdown:    {results['max_drawdown']*100:.1f}%")
    print(f"Profit Factor:   {results.get('profit_factor', 0):.2f}")
    print("="*80)
    print(f"\n✅ Results saved to {output_dir}/")

    return results


def train_all_simple_models(threshold=0.50):
    """Train all available simple ML models."""
    results = {}

    # Random Forest
    print("\n\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    results['RandomForest'] = train_simple_ml_model('RandomForest', rf_params, threshold)

    # Extra Trees
    print("\n\n" + "="*80)
    print("TRAINING EXTRA TREES")
    print("="*80)
    et_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    results['ExtraTrees'] = train_simple_ml_model('ExtraTrees', et_params, threshold)

    # Gradient Boosting
    print("\n\n" + "="*80)
    print("TRAINING GRADIENT BOOSTING")
    print("="*80)
    gb_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42
    }
    results['GradientBoosting'] = train_simple_ml_model('GradientBoosting', gb_params, threshold)

    # XGBoost (if available)
    if HAS_XGB:
        print("\n\n" + "="*80)
        print("TRAINING XGBOOST")
        print("="*80)
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        results['XGBoost'] = train_simple_ml_model('XGBoost', xgb_params, threshold)

    # LightGBM (if available)
    if HAS_LGB:
        print("\n\n" + "="*80)
        print("TRAINING LIGHTGBM")
        print("="*80)
        lgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        results['LightGBM'] = train_simple_ml_model('LightGBM', lgb_params, threshold)

    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY OF ALL SIMPLE ML MODELS")
    print("="*80)
    print(f"{'Model':<20} {'Return':>10} {'Sharpe':>8} {'WinRate':>10} {'Trades':>8}")
    print("-"*80)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['total_return_pct']:>9.2f}% {result['sharpe_ratio']:>8.2f} "
              f"{result['win_rate']*100:>9.1f}% {result['total_trades']:>8}")
    print("="*80)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.50,
                       help='Probability threshold for taking trades (default: 0.50)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'RandomForest', 'XGBoost', 'LightGBM',
                               'GradientBoosting', 'ExtraTrees'],
                       help='Which model to train (default: all)')
    args = parser.parse_args()

    if args.model == 'all':
        train_all_simple_models(threshold=args.threshold)
    else:
        # Train single model
        if args.model == 'RandomForest':
            params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
                     'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
        elif args.model == 'XGBoost':
            params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                     'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                     'n_jobs': -1, 'eval_metric': 'logloss'}
        elif args.model == 'LightGBM':
            params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                     'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                     'n_jobs': -1, 'verbose': -1}
        elif args.model == 'GradientBoosting':
            params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                     'subsample': 0.8, 'random_state': 42}
        elif args.model == 'ExtraTrees':
            params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
                     'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}

        train_simple_ml_model(args.model, params, threshold=args.threshold)
