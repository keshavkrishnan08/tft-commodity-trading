# main_meta_learning.py
"""
Two-Layer ML Trading System:
  Layer 1: Direction Model (LSTM + Attention) predicts UP/DOWN
  Layer 2: Meta-Learner (XGBoost) decides WHICH trades to take and HOW MUCH

This is the most advanced pipeline - ML deciding on ML.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path

from src.utils.config import Config
from src.data.loader import DataLoader as MultiAssetLoader
from src.data.features import FeatureEngineer
from src.data.dataset import MultiAssetDataset
from src.models.weekly_model import WeeklyPredictionModel
from src.models.meta_learner import MetaLearner
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.visualization.trade_analytics import TradeAnalytics, create_summary_dashboard


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MetaLearningBacktest(AdvancedBacktest):
    """
    Advanced backtest that uses MetaLearner for trading decisions.

    Overrides the decision-making logic to use trained ML model
    instead of heuristics.
    """

    def __init__(self, config, meta_learner):
        super().__init__(config)
        self.meta_learner = meta_learner

    def run_backtest(self, predictions, labels, dates, dataset, intraday_data=None):
        """Run backtest using meta-learner for decisions."""
        self.intraday_data = intraday_data
        if intraday_data:
            from src.evaluation.high_fidelity_simulator import HighFidelitySimulator
            self.hf_simulator = HighFidelitySimulator(self.config, intraday_data)
        else:
            self.hf_simulator = None

        n_samples = len(dates)

        # Initialize tracking
        self.capital = self.initial_capital
        self.equity_curve = [self.capital]
        self.trades = []
        self.daily_returns = []
        self.max_equity = self.capital
        self.max_drawdown = 0.0

        self.recent_wins = {asset: 0 for asset in self.config.TARGET_ASSETS}
        self.recent_losses = {asset: 0 for asset in self.config.TARGET_ASSETS}

        print(f"\nRunning META-LEARNING backtest on {n_samples} samples...")

        for i in range(n_samples):
            date = dates[i]
            daily_pnl = 0.0

            for asset in self.config.TARGET_ASSETS:
                raw_prob = predictions[asset][i]
                actual = labels[asset][i]

                # Calculate volatility from TRUE price data
                volatility = self._estimate_volatility(dataset, asset, i)

                # IMPROVED: Extract ACTUAL market state from dataset features
                try:
                    # Get raw (unscaled) feature values for current sample
                    feature_idx = i + dataset.sequence_length - 1
                    raw_features = dataset.features.iloc[feature_idx]

                    # Extract actual technical indicators
                    rsi = raw_features.get(f'{asset}_RSI', 50.0)
                    macd_hist = raw_features.get(f'{asset}_MACD_Hist', 0.0)

                    # Calculate momentum from returns
                    momentum_20d = raw_features.get(f'{asset}_Return', 0.0) * 20  # 20-day momentum proxy
                    momentum_60d = raw_features.get(f'{asset}_Return', 0.0) * 60  # 60-day momentum proxy

                    # Volume surge (if available)
                    volume_surge = 1.0  # Not available in this dataset

                except Exception as e:
                    # Fallback to default values if extraction fails
                    rsi = 50.0
                    macd_hist = 0.0
                    momentum_20d = 0.0
                    momentum_60d = 0.0
                    volume_surge = 1.0

                # Extract market state for meta-learner
                market_state = {
                    'volatility': volatility,
                    'momentum_20d': momentum_20d,
                    'momentum_60d': momentum_60d,
                    'rsi': rsi,
                    'macd_hist': macd_hist,
                    'volume_surge': volume_surge,
                }

                # Account state
                current_drawdown = (self.max_equity - self.capital) / self.max_equity if self.max_equity > 0 else 0
                account_state = {
                    'capital': self.capital,
                    'drawdown': current_drawdown,
                    'recent_win_rate': self.recent_wins[asset] / max(1, self.recent_wins[asset] + self.recent_losses[asset]),
                    'recent_sharpe': 0.0,  # TODO: Calculate rolling Sharpe
                    'consecutive_wins': self.recent_wins[asset],
                    'consecutive_losses': self.recent_losses[asset],
                }

                # ===== META-LEARNER DECISION =====
                # This is where the SECOND ML model decides!
                decision = self.meta_learner.predict_decision(
                    raw_prob,
                    market_state,
                    account_state
                )

                if decision['should_take']:
                    # Determine direction
                    direction = 'long' if raw_prob > 0.5 else 'short'

                    # Position size from meta-learner
                    position_fraction = decision['position_size']
                    position_dollars = self.capital * position_fraction

                    # Exit levels: TIGHTER stop loss for better risk management
                    stop_loss_pct = max(0.012, volatility * 1.5)  # 1.5x ATR (was 2.0x)
                    take_profit_pct = stop_loss_pct * 2.0  # Maintain 2:1 R:R

                    # Simulate trade
                    trade_decision = {
                        'take_trade': True,
                        'direction': direction,
                        'raw_probability': raw_prob,
                        'calibrated_probability': raw_prob,  # No calibration in meta-learning mode
                        'confidence': decision['take_probability'],
                        'position_fraction': position_fraction,
                        'position_dollars': position_dollars,
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                        'risk_reward': take_profit_pct / stop_loss_pct,
                        'volatility': volatility
                    }

                    # Execute trade
                    trade_result = self._simulate_trade(
                        trade_decision, actual, asset, date, volatility, i, dataset
                    )

                    if trade_result:
                        daily_pnl += trade_result['pnl']

                        # ONLINE LEARNING: Update confidence feedback
                        confidence_level = decision['confidence']
                        self.meta_learner.update_confidence_feedback(
                            confidence_level,
                            trade_result['won'],
                            trade_result['pnl']
                        )

                        # Update streaks
                        if trade_result['won']:
                            self.recent_wins[asset] = min(5, self.recent_wins[asset] + 1)
                            self.recent_losses[asset] = 0
                        else:
                            self.recent_losses[asset] = min(5, self.recent_losses[asset] + 1)
                            self.recent_wins[asset] = 0

                        # Record trade
                        self.trades.append(trade_result)

            # Update capital
            self.capital += daily_pnl
            self.equity_curve.append(self.capital)
            self.daily_returns.append(daily_pnl / self.equity_curve[-2] if self.equity_curve[-2] > 0 else 0)

            # Update max equity and drawdown
            if self.capital > self.max_equity:
                self.max_equity = self.capital

            current_dd = (self.max_equity - self.capital) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, current_dd)

        return self._generate_results(dates)


def main():
    """Main meta-learning trading pipeline"""

    config = Config()
    set_seed(config.RANDOM_SEED)

    print("\n" + "="*80)
    print("TWO-LAYER ML TRADING SYSTEM")
    print("="*80)
    print("Layer 1: Direction Model (LSTM + Attention)")
    print("         Predicts UP/DOWN for each asset")
    print()
    print("Layer 2: Meta-Learner (XGBoost)")
    print("         Decides WHICH trades to take and HOW MUCH to position")
    print()
    print("This is ML on top of ML - the meta-learner learned from 1000s of")
    print("simulated trades what works and what doesn't!")
    print("="*80 + "\n")

    # ====================================================================
    # STEP 1: LOAD MODELS
    # ====================================================================
    print("STEP 1: Loading trained models...")
    print("-" * 80)

    # Load direction model
    direction_model_path = config.MODEL_DIR / 'best_model.pth'
    if not direction_model_path.exists():
        print(f"ERROR: Direction model not found at {direction_model_path}")
        print("Run: python main_advanced.py first")
        return

    direction_model = WeeklyPredictionModel(config)
    checkpoint = torch.load(direction_model_path, map_location=config.DEVICE)
    direction_model.load_state_dict(checkpoint['model_state_dict'])
    direction_model.to(config.DEVICE)
    direction_model.eval()
    print(f"✓ Direction model loaded")

    # Load meta-learner
    meta_learner_path = config.MODEL_DIR / 'meta_learner.pkl'
    if not meta_learner_path.exists():
        print(f"ERROR: Meta-learner not found at {meta_learner_path}")
        print("Run: python train_meta_learner.py first")
        return

    meta_learner = MetaLearner(config)
    meta_learner.load(meta_learner_path)
    print(f"✓ Meta-learner loaded")

    # ====================================================================
    # STEP 2: LOAD TEST DATA
    # ====================================================================
    print("\nSTEP 2: Loading test data...")
    print("-" * 80)

    loader = MultiAssetLoader(config)
    df_raw = loader.get_data()

    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_features(df_raw)
    feature_cols = engineer.get_feature_columns()

    # Split data
    n = len(df_features)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))

    # Extract raw prices for backtesting
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    test_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[val_end:],
        labels={asset: df_features[f'{asset}_Label'].iloc[val_end:]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True,
        raw_prices=extract_raw_prices(df_features.iloc[val_end:])
    )

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"✓ Test set: {len(test_dataset)} sequences")

    # ====================================================================
    # STEP 3: GENERATE PREDICTIONS (Layer 1)
    # ====================================================================
    print("\nSTEP 3: Generating direction predictions...")
    print("-" * 80)

    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []

    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = features.to(config.DEVICE)
            preds, _ = direction_model(features)

            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(preds[asset]).cpu().numpy().flatten()
                true_labels = labels[asset].numpy().flatten()

                all_predictions[asset].extend(probs)
                all_labels[asset].extend(true_labels)

            # Get dates
            for j in range(len(features)):
                idx = i * config.BATCH_SIZE + j
                if idx < len(test_dataset):
                    all_dates.append(test_dataset.get_date(idx))

    print(f"✓ Generated predictions for {len(all_dates)} samples")

    # ====================================================================
    # STEP 4: RUN META-LEARNING BACKTEST (Layer 2)
    # ====================================================================
    print("\nSTEP 4: Running meta-learning backtest...")
    print("-" * 80)
    print("Meta-learner is now deciding which trades to take and how to size them!\n")

    backtest = MetaLearningBacktest(config, meta_learner)
    results = backtest.run_backtest(
        predictions=all_predictions,
        labels=all_labels,
        dates=all_dates,
        dataset=test_dataset
    )

    # Print results
    backtest.print_results(results)

    # ====================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # ====================================================================
    print("\nSTEP 5: Generating visualizations...")
    print("-" * 80)

    analytics = TradeAnalytics(results, save_dir=config.PLOT_DIR)
    analytics.plot_all()

    create_summary_dashboard(results, config.PLOT_DIR / 'meta_learning_dashboard.png')

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print("\n" + "="*80)
    print("META-LEARNING PIPELINE COMPLETE!")
    print("="*80)

    print(f"\n💰 FINAL RESULTS:")
    print(f"  Initial Capital:    ${results['initial_capital']:,.2f}")
    print(f"  Final Capital:      ${results['final_capital']:,.2f}")
    print(f"  Total Return:       {results['total_return_pct']:+.2f}%")
    print(f"  Win Rate:           {results['win_rate']:.1%}")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:       {results['max_drawdown']:.1%}")
    print(f"  Total Trades:       {len(results['trades'])}")

    print(f"\n📊 WHAT HAPPENED:")
    print(f"  1. Direction model predicted {len(all_dates)} samples")
    print(f"  2. Meta-learner selected {len(results['trades'])} best trades")
    print(f"  3. Selection rate: {len(results['trades'])/len(all_dates)/4:.1%}")
    print(f"  4. ML-optimized position sizing based on certainty")

    print(f"\n📁 Saved Files:")
    print(f"  Dashboard: {config.PLOT_DIR / 'meta_learning_dashboard.png'}")

    print("\n" + "="*80)
    print("The meta-learner learned which of the direction model's predictions")
    print("to trust and how much capital to risk. This is ML deciding on ML!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
