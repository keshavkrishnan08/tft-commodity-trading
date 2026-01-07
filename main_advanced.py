# main_advanced.py
"""
Advanced Trading System Pipeline
Integrates:
1. LSTM+Attention Direction Model (with label smoothing)
2. Probability Calibration (Platt Scaling)
3. ML Position Sizing (Kelly Criterion)
4. Exit Optimization (ATR-based SL/TP)
5. Advanced Backtest ($10K Capital)
6. Comprehensive Visualizations
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
from src.training.trainer import Trainer
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
    Overrides decision-making to use trained ML model instead of heuristics.
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

        # Track decisions for debugging
        decision_stats = {'total': 0, 'longs': 0, 'shorts': 0, 'rejected': 0}
        asset_decisions = {asset: {'total': 0, 'taken': 0} for asset in self.config.TARGET_ASSETS}

        for i in range(n_samples):
            date = dates[i]
            daily_pnl = 0.0

            # Print progress every 100 samples
            if i > 0 and i % 100 == 0:
                print(f"  Sample {i}/{n_samples}: Trades taken: {len(self.trades)}, "
                      f"Longs: {decision_stats['longs']}, Shorts: {decision_stats['shorts']}, "
                      f"Rejected: {decision_stats['rejected']}")

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
                    'recent_sharpe': 0.0,
                    'consecutive_wins': self.recent_wins[asset],
                    'consecutive_losses': self.recent_losses[asset],
                }

                # ===== META-LEARNER DECISION =====
                # Second ML model decides whether to take trade and how much!
                decision = self.meta_learner.predict_decision(
                    raw_prob,
                    market_state,
                    account_state
                )

                # Track decision stats
                decision_stats['total'] += 1
                asset_decisions[asset]['total'] += 1

                if decision['should_take']:
                    asset_decisions[asset]['taken'] += 1
                    # Determine direction
                    direction = 'long' if raw_prob > 0.5 else 'short'

                    # Track longs vs shorts
                    if direction == 'long':
                        decision_stats['longs'] += 1
                    else:
                        decision_stats['shorts'] += 1

                    # Position size from meta-learner
                    position_fraction = decision['position_size']
                    position_dollars = self.capital * position_fraction

                    # Exit levels: TIGHTER stop loss for better risk management
                    stop_loss_pct = max(0.012, volatility * 1.5)  # 1.5x ATR (was 2.0x)
                    take_profit_pct = stop_loss_pct * 2.0  # Maintain 2:1 R:R

                    # Create trade decision
                    trade_decision = {
                        'take_trade': True,
                        'direction': direction,
                        'raw_probability': raw_prob,
                        'calibrated_probability': raw_prob,
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
                else:
                    # Trade rejected by meta-learner
                    decision_stats['rejected'] += 1

            # Update capital
            self.capital += daily_pnl
            self.equity_curve.append(self.capital)
            self.daily_returns.append(daily_pnl / self.equity_curve[-2] if self.equity_curve[-2] > 0 else 0)

            # Update max equity and drawdown
            if self.capital > self.max_equity:
                self.max_equity = self.capital

            current_dd = (self.max_equity - self.capital) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, current_dd)

        # Print decision summary
        print(f"\n📊 DECISION SUMMARY:")
        print(f"  Total decisions: {decision_stats['total']}")
        print(f"  Longs taken: {decision_stats['longs']} ({decision_stats['longs']/decision_stats['total']*100:.1f}%)")
        print(f"  Shorts taken: {decision_stats['shorts']} ({decision_stats['shorts']/decision_stats['total']*100:.1f}%)")
        print(f"  Rejected: {decision_stats['rejected']} ({decision_stats['rejected']/decision_stats['total']*100:.1f}%)")
        print(f"\n  Per-Asset:")
        for asset in self.config.TARGET_ASSETS:
            total = asset_decisions[asset]['total']
            taken = asset_decisions[asset]['taken']
            print(f"    {asset}: {taken}/{total} ({taken/total*100:.1f}%)")

        return self._generate_results(dates)


def main():
    """Main advanced trading pipeline"""
    
    config = Config()
    set_seed(config.RANDOM_SEED)

    print("\n" + "="*80)
    print("ADVANCED TRADING SYSTEM PIPELINE")
    print("="*80)
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:,}")
    print(f"Max Position Size: {config.MAX_POSITION_SIZE:.0%}")
    print(f"Label Smoothing: {config.LABEL_SMOOTHING}")
    print(f"Buy Threshold: {config.BUY_THRESHOLD} | Sell Threshold: {config.SELL_THRESHOLD}")
    print(f"Device: {config.DEVICE}")
    print("="*80 + "\n")

    # ====================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ====================================================================
    print("STEP 1: Loading and preparing data...")
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

    # Prepare raw price data for backtesting (H/L checks)
    def extract_raw_prices(df):
        raw_prices = {}
        for asset in config.TARGET_ASSETS:
            # We want Open, High, Low, Close for H/L verification
            cols = [f'{asset}_{c}' for c in ['Open', 'High', 'Low', 'Close']]
            raw_prices[asset] = df[cols]
        return raw_prices

    # Create datasets
    train_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[:train_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[:train_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[:train_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=None,
        fit_scaler=True,
        raw_prices=extract_raw_prices(df_features.iloc[:train_end])
    )

    val_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[train_end:val_end],
        labels={asset: df_features[f'{asset}_Label'].iloc[train_end:val_end]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[train_end:val_end],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(df_features.iloc[train_end:val_end])
    )

    test_dataset = MultiAssetDataset(
        features=df_features[feature_cols].iloc[val_end:],
        labels={asset: df_features[f'{asset}_Label'].iloc[val_end:]
                for asset in config.TARGET_ASSETS},
        dates=df_features['Date'].iloc[val_end:],
        sequence_length=config.SEQUENCE_LENGTH,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        raw_prices=extract_raw_prices(df_features.iloc[val_end:])
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val:   {len(val_dataset)} sequences")
    print(f"Test:  {len(test_dataset)} sequences")

    # ====================================================================
    # STEP 2: TRAIN DIRECTION MODEL (with Label Smoothing)
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 2: Training Direction Model...")
    print("="*80)

    model = WeeklyPredictionModel(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit()

    # Load best model
    trainer.load_best_model()

    # ====================================================================
    # STEP 3: GENERATE PREDICTIONS ON TEST SET
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 3: Generating predictions...")
    print("="*80)

    model.eval()
    all_predictions = {asset: [] for asset in config.TARGET_ASSETS}
    all_labels = {asset: [] for asset in config.TARGET_ASSETS}
    all_dates = []

    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = features.to(config.DEVICE)
            preds, _ = model(features)
            
            for asset in config.TARGET_ASSETS:
                probs = torch.sigmoid(preds[asset]).cpu().numpy().flatten()
                true_labels = labels[asset].numpy().flatten()
                
                all_predictions[asset].extend(probs)
                all_labels[asset].extend(true_labels)
            
            # Get dates for this batch
            for j in range(len(features)):
                idx = i * config.BATCH_SIZE + j
                if idx < len(test_dataset):
                    all_dates.append(test_dataset.get_date(idx))

    # Check prediction distribution
    print("\nPrediction Distribution (Balanced Check):")
    for asset in config.TARGET_ASSETS:
        probs = np.array(all_predictions[asset])
        up_preds = (probs > config.BUY_THRESHOLD).sum()
        down_preds = (probs < config.SELL_THRESHOLD).sum()
        hold_preds = len(probs) - up_preds - down_preds
        
        print(f"  {asset}: UP={up_preds} ({up_preds/len(probs):.1%}), "
              f"DOWN={down_preds} ({down_preds/len(probs):.1%}), "
              f"HOLD={hold_preds} ({hold_preds/len(probs):.1%})")

    # ====================================================================
    # STEP 4: RUN ADVANCED BACKTEST (with Meta-Learner if available)
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 4: Running Advanced Backtest...")
    print("="*80)

    # Check if meta-learner exists
    meta_learner_path = config.MODEL_DIR / 'meta_learner.pkl'

    if meta_learner_path.exists():
        print("\n🧠 META-LEARNER FOUND! Using ML-based trade selection and sizing...")
        print("   The meta-learner will decide which trades to take and how much to position.")
        print("   This is ML on top of ML - second layer optimizes trading decisions!\n")

        # Load meta-learner
        meta_learner = MetaLearner(config)
        meta_learner.load(meta_learner_path)

        # Use meta-learning backtest
        backtest = MetaLearningBacktest(config, meta_learner)
        results = backtest.run_backtest(
            predictions=all_predictions,
            labels=all_labels,
            dates=all_dates,
            dataset=test_dataset
        )

        mode_used = "META-LEARNING"
    else:
        print("\n⚠️  Meta-learner not found. Using heuristic-based decisions.")
        print("   To use ML-based trade selection, run: python train_meta_learner.py\n")

        # Use standard advanced backtest
        backtest = AdvancedBacktest(config)
        results = backtest.run_backtest(
            predictions=all_predictions,
            labels=all_labels,
            dates=all_dates,
            dataset=test_dataset
        )

        mode_used = "HEURISTIC"

    # Print results
    backtest.print_results(results)

    # ====================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # ====================================================================
    print("\n" + "="*80)
    print("STEP 5: Generating visualizations...")
    print("="*80)

    analytics = TradeAnalytics(results, save_dir=config.PLOT_DIR)
    analytics.plot_all()

    # Create summary dashboard
    create_summary_dashboard(results, config.PLOT_DIR / 'trading_dashboard.png')

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

    print(f"\n💰 FINAL RESULTS ({mode_used} MODE):")
    print(f"  Initial Capital:    ${results['initial_capital']:,.2f}")
    print(f"  Final Capital:      ${results['final_capital']:,.2f}")
    print(f"  Total Return:       {results['total_return_pct']:+.2f}%")
    print(f"  Win Rate:           {results['win_rate']:.1%}")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:       {results['max_drawdown']:.1%}")
    print(f"  Total Trades:       {len(results['trades'])}")

    if mode_used == "META-LEARNING":
        print(f"\n🧠 META-LEARNING STATS:")
        print(f"  Direction model predicted {len(all_dates)} samples")
        print(f"  Meta-learner selected {len(results['trades'])} best trades")
        print(f"  Selection rate: {len(results['trades'])/len(all_dates)/4:.1%}")
        print(f"  → ML learned which predictions to trust!")

    print(f"\n📁 Saved Files:")
    print(f"  Model:              {config.MODEL_DIR / 'best_model.pth'}")
    if mode_used == "META-LEARNING":
        print(f"  Meta-learner:       {config.MODEL_DIR / 'meta_learner.pkl'}")
    print(f"  Dashboard:          {config.PLOT_DIR / 'trading_dashboard.png'}")
    print(f"  All visualizations: {config.PLOT_DIR / '*.png'}")

    if mode_used == "HEURISTIC":
        print("\n💡 TIP: Train the meta-learner for better results!")
        print("   Run: python train_meta_learner.py")
        print("   This will train an ML model to learn which trades to take")

    print("\n" + "="*80)
    print("Check the results/plots/ directory for all visualizations!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
