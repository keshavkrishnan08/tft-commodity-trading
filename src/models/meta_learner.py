# src/models/meta_learner.py
"""
Meta-Learning Model: Learns WHICH trades to take and HOW to size them.

This is a second ML layer that takes the direction model's predictions
and learns optimal trading decisions through supervised learning.

Architecture:
  Input:  Direction model probability + market features + account state
  Output: Take trade (classification), Position size (regression), Exit levels (regression)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MetaLearner:
    """
    Meta-learner: A second ML layer that decides HOW to trade based on direction predictions.
    
    The direction model tells us "up or down?" but that's not enough. We also need to know:
    - Should I actually take this trade? (Some predictions are too uncertain)
    - How much should I risk? (Position sizing based on confidence and market conditions)
    - When should I exit? (Stop loss and take profit levels)
    
    This meta-learner learns these decisions by observing what worked in the past.
    It's trained on historical trade outcomes from the training period (2013-2017),
    ensuring no data leakage into the test period (2018-2022).
    
    Uses Random Forest ensemble (200 trees) for stability and interpretability.
    """

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

        # Separate models for each decision
        self.trade_classifier = None      # Binary: take trade or not
        self.position_sizer = None        # Regression: position size (0.01-0.10)
        self.exit_optimizer = None        # Regression: optimal exit timing

        self.is_fitted = False

        # ONLINE LEARNING: Track performance by confidence level
        self.confidence_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.confidence_wins = {i: 0 for i in range(len(self.confidence_bins) - 1)}
        self.confidence_total = {i: 0 for i in range(len(self.confidence_bins) - 1)}
        self.confidence_pnl = {i: 0.0 for i in range(len(self.confidence_bins) - 1)}

    def _extract_features(self, direction_prob, market_state, account_state):
        """
        Extract features for meta-learning.

        Args:
            direction_prob: Probability from direction model (0-1)
            market_state: Dict with volatility, momentum, RSI, etc.
            account_state: Dict with capital, drawdown, recent performance

        Returns:
            Feature vector for meta-model
        """
        features = [
            # Direction model outputs
            direction_prob,
            abs(direction_prob - 0.5) * 2,  # Certainty (0=uncertain, 1=very certain)

            # Market conditions
            market_state.get('volatility', 0.02),
            market_state.get('momentum_20d', 0.0),
            market_state.get('momentum_60d', 0.0),
            market_state.get('rsi', 50.0) / 100.0,
            market_state.get('macd_hist', 0.0),
            market_state.get('volume_surge', 1.0),

            # Account state
            account_state.get('capital', 10000) / 10000,  # Normalized
            account_state.get('drawdown', 0.0),
            account_state.get('recent_win_rate', 0.5),
            account_state.get('recent_sharpe', 0.0),
            account_state.get('consecutive_wins', 0),
            account_state.get('consecutive_losses', 0),
        ]

        return np.array(features)

    def fit(self, training_data):
        """
        Train the meta-learning models.

        Args:
            training_data: DataFrame with columns:
                - direction_prob: Direction model's probability
                - market_*: Market condition features
                - account_*: Account state features
                - should_take: Label (1=trade was profitable, 0=not)
                - optimal_position: Retrospective optimal position size
                - actual_return: Actual return from the trade
        """
        print("\n" + "="*80)
        print("TRAINING META-LEARNER: Learning Optimal Trading Decisions")
        print("="*80)

        # Extract features
        X = []
        for idx, row in training_data.iterrows():
            market_state = {
                'volatility': row.get('volatility', 0.02),
                'momentum_20d': row.get('momentum_20d', 0.0),
                'momentum_60d': row.get('momentum_60d', 0.0),
                'rsi': row.get('rsi', 50.0),
                'macd_hist': row.get('macd_hist', 0.0),
                'volume_surge': row.get('volume_surge', 1.0),
            }
            account_state = {
                'capital': row.get('capital', 10000),
                'drawdown': row.get('drawdown', 0.0),
                'recent_win_rate': row.get('recent_win_rate', 0.5),
                'recent_sharpe': row.get('recent_sharpe', 0.0),
                'consecutive_wins': row.get('consecutive_wins', 0),
                'consecutive_losses': row.get('consecutive_losses', 0),
            }

            features = self._extract_features(
                row['direction_prob'],
                market_state,
                account_state
            )
            X.append(features)

        X = np.array(X, dtype=np.float32)

        # Clean data: replace NaN and inf with valid values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Labels - ensure proper types
        y_trade = training_data['should_take'].values.astype(np.int32)
        y_position = training_data['optimal_position'].values.astype(np.float32)
        y_return = training_data['actual_return'].values.astype(np.float32)

        # Clean labels
        y_position = np.nan_to_num(y_position, nan=0.01, posinf=0.10, neginf=0.01)
        y_return = np.nan_to_num(y_return, nan=0.0, posinf=1.0, neginf=-1.0)

        # 1. Train Trade Classifier
        print("\n1. Training Trade Selection Model (RandomForest)...")
        # Using RandomForest instead of XGBoost for macOS stability
        self.trade_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
            verbose=0
        )
        self.trade_classifier.fit(X_scaled, y_trade)
        print("   ✓ RandomForest trained successfully")

        train_acc = self.trade_classifier.score(X_scaled, y_trade)
        print(f"   Trade selection accuracy: {train_acc:.2%}")

        # 2. Train Position Sizer (only on trades that were taken)
        print("\n2. Training Position Sizing Model (RandomForest)...")
        mask_trades = y_trade == 1
        X_trades = X_scaled[mask_trades]
        y_pos_trades = y_position[mask_trades]

        self.position_sizer = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
            verbose=0
        )
        self.position_sizer.fit(X_trades, y_pos_trades)
        print("   ✓ RandomForest trained successfully")

        # 3. Train Exit Optimizer
        print("\n3. Training Exit Optimization Model (RandomForest)...")
        self.exit_optimizer = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
            verbose=0
        )
        self.exit_optimizer.fit(X_trades, y_return[mask_trades])
        print("   ✓ RandomForest trained successfully")

        self.is_fitted = True

        print("\n✓ Meta-learner training complete!")
        print(f"  Trained on {len(training_data)} samples")
        print(f"  Trades taken: {mask_trades.sum()} ({mask_trades.mean():.1%})")
        print("="*80 + "\n")

    def predict_decision(self, direction_prob, market_state, account_state):
        """
        Predict optimal trading decision using trained meta-model.

        Args:
            direction_prob: Direction model's probability
            market_state: Current market conditions
            account_state: Current account state

        Returns:
            Dict with:
                - should_take: Boolean (True/False)
                - take_probability: Probability of taking trade
                - position_size: Recommended position size (0.01-0.10)
                - confidence: Meta-model's confidence in decision
        """
        if not self.is_fitted:
            # Fallback to simple threshold if not trained
            return {
                'should_take': abs(direction_prob - 0.5) > 0.001,  # Take almost everything
                'take_probability': abs(direction_prob - 0.5) * 2,
                'position_size': 0.05,
                'confidence': 0.5
            }

        # Extract features
        features = self._extract_features(direction_prob, market_state, account_state)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # 1. Should take trade?
        take_prob = self.trade_classifier.predict_proba(features_scaled)[0, 1]

        # EXTREMELY AGGRESSIVE: Start at 0.30, lower after winning streaks
        base_threshold = 0.30

        # WINNING STREAK BONUS: Get more aggressive after wins
        consecutive_wins = account_state.get('consecutive_wins', 0)
        consecutive_losses = account_state.get('consecutive_losses', 0)

        # After 3+ wins, lower threshold by 0.05 per win (more trades)
        if consecutive_wins >= 3:
            base_threshold -= min(0.15, consecutive_wins * 0.05)

        # After 3+ losses, still take trades but slightly higher threshold
        if consecutive_losses >= 3:
            base_threshold += min(0.10, consecutive_losses * 0.03)

        # Ensure threshold stays reasonable
        base_threshold = np.clip(base_threshold, 0.15, 0.45)

        should_take = take_prob > base_threshold

        # 2. Position size (if taking trade)
        if should_take:
            position_size_raw = self.position_sizer.predict(features_scaled)[0]

            # AGGRESSIVE: Scale up position sizes with confidence boost
            # Calculate certainty from direction probability
            dir_certainty = abs(direction_prob - 0.5) * 2

            # Confidence boost: multiply by up to 2.0x for high certainty trades
            confidence_multiplier = 1.2 + (dir_certainty * 0.8)  # 1.2x to 2.0x

            # ONLINE LEARNING: Apply adaptive adjustment based on historical performance
            adaptive_adjustment = self.get_confidence_adjustment(dir_certainty)
            confidence_multiplier *= adaptive_adjustment

            # WINNING STREAK BONUS: Increase position size after consecutive wins
            if consecutive_wins >= 3:
                # Add up to 30% more for winning streaks
                streak_bonus = 1.0 + min(0.30, consecutive_wins * 0.08)
                confidence_multiplier *= streak_bonus

            # LOSING STREAK PENALTY: Reduce position size after losses (but don't stop trading)
            if consecutive_losses >= 3:
                # Reduce by up to 30% during losing streaks
                streak_penalty = 1.0 - min(0.30, consecutive_losses * 0.08)
                confidence_multiplier *= max(0.7, streak_penalty)  # Never go below 70%

            # Apply boost and clip to valid range
            position_size = position_size_raw * confidence_multiplier

            # IMPORTANT: Ensure minimum position is higher
            position_size = np.clip(position_size, 0.03, self.config.MAX_POSITION_SIZE)
        else:
            position_size = 0.0

        return {
            'should_take': should_take,
            'take_probability': take_prob,
            'position_size': position_size,
            'confidence': abs(take_prob - 0.5) * 2  # 0=uncertain, 1=very certain
        }

    def update_confidence_feedback(self, confidence, won, pnl):
        """
        ONLINE LEARNING: Update performance tracking by confidence level.

        Args:
            confidence: Confidence level (0-1)
            won: Whether trade was profitable
            pnl: Actual P&L from trade
        """
        # Find which bin this confidence falls into
        for i in range(len(self.confidence_bins) - 1):
            if self.confidence_bins[i] <= confidence < self.confidence_bins[i + 1]:
                self.confidence_total[i] += 1
                if won:
                    self.confidence_wins[i] += 1
                self.confidence_pnl[i] += pnl
                break

    def get_confidence_adjustment(self, confidence):
        """
        Get position size adjustment based on learned confidence performance.

        Returns multiplier: 0.5x to 1.5x based on how well this confidence level performs.
        """
        # Find which bin
        bin_idx = None
        for i in range(len(self.confidence_bins) - 1):
            if self.confidence_bins[i] <= confidence < self.confidence_bins[i + 1]:
                bin_idx = i
                break

        if bin_idx is None or self.confidence_total[bin_idx] < 5:
            # Not enough data yet, return neutral
            return 1.0

        # Calculate win rate for this confidence level
        win_rate = self.confidence_wins[bin_idx] / self.confidence_total[bin_idx]

        # Adjust position sizing based on empirical performance
        # win_rate > 0.6: increase by up to 50%
        # win_rate < 0.4: decrease by up to 50%
        if win_rate > 0.5:
            adjustment = 1.0 + (win_rate - 0.5) * 1.0  # Up to 1.5x for 100% win rate
        else:
            adjustment = 0.5 + (win_rate * 1.0)  # Down to 0.5x for 0% win rate

        return np.clip(adjustment, 0.5, 1.5)

    def get_feature_importance(self):
        """Get feature importance from the trained models."""
        if not self.is_fitted:
            return None

        feature_names = [
            'direction_prob', 'certainty', 'volatility', 'momentum_20d',
            'momentum_60d', 'rsi', 'macd_hist', 'volume_surge',
            'capital_norm', 'drawdown', 'recent_win_rate', 'recent_sharpe',
            'consecutive_wins', 'consecutive_losses'
        ]

        importance = {
            'trade_selection': dict(zip(feature_names,
                                       self.trade_classifier.feature_importances_)),
            'position_sizing': dict(zip(feature_names,
                                       self.position_sizer.feature_importances_))
        }

        return importance

    def save(self, path):
        """Save trained meta-learner to disk."""
        save_dict = {
            'trade_classifier': self.trade_classifier,
            'position_sizer': self.position_sizer,
            'exit_optimizer': self.exit_optimizer,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"✓ Meta-learner saved to {path}")

    def load(self, path):
        """Load trained meta-learner from disk."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.trade_classifier = save_dict['trade_classifier']
        self.position_sizer = save_dict['position_sizer']
        self.exit_optimizer = save_dict['exit_optimizer']
        self.scaler = save_dict['scaler']
        self.is_fitted = save_dict['is_fitted']

        print(f"✓ Meta-learner loaded from {path}")


def generate_meta_training_data(direction_model, dataset, config):
    """
    Generate training data for meta-learner by simulating trades on validation set.

    For each prediction:
    1. Get direction model's probability
    2. Extract market features
    3. Simulate different position sizes
    4. Record which decisions were profitable

    Returns:
        DataFrame with training data for meta-learner
    """
    import torch
    from torch.utils.data import DataLoader

    print("\n" + "="*80)
    print("GENERATING META-LEARNING TRAINING DATA")
    print("="*80)

    direction_model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    training_samples = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(config.DEVICE)
            predictions, _ = direction_model(features)

            # Process each sample in batch
            for i in range(len(features)):
                sample_idx = batch_idx * 32 + i
                if sample_idx >= len(dataset):
                    break

                for asset in config.TARGET_ASSETS:
                    # Get prediction and label
                    logit = predictions[asset][i].item()
                    prob = 1 / (1 + np.exp(-logit))  # Sigmoid
                    actual = int(labels[asset][i].item())

                    # IMPROVED: Extract ACTUAL market state from dataset features
                    try:
                        # Get raw (unscaled) feature values for current sample
                        feature_idx = sample_idx
                        if feature_idx < len(dataset.features):
                            raw_features = dataset.features.iloc[feature_idx]

                            # Extract actual technical indicators
                            rsi = raw_features.get(f'{asset}_RSI', 50.0)
                            macd_hist = raw_features.get(f'{asset}_MACD_Hist', 0.0)

                            # Calculate momentum from returns
                            momentum_20d = raw_features.get(f'{asset}_Return', 0.0) * 20
                            momentum_60d = raw_features.get(f'{asset}_Return', 0.0) * 60

                            # Calculate volatility from Vol_20d feature
                            volatility = raw_features.get(f'{asset}_Vol_20d', 0.02)
                        else:
                            # Defaults if out of range
                            volatility = 0.02
                            rsi = 50.0
                            macd_hist = 0.0
                            momentum_20d = 0.0
                            momentum_60d = 0.0
                    except Exception as e:
                        # Fallback to default values
                        volatility = 0.02
                        rsi = 50.0
                        macd_hist = 0.0
                        momentum_20d = 0.0
                        momentum_60d = 0.0

                    # Get market state (from raw features)
                    market_state = {
                        'volatility': volatility,
                        'momentum_20d': momentum_20d,
                        'momentum_60d': momentum_60d,
                        'rsi': rsi,
                        'macd_hist': macd_hist,
                        'volume_surge': 1.0,
                    }

                    # Simulate account state
                    account_state = {
                        'capital': 10000,
                        'drawdown': 0.0,
                        'recent_win_rate': 0.5,
                        'recent_sharpe': 0.0,
                        'consecutive_wins': 0,
                        'consecutive_losses': 0,
                    }

                    # Determine if trade would have been profitable
                    direction = 1 if prob > 0.5 else -1
                    trade_return = direction * (actual * 2 - 1)  # +1 if correct, -1 if wrong

                    # Calculate optimal position size retrospectively
                    # IMPROVED: Use Kelly-inspired sizing with volatility adjustment
                    certainty = abs(prob - 0.5) * 2

                    # Win probability (calibrated from model confidence)
                    win_prob = 0.5 + (certainty * 0.2)  # Maps 0-1 certainty to 0.5-0.7 win prob

                    # Kelly fraction: (win_prob * 2 - 1) with volatility adjustment
                    # Assuming 2:1 risk/reward ratio (50% loss, 100% gain)
                    kelly_fraction = (win_prob * 2 - 1) if win_prob > 0.5 else 0.0

                    # AGGRESSIVE: Scale to 3-10% range based on certainty and outcome
                    # Even losing trades get reasonable size if high certainty (for learning)
                    if trade_return > 0:
                        # Winning trade: full Kelly up to 10% (more aggressive multiplier)
                        optimal_position = np.clip(kelly_fraction * 0.8, 0.03, 0.10)
                    else:
                        # Losing trade: 3-6% range (higher than before)
                        # This prevents meta-learner from being too conservative
                        optimal_position = np.clip(certainty * 0.08, 0.03, 0.06)

                    # Should take this trade?
                    # VERY AGGRESSIVE: Take if any directional bias exists
                    # Include both longs AND shorts equally
                    should_take = (certainty > 0.05) or (prob > 0.52 or prob < 0.48)

                    training_samples.append({
                        'asset': asset,
                        'direction_prob': prob,
                        'volatility': market_state['volatility'],
                        'momentum_20d': market_state['momentum_20d'],
                        'momentum_60d': market_state['momentum_60d'],
                        'rsi': market_state['rsi'],
                        'macd_hist': market_state['macd_hist'],
                        'volume_surge': market_state['volume_surge'],
                        'capital': account_state['capital'],
                        'drawdown': account_state['drawdown'],
                        'recent_win_rate': account_state['recent_win_rate'],
                        'recent_sharpe': account_state['recent_sharpe'],
                        'consecutive_wins': account_state['consecutive_wins'],
                        'consecutive_losses': account_state['consecutive_losses'],
                        'should_take': int(should_take),
                        'optimal_position': optimal_position,
                        'actual_return': trade_return,
                    })

    df = pd.DataFrame(training_samples)

    print(f"\n✓ Generated {len(df)} training samples")
    print(f"  Positive samples (should take): {df['should_take'].sum()} ({df['should_take'].mean():.1%})")
    print(f"  Mean optimal position: {df['optimal_position'].mean():.3f}")
    print("="*80 + "\n")

    return df
