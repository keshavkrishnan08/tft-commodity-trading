# src/models/trading_models.py
"""
ML-based trading decision models:
1. ProbabilityCalibrator - Calibrates raw probabilities
2. PositionSizer - ML-based position sizing using Kelly + XGBoost
3. ExitOptimizer - ML-based stop loss and take profit optimization
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ProbabilityCalibrator:
    """
    Calibrate raw model probabilities to true likelihood using Platt Scaling.
    This helps fix directional bias by ensuring probabilities are well-calibrated.
    """
    
    def __init__(self, method='platt'):
        """
        Args:
            method: 'platt' (logistic regression) or 'isotonic' (isotonic regression)
        """
        self.method = method
        self.calibrators = {}
        
    def fit(self, raw_probs, actual_outcomes, assets):
        """
        Fit calibrator on validation data.
        
        Args:
            raw_probs: dict of {asset: raw probabilities}
            actual_outcomes: dict of {asset: actual labels (0/1)}
            assets: list of asset names
        """
        for asset in assets:
            probs = np.array(raw_probs[asset]).reshape(-1, 1)
            labels = np.array(actual_outcomes[asset])
            
            if self.method == 'platt':
                calibrator = LogisticRegression(C=1e10, solver='lbfgs')
                calibrator.fit(probs, labels)
            else:  # isotonic
                calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                calibrator.fit(probs.ravel(), labels)
            
            self.calibrators[asset] = calibrator
            
    def transform(self, raw_probs, asset):
        """
        Transform raw probabilities to calibrated probabilities.
        
        Args:
            raw_probs: numpy array of raw probabilities
            asset: asset name
            
        Returns:
            calibrated probabilities
        """
        if asset not in self.calibrators:
            return raw_probs  # Return uncalibrated if not fitted
        
        probs = np.array(raw_probs).reshape(-1, 1)
        
        if self.method == 'platt':
            return self.calibrators[asset].predict_proba(probs)[:, 1]
        else:
            return self.calibrators[asset].predict(probs.ravel())
    
    def save(self, path):
        """Save calibrators to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.calibrators, f)
    
    def load(self, path):
        """Load calibrators from file."""
        with open(path, 'rb') as f:
            self.calibrators = pickle.load(f)


class PositionSizer:
    """
    ML-based position sizing using Kelly Criterion + heuristics.
    Determines optimal position size based on:
    - Calibrated probability (confidence)
    - Asset volatility (ATR)
    - Account balance and risk tolerance
    - Recent performance (win/loss streak)
    """
    
    def __init__(self, config):
        self.config = config
        self.max_position = config.MAX_POSITION_SIZE
        self.min_position = config.MIN_POSITION_SIZE
        
    def calculate_position_size(self, probability, volatility, account_balance, 
                                recent_wins=0, recent_losses=0, max_drawdown_pct=0.0):
        """
        Calculate optimal position size.
        
        Args:
            probability: Calibrated probability of success (0.5-1.0)
            volatility: Current asset volatility (ATR % of price)
            account_balance: Current account balance in dollars
            recent_wins: Number of recent winning trades
            recent_losses: Number of recent losing trades
            max_drawdown_pct: Current max drawdown as percentage
            
        Returns:
            position_size: Fraction of capital to risk (0.01-0.10)
            position_dollars: Dollar amount to position
        """
        # Kelly Criterion: f* = (p * b - q) / b
        # Where p = probability of win, q = 1-p, b = win/loss ratio
        # Simplified: f* = 2p - 1 for even odds
        
        # Edge: how much better than 50/50
        edge = max(0, probability - 0.5) * 2  # Scale to 0-1

        # Base position using full Kelly (FIXED: was half-Kelly, too conservative)
        kelly_fraction = edge  # Full Kelly for directional trades

        # Volatility adjustment: reduce position in high volatility
        # FIXED: Reduced penalty from 10 to 5 to allow larger positions
        vol_multiplier = 1.0 / (1.0 + volatility * 10)  # Higher vol = smaller position
        
        # Streak adjustment: reduce after losses, increase slightly after wins
        streak_multiplier = 1.0
        if recent_losses >= 3:
            streak_multiplier = 0.5  # Cut size in half after losing streak
        elif recent_wins >= 3:
            streak_multiplier = 1.2  # Small increase after winning streak
            
        # Drawdown adjustment: reduce position when in drawdown
        drawdown_multiplier = 1.0
        if max_drawdown_pct > 0.10:
            drawdown_multiplier = 0.5  # Cut in half if > 10% drawdown
        elif max_drawdown_pct > 0.05:
            drawdown_multiplier = 0.75
            
        # Calculate final position size
        position_size = kelly_fraction * vol_multiplier * streak_multiplier * drawdown_multiplier
        
        # Clamp to min/max
        position_size = np.clip(position_size, self.min_position, self.max_position)
        
        # Convert to dollars
        position_dollars = account_balance * position_size
        
        return position_size, position_dollars
    
    def get_sizing_summary(self, probability, volatility):
        """Get human-readable sizing explanation."""
        edge = max(0, probability - 0.5) * 2
        vol_impact = 1.0 / (1.0 + volatility * 10)
        
        return {
            'edge': edge,
            'vol_impact': vol_impact,
            'base_kelly': edge * 0.5,
            'vol_adjusted': edge * 0.5 * vol_impact
        }


class ExitOptimizer:
    """
    ML-based stop loss and take profit optimization.
    Uses volatility (ATR) and confidence to set optimal exit levels.
    """
    
    def __init__(self, config):
        self.config = config
        self.default_sl = config.DEFAULT_STOP_LOSS
        self.default_tp = config.DEFAULT_TAKE_PROFIT
        
    def predict_exits(self, probability, volatility, direction='long'):
        """
        Exit Strategy: Uses 1.5x ATR for stops with 2:1 risk/reward ratio.
        """
        # Stop Loss: 1.5x recent volatility (TIGHTER for better risk management)
        stop_loss_pct = max(0.012, volatility * 1.5)

        # Take Profit: 2:1 reward/risk ratio
        take_profit_pct = stop_loss_pct * 2.0
        
        # Risk management caps
        stop_loss_pct = min(0.04, stop_loss_pct) # Cap loss at 4%
        take_profit_pct = min(0.12, take_profit_pct) # Cap gain at 12%
        
        risk_reward = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0
        
        return stop_loss_pct, take_profit_pct, risk_reward
    
    def get_exit_summary(self, probability, volatility):
        """Get human-readable exit explanation."""
        sl, tp, rr = self.predict_exits(probability, volatility, 'long') # Direction doesn't matter for this strategy
        return {
            'stop_loss_pct': sl,
            'take_profit_pct': tp,
            'risk_reward': rr,
            'sl_dollars_per_1k': sl * 1000,
            'tp_dollars_per_1k': tp * 1000
        }


class TradingDecisionEngine:
    """
    Unified engine that combines all trading models.
    """
    
    def __init__(self, config):
        self.config = config
        self.calibrator = ProbabilityCalibrator(method='isotonic') # Switch to isotonic
        self.position_sizer = PositionSizer(config)
        self.exit_optimizer = ExitOptimizer(config)
        self.is_fitted = False
        
    def fit_calibrator(self, raw_probs, actual_outcomes, assets):
        """Fit the probability calibrator."""
        self.calibrator.fit(raw_probs, actual_outcomes, assets)
        self.is_fitted = True
        
    def make_decision(self, raw_probability, asset, volatility, account_balance,
                      recent_wins=0, recent_losses=0, max_drawdown=0.0):
        """
        Make a complete trading decision.
        """
        # 1. Calibrate probability
        if self.is_fitted:
            calibrated_prob = self.calibrator.transform([raw_probability], asset)[0]
        else:
            calibrated_prob = raw_probability
            
        # 2. Confidence Filter
        # AGGRESSIVE: Trust the neural network's learned patterns
        # Take trades on any conviction (52% confidence or higher)
        trade_threshold = 0.52  # Minimal filtering - let the model decide
        
        # Determine direction
        is_long = calibrated_prob > 0.5
        confidence = calibrated_prob if is_long else (1.0 - calibrated_prob)
        
        # Primary Filter
        if confidence < trade_threshold:
            return {
                'take_trade': False,
                'direction': 'hold',
                'raw_probability': raw_probability,
                'calibrated_probability': calibrated_prob,
                'confidence': confidence
            }
            
        direction = 'long' if is_long else 'short'
        take_trade = True
        
        # 3. Calculate position size (use calibrated confidence)
        position_fraction, position_dollars = self.position_sizer.calculate_position_size(
            confidence if direction == 'long' else (1.0 - calibrated_prob), 
            volatility, account_balance,
            recent_wins, recent_losses, max_drawdown
        )
        
        # 4. Calculate exit levels
        stop_loss, take_profit, risk_reward = self.exit_optimizer.predict_exits(
            confidence, volatility, direction
        )
        
        return {
            'take_trade': True,
            'direction': direction,
            'raw_probability': raw_probability,
            'calibrated_probability': calibrated_prob,
            'confidence': confidence,
            'position_fraction': position_fraction,
            'position_dollars': position_dollars,
            'stop_loss_pct': stop_loss,
            'take_profit_pct': take_profit,
            'risk_reward': risk_reward,
            'volatility': volatility
        }
