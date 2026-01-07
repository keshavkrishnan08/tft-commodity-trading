# src/evaluation/advanced_backtest.py
"""
Advanced Backtest Engine with:
- $10K initial capital simulation
- ML-based position sizing, stop loss, take profit
- Comprehensive trade analytics and statistics
- Realistic P&L tracking
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.models.trading_models import TradingDecisionEngine


class AdvancedBacktest:
    """
    Realistic backtesting engine that simulates actual trading conditions.
    
    This isn't just "buy when prediction > 0.5" - it includes:
    - Transaction costs (0.6% round-trip: 0.3% slippage + 0.3% commission)
    - Position sizing via Kelly Criterion with volatility adjustments
    - ATR-adaptive stop losses and take profits
    - Account state tracking (drawdown, win streaks, etc.)
    - High-fidelity simulation using intraday data when available
    
    The goal is to simulate what would actually happen if you deployed this strategy
    with real money, not just theoretical returns.
    """
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.initial_capital = self.config.INITIAL_CAPITAL
        self.reset()
        
    def reset(self):
        """Reset all state for a new backtest."""
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.daily_returns = []
        self.positions = {asset: None for asset in self.config.TARGET_ASSETS}
        self.decision_engine = TradingDecisionEngine(self.config)
        
        # Performance tracking
        self.recent_wins = {asset: 0 for asset in self.config.TARGET_ASSETS}
        self.recent_losses = {asset: 0 for asset in self.config.TARGET_ASSETS}
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.hf_simulator = None
        
    def run_backtest(self, predictions, labels, dates, dataset, price_data=None, intraday_data=None, calibrate=True):
        """
        Run full backtest with ML-based trading decisions.
        
        Args:
            predictions: dict of {asset: probabilities}
            labels: dict of {asset: actual labels}
            dates: list of dates
            dataset: the dataset object (for volatility calculation)
            price_data: optional price data for realistic P&L
            intraday_data: optional dict of {asset: 1h_bars_df} for high-fidelity
            
        Returns:
            results: comprehensive backtest results
        """
        self.reset()
        self.intraday_data = intraday_data
        if intraday_data:
            from src.evaluation.high_fidelity_simulator import HighFidelitySimulator
            self.hf_simulator = HighFidelitySimulator(self.config, intraday_data)
        else:
            self.hf_simulator = None

        n_samples = len(dates)
        
        if calibrate:
            # First, calibrate the probability model on first 20% of data
            calibration_end = int(n_samples * 0.2)
            raw_probs_cal = {asset: predictions[asset][:calibration_end] 
                            for asset in self.config.TARGET_ASSETS}
            labels_cal = {asset: labels[asset][:calibration_end] 
                         for asset in self.config.TARGET_ASSETS}
            self.decision_engine.fit_calibrator(raw_probs_cal, labels_cal, 
                                                self.config.TARGET_ASSETS)
        else:
            calibration_end = 0
        
        # Run backtest from calibration end onwards
        for i in range(calibration_end, n_samples):
            date = dates[i]
            daily_pnl = 0.0
            
            for asset in self.config.TARGET_ASSETS:
                raw_prob = predictions[asset][i]
                actual = labels[asset][i]
                
                # Calculate volatility from recent price data (TRUE ATR)
                volatility = self._estimate_volatility(dataset, asset, i)
                
                # Get current drawdown
                current_drawdown = (self.max_equity - self.capital) / self.max_equity
                
                # Make trading decision
                decision = self.decision_engine.make_decision(
                    raw_probability=raw_prob,
                    asset=asset,
                    volatility=volatility,
                    account_balance=self.capital,
                    recent_wins=self.recent_wins[asset],
                    recent_losses=self.recent_losses[asset],
                    max_drawdown=current_drawdown
                )
                
                if decision['take_trade']:
                    # Simulate trade outcome
                    trade_result = self._simulate_trade(
                        decision, actual, asset, date, volatility, i, dataset
                    )
                    
                    if trade_result:
                        daily_pnl += trade_result['pnl']
                        
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
        
        return self._generate_results(dates[calibration_end:])
    
    def _estimate_volatility(self, dataset, asset, current_idx, window=20):
        """
        Calculate TRUE volatility from price data using Average True Range (ATR).

        FIXED: Was incorrectly using std(predictions). Now uses actual price volatility.
        """
        # Need at least window+1 samples for TR calculation
        if current_idx < window + 1:
            return 0.02  # Default 2% volatility for early samples

        # Get recent price data from dataset
        # raw_prices format: [Open, High, Low, Close] per row
        start_idx = max(0, current_idx - window)

        true_ranges = []
        for i in range(start_idx + 1, current_idx + 1):
            prices = dataset.get_raw_prices(i, asset)
            prev_prices = dataset.get_raw_prices(i - 1, asset)

            if prices is None or prev_prices is None:
                continue

            # Extract OHLC (assuming order: Open, High, Low, Close)
            high = float(prices[1])
            low = float(prices[2])
            close = float(prices[3])
            prev_close = float(prev_prices[3])

            # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)

        if len(true_ranges) < 5:
            return 0.02  # Fallback

        # ATR = average true range as percentage of current price
        avg_true_range = np.mean(true_ranges)
        # FIXED: Use correct index for current price (no sequence_length subtraction)
        current_price = float(dataset.get_raw_prices(current_idx, asset)[3])
        volatility = avg_true_range / current_price

        # Sanity check: volatility should be between 0.5% and 15%
        volatility = np.clip(volatility, 0.005, 0.15)

        return volatility
    
    def _simulate_trade(self, decision, actual, asset, date, volatility, current_idx, dataset):
        """
        Simulate a single trade with stop loss, take profit, and friction.
        Enhanced with Daily High/Low verification for realism.
        """
        if self.hf_simulator:
            # 1h Intraday Logic (if data available)
            result = self.hf_simulator.simulate_trade(
                asset=asset,
                entry_date=date,
                direction=decision['direction'],
                position_dollars=decision['position_dollars'],
                stop_loss_pct=decision['stop_loss_pct'],
                take_profit_pct=decision['take_profit_pct']
            )
            if result:
                result.update({
                    'asset': asset, 'direction': decision['direction'],
                    'position_dollars': decision['position_dollars'],
                    'position_fraction': decision['position_fraction'],
                    'stop_loss_pct': decision['stop_loss_pct'],
                    'take_profit_pct': decision['take_profit_pct'],
                    'risk_reward': decision['risk_reward'],
                    'raw_probability': decision['raw_probability'],
                    'calibrated_probability': decision['calibrated_probability'],
                    'actual_outcome': actual,
                    'correct': result['net_pnl'] > 0,
                    'won': result['net_pnl'] > 0,
                    'pnl': result['net_pnl'],
                    'volatility': volatility
                })
                return result

        # --- Daily H/L Fidelity Upgrade ---
        direction = decision['direction']
        position_dollars = decision['position_dollars']
        stop_loss_pct = decision['stop_loss_pct']
        take_profit_pct = decision['take_profit_pct']
        
        # Friction
        fee_pct = self.config.SLIPPAGE_PCT + self.config.COMMISSION_PCT
        total_friction = position_dollars * fee_pct * 2

        # Get entry/horizon prices
        horizon = self.config.PREDICTION_HORIZONS[self.config.PREDICTION_HORIZON]
        entry_raw = dataset.get_raw_prices(current_idx, asset)
        
        if entry_raw is None:
            return self._fallback_sim(decision, actual, asset, date, volatility, total_friction)

        entry_price = float(entry_raw[0]) # Open
        sl_price = entry_price * (1.0 - stop_loss_pct) if direction == 'long' else entry_price * (1.0 + stop_loss_pct)
        tp_price = entry_price * (1.0 + take_profit_pct) if direction == 'long' else entry_price * (1.0 - take_profit_pct)

        exit_type = 'horizon_expiry'
        exit_price = None
        
        # Step through 5-day horizon
        for d in range(horizon):
            price_row = dataset.get_raw_prices(current_idx + d, asset)
            if price_row is None: break
                
            p_high, p_low, p_close = float(price_row[1]), float(price_row[2]), float(price_row[3])
            
            # Check Stop Loss (Worst Case First)
            # FIXED: Check for GAP fills (Open vs SL)
            if direction == 'long':
                # Check for gap down
                if float(price_row[0]) <= sl_price: # Gap down below SL
                    exit_price = float(price_row[0]) # Exit at Open
                    exit_type = 'stop_loss_gap'
                    break
                elif p_low <= sl_price: # Intraday hit
                    exit_price = sl_price
                    exit_type = 'stop_loss'
                    break
                # Check TP
                if float(price_row[0]) >= tp_price: # Gap up above TP
                     exit_price = float(price_row[0])
                     exit_type = 'take_profit_gap'
                     break
                elif p_high >= tp_price:
                     exit_price = tp_price
                     exit_type = 'take_profit'
                     break
            
            else: # Short
                # Check for gap up
                if float(price_row[0]) >= sl_price: # Gap up above SL
                    exit_price = float(price_row[0])
                    exit_type = 'stop_loss_gap'
                    break
                elif p_high >= sl_price:
                    exit_price = sl_price
                    exit_type = 'stop_loss'
                    break
                # Check TP
                if float(price_row[0]) <= tp_price: # Gap down below TP
                    exit_price = float(price_row[0])
                    exit_type = 'take_profit_gap'
                    break
                elif p_low <= tp_price:
                    exit_price = tp_price
                    exit_type = 'take_profit'
                    break
            
            # DYNAMIC Trailing Stop: Tightens over time and as profit increases
            if self.config.ENABLE_TRAILING_STOP:
                # Calculate days elapsed (0-4 for 5-day horizon)
                days_elapsed = d
                time_factor = days_elapsed / horizon  # 0.0 to 1.0

                # Calculate current unrealized P&L
                current_price = p_close  # Close price
                if direction == 'long':
                    unrealized_pnl_pct = (current_price - entry_price) / entry_price
                    profit_threshold = take_profit_pct * 0.3  # Start trailing at 30% of TP

                    if unrealized_pnl_pct > profit_threshold:
                        # Trail tighter as time passes and profit increases
                        # Base trailing distance: 50% of SL, reduced by time and profit
                        trail_distance_pct = (stop_loss_pct * 0.5) * (1.0 - time_factor * 0.5)

                        # New stop: lock in some profit, getting tighter over time
                        new_sl_price = current_price - (current_price * trail_distance_pct)
                        sl_price = max(sl_price, new_sl_price)

                else:  # short
                    unrealized_pnl_pct = (entry_price - current_price) / entry_price
                    profit_threshold = take_profit_pct * 0.3

                    if unrealized_pnl_pct > profit_threshold:
                        trail_distance_pct = (stop_loss_pct * 0.5) * (1.0 - time_factor * 0.5)
                        new_sl_price = current_price + (current_price * trail_distance_pct)
                        sl_price = min(sl_price, new_sl_price)

        if exit_price is None:
            final_row = dataset.get_raw_prices(current_idx + horizon - 1, asset)
            exit_price = float(final_row[3]) if final_row is not None else entry_price
            exit_type = 'horizon_expiry'

        gross_pnl_pct = (exit_price - entry_price)/entry_price if direction == 'long' else (entry_price - exit_price)/entry_price
        net_pnl = (position_dollars * gross_pnl_pct) - total_friction
        
        return {
            'date': date, 'asset': asset, 'direction': direction,
            'position_dollars': position_dollars, 'position_fraction': decision['position_fraction'],
            'stop_loss_pct': stop_loss_pct, 'take_profit_pct': take_profit_pct,
            'risk_reward': decision['risk_reward'], 'raw_probability': decision['raw_probability'],
            'calibrated_probability': decision['calibrated_probability'],
            'actual_outcome': actual, 'correct': net_pnl > 0, 'won': net_pnl > 0,
            'gross_pnl': position_dollars * gross_pnl_pct, 'friction': total_friction,
            'pnl': net_pnl, 'exit_type': exit_type, 'volatility': volatility
        }

    def _fallback_sim(self, decision, actual, asset, date, volatility, total_friction):
        """Standard simulation fallback."""
        direction, pos_dollars = decision['direction'], decision['position_dollars']
        sl, tp = decision['stop_loss_pct'], decision['take_profit_pct']
        won = (actual == 1 if direction == 'long' else actual == 0)
        
        if won:
            gross_pnl = pos_dollars * tp * np.random.uniform(0.7, 1.2)
            exit_type = 'take_profit'
        else:
            gross_pnl = -pos_dollars * sl * np.random.uniform(0.9, 1.1)
            exit_type = 'stop_loss'
            
        return {
            'date': date, 'asset': asset, 'direction': direction,
            'position_dollars': pos_dollars, 'position_fraction': decision['position_fraction'],
            'stop_loss_pct': sl, 'take_profit_pct': tp, 
            'pnl': gross_pnl - total_friction,
            'friction': total_friction,
            'gross_pnl': gross_pnl,
            'won': (gross_pnl - total_friction) > 0, 
            'exit_type': exit_type, 'volatility': volatility,
            'raw_probability': decision['raw_probability'], 
            'calibrated_probability': decision['calibrated_probability'],
            'risk_reward': decision.get('risk_reward', tp/sl if sl > 0 else 0),
            'actual_outcome': actual
        }
        # v2 Implementation: Trailing stop logic (simplified for daily data)
        if correct:
            # Win: Assume we hit take profit
            # But with trailing stops, we might catch a "runner"
            is_runner = False
            if self.config.ENABLE_TRAILING_STOP and np.random.random() < 0.3: # 30% chance to catch a runner
                is_runner = True
                # A runner can go 2x - 5x the target TP
                win_multiplier = np.random.uniform(1.5, 4.0)
                exit_type = 'trailing_stop_exit'
            else:
                win_multiplier = np.random.uniform(0.7, 1.2) # 70-120% of target TP
                exit_type = 'take_profit'
            
            gross_pnl = position_dollars * take_profit * win_multiplier
        else:
            # Loss: Assume we hit stop loss
            # But sometimes we get stopped out even tighter or with slippage
            loss_multiplier = np.random.uniform(0.9, 1.1) # 90-110% of target SL due to slippage
            gross_pnl = -position_dollars * stop_loss * loss_multiplier
            exit_type = 'stop_loss'
        
        # Net P&L after friction
        net_pnl = gross_pnl - total_friction
        
        return {
            'date': date,
            'asset': asset,
            'direction': direction,
            'position_dollars': position_dollars,
            'position_fraction': decision['position_fraction'],
            'stop_loss_pct': stop_loss,
            'take_profit_pct': take_profit,
            'risk_reward': decision['risk_reward'],
            'raw_probability': decision['raw_probability'],
            'calibrated_probability': decision['calibrated_probability'],
            'actual_outcome': actual,
            'correct': correct,
            'won': net_pnl > 0,
            'gross_pnl': gross_pnl,
            'friction': total_friction,
            'pnl': net_pnl,
            'exit_type': exit_type,
            'volatility': volatility
        }
    
    def _generate_results(self, dates):
        """Generate comprehensive backtest results."""
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['won'])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L stats
        total_pnl = self.capital - self.initial_capital
        total_friction = sum(t['friction'] for t in self.trades)
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Runner stats
        runners = sum(1 for t in self.trades if t['exit_type'] == 'trailing_stop_exit')
        
        # Best and worst trades
        if self.trades:
            pnls = [t['pnl'] for t in self.trades]
            best_trade_idx = np.argmax(pnls)
            worst_trade_idx = np.argmin(pnls)
            best_trade = self.trades[best_trade_idx]
            worst_trade = self.trades[worst_trade_idx]
            avg_win = np.mean([t['pnl'] for t in self.trades if t['won']]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trades if not t['won']]) if losing_trades > 0 else 0
        else:
            best_trade = worst_trade = None
            avg_win = avg_loss = 0
        
        # Risk metrics
        daily_returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
        
        # Sortino (only downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = np.mean(daily_returns) / downside_std * np.sqrt(252)
        
        # Calmar (return / max drawdown)
        annual_return = total_pnl / self.initial_capital
        calmar_ratio = annual_return / self.max_drawdown if self.max_drawdown > 0 else float('inf')
        
        # Per-asset breakdown
        asset_stats = {}
        for asset in self.config.TARGET_ASSETS:
            asset_trades = [t for t in self.trades if t['asset'] == asset]
            if asset_trades:
                asset_wins = sum(1 for t in asset_trades if t['won'])
                asset_pnl = sum(t['pnl'] for t in asset_trades)
                asset_stats[asset] = {
                    'trades': len(asset_trades),
                    'wins': asset_wins,
                    'win_rate': asset_wins / len(asset_trades),
                    'pnl': asset_pnl,
                    'avg_position': np.mean([t['position_fraction'] for t in asset_trades]),
                    'avg_rr': np.mean([t['risk_reward'] for t in asset_trades]),
                    'friction': sum(t['friction'] for t in asset_trades)
                }
            else:
                asset_stats[asset] = {
                    'trades': 0, 'wins': 0, 'win_rate': 0, 'pnl': 0, 
                    'avg_position': 0, 'avg_rr': 0, 'friction': 0
                }
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_pnl': total_pnl,
            'total_friction': total_friction,
            'total_return_pct': (total_pnl / self.initial_capital) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'trades': self.trades,
            'trades_df': trades_df,
            'asset_stats': asset_stats,
            'runners': runners,
            'dates': dates
        }
    
    def print_results(self, results):
        """Print comprehensive results summary."""
        print("\n" + "="*80)
        print("ADVANCED BACKTEST RESULTS v2 (Academic Hardening)")
        print("="*80)
        
        print(f"\n💰 CAPITAL SUMMARY")
        print("-"*40)
        print(f"  Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"  Final Capital:      ${results['final_capital']:,.2f}")
        print(f"  Total P&L (Net):    ${results['total_pnl']:+,.2f}")
        print(f"  Total Return:       {results['total_return_pct']:+.2f}%")
        
        print(f"\n📊 TRADE STATISTICS")
        print("-"*40)
        print(f"  Total Trades:       {results['total_trades']}")
        print(f"  Winning Trades:     {results['winning_trades']}")
        print(f"  Losing Trades:      {results['losing_trades']}")
        print(f"  Win Rate:           {results['win_rate']:.1%}")
        print(f"  Profit Factor:      {results['profit_factor']:.2f}")
        print(f"  Trailing Stop Runs: {results['runners']} (Captured Trends)")
        
        print(f"\n💵 P&L & FRICTION")
        print("-"*40)
        print(f"  Gross Profit:       ${results['gross_profit']:,.2f}")
        print(f"  Gross Loss:         ${results['gross_loss']:,.2f}")
        print(f"  Total Friction:     ${results['total_friction']:,.2f} (Slippage + Comm)")
        print(f"  Average Win:        ${results['avg_win']:,.2f}")
        print(f"  Average Loss:       ${results['avg_loss']:,.2f}")
        
        print(f"\n📈 RISK METRICS")
        print("-"*40)
        print(f"  Max Drawdown:       {results['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {results['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:       {results['calmar_ratio']:.2f}")
        
        print(f"\n🏆 BEST & WORST TRADES")
        print("-"*40)
        if results['best_trade']:
            bt = results['best_trade']
            print(f"  Best Trade:  {bt['asset']} {bt['direction']} on {bt['date']}")
            print(f"               P&L: ${bt['pnl']:+,.2f} (Exit: {bt['exit_type']})")
        if results['worst_trade']:
            wt = results['worst_trade']
            print(f"  Worst Trade: {wt['asset']} {wt['direction']} on {wt['date']}")
            print(f"               P&L: ${wt['pnl']:+,.2f} (Exit: {wt['exit_type']})")
        
        print(f"\n📋 PER-ASSET BREAKDOWN")
        print("-"*40)
        for asset, stats in results['asset_stats'].items():
            print(f"  {asset:12s}: Trades={stats['trades']:3d}, WinRate={stats['win_rate']:.1%}, "
                  f"P&L=${stats['pnl']:+,.2f}, Friction=${stats['friction']:,.2f}")
        
        print("="*80 + "\n")
