# src/evaluation/backtest.py
"""
Simple backtest for directional predictions
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class SimpleBacktest:
    """Simple threshold-based backtest"""

    def __init__(self, model, test_dataset, config=None):
        self.config = config if config else Config()
        self.model = model.to(self.config.DEVICE)
        self.test_dataset = test_dataset

    def run_backtest(self):
        """Run simple backtest"""
        self.model.eval()

        results = {asset: {
            'dates': [],
            'predictions': [],
            'probabilities': [],
            'actuals': [],
            'raw_positions': [],
            'raw_returns': [],
            'gated_positions': [],
            'gated_returns': []
        } for asset in self.config.TARGET_ASSETS}

        with torch.no_grad():
            for idx in range(len(self.test_dataset)):
                features, labels = self.test_dataset[idx]
                features = features.unsqueeze(0).to(self.config.DEVICE)

                # Predict
                predictions, _ = self.model(features)

                # Get date
                date = self.test_dataset.get_date(idx)

                # Store results per asset
                for asset in self.config.TARGET_ASSETS:
                    # FIX: Apply sigmoid to convert logit to probability
                    prob = torch.sigmoid(predictions[asset]).item()
                    pred = 1 if prob > 0.5 else 0
                    actual = int(labels[asset].item())

                    # Trading signal
                    if prob > self.config.BUY_THRESHOLD:
                        position = 1  # Long
                    elif prob < self.config.SELL_THRESHOLD:
                        position = -1  # Short
                    else:
                        position = 0  # Neutral

                    results[asset]['dates'].append(date)
                    results[asset]['probabilities'].append(prob)
                    results[asset]['predictions'].append(pred)
                    results[asset]['actuals'].append(actual)

                    # --- PHASE 1: Raw Signal ---
                    if prob > self.config.BUY_THRESHOLD:
                        raw_pos = 1
                    elif prob < self.config.SELL_THRESHOLD:
                        raw_pos = -1
                    else:
                        raw_pos = 0
                    
                    results[asset]['raw_positions'].append(raw_pos)
                    raw_return = raw_pos * (actual * 2 - 1)
                    results[asset]['raw_returns'].append(raw_return)

                    # --- PHASE 2: Sharpe-Gated Execution ---
                    # Calculate rolling Sharpe of RAW returns so far
                    past_raw_returns = results[asset]['raw_returns']
                    if len(past_raw_returns) >= self.config.ROLLING_SHARPE_WINDOW:
                        recent_returns = np.array(past_raw_returns[-self.config.ROLLING_SHARPE_WINDOW:])
                        rolling_sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-10) * np.sqrt(252)
                    else:
                        if len(past_raw_returns) > 5:
                            recent_returns = np.array(past_raw_returns)
                            rolling_sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-10) * np.sqrt(252)
                        else:
                            rolling_sharpe = 2.0  # Initial optimism

                    gated_pos = raw_pos if rolling_sharpe >= self.config.MIN_SHARPE_THRESHOLD else 0
                    results[asset]['gated_positions'].append(gated_pos)
                    gated_return = gated_pos * (actual * 2 - 1)
                    results[asset]['gated_returns'].append(gated_return)

        # Calculate metrics
        backtest_metrics = {}

        for asset in self.config.TARGET_ASSETS:
            # Phase 1 Metrics (Raw)
            raw_ret = np.array(results[asset]['raw_returns'])
            raw_pos = np.array(results[asset]['raw_positions'])
            raw_active = raw_ret[raw_pos != 0]
            
            p1_metrics = self._calculate_phase_metrics(raw_active, len(raw_ret))

            # Phase 2 Metrics (Gated)
            gated_ret = np.array(results[asset]['gated_returns'])
            gated_pos = np.array(results[asset]['gated_positions'])
            gated_active = gated_ret[gated_pos != 0]
            
            p2_metrics = self._calculate_phase_metrics(gated_active, len(gated_ret))
            p2_metrics['skipped'] = np.sum((raw_pos != 0) & (gated_pos == 0))

            backtest_metrics[asset] = {
                'phase1': p1_metrics,
                'phase2': p2_metrics
            }

        return results, backtest_metrics

    def _calculate_phase_metrics(self, active_returns, total_days):
        if len(active_returns) > 0:
            cum_ret = np.sum(active_returns)
            sharpe = np.mean(active_returns) / (np.std(active_returns) + 1e-10) * np.sqrt(252)
            win_rate = np.sum(active_returns > 0) / len(active_returns)
            num_trades = len(active_returns)
        else:
            cum_ret, sharpe, win_rate, num_trades = 0, 0, 0, 0

        return {
            'cumulative_return': cum_ret,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'activity_rate': num_trades / total_days if total_days > 0 else 0
        }

    def print_backtest_results(self, metrics):
        """Print backtest results"""
        print("\n" + "="*80)
        print("TWO-PHASE BACKTEST RESULTS")
        print("="*80)
        print(f"Strategy: Buy if P(up) > {self.config.BUY_THRESHOLD}, "
              f"Sell if P(up) < {self.config.SELL_THRESHOLD}")
        print(f"Filter:   Take trade if Rolling Sharpe (20d) > {self.config.MIN_SHARPE_THRESHOLD}")
        print("="*80)

        for asset in self.config.TARGET_ASSETS:
            p1 = metrics[asset]['phase1']
            p2 = metrics[asset]['phase2']
            print(f"\n{asset:12s} | Phase 1 (Raw)      | Phase 2 (Gated)")
            print("-" * 65)
            print(f"  Return:      {p1['cumulative_return']:+12.2f} pts | {p2['cumulative_return']:+12.2f} pts")
            print(f"  Sharpe:      {p1['sharpe_ratio']:12.3f}     | {p2['sharpe_ratio']:12.3f}")
            print(f"  Win Rate:    {p1['win_rate']:12.1%}     | {p2['win_rate']:12.1%}")
            print(f"  Trades:      {p1['num_trades']:12d}     | {p2['num_trades']:12d}")
            print(f"  Skipped:     {'':12s}     | {p2['skipped']:12d}")

        # Overall
        p1_ret = sum([metrics[asset]['phase1']['cumulative_return'] for asset in self.config.TARGET_ASSETS])
        p2_ret = sum([metrics[asset]['phase2']['cumulative_return'] for asset in self.config.TARGET_ASSETS])
        p1_sr = np.mean([metrics[asset]['phase1']['sharpe_ratio'] for asset in self.config.TARGET_ASSETS])
        p2_sr = np.mean([metrics[asset]['phase2']['sharpe_ratio'] for asset in self.config.TARGET_ASSETS])

        print(f"\nOVERALL COMPARISON:")
        print(f"  Total Return:  {p1_ret:+12.2f} pts (Raw) vs {p2_ret:+12.2f} pts (Gated)")
        print(f"  Avg Sharpe:    {p1_sr:12.3f}       vs {p2_sr:12.3f}")

        print("="*80 + "\n")

    def plot_cumulative_returns(self, results, save_path=None):
        """Plot cumulative returns over time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, asset in enumerate(self.config.TARGET_ASSETS):
            dates = pd.to_datetime(results[asset]['dates'])
            
            # Phase 1: Raw
            raw_cum = np.cumsum(results[asset]['raw_returns'])
            axes[i].plot(dates, raw_cum, label='Phase 1 (Raw)', color='blue', alpha=0.5, linestyle='--')
            
            # Phase 2: Gated
            gated_cum = np.cumsum(results[asset]['gated_returns'])
            axes[i].plot(dates, gated_cum, label='Phase 2 (Gated)', color='red', linewidth=2)
            
            axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[i].set_title(f'{asset} Two-Phase Returns', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Cumulative Return (points)')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend()

            # Final values label
            f1 = raw_cum[-1]
            f2 = gated_cum[-1]
            axes[i].text(0.02, 0.98, f'Raw: {f1:+.1f}\nGated: {f2:+.1f}',
                        transform=axes[i].transAxes,
                        va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path is None:
            save_path = self.config.PLOT_DIR / 'backtest_returns.png'
            self.config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Backtest returns plot saved to {save_path}")
        plt.close()
