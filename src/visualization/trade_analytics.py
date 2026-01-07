# src/visualization/trade_analytics.py
"""
Comprehensive trade analytics visualizations:
1. Equity Curve
2. Drawdown Chart
3. Trade Distribution
4. Best/Worst Trades
5. Position Size Over Time
6. Win Rate by Confidence
7. Monthly Returns Heatmap
8. Risk/Reward Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


class TradeAnalytics:
    """Generate comprehensive trade analytics visualizations."""
    
    def __init__(self, results, save_dir=None):
        """
        Args:
            results: dict from AdvancedBacktest.run_backtest()
            save_dir: directory to save plots
        """
        self.results = results
        self.save_dir = Path(save_dir) if save_dir else Path('results/plots')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'profit': '#2ecc71',
            'loss': '#e74c3c',
            'neutral': '#3498db',
            'equity': '#2c3e50',
            'drawdown': '#c0392b'
        }
    
    def plot_all(self):
        """Generate all visualizations."""
        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_trade_distribution()
        self.plot_monthly_returns()
        self.plot_position_sizing()
        self.plot_confidence_analysis()
        self.plot_asset_performance()
        self.plot_risk_reward()
        print(f"✓ All visualizations saved to {self.save_dir}")
    
    def plot_equity_curve(self):
        """Plot equity curve from $10K to final value."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        equity = self.results['equity_curve']
        x = range(len(equity))
        
        # Main equity line
        ax.plot(x, equity, color=self.colors['equity'], linewidth=2, label='Equity')
        
        # Fill between for visual effect
        ax.fill_between(x, self.results['initial_capital'], equity, 
                        where=[e >= self.results['initial_capital'] for e in equity],
                        color=self.colors['profit'], alpha=0.3, label='Profit')
        ax.fill_between(x, self.results['initial_capital'], equity,
                        where=[e < self.results['initial_capital'] for e in equity],
                        color=self.colors['loss'], alpha=0.3, label='Loss')
        
        # Horizontal line at initial capital
        ax.axhline(y=self.results['initial_capital'], color='gray', 
                   linestyle='--', alpha=0.7, label='Initial Capital')
        
        # Annotations
        ax.annotate(f"Start: ${self.results['initial_capital']:,.0f}",
                    xy=(0, self.results['initial_capital']),
                    xytext=(10, -30), textcoords='offset points',
                    fontsize=10, fontweight='bold')
        ax.annotate(f"End: ${equity[-1]:,.0f}\n({self.results['total_return_pct']:+.1f}%)",
                    xy=(len(equity)-1, equity[-1]),
                    xytext=(-80, 20), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Account Value ($)', fontsize=12)
        ax.set_title('Equity Curve: $10,000 Initial Capital', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown(self):
        """Plot drawdown chart."""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        equity = np.array(self.results['equity_curve'])
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        x = range(len(drawdown))
        
        ax.fill_between(x, 0, drawdown, color=self.colors['drawdown'], alpha=0.7)
        ax.plot(x, drawdown, color=self.colors['drawdown'], linewidth=1)
        
        # Mark max drawdown
        max_dd_idx = np.argmin(drawdown)
        ax.scatter([max_dd_idx], [drawdown[max_dd_idx]], color='black', s=100, zorder=5)
        ax.annotate(f'Max DD: {drawdown[max_dd_idx]:.1f}%',
                    xy=(max_dd_idx, drawdown[max_dd_idx]),
                    xytext=(20, -20), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Underwater Equity (Drawdown from Peak)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(drawdown) * 1.1, 5)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'drawdown_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_trade_distribution(self):
        """Plot histogram of trade returns."""
        if not self.results['trades']:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        pnls = [t['pnl'] for t in self.results['trades']]
        
        # Histogram
        ax = axes[0]
        colors = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls]
        n, bins, patches = ax.hist(pnls, bins=30, edgecolor='white', linewidth=0.5)
        
        # Color bins
        for patch, left, right in zip(patches, bins[:-1], bins[1:]):
            if left + right > 0:
                patch.set_facecolor(self.colors['profit'])
            else:
                patch.set_facecolor(self.colors['loss'])
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.axvline(x=np.mean(pnls), color=self.colors['neutral'], 
                   linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
        
        ax.set_xlabel('Trade P&L ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Trade Returns', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Cumulative P&L
        ax = axes[1]
        cumulative = np.cumsum(pnls)
        ax.plot(cumulative, color=self.colors['equity'], linewidth=2)
        ax.fill_between(range(len(cumulative)), 0, cumulative,
                        where=[c >= 0 for c in cumulative],
                        color=self.colors['profit'], alpha=0.3)
        ax.fill_between(range(len(cumulative)), 0, cumulative,
                        where=[c < 0 for c in cumulative],
                        color=self.colors['loss'], alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax.set_title('Cumulative Trade Returns', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'trade_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_returns(self):
        """Plot monthly returns heatmap."""
        if not self.results['trades']:
            return
            
        trades_df = self.results['trades_df']
        if trades_df.empty or 'date' not in trades_df.columns:
            return
        
        # Convert dates and group by month
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df['year'] = trades_df['date'].dt.year
        trades_df['month'] = trades_df['date'].dt.month
        
        monthly = trades_df.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list('rg', 
            [self.colors['loss'], 'white', self.colors['profit']])
        
        # Heatmap
        im = ax.imshow(monthly.values, cmap=cmap, aspect='auto',
                       vmin=-np.abs(monthly.values).max(),
                       vmax=np.abs(monthly.values).max())
        
        # Labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index)
        
        # Annotate cells
        for i in range(len(monthly.index)):
            for j in range(len(monthly.columns)):
                val = monthly.iloc[i, j]
                color = 'white' if abs(val) > monthly.values.max() * 0.5 else 'black'
                ax.text(j, i, f'${val:.0f}', ha='center', va='center', 
                       color=color, fontsize=9, fontweight='bold')
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='P&L ($)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'monthly_returns.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_position_sizing(self):
        """Plot position sizing over time."""
        if not self.results['trades']:
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        positions = [t['position_fraction'] * 100 for t in self.results['trades']]
        pnls = [t['pnl'] for t in self.results['trades']]
        
        # Position size over time
        ax = axes[0]
        colors = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls]
        ax.bar(range(len(positions)), positions, color=colors, alpha=0.7, width=1)
        ax.axhline(y=np.mean(positions), color=self.colors['neutral'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(positions):.1f}%')
        
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Position Size (% of Capital)', fontsize=12)
        ax.set_title('Position Sizing Over Time (Green=Win, Red=Loss)', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        
        # Position size vs P&L scatter
        ax = axes[1]
        colors_scatter = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls]
        ax.scatter(positions, pnls, c=colors_scatter, alpha=0.6, s=50)
        
        # Trend line
        z = np.polyfit(positions, pnls, 1)
        p = np.poly1d(z)
        ax.plot(sorted(positions), p(sorted(positions)), 
                color=self.colors['neutral'], linestyle='--', linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Position Size (%)', fontsize=12)
        ax.set_ylabel('Trade P&L ($)', fontsize=12)
        ax.set_title('Position Size vs Trade Outcome', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'position_sizing.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_analysis(self):
        """Plot win rate by confidence level."""
        if not self.results['trades']:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bin trades by confidence
        trades = self.results['trades']
        confidence_bins = np.arange(0.5, 1.05, 0.05)
        bin_labels = [f'{b:.0%}-{b+0.05:.0%}' for b in confidence_bins[:-1]]
        
        win_rates = []
        trade_counts = []
        
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i+1]
            bin_trades = [t for t in trades 
                         if low <= t['calibrated_probability'] < high]
            if bin_trades:
                win_rate = sum(1 for t in bin_trades if t['won']) / len(bin_trades)
                win_rates.append(win_rate * 100)
                trade_counts.append(len(bin_trades))
            else:
                win_rates.append(0)
                trade_counts.append(0)
        
        # Win rate bar chart
        ax = axes[0]
        colors = [self.colors['profit'] if wr > 50 else self.colors['loss'] 
                  for wr in win_rates]
        bars = ax.bar(range(len(bin_labels)), win_rates, color=colors, alpha=0.7)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% Baseline')
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_xlabel('Confidence Level', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rate by Model Confidence', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Trade count overlay
        ax2 = ax.twinx()
        ax2.plot(range(len(trade_counts)), trade_counts, 
                 color=self.colors['neutral'], marker='o', linewidth=2)
        ax2.set_ylabel('Number of Trades', fontsize=12, color=self.colors['neutral'])
        
        # Confidence vs P&L
        ax = axes[1]
        confidences = [t['calibrated_probability'] for t in trades]
        pnls = [t['pnl'] for t in trades]
        colors_scatter = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls]
        
        ax.scatter(confidences, pnls, c=colors_scatter, alpha=0.5, s=40)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Calibrated Probability', fontsize=12)
        ax.set_ylabel('Trade P&L ($)', fontsize=12)
        ax.set_title('Model Confidence vs Trade Outcome', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confidence_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_asset_performance(self):
        """Plot per-asset performance."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        assets = list(self.results['asset_stats'].keys())
        stats = self.results['asset_stats']
        
        # P&L by asset
        ax = axes[0]
        pnls = [stats[a]['pnl'] for a in assets]
        colors = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in pnls]
        bars = ax.bar(assets, pnls, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Total P&L ($)', fontsize=12)
        ax.set_title('P&L by Asset', fontsize=14, fontweight='bold')
        for bar, pnl in zip(bars, pnls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (50 if pnl >= 0 else -100),
                   f'${pnl:,.0f}', ha='center', va='bottom' if pnl >= 0 else 'top', fontweight='bold')
        
        # Win rate by asset
        ax = axes[1]
        win_rates = [stats[a]['win_rate'] * 100 for a in assets]
        colors = [self.colors['profit'] if wr > 50 else self.colors['loss'] for wr in win_rates]
        bars = ax.bar(assets, win_rates, color=colors, alpha=0.7)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=2)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rate by Asset', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Trade count by asset
        ax = axes[2]
        trades = [stats[a]['trades'] for a in assets]
        ax.bar(assets, trades, color=self.colors['neutral'], alpha=0.7)
        ax.set_ylabel('Number of Trades', fontsize=12)
        ax.set_title('Trade Count by Asset', fontsize=14, fontweight='bold')
        
        # Avg position by asset
        ax = axes[3]
        positions = [stats[a]['avg_position'] * 100 for a in assets]
        ax.bar(assets, positions, color=self.colors['equity'], alpha=0.7)
        ax.set_ylabel('Avg Position Size (%)', fontsize=12)
        ax.set_title('Average Position Size by Asset', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'asset_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_risk_reward(self):
        """Plot risk/reward analysis."""
        if not self.results['trades']:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        trades = self.results['trades']
        
        # R:R ratio distribution
        ax = axes[0]
        rr_ratios = [t['risk_reward'] for t in trades]
        ax.hist(rr_ratios, bins=20, color=self.colors['neutral'], 
                edgecolor='white', alpha=0.7)
        ax.axvline(x=np.mean(rr_ratios), color=self.colors['profit'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(rr_ratios):.2f}')
        ax.axvline(x=1.5, color='orange', linestyle='--', linewidth=2, 
                   label='Target: 1.5')
        ax.set_xlabel('Risk/Reward Ratio', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Risk/Reward Ratios', fontsize=14, fontweight='bold')
        ax.legend()
        
        # SL vs TP scatter
        ax = axes[1]
        sls = [t['stop_loss_pct'] * 100 for t in trades]
        tps = [t['take_profit_pct'] * 100 for t in trades]
        colors = [self.colors['profit'] if t['won'] else self.colors['loss'] for t in trades]
        
        ax.scatter(sls, tps, c=colors, alpha=0.5, s=50)
        
        # 1:1 line
        max_val = max(max(sls), max(tps))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
        ax.plot([0, max_val], [0, max_val * 1.5], 'g--', alpha=0.5, label='1:1.5')
        ax.plot([0, max_val], [0, max_val * 2], 'b--', alpha=0.5, label='1:2')
        
        ax.set_xlabel('Stop Loss (%)', fontsize=12)
        ax.set_ylabel('Take Profit (%)', fontsize=12)
        ax.set_title('Stop Loss vs Take Profit (Green=Win, Red=Loss)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'risk_reward_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_summary_dashboard(results, save_path):
    """Create a single-page summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Equity curve (top, spanning 3 columns)
    ax1 = fig.add_subplot(gs[0, :3])
    equity = results['equity_curve']
    ax1.plot(equity, color='#2c3e50', linewidth=2)
    ax1.fill_between(range(len(equity)), results['initial_capital'], equity,
                     where=[e >= results['initial_capital'] for e in equity],
                     color='#2ecc71', alpha=0.3)
    ax1.fill_between(range(len(equity)), results['initial_capital'], equity,
                     where=[e < results['initial_capital'] for e in equity],
                     color='#e74c3c', alpha=0.3)
    ax1.axhline(y=results['initial_capital'], color='gray', linestyle='--')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Key metrics box (top right)
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.axis('off')
    metrics_text = f"""
    Initial: ${results['initial_capital']:,.0f}
    Final:   ${results['final_capital']:,.0f}
    Return:  {results['total_return_pct']:+.1f}%
    
    Trades:  {results['total_trades']}
    Win Rate: {results['win_rate']:.1%}
    
    Sharpe:  {results['sharpe_ratio']:.2f}
    Sortino: {results['sortino_ratio']:.2f}
    Max DD:  {results['max_drawdown']:.1%}
    """
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('Key Metrics', fontsize=14, fontweight='bold')
    
    # 3. Trade distribution (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    if results['trades']:
        pnls = [t['pnl'] for t in results['trades']]
        colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, width=1, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-')
    ax3.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('P&L ($)')
    
    # 4. Asset performance (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    assets = list(results['asset_stats'].keys())
    pnls = [results['asset_stats'][a]['pnl'] for a in assets]
    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in pnls]
    ax4.barh(assets, pnls, color=colors, alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='-')
    ax4.set_title('P&L by Asset', fontsize=14, fontweight='bold')
    ax4.set_xlabel('P&L ($)')
    
    # 5. Win rate by asset (bottom left)
    ax5 = fig.add_subplot(gs[2, :2])
    win_rates = [results['asset_stats'][a]['win_rate'] * 100 for a in assets]
    colors = ['#2ecc71' if wr > 50 else '#e74c3c' for wr in win_rates]
    ax5.bar(assets, win_rates, color=colors, alpha=0.7)
    ax5.axhline(y=50, color='black', linestyle='--')
    ax5.set_ylim(0, 100)
    ax5.set_title('Win Rate by Asset', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Win Rate (%)')
    
    # 6. Best/Worst trades (bottom right)
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    if results['best_trade'] and results['worst_trade']:
        bt = results['best_trade']
        wt = results['worst_trade']
        trade_text = f"""
        BEST TRADE
        Asset: {bt['asset']} | Dir: {bt['direction']}
        P&L: ${bt['pnl']:+,.2f}
        Conf: {bt['calibrated_probability']:.1%}
        
        WORST TRADE
        Asset: {wt['asset']} | Dir: {wt['direction']}
        P&L: ${wt['pnl']:+,.2f}
        Conf: {wt['calibrated_probability']:.1%}
        """
        ax6.text(0.1, 0.9, trade_text, transform=ax6.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Best & Worst Trades', fontsize=14, fontweight='bold')
    
    plt.suptitle('TRADING SYSTEM PERFORMANCE DASHBOARD', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Dashboard saved to {save_path}")
