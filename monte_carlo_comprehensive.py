"""
Comprehensive Monte Carlo Analysis: LSTM+Attention (V7) vs TFT Sliding (V8)
Enhanced version with extensive statistics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Paths
V7_TRADES = Path("experiments/sliding_window_v7/trades.csv")
V8_TRADES = Path("experiments/tft_v8_sliding/trades.csv")
OUTPUT_DIR = Path("experiments/monte_carlo_comprehensive")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Monte Carlo parameters
N_SIMULATIONS = 250000
INITIAL_CAPITAL = 10000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_trades(path):
    """Load trade data and extract P&L values"""
    df = pd.read_csv(path)
    if 'pnl' not in df.columns:
        raise ValueError(f"Cannot find 'pnl' column in {path}")
    return df

def monte_carlo_simulation(pnl_values, n_simulations=N_SIMULATIONS, initial_capital=INITIAL_CAPITAL):
    """
    Perform Monte Carlo simulation by bootstrap resampling trade P&L sequences
    """
    results = {
        'final_returns': [],
        'final_equity': [],
        'max_drawdowns': [],
        'sharpe_ratios': [],
        'sortino_ratios': [],
        'win_rates': [],
        'profit_factors': [],
        'avg_win': [],
        'avg_loss': [],
        'max_win': [],
        'max_loss': [],
        'equity_curves': []
    }

    n_trades = len(pnl_values)

    for i in tqdm(range(n_simulations), desc="Running simulations"):
        # Bootstrap resample
        sampled_pnl = np.random.choice(pnl_values, size=n_trades, replace=True)

        # Build equity curve
        equity_curve = np.zeros(n_trades + 1)
        equity_curve[0] = initial_capital

        for j in range(n_trades):
            equity_curve[j + 1] = equity_curve[j] + sampled_pnl[j]

        # Final metrics
        final_equity = equity_curve[-1]
        final_return = (final_equity / initial_capital - 1) * 100

        # Drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min() * 100

        # Sharpe & Sortino
        trade_returns = sampled_pnl / initial_capital
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            avg_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            sharpe = (avg_return / std_return * np.sqrt(250)) if std_return > 0 else 0

            # Sortino (downside deviation)
            negative_returns = trade_returns[trade_returns < 0]
            if len(negative_returns) > 0:
                downside_dev = np.std(negative_returns)
                sortino = (avg_return / downside_dev * np.sqrt(250)) if downside_dev > 0 else 0
            else:
                sortino = sharpe
        else:
            sharpe = 0
            sortino = 0

        # Win rate and profit factor
        wins = sampled_pnl[sampled_pnl > 0]
        losses = sampled_pnl[sampled_pnl < 0]
        win_rate = len(wins) / len(sampled_pnl) * 100

        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0001
        profit_factor = gross_profit / gross_loss

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        max_win = wins.max() if len(wins) > 0 else 0
        max_loss = losses.min() if len(losses) > 0 else 0

        # Store results
        results['final_returns'].append(final_return)
        results['final_equity'].append(final_equity)
        results['max_drawdowns'].append(max_dd)
        results['sharpe_ratios'].append(sharpe)
        results['sortino_ratios'].append(sortino)
        results['win_rates'].append(win_rate)
        results['profit_factors'].append(profit_factor)
        results['avg_win'].append(avg_win)
        results['avg_loss'].append(avg_loss)
        results['max_win'].append(max_win)
        results['max_loss'].append(max_loss)

        # Store sample equity curves
        if i < 10000:  # Store 10000 curves for visualization
            results['equity_curves'].append(equity_curve)

    return results

def calculate_statistics(results):
    """Calculate comprehensive summary statistics"""
    stats = {}

    metrics = ['final_returns', 'max_drawdowns', 'sharpe_ratios', 'sortino_ratios',
               'win_rates', 'profit_factors', 'avg_win', 'avg_loss', 'max_win', 'max_loss']

    for metric in metrics:
        data = np.array(results[metric])
        stats[metric] = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'ci_90_lower': np.percentile(data, 5),
            'ci_90_upper': np.percentile(data, 95),
            'ci_95_lower': np.percentile(data, 2.5),
            'ci_95_upper': np.percentile(data, 97.5),
            'ci_99_lower': np.percentile(data, 0.5),
            'ci_99_upper': np.percentile(data, 99.5),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': calculate_skewness(data),
            'kurtosis': calculate_kurtosis(data)
        }

        if metric == 'final_returns':
            stats[metric]['prob_negative'] = (data < 0).sum() / len(data) * 100
            stats[metric]['prob_loss_50pct'] = (data < -50).sum() / len(data) * 100
            stats[metric]['prob_gain_100pct'] = (data > 100).sum() / len(data) * 100

    return stats

def calculate_skewness(data):
    """Calculate skewness"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def plot_comprehensive_distributions(v7_results, v8_results, n_sims):
    """Create comprehensive distribution plots"""
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)

    metrics = [
        ('final_returns', 'Final Return (%)', 'Distribution of Final Returns'),
        ('sharpe_ratios', 'Sharpe Ratio', 'Distribution of Sharpe Ratios'),
        ('sortino_ratios', 'Sortino Ratio', 'Distribution of Sortino Ratios'),
        ('max_drawdowns', 'Max Drawdown (%)', 'Distribution of Max Drawdowns'),
        ('win_rates', 'Win Rate (%)', 'Distribution of Win Rates'),
        ('profit_factors', 'Profit Factor', 'Distribution of Profit Factors'),
        ('avg_win', 'Average Win ($)', 'Distribution of Average Wins'),
        ('avg_loss', 'Average Loss ($)', 'Distribution of Average Losses'),
        ('max_win', 'Max Win ($)', 'Distribution of Maximum Wins'),
    ]

    for idx, (metric, xlabel, title) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        v7_data = v7_results[metric]
        v8_data = v8_results[metric]

        # Plot histograms with transparency
        ax.hist(v7_data, bins=60, alpha=0.5, label='V7 LSTM+Attention',
                color='steelblue', density=True, edgecolor='darkblue', linewidth=0.5)
        ax.hist(v8_data, bins=60, alpha=0.5, label='V8 TFT Sliding',
                color='darkorange', density=True, edgecolor='darkred', linewidth=0.5)

        # Add median and mean lines
        ax.axvline(np.median(v7_data), color='steelblue', linestyle='--',
                   linewidth=2.5, label=f'V7 Median: {np.median(v7_data):.2f}')
        ax.axvline(np.median(v8_data), color='darkorange', linestyle='--',
                   linewidth=2.5, label=f'V8 Median: {np.median(v8_data):.2f}')

        ax.axvline(np.mean(v7_data), color='steelblue', linestyle=':',
                   linewidth=2, alpha=0.7, label=f'V7 Mean: {np.mean(v7_data):.2f}')
        ax.axvline(np.mean(v8_data), color='darkorange', linestyle=':',
                   linewidth=2, alpha=0.7, label=f'V8 Mean: {np.mean(v8_data):.2f}')

        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Comprehensive Monte Carlo Distributions ({n_sims:,} Simulations)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_DIR / 'comprehensive_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_intervals(v7_stats, v8_stats, n_sims):
    """Plot detailed confidence interval comparisons"""
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle(f'Confidence Interval Analysis ({n_sims:,} Simulations)',
                 fontsize=16, fontweight='bold')

    metrics_plot = [
        ('final_returns', 'Final Return (%)'),
        ('sharpe_ratios', 'Sharpe Ratio'),
        ('sortino_ratios', 'Sortino Ratio'),
        ('max_drawdowns', 'Max Drawdown (%)'),
        ('win_rates', 'Win Rate (%)'),
        ('profit_factors', 'Profit Factor'),
        ('avg_win', 'Average Win ($)'),
        ('avg_loss', 'Average Loss ($)'),
        ('max_win', 'Max Win ($)')
    ]

    for idx, (metric, title) in enumerate(metrics_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        models = ['V7\nLSTM+Attention', 'V8\nTFT Sliding']
        means = [v7_stats[metric]['mean'], v8_stats[metric]['mean']]
        medians = [v7_stats[metric]['median'], v8_stats[metric]['median']]
        ci_95_lower = [v7_stats[metric]['ci_95_lower'], v8_stats[metric]['ci_95_lower']]
        ci_95_upper = [v7_stats[metric]['ci_95_upper'], v8_stats[metric]['ci_95_upper']]
        ci_90_lower = [v7_stats[metric]['ci_90_lower'], v8_stats[metric]['ci_90_lower']]
        ci_90_upper = [v7_stats[metric]['ci_90_upper'], v8_stats[metric]['ci_90_upper']]

        x_pos = np.arange(len(models))

        # Plot median bars
        bars = ax.bar(x_pos, medians, color=['steelblue', 'darkorange'],
                      alpha=0.7, width=0.5, label='Median')

        # Plot 95% CI error bars
        errors_95 = [
            [medians[i] - ci_95_lower[i] for i in range(2)],
            [ci_95_upper[i] - medians[i] for i in range(2)]
        ]
        ax.errorbar(x_pos, medians, yerr=errors_95, fmt='none', ecolor='black',
                    capsize=8, capthick=2, linewidth=2, label='95% CI')

        # Plot 90% CI error bars
        errors_90 = [
            [medians[i] - ci_90_lower[i] for i in range(2)],
            [ci_90_upper[i] - medians[i] for i in range(2)]
        ]
        ax.errorbar(x_pos, medians, yerr=errors_90, fmt='none', ecolor='gray',
                    capsize=6, capthick=1.5, linewidth=1.5, alpha=0.7, label='90% CI')

        # Add mean markers
        ax.scatter(x_pos, means, color='red', s=100, zorder=5,
                   marker='D', label='Mean', edgecolors='darkred', linewidths=2)

        # Add value labels
        for i, (bar, median, mean) in enumerate(zip(bars, medians, means)):
            ax.text(bar.get_x() + bar.get_width()/2, median,
                   f'{median:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax.text(bar.get_x() + bar.get_width()/2, mean,
                   f'{mean:.2f}',
                   ha='center', va='top', fontweight='bold', fontsize=8, color='red')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=10, fontweight='bold')
        ax.set_ylabel(title, fontsize=10, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_intervals_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_equity_curves(v7_results, v8_results, n_sims):
    """Plot sample equity curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    n_curves = min(10000, len(v7_results['equity_curves']))

    # V7 equity curves - all
    for curve in v7_results['equity_curves'][:n_curves]:
        ax1.plot(curve, alpha=0.02, color='steelblue', linewidth=0.5)
    ax1.set_title(f'V7 LSTM+Attention: {n_curves} Equity Curves', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax1.axhline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2.5, label='Initial Capital')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # V8 equity curves - all
    for curve in v8_results['equity_curves'][:n_curves]:
        ax2.plot(curve, alpha=0.02, color='darkorange', linewidth=0.5)
    ax2.set_title(f'V8 TFT Sliding: {n_curves} Equity Curves', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax2.axhline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2.5, label='Initial Capital')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # V7 - percentile bands
    equity_array = np.array(v7_results['equity_curves'][:n_curves])
    median_curve = np.median(equity_array, axis=0)
    p5_curve = np.percentile(equity_array, 5, axis=0)
    p25_curve = np.percentile(equity_array, 25, axis=0)
    p75_curve = np.percentile(equity_array, 75, axis=0)
    p95_curve = np.percentile(equity_array, 95, axis=0)

    x = np.arange(len(median_curve))
    ax3.fill_between(x, p5_curve, p95_curve, alpha=0.2, color='steelblue', label='5th-95th Percentile')
    ax3.fill_between(x, p25_curve, p75_curve, alpha=0.3, color='steelblue', label='25th-75th Percentile')
    ax3.plot(median_curve, color='darkblue', linewidth=3, label='Median', zorder=5)
    ax3.axhline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    ax3.set_title('V7 LSTM+Attention: Percentile Bands', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # V8 - percentile bands
    equity_array = np.array(v8_results['equity_curves'][:n_curves])
    median_curve = np.median(equity_array, axis=0)
    p5_curve = np.percentile(equity_array, 5, axis=0)
    p25_curve = np.percentile(equity_array, 25, axis=0)
    p75_curve = np.percentile(equity_array, 75, axis=0)
    p95_curve = np.percentile(equity_array, 95, axis=0)

    x = np.arange(len(median_curve))
    ax4.fill_between(x, p5_curve, p95_curve, alpha=0.2, color='darkorange', label='5th-95th Percentile')
    ax4.fill_between(x, p25_curve, p75_curve, alpha=0.3, color='darkorange', label='25th-75th Percentile')
    ax4.plot(median_curve, color='darkred', linewidth=3, label='Median', zorder=5)
    ax4.axhline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    ax4.set_title('V8 TFT Sliding: Percentile Bands', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Monte Carlo Equity Curve Analysis ({n_sims:,} Simulations)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'equity_curves_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_return_scatter(v7_results, v8_results, n_sims):
    """Risk-return scatter plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Return vs Drawdown
    ax1.scatter(v7_results['max_drawdowns'], v7_results['final_returns'],
                alpha=0.3, s=5, color='steelblue', label='V7 LSTM+Attention')
    ax1.scatter(v8_results['max_drawdowns'], v8_results['final_returns'],
                alpha=0.3, s=5, color='darkorange', label='V8 TFT Sliding')

    ax1.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Risk-Return Profile: Return vs Drawdown', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, markerscale=5)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Return vs Sharpe
    ax2.scatter(v7_results['sharpe_ratios'], v7_results['final_returns'],
                alpha=0.3, s=5, color='steelblue', label='V7 LSTM+Attention')
    ax2.scatter(v8_results['sharpe_ratios'], v8_results['final_returns'],
                alpha=0.3, s=5, color='darkorange', label='V8 TFT Sliding')

    ax2.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Final Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Risk-Adjusted Profile: Return vs Sharpe', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, markerscale=5)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    fig.suptitle(f'Risk-Return Analysis ({n_sims:,} Simulations)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_percentile_comparison(v7_stats, v8_stats, n_sims):
    """Plot percentile comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    metrics = [
        ('final_returns', 'Final Return (%)'),
        ('sharpe_ratios', 'Sharpe Ratio'),
        ('max_drawdowns', 'Max Drawdown (%)'),
        ('profit_factors', 'Profit Factor')
    ]

    for idx, (metric, title) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Use key percentiles
        key_percentiles = ['Min', '5%', '25%', '50%', '75%', '95%', 'Max']
        v7_key = [v7_stats[metric]['min'], v7_stats[metric]['ci_95_lower'],
                  v7_stats[metric]['q25'], v7_stats[metric]['median'],
                  v7_stats[metric]['q75'], v7_stats[metric]['ci_95_upper'],
                  v7_stats[metric]['max']]
        v8_key = [v8_stats[metric]['min'], v8_stats[metric]['ci_95_lower'],
                  v8_stats[metric]['q25'], v8_stats[metric]['median'],
                  v8_stats[metric]['q75'], v8_stats[metric]['ci_95_upper'],
                  v8_stats[metric]['max']]

        x = np.arange(len(key_percentiles))
        width = 0.35

        ax.bar(x - width/2, v7_key, width, label='V7 LSTM+Attention',
               color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
        ax.bar(x + width/2, v8_key, width, label='V8 TFT Sliding',
               color='darkorange', alpha=0.7, edgecolor='darkred', linewidth=1.5)

        ax.set_xlabel('Percentile', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} by Percentile', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(key_percentiles, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add median line
        ax.axhline(v7_stats[metric]['median'], color='steelblue',
                   linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(v8_stats[metric]['median'], color='darkorange',
                   linestyle='--', linewidth=1.5, alpha=0.5)

    fig.suptitle(f'Percentile Distribution Comparison ({n_sims:,} Simulations)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'percentile_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_curves(v7_results, v8_results, n_sims):
    """Plot cumulative probability curves"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    metrics = [
        ('final_returns', 'Final Return (%)', 'Probability of Return >'),
        ('sharpe_ratios', 'Sharpe Ratio', 'Probability of Sharpe >'),
        ('max_drawdowns', 'Max Drawdown (%)', 'Probability of Drawdown <'),
        ('profit_factors', 'Profit Factor', 'Probability of Profit Factor >')
    ]

    for idx, (metric, xlabel, ylabel) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        v7_data = sorted(v7_results[metric])
        v8_data = sorted(v8_results[metric])

        if 'drawdown' in metric:
            # For drawdowns, reverse (less negative is better)
            v7_probs = [100 * (1 - i/len(v7_data)) for i in range(len(v7_data))]
            v8_probs = [100 * (1 - i/len(v8_data)) for i in range(len(v8_data))]
        else:
            v7_probs = [100 * (1 - i/len(v7_data)) for i in range(len(v7_data))]
            v8_probs = [100 * (1 - i/len(v8_data)) for i in range(len(v8_data))]

        ax.plot(v7_data, v7_probs, color='steelblue', linewidth=2.5,
                label='V7 LSTM+Attention', alpha=0.8)
        ax.plot(v8_data, v8_probs, color='darkorange', linewidth=2.5,
                label='V8 TFT Sliding', alpha=0.8)

        # Add reference lines
        ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='50% Probability')
        ax.axhline(95, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='95% Probability')
        ax.axhline(5, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='5% Probability')

        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{ylabel} (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Cumulative Probability: {xlabel}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    fig.suptitle(f'Cumulative Probability Analysis ({n_sims:,} Simulations)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'probability_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_comprehensive_report(v7_stats, v8_stats, v7_trades, v8_trades, n_sims):
    """Save comprehensive statistical report"""
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE MONTE CARLO ANALYSIS REPORT")
    report.append(f"V7 (LSTM+Attention) vs V8 (TFT Sliding Window)")
    report.append(f"Simulations: {n_sims:,} | Initial Capital: ${INITIAL_CAPITAL:,}")
    report.append("=" * 100)
    report.append("")

    # Actual backtest performance
    v7_actual_return = (v7_trades['pnl'].sum() / INITIAL_CAPITAL) * 100
    v8_actual_return = (v8_trades['pnl'].sum() / INITIAL_CAPITAL) * 100

    report.append("ACTUAL HISTORICAL BACKTEST PERFORMANCE")
    report.append("-" * 100)
    report.append(f"{'Model':<30} {'Trades':<15} {'Total P&L':<20} {'Return %':<15}")
    report.append("-" * 100)
    report.append(f"{'V7 LSTM+Attention':<30} {len(v7_trades):<15,} ${v7_trades['pnl'].sum():<19,.2f} {v7_actual_return:<14.2f}%")
    report.append(f"{'V8 TFT Sliding':<30} {len(v8_trades):<15,} ${v8_trades['pnl'].sum():<19,.2f} {v8_actual_return:<14.2f}%")
    report.append("")

    # Monte Carlo simulation results
    metrics_display = [
        ('final_returns', 'FINAL RETURN (%)'),
        ('sharpe_ratios', 'SHARPE RATIO'),
        ('sortino_ratios', 'SORTINO RATIO'),
        ('max_drawdowns', 'MAX DRAWDOWN (%)'),
        ('win_rates', 'WIN RATE (%)'),
        ('profit_factors', 'PROFIT FACTOR'),
        ('avg_win', 'AVERAGE WIN ($)'),
        ('avg_loss', 'AVERAGE LOSS ($)'),
        ('max_win', 'MAXIMUM WIN ($)'),
        ('max_loss', 'MAXIMUM LOSS ($)')
    ]

    for metric_name, display_name in metrics_display:
        report.append(f"\n{display_name}")
        report.append("-" * 100)

        v7_m = v7_stats[metric_name]
        v8_m = v8_stats[metric_name]

        report.append(f"{'Statistic':<20} {'V7 LSTM+Attention':<25} {'V8 TFT Sliding':<25} {'V8 Advantage':<25}")
        report.append("-" * 100)

        stat_keys = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75',
                     'ci_90_lower', 'ci_90_upper', 'ci_95_lower', 'ci_95_upper',
                     'ci_99_lower', 'ci_99_upper', 'iqr', 'skewness', 'kurtosis']

        for stat in stat_keys:
            v7_val = v7_m[stat]
            v8_val = v8_m[stat]

            # Calculate advantage (for drawdown, lower is better)
            if 'drawdown' in metric_name or 'loss' in metric_name:
                advantage = v7_val - v8_val  # Less negative/loss is better
            else:
                advantage = v8_val - v7_val

            report.append(f"{stat.replace('_', ' ').title():<20} {v7_val:>24.4f}  {v8_val:>24.4f}  {advantage:>+24.4f}")

        # Special metrics
        if metric_name == 'final_returns':
            report.append(f"{'Probability of Loss':<20} {v7_m['prob_negative']:>24.2f}% {v8_m['prob_negative']:>24.2f}%")
            report.append(f"{'Prob Loss > 50%':<20} {v7_m['prob_loss_50pct']:>24.2f}% {v8_m['prob_loss_50pct']:>24.2f}%")
            report.append(f"{'Prob Gain > 100%':<20} {v7_m['prob_gain_100pct']:>24.2f}% {v8_m['prob_gain_100pct']:>24.2f}%")

    report.append("\n" + "=" * 100)
    report.append("KEY INSIGHTS")
    report.append("=" * 100)

    return_advantage = v8_stats['final_returns']['median'] - v7_stats['final_returns']['median']
    sharpe_advantage = v8_stats['sharpe_ratios']['median'] - v7_stats['sharpe_ratios']['median']
    dd_advantage = v7_stats['max_drawdowns']['median'] - v8_stats['max_drawdowns']['median']

    report.append(f"\n1. Return Performance:")
    report.append(f"   - V8 delivers {return_advantage:+.2f}% higher median returns than V7")
    report.append(f"   - V8 has {v8_stats['final_returns']['prob_negative']:.2f}% probability of loss vs V7's {v7_stats['final_returns']['prob_negative']:.2f}%")
    report.append(f"   - V8 has {v8_stats['final_returns']['prob_gain_100pct']:.2f}% chance of >100% gains vs V7's {v7_stats['final_returns']['prob_gain_100pct']:.2f}%")

    report.append(f"\n2. Risk-Adjusted Performance:")
    report.append(f"   - V8 Sharpe Ratio is {sharpe_advantage:+.2f} higher ({v8_stats['sharpe_ratios']['median']:.2f} vs {v7_stats['sharpe_ratios']['median']:.2f})")
    report.append(f"   - V8 has {dd_advantage:+.2f}% better median max drawdown ({v8_stats['max_drawdowns']['median']:.2f}% vs {v7_stats['max_drawdowns']['median']:.2f}%)")

    report.append(f"\n3. Consistency:")
    report.append(f"   - V8 Return StdDev: {v8_stats['final_returns']['std']:.2f}% vs V7: {v7_stats['final_returns']['std']:.2f}%")
    report.append(f"   - V8 Return IQR: {v8_stats['final_returns']['iqr']:.2f}% vs V7: {v7_stats['final_returns']['iqr']:.2f}%")

    report.append(f"\n4. Win/Loss Profile:")
    report.append(f"   - V8 Median Win Rate: {v8_stats['win_rates']['median']:.2f}% vs V7: {v7_stats['win_rates']['median']:.2f}%")
    report.append(f"   - V8 Median Profit Factor: {v8_stats['profit_factors']['median']:.2f} vs V7: {v7_stats['profit_factors']['median']:.2f}")

    report.append("\n" + "=" * 100)

    # Save report
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / 'comprehensive_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)

    # Save detailed CSV
    stats_rows = []
    for metric in ['final_returns', 'sharpe_ratios', 'sortino_ratios', 'max_drawdowns',
                   'win_rates', 'profit_factors', 'avg_win', 'avg_loss', 'max_win', 'max_loss']:
        for model, stats in [('V7_LSTM', v7_stats), ('V8_TFT', v8_stats)]:
            row = {
                'Model': model,
                'Metric': metric,
                'Mean': stats[metric]['mean'],
                'Median': stats[metric]['median'],
                'Std': stats[metric]['std'],
                'Min': stats[metric]['min'],
                'Max': stats[metric]['max'],
                'Q25': stats[metric]['q25'],
                'Q75': stats[metric]['q75'],
                'IQR': stats[metric]['iqr'],
                'CI_90_Lower': stats[metric]['ci_90_lower'],
                'CI_90_Upper': stats[metric]['ci_90_upper'],
                'CI_95_Lower': stats[metric]['ci_95_lower'],
                'CI_95_Upper': stats[metric]['ci_95_upper'],
                'CI_99_Lower': stats[metric]['ci_99_lower'],
                'CI_99_Upper': stats[metric]['ci_99_upper'],
                'Skewness': stats[metric]['skewness'],
                'Kurtosis': stats[metric]['kurtosis']
            }
            stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_DIR / 'comprehensive_statistics.csv', index=False)

def main():
    print("="*100)
    print("COMPREHENSIVE MONTE CARLO ANALYSIS: V7 (LSTM+Attention) vs V8 (TFT Sliding)")
    print("="*100)

    # Load trade data
    print("\n[1/7] Loading trade data...")
    v7_trades = load_trades(V7_TRADES)
    v8_trades = load_trades(V8_TRADES)

    print(f"  - V7 LSTM+Attention: {len(v7_trades):,} trades")
    print(f"  - V8 TFT Sliding:    {len(v8_trades):,} trades")

    # Extract P&L
    v7_pnl = v7_trades['pnl'].values
    v8_pnl = v8_trades['pnl'].values

    # Run Monte Carlo simulations
    print(f"\n[2/7] Running {N_SIMULATIONS:,} Monte Carlo simulations for V7...")
    v7_results = monte_carlo_simulation(v7_pnl, N_SIMULATIONS, INITIAL_CAPITAL)

    print(f"\n[3/7] Running {N_SIMULATIONS:,} Monte Carlo simulations for V8...")
    v8_results = monte_carlo_simulation(v8_pnl, N_SIMULATIONS, INITIAL_CAPITAL)

    # Calculate statistics
    print("\n[4/7] Calculating comprehensive statistics...")
    v7_stats = calculate_statistics(v7_results)
    v8_stats = calculate_statistics(v8_results)

    # Generate visualizations
    print("\n[5/7] Generating comprehensive visualizations...")

    print("  → Comprehensive distributions...")
    plot_comprehensive_distributions(v7_results, v8_results, N_SIMULATIONS)

    print("  → Confidence intervals...")
    plot_confidence_intervals(v7_stats, v8_stats, N_SIMULATIONS)

    print("  → Equity curves...")
    plot_equity_curves(v7_results, v8_results, N_SIMULATIONS)

    print("  → Risk-return scatter...")
    plot_risk_return_scatter(v7_results, v8_results, N_SIMULATIONS)

    print("  → Percentile comparison...")
    plot_percentile_comparison(v7_stats, v8_stats, N_SIMULATIONS)

    print("  → Probability curves...")
    plot_probability_curves(v7_results, v8_results, N_SIMULATIONS)

    # Save comprehensive report
    print("\n[6/7] Saving comprehensive report...")
    save_comprehensive_report(v7_stats, v8_stats, v7_trades, v8_trades, N_SIMULATIONS)

    print("\n[7/7] Analysis complete!")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("="*100)

    print("\nGenerated files:")
    print("  - comprehensive_report.txt")
    print("  - comprehensive_statistics.csv")
    print("  - comprehensive_distributions.png")
    print("  - confidence_intervals_detailed.png")
    print("  - equity_curves_comprehensive.png")
    print("  - risk_return_scatter.png")
    print("  - percentile_comparison.png")
    print("  - probability_curves.png")
    print("="*100)

if __name__ == "__main__":
    main()
