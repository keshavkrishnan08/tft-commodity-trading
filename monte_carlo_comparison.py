"""
Monte Carlo Comparative Analysis for V7 (LSTM+Attention) vs V8 (TFT Sliding)

This script performs bootstrap resampling on the trade sequences from both models
to evaluate statistical robustness, confidence intervals, and risk metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Paths
V7_TRADES = Path("experiments/sliding_window_v7/trades.csv")
V8_TRADES = Path("experiments/tft_v8_sliding/trades.csv")
OUTPUT_DIR = Path("experiments/monte_carlo_comparison")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Monte Carlo parameters
N_SIMULATIONS = 25000
INITIAL_CAPITAL = 10000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_trades(path):
    """Load trade data and extract P&L values"""
    df = pd.read_csv(path)

    # Extract P&L column
    if 'pnl' not in df.columns:
        raise ValueError(f"Cannot find 'pnl' column in {path}")

    return df

def monte_carlo_simulation(pnl_values, n_simulations=N_SIMULATIONS, initial_capital=INITIAL_CAPITAL):
    """
    Perform Monte Carlo simulation by bootstrap resampling trade P&L sequences

    CORRECTED METHOD: Bootstrap resample P&L values and build equity curve
    by adding them sequentially, NOT by compounding percentage returns.

    Args:
        pnl_values: Array of trade P&L values (dollar amounts)
        n_simulations: Number of simulation runs
        initial_capital: Starting capital

    Returns:
        dict with simulation results
    """
    results = {
        'final_returns': [],
        'max_drawdowns': [],
        'sharpe_ratios': [],
        'win_rates': [],
        'equity_curves': []
    }

    n_trades = len(pnl_values)

    for i in tqdm(range(n_simulations), desc="Running simulations"):
        # Bootstrap resample: sample P&L values with replacement
        sampled_pnl = np.random.choice(pnl_values, size=n_trades, replace=True)

        # Build equity curve by adding P&L sequentially
        equity_curve = np.zeros(n_trades + 1)
        equity_curve[0] = initial_capital

        for j in range(n_trades):
            equity_curve[j + 1] = equity_curve[j] + sampled_pnl[j]

        # Calculate metrics
        final_equity = equity_curve[-1]
        final_return = (final_equity / initial_capital - 1) * 100

        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min() * 100

        # Sharpe ratio - calculate per-trade returns for Sharpe
        trade_returns = sampled_pnl / initial_capital  # Simple approximation
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            avg_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            # Annualize assuming ~250 trades per year (rough estimate)
            sharpe = (avg_return / std_return * np.sqrt(250)) if std_return > 0 else 0
        else:
            sharpe = 0

        # Win rate
        win_rate = (sampled_pnl > 0).sum() / len(sampled_pnl) * 100

        results['final_returns'].append(final_return)
        results['max_drawdowns'].append(max_dd)
        results['sharpe_ratios'].append(sharpe)
        results['win_rates'].append(win_rate)

        # Store a sample of equity curves for visualization
        if i < 100:
            results['equity_curves'].append(equity_curve)

    return results

def calculate_statistics(results):
    """Calculate summary statistics from simulation results"""
    stats = {}
    
    for metric in ['final_returns', 'max_drawdowns', 'sharpe_ratios', 'win_rates']:
        data = np.array(results[metric])
        stats[metric] = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'ci_95_lower': np.percentile(data, 2.5),
            'ci_95_upper': np.percentile(data, 97.5),
            'ci_99_lower': np.percentile(data, 0.5),
            'ci_99_upper': np.percentile(data, 99.5),
            'min': np.min(data),
            'max': np.max(data),
            'prob_negative': (data < 0).sum() / len(data) * 100 if metric == 'final_returns' else None
        }
    
    return stats

def plot_comparative_distributions(v7_results, v8_results):
    """Create comparative distribution plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('final_returns', 'Final Return (%)', 'Distribution of Final Returns'),
        ('sharpe_ratios', 'Sharpe Ratio', 'Distribution of Sharpe Ratios'),
        ('max_drawdowns', 'Max Drawdown (%)', 'Distribution of Max Drawdowns'),
        ('win_rates', 'Win Rate (%)', 'Distribution of Win Rates')
    ]
    
    for idx, (metric, xlabel, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        v7_data = v7_results[metric]
        v8_data = v8_results[metric]
        
        # Plot histograms
        ax.hist(v7_data, bins=50, alpha=0.6, label='V7 LSTM+Attention', color='steelblue', density=True)
        ax.hist(v8_data, bins=50, alpha=0.6, label='V8 TFT Sliding', color='darkorange', density=True)
        
        # Add median lines
        ax.axvline(np.median(v7_data), color='steelblue', linestyle='--', linewidth=2, label=f'V7 Median: {np.median(v7_data):.2f}')
        ax.axvline(np.median(v8_data), color='darkorange', linestyle='--', linewidth=2, label=f'V8 Median: {np.median(v8_data):.2f}')
        
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparative_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_intervals(v7_stats, v8_stats):
    """Plot confidence interval comparisons"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics = ['final_returns', 'sharpe_ratios']
    titles = ['Final Return % (95% CI)', 'Sharpe Ratio (95% CI)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        models = ['V7\nLSTM+Attention', 'V8\nTFT Sliding']
        means = [v7_stats[metric]['mean'], v8_stats[metric]['mean']]
        ci_lower = [v7_stats[metric]['ci_95_lower'], v8_stats[metric]['ci_95_lower']]
        ci_upper = [v7_stats[metric]['ci_95_upper'], v8_stats[metric]['ci_95_upper']]
        
        x_pos = np.arange(len(models))
        
        # Plot bars
        bars = ax.bar(x_pos, means, color=['steelblue', 'darkorange'], alpha=0.7, width=0.6)
        
        # Plot error bars for 95% CI
        errors = [[means[i] - ci_lower[i] for i in range(2)],
                  [ci_upper[i] - means[i] for i in range(2)]]
        ax.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', 
                    capsize=10, capthick=2, linewidth=2)
        
        # Add value labels
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, mean, 
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_equity_curves(v7_results, v8_results):
    """Plot sample equity curves from simulations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # V7 equity curves
    for curve in v7_results['equity_curves'][:50]:
        ax1.plot(curve, alpha=0.3, color='steelblue', linewidth=0.5)
    ax1.set_title('V7 LSTM+Attention: Sample Equity Curves (50 runs)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax1.axhline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # V8 equity curves
    for curve in v8_results['equity_curves'][:50]:
        ax2.plot(curve, alpha=0.3, color='darkorange', linewidth=0.5)
    ax2.set_title('V8 TFT Sliding: Sample Equity Curves (50 runs)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax2.axhline(INITIAL_CAPITAL, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_equity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_statistics_report(v7_stats, v8_stats, v7_trades, v8_trades):
    """Save comprehensive statistics report"""
    report = []
    report.append("=" * 80)
    report.append("MONTE CARLO COMPARATIVE ANALYSIS REPORT")
    report.append(f"V7 (LSTM+Attention) vs V8 (TFT Sliding Window)")
    report.append(f"Simulations: {N_SIMULATIONS:,} | Initial Capital: ${INITIAL_CAPITAL:,}")
    report.append("=" * 80)
    report.append("")
    
    # Model summaries
    report.append("ACTUAL HISTORICAL PERFORMANCE")
    report.append("-" * 80)
    report.append(f"V7 LSTM+Attention: {len(v7_trades):,} trades")
    report.append(f"V8 TFT Sliding:    {len(v8_trades):,} trades")
    report.append("")
    
    # Simulation results
    for metric_name, display_name in [
        ('final_returns', 'Final Return (%)'),
        ('sharpe_ratios', 'Sharpe Ratio'),
        ('max_drawdowns', 'Max Drawdown (%)'),
        ('win_rates', 'Win Rate (%)')
    ]:
        report.append(f"\n{display_name.upper()}")
        report.append("-" * 80)
        
        v7_m = v7_stats[metric_name]
        v8_m = v8_stats[metric_name]
        
        report.append(f"{'Metric':<20} {'V7 LSTM':<20} {'V8 TFT':<20} {'V8 Advantage':<20}")
        report.append("-" * 80)
        
        for stat in ['mean', 'median', 'std', 'ci_95_lower', 'ci_95_upper', 'min', 'max']:
            v7_val = v7_m[stat]
            v8_val = v8_m[stat]
            advantage = v8_val - v7_val if stat != 'max_drawdowns' else v7_val - v8_val  # Lower DD is better
            
            report.append(f"{stat.replace('_', ' ').title():<20} {v7_val:>18.2f}  {v8_val:>18.2f}  {advantage:>+18.2f}")
        
        if metric_name == 'final_returns':
            report.append(f"{'Probability of Loss':<20} {v7_m['prob_negative']:>18.2f}% {v8_m['prob_negative']:>18.2f}%")
    
    report.append("\n" + "=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / 'monte_carlo_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    # Save CSV statistics
    stats_df = pd.DataFrame({
        'Model': ['V7_LSTM', 'V8_TFT'],
        'N_Trades': [len(v7_trades), len(v8_trades)],
        'Mean_Return': [v7_stats['final_returns']['mean'], v8_stats['final_returns']['mean']],
        'Median_Return': [v7_stats['final_returns']['median'], v8_stats['final_returns']['median']],
        'Return_95CI_Lower': [v7_stats['final_returns']['ci_95_lower'], v8_stats['final_returns']['ci_95_lower']],
        'Return_95CI_Upper': [v7_stats['final_returns']['ci_95_upper'], v8_stats['final_returns']['ci_95_upper']],
        'Mean_Sharpe': [v7_stats['sharpe_ratios']['mean'], v8_stats['sharpe_ratios']['mean']],
        'Median_Sharpe': [v7_stats['sharpe_ratios']['median'], v8_stats['sharpe_ratios']['median']],
        'Mean_MaxDD': [v7_stats['max_drawdowns']['mean'], v8_stats['max_drawdowns']['mean']],
        'Median_MaxDD': [v7_stats['max_drawdowns']['median'], v8_stats['max_drawdowns']['median']],
        'Mean_WinRate': [v7_stats['win_rates']['mean'], v8_stats['win_rates']['mean']],
        'Prob_of_Loss': [v7_stats['final_returns']['prob_negative'], v8_stats['final_returns']['prob_negative']]
    })
    stats_df.to_csv(OUTPUT_DIR / 'monte_carlo_stats.csv', index=False)

def main():
    print("="*80)
    print("MONTE CARLO COMPARATIVE ANALYSIS: V7 vs V8")
    print("="*80)
    
    # Load trade data
    print("\n[1/6] Loading trade data...")
    v7_trades = load_trades(V7_TRADES)
    v8_trades = load_trades(V8_TRADES)
    
    print(f"  - V7 LSTM+Attention: {len(v7_trades):,} trades")
    print(f"  - V8 TFT Sliding:    {len(v8_trades):,} trades")
    
    # Extract P&L values
    v7_pnl = v7_trades['pnl'].values
    v8_pnl = v8_trades['pnl'].values

    # Run Monte Carlo simulations
    print(f"\n[2/6] Running {N_SIMULATIONS:,} Monte Carlo simulations for V7...")
    v7_results = monte_carlo_simulation(v7_pnl, N_SIMULATIONS, INITIAL_CAPITAL)

    print(f"\n[3/6] Running {N_SIMULATIONS:,} Monte Carlo simulations for V8...")
    v8_results = monte_carlo_simulation(v8_pnl, N_SIMULATIONS, INITIAL_CAPITAL)
    
    # Calculate statistics
    print("\n[4/6] Calculating statistics...")
    v7_stats = calculate_statistics(v7_results)
    v8_stats = calculate_statistics(v8_results)
    
    # Generate plots
    print("\n[5/6] Generating comparative visualizations...")
    plot_comparative_distributions(v7_results, v8_results)
    print(f"  ✓ Saved: comparative_distributions.png")
    
    plot_confidence_intervals(v7_stats, v8_stats)
    print(f"  ✓ Saved: confidence_intervals.png")
    
    plot_sample_equity_curves(v7_results, v8_results)
    print(f"  ✓ Saved: sample_equity_curves.png")
    
    # Save report
    print("\n[6/6] Saving statistical report...")
    save_statistics_report(v7_stats, v8_stats, v7_trades, v8_trades)
    print(f"  ✓ Saved: monte_carlo_report.txt")
    print(f"  ✓ Saved: monte_carlo_stats.csv")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE! Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
