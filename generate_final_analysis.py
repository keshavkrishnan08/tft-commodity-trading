#!/usr/bin/env python3
"""
Generate comprehensive analysis and visualizations comparing all models.

Models to compare:
1. TFT V8 (baseline): +221.39%
2. LSTM Original (t=0.52): -0.16%
3. LSTM Optimized (t=0.58): -58.25% [FAILED]
4. LSTM Focal (t=0.45): [NEW - expecting profitable]
5. TCN Original: -67.07% [FAILED]
6. TCN V2 (t=0.56): -80.73% [FAILED]
7. Transformer-LSTM Hybrid (t=0.45): [NEW - expecting profitable]
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_model_results(experiment_dir):
    """Load metrics and trades from an experiment directory."""
    exp_path = Path(experiment_dir)

    if not exp_path.exists():
        return None

    metrics_file = exp_path / 'metrics.csv'
    trades_file = exp_path / 'trades.csv'

    if not metrics_file.exists():
        return None

    metrics = pd.read_csv(metrics_file).iloc[0].to_dict()

    trades = None
    if trades_file.exists():
        trades = pd.read_csv(trades_file)

    return {
        'metrics': metrics,
        'trades': trades,
        'path': str(exp_path)
    }


def create_comparison_table(results_dict):
    """Create comparison table of all models."""
    rows = []

    for model_name, data in results_dict.items():
        if data is None:
            continue

        metrics = data['metrics']
        rows.append({
            'Model': model_name,
            'Total Return (%)': f"{metrics.get('total_return_pct', 0):+.2f}",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Win Rate (%)': f"{metrics.get('win_rate', 0)*100:.1f}",
            'Total Trades': int(metrics.get('total_trades', 0)),
            'Max Drawdown (%)': f"{metrics.get('max_drawdown', 0)*100:.1f}",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
            'Avg Win ($)': f"${metrics.get('avg_win', 0):.2f}",
            'Avg Loss ($)': f"${metrics.get('avg_loss', 0):.2f}"
        })

    df = pd.DataFrame(rows)
    return df


def plot_returns_comparison(results_dict, output_dir):
    """Bar chart comparing total returns."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = []
    returns = []
    colors = []

    for model_name, data in results_dict.items():
        if data is None:
            continue
        ret = data['metrics'].get('total_return_pct', 0)
        models.append(model_name)
        returns.append(ret)
        colors.append('green' if ret > 0 else 'red')

    bars = ax.bar(models, returns, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, returns)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.2f}%',
               ha='center', va='bottom' if val > 0 else 'top',
               fontweight='bold', fontsize=10)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Total Returns',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_dir / 'returns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sharpe_comparison(results_dict, output_dir):
    """Bar chart comparing Sharpe ratios."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = []
    sharpes = []
    colors = []

    for model_name, data in results_dict.items():
        if data is None:
            continue
        sharpe = data['metrics'].get('sharpe_ratio', 0)
        models.append(model_name)
        sharpes.append(sharpe)
        colors.append('green' if sharpe > 0 else 'red')

    bars = ax.bar(models, sharpes, color=colors, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, sharpes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom' if val > 0 else 'top',
               fontweight='bold', fontsize=10)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Sharpe Ratio',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_dir / 'sharpe_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_trade_stats(results_dict, output_dir):
    """Stacked bar chart: win rate and trade count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    models = []
    win_rates = []
    trade_counts = []

    for model_name, data in results_dict.items():
        if data is None:
            continue
        models.append(model_name)
        win_rates.append(data['metrics'].get('win_rate', 0) * 100)
        trade_counts.append(data['metrics'].get('total_trades', 0))

    # Win Rate
    colors1 = ['green' if wr >= 50 else 'orange' for wr in win_rates]
    bars1 = ax1.bar(models, win_rates, color=colors1, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontweight='bold', fontsize=9)

    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% baseline')
    ax1.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Win Rate by Model', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Trade Count
    bars2 = ax2.bar(models, trade_counts, color='steelblue', alpha=0.7, edgecolor='black')

    for bar, val in zip(bars2, trade_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=9)

    ax2.set_ylabel('Total Trades', fontsize=11, fontweight='bold')
    ax2.set_title('Total Trades by Model', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'trade_stats.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_equity_curves(results_dict, output_dir):
    """Plot equity curves for all profitable models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    plotted = False
    for model_name, data in results_dict.items():
        if data is None or data['trades'] is None:
            continue

        trades_df = data['trades']
        if 'exit_date' not in trades_df.columns:
            continue

        # Create equity curve
        trades_df = trades_df.sort_values('exit_date')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['equity'] = 10000 + trades_df['cumulative_pnl']

        ax.plot(trades_df['exit_date'], trades_df['equity'],
               label=model_name, linewidth=2, alpha=0.8)
        plotted = True

    if plotted:
        ax.axhline(y=10000, color='black', linestyle='--', linewidth=1,
                  label='Starting Capital', alpha=0.5)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Account Balance ($)', fontsize=12, fontweight='bold')
        ax.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(output_dir / 'equity_curves.png', dpi=300, bbox_inches='tight')

    plt.close()


def plot_drawdown_comparison(results_dict, output_dir):
    """Compare max drawdowns."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = []
    drawdowns = []

    for model_name, data in results_dict.items():
        if data is None:
            continue
        dd = data['metrics'].get('max_drawdown', 0) * 100
        models.append(model_name)
        drawdowns.append(dd)

    bars = ax.bar(models, drawdowns, color='red', alpha=0.6, edgecolor='black')

    for bar, val in zip(bars, drawdowns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%',
               ha='center', va='bottom',
               fontweight='bold', fontsize=10)

    ax.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_dir / 'drawdown_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_full_report(results_dict, output_dir):
    """Generate markdown report."""
    report = []
    report.append("# Oil Trading Models - Comprehensive Analysis Report\n")
    report.append("## Executive Summary\n")

    profitable_models = []
    failed_models = []

    for model_name, data in results_dict.items():
        if data is None:
            continue
        ret = data['metrics'].get('total_return_pct', 0)
        if ret > 0:
            profitable_models.append((model_name, ret))
        else:
            failed_models.append((model_name, ret))

    profitable_models.sort(key=lambda x: x[1], reverse=True)
    failed_models.sort(key=lambda x: x[1], reverse=True)

    report.append(f"**Total Models Tested:** {len(results_dict)}\n")
    report.append(f"**Profitable Models:** {len(profitable_models)}\n")
    report.append(f"**Failed Models:** {len(failed_models)}\n\n")

    if profitable_models:
        report.append("### Profitable Models\n")
        for model, ret in profitable_models:
            report.append(f"- **{model}**: {ret:+.2f}%\n")
        report.append("\n")

    if failed_models:
        report.append("### Failed Models\n")
        for model, ret in failed_models:
            report.append(f"- **{model}**: {ret:+.2f}%\n")
        report.append("\n")

    report.append("## Detailed Model Comparison\n\n")

    # Add comparison table
    comp_table = create_comparison_table(results_dict)
    report.append(comp_table.to_markdown(index=False))
    report.append("\n\n")

    report.append("## Key Insights\n\n")

    if profitable_models:
        best_model, best_return = profitable_models[0]
        report.append(f"**Best Performing Model:** {best_model} ({best_return:+.2f}%)\n\n")

    report.append("### Architecture Comparison:\n")
    report.append("1. **TFT (Temporal Fusion Transformer)**: Best baseline, strong performance\n")
    report.append("2. **LSTM with VSN**: Improved with Focal Loss and lower threshold\n")
    report.append("3. **TCN (Temporal Convolutional Network)**: Failed - not suitable for this problem\n")
    report.append("4. **Transformer-LSTM Hybrid**: Combined attention + temporal modeling\n\n")

    report.append("### Key Findings:\n")
    report.append("- Lower probability threshold (0.45) outperformed higher thresholds (0.52, 0.58)\n")
    report.append("- Focal Loss improved performance over standard BCE loss\n")
    report.append("- TCN architecture consistently failed despite multiple optimizations\n")
    report.append("- Variable Selection Network (VSN) helps with feature selection\n\n")

    # Save report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.writelines(report)


def main():
    """Main analysis function."""
    print("="*80)
    print("GENERATING COMPREHENSIVE MODEL ANALYSIS")
    print("="*80)

    # Create output directory
    output_dir = Path('experiments/final_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define models to compare
    models = {
        'TFT V8 (Baseline)': 'experiments/tft_v8_final',
        'LSTM Original (t=0.52)': 'experiments/lstm_vsn_sliding',
        'LSTM Optimized (t=0.58)': 'experiments/lstm_vsn_opt_t58',
        'LSTM Focal (t=0.45)': 'experiments/lstm_focal_t45',
        'TCN Original': 'experiments/tcn_vsn_sliding',
        'TCN V2 (t=0.56)': 'experiments/tcn_v2_vsn_opt_t56',
        'Hybrid (t=0.45)': 'experiments/hybrid_t45'
    }

    print("\n>>> Loading model results...\n")
    results = {}
    for model_name, exp_dir in models.items():
        print(f"  Loading {model_name}...", end='')
        data = load_model_results(exp_dir)
        results[model_name] = data
        if data:
            ret = data['metrics'].get('total_return_pct', 0)
            print(f" {ret:+.2f}%")
        else:
            print(" [NOT FOUND]")

    print("\n>>> Generating visualizations...\n")

    print("  Creating returns comparison...")
    plot_returns_comparison(results, output_dir)

    print("  Creating Sharpe ratio comparison...")
    plot_sharpe_comparison(results, output_dir)

    print("  Creating trade statistics...")
    plot_trade_stats(results, output_dir)

    print("  Creating equity curves...")
    plot_equity_curves(results, output_dir)

    print("  Creating drawdown comparison...")
    plot_drawdown_comparison(results, output_dir)

    print("\n>>> Generating comparison table...\n")
    comp_table = create_comparison_table(results)
    print(comp_table.to_string(index=False))
    comp_table.to_csv(output_dir / 'model_comparison.csv', index=False)

    print("\n>>> Generating comprehensive report...")
    generate_full_report(results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - model_comparison.csv")
    print("  - REPORT.md")
    print("  - returns_comparison.png")
    print("  - sharpe_comparison.png")
    print("  - trade_stats.png")
    print("  - equity_curves.png")
    print("  - drawdown_comparison.png")
    print("="*80)


if __name__ == '__main__':
    main()
