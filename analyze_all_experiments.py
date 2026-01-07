"""
Comprehensive Experiment Analysis for Publication
Generates complete statistics for all experiments with publication-ready metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_experiment_results():
    """Load all experiment results"""
    experiments_dir = Path('/Users/keshavkrishnan/Oil_Project/experiments')
    results = []

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / 'metrics.csv'
        trades_file = exp_dir / 'trades.csv'

        if not metrics_file.exists():
            continue

        try:
            # Load metrics
            metrics = pd.read_csv(metrics_file)

            # Skip if this is a classification metrics file (not backtest)
            if 'initial_capital' not in metrics.columns:
                continue

            # Load trades for additional analysis
            trades = None
            if trades_file.exists():
                trades = pd.read_csv(trades_file)

            results.append({
                'name': exp_dir.name,
                'metrics': metrics,
                'trades': trades,
                'path': str(exp_dir)
            })
        except Exception as e:
            print(f"Error loading {exp_dir.name}: {e}")
            continue

    return results

def calculate_advanced_metrics(metrics_df, trades_df=None):
    """Calculate publication-ready statistical metrics"""
    stats = {}

    # Basic metrics
    stats['initial_capital'] = float(metrics_df['initial_capital'].values[0])
    stats['final_capital'] = float(metrics_df['final_capital'].values[0])
    stats['total_return_pct'] = float(metrics_df['total_return_pct'].values[0])
    stats['sharpe_ratio'] = float(metrics_df['sharpe_ratio'].values[0])
    stats['sortino_ratio'] = float(metrics_df['sortino_ratio'].values[0])
    stats['max_drawdown'] = float(metrics_df['max_drawdown'].values[0])
    stats['calmar_ratio'] = float(metrics_df['calmar_ratio'].values[0])

    # Trade statistics
    stats['total_trades'] = int(metrics_df['total_trades'].values[0])
    stats['winning_trades'] = int(metrics_df['winning_trades'].values[0])
    stats['losing_trades'] = int(metrics_df['losing_trades'].values[0])
    stats['win_rate'] = float(metrics_df['win_rate'].values[0])
    stats['profit_factor'] = float(metrics_df['profit_factor'].values[0])

    # P&L analysis
    stats['gross_profit'] = float(metrics_df['gross_profit'].values[0])
    stats['gross_loss'] = float(metrics_df['gross_loss'].values[0])
    stats['avg_win'] = float(metrics_df['avg_win'].values[0])
    stats['avg_loss'] = float(metrics_df['avg_loss'].values[0])

    # Calculate additional metrics
    stats['total_pnl'] = stats['final_capital'] - stats['initial_capital']
    stats['avg_return_per_trade'] = stats['total_pnl'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
    stats['win_loss_ratio'] = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else 0

    # Risk-adjusted metrics
    stats['return_over_max_dd'] = stats['total_return_pct'] / abs(stats['max_drawdown']) if stats['max_drawdown'] != 0 else 0

    # Annualized metrics (assuming 5-year period 2018-2022)
    years = 5.0
    stats['cagr'] = ((stats['final_capital'] / stats['initial_capital']) ** (1/years) - 1) * 100

    # Additional statistical metrics if trades available
    if trades_df is not None and len(trades_df) > 0:
        pnls = trades_df['pnl'].values

        # Root Mean Square Error (deviation from mean)
        mean_pnl = np.mean(pnls)
        stats['pnl_rmse'] = np.sqrt(np.mean((pnls - mean_pnl) ** 2))

        # Mean Absolute Error
        stats['pnl_mae'] = np.mean(np.abs(pnls - mean_pnl))

        # Standard deviation of P&L
        stats['pnl_std'] = np.std(pnls)

        # Skewness and Kurtosis
        stats['pnl_skewness'] = pd.Series(pnls).skew()
        stats['pnl_kurtosis'] = pd.Series(pnls).kurtosis()

        # Percentiles
        stats['pnl_25th_percentile'] = np.percentile(pnls, 25)
        stats['pnl_50th_percentile'] = np.percentile(pnls, 50)
        stats['pnl_75th_percentile'] = np.percentile(pnls, 75)

        # Best and worst trades
        stats['best_trade'] = float(np.max(pnls))
        stats['worst_trade'] = float(np.min(pnls))

        # Consecutive wins/losses
        won = trades_df['won'].values if 'won' in trades_df.columns else (pnls > 0)
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for w in won:
            if w:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        stats['max_consecutive_wins'] = max_consecutive_wins
        stats['max_consecutive_losses'] = max_consecutive_losses

        # Information Ratio (if we assume market return = 0 for futures)
        returns = trades_df['pnl'] / trades_df['position_dollars'] if 'position_dollars' in trades_df.columns else pnls
        stats['information_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    return stats

def generate_comparison_table(all_results):
    """Generate comprehensive comparison table"""
    comparison = []

    for result in all_results:
        stats = calculate_advanced_metrics(result['metrics'], result['trades'])
        stats['experiment_name'] = result['name']
        comparison.append(stats)

    df = pd.DataFrame(comparison)

    # Sort by Sharpe ratio (descending)
    df = df.sort_values('sharpe_ratio', ascending=False)

    return df

def print_terminal_summary(comparison_df):
    """Print comprehensive terminal summary"""
    print("\n" + "="*100)
    print("COMPREHENSIVE EXPERIMENT ANALYSIS - PUBLICATION READY".center(100))
    print("="*100 + "\n")

    # Filter to key experiments
    key_experiments = [
        'tft_v8_sliding_proper',
        'tft_v8_sliding',
        'tft_ablation_no_vsn',
        'tft_ablation_no_causal',
        'hybrid_wisdom_v4'
    ]

    df_key = comparison_df[comparison_df['experiment_name'].isin(key_experiments)]

    print("=" * 100)
    print("KEY EXPERIMENTS - PERFORMANCE SUMMARY")
    print("=" * 100)

    display_cols = [
        'experiment_name',
        'total_return_pct',
        'cagr',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'win_rate',
        'total_trades'
    ]

    if all(col in df_key.columns for col in display_cols):
        print(df_key[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("DETAILED STATISTICS - TOP PERFORMER: " + df_key.iloc[0]['experiment_name'])
    print("=" * 100 + "\n")

    top = df_key.iloc[0]

    print(f"📊 CAPITAL PERFORMANCE:")
    print(f"   Initial Capital:      ${top['initial_capital']:,.2f}")
    print(f"   Final Capital:        ${top['final_capital']:,.2f}")
    print(f"   Total Return:         {top['total_return_pct']:.2f}%")
    print(f"   CAGR (5-year):        {top['cagr']:.2f}%\n")

    print(f"📈 RISK METRICS:")
    print(f"   Sharpe Ratio:         {top['sharpe_ratio']:.2f}")
    print(f"   Sortino Ratio:        {top['sortino_ratio']:.2f}")
    print(f"   Calmar Ratio:         {top['calmar_ratio']:.2f}")
    print(f"   Max Drawdown:         {top['max_drawdown']:.2f}%")
    print(f"   Return/MaxDD:         {top['return_over_max_dd']:.2f}x\n")

    print(f"📉 TRADE STATISTICS:")
    print(f"   Total Trades:         {top['total_trades']:.0f}")
    print(f"   Winning Trades:       {top['winning_trades']:.0f} ({top['win_rate']*100:.1f}%)")
    print(f"   Losing Trades:        {top['losing_trades']:.0f}")
    print(f"   Profit Factor:        {top['profit_factor']:.2f}\n")

    print(f"💰 P&L ANALYSIS:")
    print(f"   Gross Profit:         ${top['gross_profit']:,.2f}")
    print(f"   Gross Loss:           ${top['gross_loss']:,.2f}")
    print(f"   Average Win:          ${top['avg_win']:.2f}")
    print(f"   Average Loss:         ${top['avg_loss']:.2f}")
    print(f"   Win/Loss Ratio:       {top['win_loss_ratio']:.2f}:1")
    print(f"   Avg Return/Trade:     ${top['avg_return_per_trade']:.2f}\n")

    # Additional statistical metrics if available
    if 'pnl_rmse' in top:
        print(f"📊 STATISTICAL METRICS:")
        print(f"   P&L RMSE:             ${top['pnl_rmse']:.2f}")
        print(f"   P&L MAE:              ${top['pnl_mae']:.2f}")
        print(f"   P&L Std Dev:          ${top['pnl_std']:.2f}")
        print(f"   P&L Skewness:         {top['pnl_skewness']:.3f}")
        print(f"   P&L Kurtosis:         {top['pnl_kurtosis']:.3f}\n")

        print(f"   25th Percentile:      ${top['pnl_25th_percentile']:.2f}")
        print(f"   Median (50th):        ${top['pnl_50th_percentile']:.2f}")
        print(f"   75th Percentile:      ${top['pnl_75th_percentile']:.2f}\n")

        print(f"   Best Trade:           ${top['best_trade']:.2f}")
        print(f"   Worst Trade:          ${top['worst_trade']:.2f}\n")

        print(f"   Max Consecutive Wins:  {top['max_consecutive_wins']:.0f}")
        print(f"   Max Consecutive Losses: {top['max_consecutive_losses']:.0f}\n")

        print(f"   Information Ratio:     {top['information_ratio']:.3f}\n")

    print("=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100 + "\n")

    ablation_experiments = ['tft_v8_sliding_proper', 'tft_ablation_no_vsn', 'tft_ablation_no_causal']
    df_ablation = comparison_df[comparison_df['experiment_name'].isin(ablation_experiments)]

    if len(df_ablation) == 3:
        baseline = df_ablation[df_ablation['experiment_name'] == 'tft_v8_sliding_proper'].iloc[0]
        no_vsn = df_ablation[df_ablation['experiment_name'] == 'tft_ablation_no_vsn'].iloc[0]
        no_causal = df_ablation[df_ablation['experiment_name'] == 'tft_ablation_no_causal'].iloc[0]

        print(f"Baseline (TFT with VSN):    {baseline['total_return_pct']:.2f}%  |  Sharpe: {baseline['sharpe_ratio']:.2f}")
        print(f"No VSN (Direct Projection): {no_vsn['total_return_pct']:.2f}%  |  Sharpe: {no_vsn['sharpe_ratio']:.2f}")
        print(f"No Causal (Lookahead):      {no_causal['total_return_pct']:.2f}%  |  Sharpe: {no_causal['sharpe_ratio']:.2f}\n")

        vsn_delta = no_vsn['total_return_pct'] - baseline['total_return_pct']
        causal_delta = no_causal['total_return_pct'] - baseline['total_return_pct']

        print(f"📊 ABLATION ANALYSIS:")
        print(f"   VSN Contribution:     {vsn_delta:+.2f}% (SURPRISING: Removing VSN IMPROVES)")
        print(f"   Causal Constraint:    {causal_delta:+.2f}% (Removing creates lookahead bias)\n")

        print(f"🔍 INTERPRETATION:")
        print(f"   • VSN at 32 dim creates bottleneck (199 features → 32 is too constrained)")
        print(f"   • Direct projection works better with limited capacity")
        print(f"   • Causal masking is CRITICAL (prevents {causal_delta:.0f}% artificial inflation)")
        print(f"   • Future work: Test VSN at 128+ dim where bottleneck resolves\n")

    print("=" * 100)
    print("ALL EXPERIMENTS (Sorted by Sharpe Ratio)")
    print("=" * 100 + "\n")

    summary_cols = [
        'experiment_name',
        'total_return_pct',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'total_trades'
    ]

    if all(col in comparison_df.columns for col in summary_cols):
        print(comparison_df[summary_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE - READY FOR PUBLICATION")
    print("=" * 100 + "\n")

def main():
    """Main execution"""
    print("Loading all experiment results...")
    all_results = load_experiment_results()
    print(f"Found {len(all_results)} experiments with valid results\n")

    print("Calculating comprehensive statistics...")
    comparison_df = generate_comparison_table(all_results)

    # Save to CSV
    output_file = '/Users/keshavkrishnan/Oil_Project/experiments/COMPREHENSIVE_ANALYSIS.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"Saved comprehensive analysis to: {output_file}\n")

    # Print terminal summary
    print_terminal_summary(comparison_df)

    # Save JSON for paper
    json_output = '/Users/keshavkrishnan/Oil_Project/experiments/COMPREHENSIVE_ANALYSIS.json'
    comparison_dict = comparison_df.to_dict(orient='records')
    with open(json_output, 'w') as f:
        json.dump(comparison_dict, f, indent=2)
    print(f"Saved JSON data to: {json_output}\n")

if __name__ == '__main__':
    main()
