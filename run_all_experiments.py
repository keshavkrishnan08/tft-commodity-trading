# run_all_experiments.py
"""
Master Script: Run All TFT Experiments and Generate Comparison Report
- Runs all variants (different hidden dims, ablations)
- Collects results
- Generates comparison graphs
- Creates summary report
"""
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

EXPERIMENTS = {
    'TFT Expanding 32 dim': 'main_tft_v8_expanding_dim32.py',
    'TFT Expanding 64 dim': 'main_tft_v8_expanding_dim64.py',
    'TFT Expanding 128 dim': 'main_tft_v8_expanding_dim128.py',
}

RESULTS_DIR = Path('experiments/comparison_results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_experiment(name, script):
    """Run a single experiment and return results"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {name}")
    print(f"Script: {script}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            ['python', script],
            timeout=7200  # 2 hour timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {name} completed in {elapsed_time/60:.1f} minutes\n")
            return {
                'name': name,
                'status': 'success',
                'time': elapsed_time
            }
        else:
            print(f"\n❌ {name} failed!\n")
            return {
                'name': name,
                'status': 'failed',
                'time': elapsed_time
            }
    except subprocess.TimeoutExpired:
        print(f"⏱️  {name} timed out after 2 hours")
        return {
            'name': name,
            'status': 'timeout',
            'time': 7200
        }
    except Exception as e:
        print(f"\n❌ {name} crashed: {str(e)}\n")
        return {
            'name': name,
            'status': 'crashed',
            'time': time.time() - start_time
        }

def collect_results():
    """Collect results from all experiments"""
    results = []

    exp_map = {
        'TFT Expanding 32 dim': 'experiments/tft_v8_expanding_dim32/metrics.csv',
        'TFT Expanding 64 dim': 'experiments/tft_v8_expanding_dim64/metrics.csv',
        'TFT Expanding 128 dim': 'experiments/tft_v8_expanding_dim128/metrics.csv',
    }

    for name, metrics_file in exp_map.items():
        metrics_path = Path(metrics_file)
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            row = df.iloc[0].to_dict()
            row['experiment'] = name
            row['hidden_dim'] = int(name.split()[-2])
            results.append(row)
        else:
            print(f"⚠️  No results found for {name}")

    return pd.DataFrame(results)

def create_comparison_graphs(df):
    """Generate comparison visualizations"""

    # 1. Return vs Hidden Dimension
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Total Return
    ax = axes[0, 0]
    bars = ax.bar(df['experiment'], df['total_return_pct'], color=['#2E86AB', '#A23B72', '#F18F01'])
    ax.set_title('Total Return by Hidden Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, df['total_return_pct'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Plot 2: Sharpe Ratio
    ax = axes[0, 1]
    bars = ax.bar(df['experiment'], df['sharpe_ratio'], color=['#2E86AB', '#A23B72', '#F18F01'])
    ax.set_title('Sharpe Ratio by Hidden Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, df['sharpe_ratio']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Plot 3: Win Rate
    ax = axes[1, 0]
    bars = ax.bar(df['experiment'], df['win_rate'] * 100, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax.set_title('Win Rate by Hidden Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.legend()
    for bar, val in zip(bars, df['win_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Plot 4: Trade Count
    ax = axes[1, 1]
    bars = ax.bar(df['experiment'], df['total_trades'], color=['#2E86AB', '#A23B72', '#F18F01'])
    ax.set_title('Total Trades by Hidden Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Trades', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, df['total_trades']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'comparison_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {RESULTS_DIR / 'comparison_metrics.png'}")

    # 2. Hidden Dim Scaling Analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Return vs Hidden Dim
    ax = axes[0]
    ax.plot(df['hidden_dim'], df['total_return_pct'], 'o-', linewidth=3, markersize=12, color='#2E86AB')
    ax.set_title('Return Scaling with Hidden Dimension', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hidden Dimension', fontsize=12)
    ax.set_ylabel('Total Return (%)', fontsize=12)
    ax.grid(alpha=0.3)
    for x, y in zip(df['hidden_dim'], df['total_return_pct']):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11)

    # Sharpe vs Hidden Dim
    ax = axes[1]
    ax.plot(df['hidden_dim'], df['sharpe_ratio'], 'o-', linewidth=3, markersize=12, color='#A23B72')
    ax.set_title('Sharpe Scaling with Hidden Dimension', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hidden Dimension', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.grid(alpha=0.3)
    for x, y in zip(df['hidden_dim'], df['sharpe_ratio']):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'hidden_dim_scaling.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {RESULTS_DIR / 'hidden_dim_scaling.png'}")

    # 3. Detailed Metrics Table
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['experiment'],
            f"{row['total_return_pct']:.2f}%",
            f"{row['sharpe_ratio']:.2f}",
            f"{row['sortino_ratio']:.2f}",
            f"{row['max_drawdown']:.2f}%",
            f"{row['win_rate']*100:.1f}%",
            f"{int(row['total_trades'])}"
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['Experiment', 'Return', 'Sharpe', 'Sortino', 'Max DD', 'Win Rate', 'Trades'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(7):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    colors = ['#E8F4F8', '#FFF5E1', '#FFE6E6']
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            table[(i, j)].set_facecolor(colors[i-1])

    plt.savefig(RESULTS_DIR / 'detailed_metrics_table.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {RESULTS_DIR / 'detailed_metrics_table.png'}")

def generate_summary_report(df, run_results):
    """Generate text summary report"""

    report = []
    report.append("="*80)
    report.append("TFT HIDDEN DIMENSION COMPARISON - FINAL REPORT")
    report.append("="*80)
    report.append("")

    # Execution Summary
    report.append("EXECUTION SUMMARY:")
    report.append("-" * 80)
    for result in run_results:
        status_emoji = {'success': '✅', 'failed': '❌', 'timeout': '⏱️', 'crashed': '💥'}.get(result['status'], '❓')
        report.append(f"{status_emoji} {result['name']}: {result['status']} ({result['time']/60:.1f} min)")
    report.append("")

    # Performance Ranking
    report.append("PERFORMANCE RANKING (by Total Return):")
    report.append("-" * 80)
    df_sorted = df.sort_values('total_return_pct', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        report.append(f"{i}. {row['experiment']}")
        report.append(f"   Return: {row['total_return_pct']:+.2f}% | Sharpe: {row['sharpe_ratio']:.2f} | Win Rate: {row['win_rate']*100:.1f}% | Trades: {int(row['total_trades'])}")
    report.append("")

    # VSN Analysis
    report.append("VSN CAPACITY ANALYSIS:")
    report.append("-" * 80)
    if len(df) >= 2:
        best_idx = df['total_return_pct'].idxmax()
        worst_idx = df['total_return_pct'].idxmin()
        best = df.loc[best_idx]
        worst = df.loc[worst_idx]

        report.append(f"Best:  {best['experiment']} → {best['total_return_pct']:.2f}%")
        report.append(f"Worst: {worst['experiment']} → {worst['total_return_pct']:.2f}%")
        report.append(f"Delta: {best['total_return_pct'] - worst['total_return_pct']:.2f}% improvement")
        report.append("")

        if best['hidden_dim'] > worst['hidden_dim']:
            report.append(f"✅ CONCLUSION: VSN benefits from larger hidden dimensions!")
            report.append(f"   {best['hidden_dim']} dim improves over {worst['hidden_dim']} dim by {best['total_return_pct'] - worst['total_return_pct']:.2f}%")
        else:
            report.append(f"⚠️  SURPRISING: Smaller hidden dim performed better!")
            report.append(f"   {best['hidden_dim']} dim outperformed {worst['hidden_dim']} dim")
    report.append("")

    # Key Metrics Summary
    report.append("AGGREGATE STATISTICS:")
    report.append("-" * 80)
    report.append(f"Average Return:    {df['total_return_pct'].mean():.2f}%")
    report.append(f"Average Sharpe:    {df['sharpe_ratio'].mean():.2f}")
    report.append(f"Average Win Rate:  {df['win_rate'].mean()*100:.1f}%")
    report.append(f"Return Std Dev:    {df['total_return_pct'].std():.2f}%")
    report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS FOR PAPER:")
    report.append("-" * 80)
    best_config = df_sorted.iloc[0]
    report.append(f"1. Use {best_config['experiment']} as main result ({best_config['total_return_pct']:.2f}%)")
    report.append(f"2. Report hidden dimension scaling in ablation study")
    report.append(f"3. Include all variants in supplementary materials")
    report.append(f"4. Discuss VSN capacity requirements in paper")
    report.append("")

    report.append("="*80)
    report.append("Files Generated:")
    report.append(f"  - {RESULTS_DIR / 'comparison_metrics.png'}")
    report.append(f"  - {RESULTS_DIR / 'hidden_dim_scaling.png'}")
    report.append(f"  - {RESULTS_DIR / 'detailed_metrics_table.png'}")
    report.append(f"  - {RESULTS_DIR / 'results_summary.csv'}")
    report.append(f"  - {RESULTS_DIR / 'SUMMARY_REPORT.txt'}")
    report.append("="*80)

    # Save report
    report_text = '\n'.join(report)
    with open(RESULTS_DIR / 'SUMMARY_REPORT.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)

def main():
    print("\n" + "="*80)
    print("MASTER EXPERIMENT RUNNER")
    print("Running all TFT variants with different hidden dimensions")
    print("="*80 + "\n")

    total_start = time.time()
    run_results = []

    # Run all experiments
    for name, script in EXPERIMENTS.items():
        result = run_experiment(name, script)
        run_results.append(result)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")

    # Collect and analyze results
    print("Collecting results...")
    df = collect_results()

    if len(df) == 0:
        print("❌ No results to analyze!")
        return

    # Save raw results
    df.to_csv(RESULTS_DIR / 'results_summary.csv', index=False)
    print(f"✅ Saved: {RESULTS_DIR / 'results_summary.csv'}")

    # Generate visualizations
    print("\nGenerating comparison graphs...")
    create_comparison_graphs(df)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, run_results)

    print(f"\n{'='*80}")
    print("COMPLETE! Check the 'experiments/comparison_results' folder")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
