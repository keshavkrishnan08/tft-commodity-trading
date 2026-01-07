"""
Generate all publication-ready figures for experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

OUTPUT_DIR = Path('/Users/keshavkrishnan/Oil_Project/paper_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_comprehensive_analysis():
    """Load the comprehensive analysis CSV"""
    return pd.read_csv('/Users/keshavkrishnan/Oil_Project/experiments/COMPREHENSIVE_ANALYSIS.csv')

def plot_performance_comparison(df):
    """Figure 1: Performance comparison across key experiments"""
    key_experiments = [
        'tft_ablation_no_vsn',
        'tft_ablation_no_causal',
        'tft_v8_sliding',
        'tft_v8_sliding_proper',
        'hybrid_wisdom_v4'
    ]

    df_key = df[df['experiment_name'].isin(key_experiments)]

    # Rename for clarity
    rename_map = {
        'tft_ablation_no_vsn': 'TFT (No-VSN)',
        'tft_ablation_no_causal': 'TFT (No-Causal)',
        'tft_v8_sliding': 'TFT V8',
        'tft_v8_sliding_proper': 'TFT V8 Proper',
        'hybrid_wisdom_v4': 'Random Forest'
    }
    df_key['name'] = df_key['experiment_name'].map(rename_map)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # Total Return
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF6B6B']
    bars1 = ax1.bar(df_key['name'], df_key['total_return_pct'], color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Return (2018-2022)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars1, df_key['total_return_pct']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Sharpe Ratio
    bars2 = ax2.bar(df_key['name'], df_key['sharpe_ratio'], color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Risk-Adjusted Returns', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Typical Hedge Fund (1.0)')
    ax2.legend()

    for bar, val in zip(bars2, df_key['sharpe_ratio']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Max Drawdown
    bars3 = ax3.bar(df_key['name'], df_key['max_drawdown']*100, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Risk Control', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.invert_yaxis()

    for bar, val in zip(bars3, df_key['max_drawdown']*100):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height - 0.5,
                f'{val:.1f}%', ha='center', va='top', fontsize=10, fontweight='bold', color='white')

    # Win Rate
    bars4 = ax4.bar(df_key['name'], df_key['win_rate']*100, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Trading Accuracy', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax4.legend()

    for bar, val in zip(bars4, df_key['win_rate']*100):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: performance_comparison.pdf")
    plt.close()

def plot_ablation_study(df):
    """Figure 2: Ablation study results"""
    ablation_data = [
        ('TFT V8\n(Baseline)', 221.39, 4.11),
        ('TFT\n(No-VSN)', 372.10, 4.75),
        ('TFT\n(No-Causal)', 294.82, 4.93)
    ]

    names, returns, sharpes = zip(*ablation_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Returns
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars1 = ax1.bar(names, returns, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation Study: Return Impact', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add delta annotations
    ax1.annotate('', xy=(1, 372.10), xytext=(0, 221.39),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(0.5, 300, '+150.7%\n(VSN Removal)', ha='center', va='center',
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # Sharpe
    bars2 = ax2.bar(names, sharpes, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Ablation Study: Risk-Adjusted Impact', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, sharpes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_study.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ablation_study.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: ablation_study.pdf")
    plt.close()

def plot_risk_return_scatter(df):
    """Figure 3: Risk-return scatter plot"""
    df_filtered = df[df['total_trades'] > 100]  # Filter out experiments with too few trades

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by experiment type
    def get_color(name):
        if 'tft' in name.lower():
            return '#2E86AB'
        elif 'lstm' in name.lower() or 'hybrid' in name.lower():
            return '#4CAF50'
        else:
            return '#FF6B6B'

    colors = [get_color(name) for name in df_filtered['experiment_name']]

    scatter = ax.scatter(df_filtered['max_drawdown']*100, df_filtered['total_return_pct'],
                        s=df_filtered['sharpe_ratio']*100, c=colors, alpha=0.6, edgecolors='black')

    # Annotate key points
    key_experiments = ['tft_ablation_no_vsn', 'tft_v8_sliding_proper', 'hybrid_wisdom_v4']
    for exp in key_experiments:
        if exp in df_filtered['experiment_name'].values:
            row = df_filtered[df_filtered['experiment_name'] == exp].iloc[0]
            ax.annotate(exp.replace('_', ' ').title(),
                       xy=(row['max_drawdown']*100, row['total_return_pct']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

    ax.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Profile (Bubble Size = Sharpe Ratio)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add diagonal lines for risk/return ratios
    max_dd = df_filtered['max_drawdown'].max() * 100
    for ratio in [5, 10, 20]:
        x = np.linspace(0, max_dd, 100)
        y = ratio * x
        ax.plot(x, y, '--', alpha=0.3, color='gray', linewidth=1)
        ax.text(max_dd*0.9, ratio*max_dd*0.9, f'{ratio}x', fontsize=8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_return_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: risk_return_scatter.pdf")
    plt.close()

def plot_metrics_heatmap(df):
    """Figure 4: Metrics heatmap"""
    key_experiments = [
        'tft_ablation_no_vsn',
        'tft_ablation_no_causal',
        'tft_v8_sliding',
        'tft_v8_sliding_proper',
        'hybrid_wisdom_v4'
    ]

    df_key = df[df['experiment_name'].isin(key_experiments)]

    metrics = ['total_return_pct', 'sharpe_ratio', 'sortino_ratio',
               'win_rate', 'profit_factor', 'calmar_ratio']

    data = df_key[metrics].values

    # Normalize each column to 0-1
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data_norm.T, cmap='RdYlGn', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(df_key)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([name.replace('_', ' ').title() for name in df_key['experiment_name']], rotation=45, ha='right')
    ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Score', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(df_key)):
            text = ax.text(j, i, f'{data_norm[j, i]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    ax.set_title('Performance Metrics Heatmap (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: metrics_heatmap.pdf")
    plt.close()

def plot_statistical_metrics(df):
    """Figure 5: Statistical metrics comparison"""
    # Filter experiments with statistical metrics
    df_stats = df[df['pnl_rmse'].notna()]

    if len(df_stats) == 0:
        print("⚠ No statistical metrics available")
        return

    key_experiments = ['tft_ablation_no_vsn', 'tft_ablation_no_causal', 'tft_v8_sliding_proper']
    df_key = df_stats[df_stats['experiment_name'].isin(key_experiments)]

    if len(df_key) == 0:
        print("⚠ No statistical metrics for key experiments")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    names = [n.replace('_', ' ').title() for n in df_key['experiment_name']]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # RMSE
    ax1.bar(names, df_key['pnl_rmse'], color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax1.set_title('P&L Root Mean Square Error', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Skewness
    ax2.bar(names, df_key['pnl_skewness'], color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Skewness', fontsize=12, fontweight='bold')
    ax2.set_title('P&L Distribution Skewness', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Kurtosis
    ax3.bar(names, df_key['pnl_kurtosis'], color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Kurtosis', fontsize=12, fontweight='bold')
    ax3.set_title('P&L Distribution Kurtosis', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Normal (0)')
    ax3.legend()

    # Information Ratio
    ax4.bar(names, df_key['information_ratio'], color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Information Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Information Ratio', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'statistical_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'statistical_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: statistical_metrics.pdf")
    plt.close()

def main():
    """Generate all figures"""
    print("Loading comprehensive analysis...")
    df = load_comprehensive_analysis()
    print(f"Loaded {len(df)} experiments\n")

    print("Generating publication-ready figures...\n")

    plot_performance_comparison(df)
    plot_ablation_study(df)
    plot_risk_return_scatter(df)
    plot_metrics_heatmap(df)
    plot_statistical_metrics(df)

    print(f"\n✅ All figures generated in {OUTPUT_DIR}/")
    print(f"\nFigures created:")
    print(f"  1. performance_comparison.pdf")
    print(f"  2. ablation_study.pdf")
    print(f"  3. risk_return_scatter.pdf")
    print(f"  4. metrics_heatmap.pdf")
    print(f"  5. statistical_metrics.pdf")

if __name__ == '__main__':
    main()
