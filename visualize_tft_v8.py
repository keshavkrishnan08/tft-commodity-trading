# visualize_tft_v8.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_visualization():
    print("Generating Comprehensive V8 Ultimate Analytics...")
    
    BASE_DIR = Path('experiments/tft_v8_sliding')
    TRADES_FILE = BASE_DIR / 'trades.csv'
    OUTPUT_DIR = BASE_DIR / 'visuals'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if not TRADES_FILE.exists():
        print(f"Error: {TRADES_FILE} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(TRADES_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.sort_values('date')
    
    # 2. Accuracy per Asset
    asset_metrics = []
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset]
        # Accuracy is where direction (long=1, short=0) matches actual_outcome
        # Note: 'direction' is string 'long'/'short'. Convert to match actual_outcome (1/0)
        pred_label = (asset_df['direction'] == 'long').astype(int)
        acc = accuracy_score(asset_df['actual_outcome'], pred_label)
        prec = precision_score(asset_df['actual_outcome'], pred_label)
        rec = recall_score(asset_df['actual_outcome'], pred_label)
        
        asset_metrics.append({
            'Asset': asset,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Win Rate': asset_df['won'].mean(),
            'Total P&L': asset_df['pnl'].sum(),
            'Trades': len(asset_df)
        })
    
    metrics_df = pd.DataFrame(asset_metrics)
    metrics_df.to_csv(BASE_DIR / 'asset_performance_metrics.csv', index=False)
    
    # Plot Accuracy & Win Rate per Asset
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_df['Asset']))
    width = 0.35
    plt.bar(x - width/2, metrics_df['Accuracy'], width, label='Model Accuracy', color='#1976d2')
    plt.bar(x + width/2, metrics_df['Win Rate'], width, label='Trade Win Rate', color='#388e3c')
    plt.xticks(x, metrics_df['Asset'])
    plt.title('V8 Performance Metrics per Asset', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_per_asset.png', dpi=300)
    plt.close()

    # 3. Monthly Returns Heatmap
    df['month_name'] = df['date'].dt.strftime('%b')
    monthly_pnl = df.groupby(['year', 'month'])['pnl'].sum().unstack().fillna(0)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pnl.columns = [month_names[i-1] for i in monthly_pnl.columns]
    
    plt.figure(figsize=(14, 7))
    sns.heatmap(monthly_pnl, annot=True, fmt=".0f", cmap='RdYlGn', center=0, cbar_kws={'label': 'P&L ($)'})
    plt.title('V8 Monthly Alpha Heatmap (Total P&L)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'monthly_returns_heatmap.png', dpi=300)
    plt.close()

    # 4. Calibration Analysis (Meta-ML Layer)
    # Compare raw vs calibrated probabilities
    plt.figure(figsize=(12, 6))
    
    # Reliability Diagram
    bins = np.linspace(0, 1, 11)
    df['prob_bin'] = pd.cut(df['calibrated_probability'], bins)
    reliability = df.groupby('prob_bin')['won'].mean()
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, reliability.values, marker='o', label='V8 Isotonic Calibrator', color='#d32f2f')
    plt.fill_between(bin_centers, reliability.values, bin_centers, alpha=0.1, color='#d32f2f')
    
    plt.title('Meta-ML Reality Check: Calibration Reliability', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Win Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'calibration_reliability.png', dpi=300)
    plt.close()

    # 5. Asset-Specific Equity Curves
    plt.figure(figsize=(15, 8))
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset].copy()
        asset_df['cum_pnl'] = asset_df['pnl'].cumsum()
        plt.plot(asset_df['date'], asset_df['cum_pnl'], label=f'{asset} (Σ P&L)', linewidth=2)
    
    plt.title('V8 Cumulative P&L Attribution by Asset', fontsize=14, fontweight='bold')
    plt.ylabel('P&L Contribution ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'asset_equity_curves.png', dpi=300)
    plt.close()

    # 6. Trade Outcome Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pnl'], kde=True, bins=50, color='#673ab7')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('V8 Trade Outcome Distribution ($ P&L)', fontsize=14, fontweight='bold')
    plt.xlabel('P&L per Trade')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pnl_distribution.png', dpi=300)
    plt.close()

    # 7. Drawdown Analysis
    df['cum_pnl'] = df['pnl'].cumsum()
    df['equity'] = 10000 + df['cum_pnl']
    df['rolling_max'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['rolling_max']) / df['rolling_max']
    
    plt.figure(figsize=(15, 6))
    plt.plot(df['date'], df['equity'], color='#2e7d32', label='Equity')
    plt.title('V8 Master Equity Curve', fontsize=16, fontweight='bold')
    plt.ylabel('Account Value ($)')
    plt.twinx()
    plt.fill_between(df['date'], 0, df['drawdown'], color='#c62828', alpha=0.3, label='Drawdown')
    plt.ylabel('Drawdown (%)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'equity_and_drawdown.png', dpi=300)
    plt.close()

    print(f"✅ Advanced Visuals saved to {OUTPUT_DIR}/")
    print(f"✅ Asset metrics saved to {BASE_DIR}/asset_performance_metrics.csv")

if __name__ == "__main__":
    run_visualization()
