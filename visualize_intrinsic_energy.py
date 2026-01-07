
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.config import Config
from src.data.loader import DataLoader

# Set high-fidelity style using standard matplotlib
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_intrinsic_visuals():
    EXP_DIR = Path('experiments/tft_v8_intrinsic_energy')
    VIS_DIR = Path('visuals/intrinsic_energy')
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    trades_df = pd.read_csv(EXP_DIR / 'trades.csv')
    entropy_df = pd.read_csv(EXP_DIR / 'vsn_entropy.csv')
    
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    entropy_df['Date'] = pd.to_datetime(entropy_df['Date'])
    
    # Load raw prices for context
    config = Config()
    config.TARGET_ASSETS = ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']
    loader = DataLoader(config)
    price_df = loader.get_data()
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Filter prices to match test period
    test_start = trades_df['date'].min()
    price_df = price_df[price_df['Date'] >= test_start]
    entropy_df = entropy_df[entropy_df['Date'] >= test_start]

    # --- FIGURE 1: CUMULATIVE EQUITY CURVE ---
    print("Generating Cumulative Equity Curve...")
    all_dates = pd.date_range(start=test_start, end=trades_df['date'].max(), freq='D')
    equity_curve = pd.Series(index=all_dates, data=0.0)
    
    initial_total = 10000 
    
    daily_returns = trades_df.groupby('date')['pnl'].sum()
    for date, profit in daily_returns.items():
        if date in equity_curve.index:
            equity_curve[date:] += profit
            
    equity_curve += initial_total
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, color='#1A5276', linewidth=2.5, label='TFT-VSN Portfolio')
    plt.fill_between(equity_curve.index, initial_total, equity_curve.values, color='#1A5276', alpha=0.1)
    
    plt.title('Energy Markets: Cumulative Equity (2018-2022)', pad=20)
    plt.ylabel('Portfolio Value ($)', labelpad=10)
    plt.xlabel('Date', labelpad=10)
    plt.axhline(y=initial_total, color='black', linestyle='--', alpha=0.5, label='Initial Capital ($10k)')
    
    final_ret = ((equity_curve.iloc[-1] / initial_total) - 1) * 100
    plt.annotate(f'Total Return: {final_ret:+.1f}%', 
                 xy=(equity_curve.index[-1], equity_curve.iloc[-1]),
                 xytext=(-120, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=14, fontweight='bold', color='#1A5276')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'cumulative_equity.png')
    plt.close()

    # --- FIGURE 2: VSN UNCERTAINTY SIGNATURE ---
    print("Generating VSN Uncertainty Signature...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.plot(price_df['Date'], price_df['WTI_Close'], color='black', alpha=0.3, linewidth=1, label='WTI Crude Oil (Market Context)')
    ax1.set_ylabel('Asset Price ($)', color='gray', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='gray')
    
    ax2 = ax1.twinx()
    entropy_df['Entropy_SM'] = entropy_df['Entropy'].rolling(window=10).mean()
    
    ax2.plot(entropy_df['Date'], entropy_df['Entropy_SM'], color='#8E44AD', linewidth=2, label='VSN Selection Entropy (Uncertainty)')
    ax2.fill_between(entropy_df['Date'], 0, entropy_df['Entropy_SM'], color='#8E44AD', alpha=0.15)
    
    ax2.axvspan(pd.to_datetime('2020-02-15'), pd.to_datetime('2020-05-15'), color='red', alpha=0.1, label='Structural Instability (High Entropy)')
    
    ax2.set_ylabel('Variable Selection Entropy (bits)', color='#8E44AD', labelpad=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#8E44AD')
    
    plt.title('TFT VSN Uncertainty Signature: Real-time Structural Change Detection', pad=20)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'uncertainty_signature.png')
    plt.close()

    # --- FIGURE 3: TRADE DISTRIBUTION ---
    print("Generating Trade Distribution...")
    plt.figure(figsize=(10, 6))
    
    plt.hist(trades_df['return_pct'] if 'return_pct' in trades_df.columns else (trades_df['pnl'] / trades_df['position_dollars'] * 100), 
             bins=50, density=True, alpha=0.6, color='#27AE60', edgecolor='white', label='Trade Density')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    win_rate = trades_df['won'].mean() * 100
    plt.title(f'Energy Portfolio: Trade Return Distribution (Win Rate: {win_rate:.1f}%)', pad=15)
    plt.xlabel('Return Per Trade (%)', labelpad=10)
    plt.ylabel('Frequency Density', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'trade_distribution.png')
    plt.close()
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'trade_distribution.png')
    plt.close()
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'trade_distribution.png')
    plt.close()

    print(f"All visuals saved to {VIS_DIR}")

if __name__ == "__main__":
    create_intrinsic_visuals()
