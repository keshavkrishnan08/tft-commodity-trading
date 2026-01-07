# visualize_v6.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EXPERIMENT_DIR = Path('experiments/tft_v8')
PLOTS_DIR = EXPERIMENT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_data():
    equity = pd.read_csv(EXPERIMENT_DIR / 'equity_curve.csv', parse_dates=['Date'])
    trades = pd.read_csv(EXPERIMENT_DIR / 'trades.csv', parse_dates=['date'])
    cal_path = EXPERIMENT_DIR / 'calibration_curve.csv'
    calibration = pd.read_csv(cal_path) if cal_path.exists() else None
    return equity, trades, calibration

def plot_equity_drawdown(equity):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Equity Curve
    sns.lineplot(data=equity, x='Date', y='Equity', ax=ax1, color='#1f77b4', linewidth=1.5)
    ax1.set_title("Hybrid Wisdom V6: Cumulative Wealth (199 Features + Calibrated)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Account Balance ($)")
    
    # Highlight max equity
    max_eq = equity['Equity'].max()
    max_date = equity.loc[equity['Equity'].idxmax(), 'Date']
    ax1.annotate(f'Peak: ${max_eq:,.0f}', xy=(max_date, max_eq), xytext=(max_date, max_eq*1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    # 2. Drawdown
    sns.lineplot(data=equity, x='Date', y='Drawdown', ax=ax2, color='#d62728', linewidth=1)
    ax2.fill_between(equity['Date'], equity['Drawdown'], 0, color='#d62728', alpha=0.3)
    ax2.set_title("Drawdown Profile", fontsize=12)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.invert_yaxis()  # Drawdown goes down from 0
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'v6_equity_drawdown.png')
    print("Saved equity_drawdown.png")

def plot_calibration_curve(calibration):
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    # Plot curves per asset
    for asset in calibration['Asset'].unique():
        subset = calibration[calibration['Asset'] == asset]
        plt.plot(subset['Raw_Prob'], subset['Calibrated_Prob'], label=f'{asset} Calibration', linewidth=2)
        
    plt.title("Isotonic Probability Calibration (Truth vs. Confidence)", fontsize=14, fontweight='bold')
    plt.xlabel("Model Confidence (Raw Output)", fontsize=12)
    plt.ylabel("Real Probability (Calibrated)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Annotate correction
    plt.annotate('Model Overconfident\n(Pulled Down)', xy=(0.8, 0.6), xytext=(0.9, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Model Underconfident\n(Pulled Up)', xy=(0.2, 0.4), xytext=(0.05, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.savefig(PLOTS_DIR / 'v6_calibration_curve.png')
    print("Saved calibration_curve.png")

def plot_atr_impact(trades):
    # This assumes 'trades' contains TRADED assets. 
    # To Visualize the filter, we'd ideally explicitly show what was skipped.
    # But showing the "Golden Zone" of executed trades is still valuable.
    
    # We will simulate the "skipped" zone visually
    plt.figure(figsize=(12, 8))
    
    # We don't have the skipped trades directly in 'trades.csv', only executed ones.
    # But we can look at the ATR distribution of the trades we took.
    
    # 1. Histogram of ATR Ranks for Winning vs Losing Trades
    # NOTE: We need to reconstruct ATR Rank for this plot if it's not in trades.csv
    # But 'trades.csv' doesn't contain ATR_Rank explicitly? 
    # Ah, the updated backtest script didn't save ATR_Rank into the trade dict.
    # Wait! 'main_hybrid_v6.py' saves 'volatility' which is ATR value, but not rank.
    # Let's plot PnL vs Volatility (Absolute) instead.
    
    # Filter out extreme outliers for better visualization
    trades_filtered = trades[trades['pnl'].abs() < trades['pnl'].std() * 3]
    
    sns.scatterplot(data=trades_filtered, x='volatility', y='pnl', hue='won', palette={True: 'green', False: 'red'}, alpha=0.6)
    
    plt.title("Trade PnL vs. Volatility (ATR)", fontsize=14, fontweight='bold')
    plt.xlabel("Volatility (ATR %)", fontsize=12)
    plt.ylabel("Trade PnL ($)", fontsize=12)
    plt.axhline(0, color='black', linestyle='--')
    
    plt.savefig(PLOTS_DIR / 'v6_pnl_volatility.png')
    print("Saved pnl_volatility.png")

def plot_monthly_heatmap(equity):
    # Calculate monthly returns
    equity['Year'] = equity['Date'].dt.year
    equity['Month'] = equity['Date'].dt.month
    
    monthly_returns = equity.groupby(['Year', 'Month'])['Daily_Return'].sum().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', center=0, cbar_kws={'label': 'Monthly Return'})
    plt.title("Monthly Returns Heatmap", fontsize=14, fontweight='bold')
    plt.ylabel("Year")
    plt.xlabel("Month")
    
    plt.savefig(PLOTS_DIR / 'v6_monthly_heatmap.png')
    print("Saved monthly_heatmap.png")

if __name__ == "__main__":
    try:
        equity, trades, calibration = load_data()
        plot_equity_drawdown(equity)
        if calibration is not None:
            plot_calibration_curve(calibration)
        plot_atr_impact(trades)
        plot_monthly_heatmap(equity)
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
