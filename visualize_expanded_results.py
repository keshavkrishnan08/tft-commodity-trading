import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_expanded_equity():
    print("Generating Expanded Equity Curve...")
    
    # Load data
    try:
        df = pd.read_csv('experiments/tft_v8_expanded/equity_curve.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("Error: equity_curve.csv not found. Run run_backtest_expanded.py first.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Strategy Equity
    ax.plot(df['Date'], df['Equity'], label='TFT Expanded Portfolio (7 Assets)', color='purple', linewidth=2)
    
    # Plot Buy & Hold (if available, or just absolute return)
    # We don't have B&H in this CSV, so just plot Strategy.
    
    # Highlight Crisis Periods
    # Crypto Winter / Inflation (2022)
    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-06-30'), color='gray', alpha=0.1, label='Market Stress')
    
    # Formatting
    ax.set_title('Expanded Portfolio Performance (Energy + Metals + Crypto)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('paper_figures/figure_expanded_equity.png', dpi=300)
    print("Saved paper_figures/figure_expanded_equity.png")

if __name__ == "__main__":
    plot_expanded_equity()
