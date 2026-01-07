import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def generate_deep_dives():
    print("Generating Asset Deep Dives...")
    
    EXP_DIR = Path('experiments/tft_v8_expanded_sliding')
    VIS_DIR = Path('visuals')
    
    # 1. Load Data
    try:
        equity_df = pd.read_csv(EXP_DIR / 'equity_curve.csv')
        # Skip 'Entry' row for date parsing
        equity_df = equity_df[equity_df['Date'] != 'Entry'].copy()
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        
        trades_df = pd.read_csv(EXP_DIR / 'trades.csv')
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        entropy_df = pd.read_csv(EXP_DIR / 'vsn_entropy.csv')
        entropy_df['Date'] = pd.to_datetime(entropy_df['Date'])
        
        # Load raw prices for charting (need full history or at least test period)
        # Using a subset of target assets
        from src.utils.config import Config
        from src.data.loader import DataLoader
        config = Config()
        loader = DataLoader(config)
        df_raw = loader.get_data()
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        
    except FileNotFoundError as e:
        print(f"Error: Required data files not found ({e}). Run backtest and VSN analysis first.")
        return

    assets = {
        'Gold': 'Gold',
        'Silver': 'Silver',
        'Bitcoin': 'BTC'
    }

    for folder_name, symbol in assets.items():
        print(f"Processing {folder_name} ({symbol})...")
        out_path = VIS_DIR / folder_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        # --- A. Price vs VSN Entropy ---
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        asset_price = df_raw[['Date', f'{symbol}_Close']].copy()
        # Filter to test period matching entropy
        asset_price = asset_price[asset_price['Date'].isin(entropy_df['Date'])]
        
        ax1.plot(asset_price['Date'], asset_price[f'{symbol}_Close'], color='black', label=f'{folder_name} Price', alpha=0.8)
        ax1.set_ylabel('Price (USD)', fontweight='bold')
        ax1.set_xlabel('Date', fontweight='bold')
        
        ax2 = ax1.twinx()
        ax2.fill_between(entropy_df['Date'], entropy_df[f'{symbol}_Entropy'], color='purple', alpha=0.3, label='VSN Entropy (Risk)')
        ax2.set_ylabel('VSN Entropy', color='purple', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        plt.title(f'{folder_name}: The Risk-Avoidance Signature', fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.savefig(out_path / 'risk_signature.png', dpi=300)
        plt.close()

        # --- B. Asset Specific Equity ---
        # We need to calculate what this asset contributed to the $10k
        asset_trades = trades_df[trades_df['asset'] == symbol].copy()
        asset_trades = asset_trades.sort_values('date')
        
        if not asset_trades.empty:
            # Asset specific equity curve (starting from 0 pnl)
            asset_trades['Cumulative_PnL'] = asset_trades['pnl'].cumsum()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(asset_trades['date'], asset_trades['Cumulative_PnL'], color='green' if asset_trades['pnl'].sum() > 0 else 'red', linewidth=2)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'{folder_name} Strategy P&L Contribution', fontweight='bold')
            ax.set_ylabel('Net P&L (USD)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out_path / 'equity_contribution.png', dpi=300)
            plt.close()
        
        # --- C. Detailed Trade Map ---
        # Price chart with green/red arrows for entry
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(asset_price['Date'], asset_price[f'{symbol}_Close'], color='gray', alpha=0.4, label='Price')
        
        # Buy/Sell markers
        longs = asset_trades[asset_trades['direction'] == 'long']
        shorts = asset_trades[asset_trades['direction'] == 'short']
        
        ax.scatter(longs['date'], asset_price.set_index('Date').loc[longs['date'], f'{symbol}_Close'], 
                   marker='^', color='green', s=100, label='Long Entry', zorder=5)
        ax.scatter(shorts['date'], asset_price.set_index('Date').loc[shorts['date'], f'{symbol}_Close'], 
                   marker='v', color='red', s=100, label='Short Entry', zorder=5)
        
        ax.set_title(f'{folder_name}: Deployment Analysis (Entry Map)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_path / 'trade_markers.png', dpi=300)
        plt.close()

    print(f"Deep dives generated in {VIS_DIR}/")

if __name__ == "__main__":
    generate_deep_dives()
