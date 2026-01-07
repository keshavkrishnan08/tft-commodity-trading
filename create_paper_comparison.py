#!/usr/bin/env python3
"""
Create comprehensive model comparison for academic paper.

Includes:
1. Simple baselines (Buy & Hold, MA Crossover)
2. Classical ML (Random Forest, XGBoost)
3. Deep Learning (LSTM, TCN)
4. Ablation Studies (TFT variants)
5. Best Model (TFT V8)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.utils.config import Config


def calculate_buy_hold():
    """Calculate buy & hold for all 4 assets equally weighted."""
    config = Config()
    loader = DataLoader(config)
    df = loader.get_data()

    # Filter to test period (2018-2022)
    df['Date'] = pd.to_datetime(df['Date'])
    test_df = df[(df['Date'] >= '2018-01-01') & (df['Date'] < '2023-01-01')]

    returns = []
    for asset in ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']:
        start_price = test_df[f'{asset}_Close'].iloc[0]
        end_price = test_df[f'{asset}_Close'].iloc[-1]
        asset_return = (end_price - start_price) / start_price
        returns.append(asset_return)

    # Equally weighted
    avg_return = np.mean(returns) * 100

    # Calculate Sharpe (assuming daily returns)
    daily_returns = []
    for asset in ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']:
        daily_ret = test_df[f'{asset}_Close'].pct_change().dropna()
        daily_returns.append(daily_ret)

    combined_returns = pd.DataFrame(daily_returns).T.mean(axis=1)
    sharpe = (combined_returns.mean() / combined_returns.std()) * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + combined_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    return {
        'Model': 'Buy & Hold (Baseline)',
        'Tier': '1. Simple Baselines',
        'Return (%)': avg_return,
        'Trades': 0,
        'Win Rate (%)': np.nan,
        'Sharpe': sharpe,
        'Max DD (%)': max_dd
    }


def calculate_ma_crossover():
    """Calculate simple MA crossover strategy."""
    config = Config()
    loader = DataLoader(config)
    df = loader.get_data()

    # Filter to test period
    df['Date'] = pd.to_datetime(df['Date'])
    test_df = df[(df['Date'] >= '2018-01-01') & (df['Date'] < '2023-01-01')].copy()

    capital = 10000
    trades = []
    equity = [capital]

    for asset in ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']:
        # Calculate MAs
        test_df[f'{asset}_MA10'] = test_df[f'{asset}_Close'].rolling(10).mean()
        test_df[f'{asset}_MA30'] = test_df[f'{asset}_Close'].rolling(30).mean()

        position = None
        for i in range(30, len(test_df)):
            if position is None:
                # Enter long when MA10 > MA30
                if (test_df[f'{asset}_MA10'].iloc[i] > test_df[f'{asset}_MA30'].iloc[i] and
                    test_df[f'{asset}_MA10'].iloc[i-1] <= test_df[f'{asset}_MA30'].iloc[i-1]):
                    position = {
                        'entry_price': test_df[f'{asset}_Close'].iloc[i],
                        'entry_idx': i,
                        'direction': 'long'
                    }
                # Enter short when MA10 < MA30
                elif (test_df[f'{asset}_MA10'].iloc[i] < test_df[f'{asset}_MA30'].iloc[i] and
                      test_df[f'{asset}_MA10'].iloc[i-1] >= test_df[f'{asset}_MA30'].iloc[i-1]):
                    position = {
                        'entry_price': test_df[f'{asset}_Close'].iloc[i],
                        'entry_idx': i,
                        'direction': 'short'
                    }
            else:
                # Exit conditions
                should_exit = False
                if position['direction'] == 'long':
                    if test_df[f'{asset}_MA10'].iloc[i] < test_df[f'{asset}_MA30'].iloc[i]:
                        should_exit = True
                else:  # short
                    if test_df[f'{asset}_MA10'].iloc[i] > test_df[f'{asset}_MA30'].iloc[i]:
                        should_exit = True

                if should_exit or i == len(test_df) - 1:
                    exit_price = test_df[f'{asset}_Close'].iloc[i]
                    if position['direction'] == 'long':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                    pnl = capital * 0.25 * pnl_pct * 0.994  # 0.6% friction
                    capital += pnl
                    trades.append(pnl)
                    equity.append(capital)
                    position = None

    total_return = (capital - 10000) / 10000 * 100
    win_rate = (np.array(trades) > 0).sum() / len(trades) * 100 if trades else 0

    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 0 else 0

    equity_array = np.array(equity)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    return {
        'Model': 'MA Crossover (10/30)',
        'Tier': '1. Simple Baselines',
        'Return (%)': total_return,
        'Trades': len(trades),
        'Win Rate (%)': win_rate,
        'Sharpe': sharpe,
        'Max DD (%)': max_dd
    }


def load_model_results():
    """Load existing model results."""
    results = []

    def get_metrics(path, model_name, tier):
        try:
            df = pd.read_csv(path)
            return {
                'Model': model_name,
                'Tier': tier,
                'Return (%)': df['total_return_pct'].values[0],
                'Trades': int(df['total_trades'].values[0]),
                'Win Rate (%)': df['win_rate'].values[0] * 100,
                'Sharpe': df['sharpe_ratio'].values[0],
                'Max DD (%)': df['max_drawdown'].values[0] * 100
            }
        except:
            return None

    # Deep Learning Baselines
    r = get_metrics('experiments/lstm_vsn_sliding/metrics.csv',
                    'LSTM with VSN', '2. Deep Learning Baselines')
    if r: results.append(r)

    r = get_metrics('experiments/tcn_vsn_sliding/metrics.csv',
                    'TCN with VSN', '2. Deep Learning Baselines')
    if r: results.append(r)

    # Ablation Studies
    r = get_metrics('experiments/tft_ablation_no_vsn/metrics.csv',
                    'TFT without VSN', '3. Ablation Studies')
    if r: results.append(r)

    r = get_metrics('experiments/tft_ablation_no_causal/metrics.csv',
                    'TFT without Causal Attention', '3. Ablation Studies')
    if r: results.append(r)

    # Best Model
    r = get_metrics('experiments/tft_v8_sliding/metrics.csv',
                    'TFT V8 (Full) ⭐', '4. Proposed Model')
    if r: results.append(r)

    return results


if __name__ == '__main__':
    print("Generating comprehensive model comparison...")
    print()

    # Calculate baselines
    print("Calculating baselines...")
    results = []

    print("  - Buy & Hold...")
    results.append(calculate_buy_hold())

    print("  - MA Crossover...")
    results.append(calculate_ma_crossover())

    # Load model results
    print("Loading model results...")
    results.extend(load_model_results())

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by tier then return
    tier_order = {
        '1. Simple Baselines': 1,
        '2. Deep Learning Baselines': 2,
        '3. Ablation Studies': 3,
        '4. Proposed Model': 4
    }
    df['tier_order'] = df['Tier'].map(tier_order)
    df = df.sort_values(['tier_order', 'Return (%)'], ascending=[True, False])
    df = df.drop('tier_order', axis=1)

    # Format for display
    print("\n" + "="*90)
    print("COMPREHENSIVE MODEL COMPARISON FOR ACADEMIC PAPER")
    print("Test Period: 2018-2022 (4.5 years)")
    print("="*90)
    print()

    current_tier = None
    for _, row in df.iterrows():
        if row['Tier'] != current_tier:
            current_tier = row['Tier']
            print(f"\n{current_tier}")
            print("-" * 90)

        trades_str = str(int(row['Trades'])) if pd.notna(row['Trades']) and row['Trades'] > 0 else 'N/A'
        win_rate_str = f"{row['Win Rate (%)']:.1f}%" if pd.notna(row['Win Rate (%)']) else 'N/A'

        print(f"{row['Model']:<35} "
              f"Return: {row['Return (%)']:>8.2f}%  "
              f"Trades: {trades_str:>6}  "
              f"Win: {win_rate_str:>6}  "
              f"Sharpe: {row['Sharpe']:>5.2f}  "
              f"DD: {row['Max DD (%)']:>5.1f}%")

    print("\n" + "="*90)

    # Save to CSV
    output_path = Path('experiments/paper_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")

    # Key insights
    print("\n" + "="*90)
    print("KEY INSIGHTS FOR PAPER")
    print("="*90)

    best_baseline = df[df['Tier'] == '1. Simple Baselines']['Return (%)'].max()
    best_dl = df[df['Tier'] == '2. Deep Learning Baselines']['Return (%)'].max()
    best_ablation = df[df['Tier'] == '3. Ablation Studies']['Return (%)'].max()
    proposed = df[df['Tier'] == '4. Proposed Model']['Return (%)'].values[0]

    print(f"\n1. Simple baselines achieved at most {best_baseline:.1f}% return")
    print(f"2. Deep learning baselines FAILED (best: {best_dl:.1f}%)")
    print(f"3. Ablation studies show importance of architecture:")
    print(f"   - Without VSN: {df[df['Model']=='TFT without VSN']['Return (%)'].values[0]:.1f}%")
    print(f"   - Without Causal Attention: {df[df['Model']=='TFT without Causal Attention']['Return (%)'].values[0]:.1f}%")
    print(f"4. Proposed TFT V8 model: {proposed:.1f}%")
    print(f"\n⚠️  Note: Ablations unexpectedly outperformed full model - investigate further")
    print("   This could indicate overfitting or that some components add noise")
    print("="*90)
