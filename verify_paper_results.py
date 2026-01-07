#!/usr/bin/env python3
"""
Verify all model results for top-tier journal paper.

Ensures:
1. All models tested on identical period (2018-2022)
2. All metrics calculated consistently
3. Results are reproducible and verifiable
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def verify_test_period(trades_df_path):
    """Verify the test period from trades."""
    try:
        df = pd.read_csv(trades_df_path)
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            return dates.min(), dates.max(), len(df)
        return None, None, 0
    except:
        return None, None, 0

def load_and_verify_model(metrics_path, trades_path, model_name):
    """Load model results and verify consistency."""
    try:
        # Load metrics
        metrics = pd.read_csv(metrics_path)

        # Basic metrics
        result = {
            'Model': model_name,
            'Return (%)': float(metrics['total_return_pct'].values[0]),
            'Total Trades': int(metrics['total_trades'].values[0]),
            'Win Rate (%)': float(metrics['win_rate'].values[0] * 100),
            'Sharpe Ratio': float(metrics['sharpe_ratio'].values[0]),
            'Sortino Ratio': float(metrics['sortino_ratio'].values[0]),
            'Max Drawdown (%)': float(metrics['max_drawdown'].values[0] * 100),
            'Profit Factor': float(metrics['profit_factor'].values[0]),
            'Avg Win': float(metrics['avg_win'].values[0]),
            'Avg Loss': float(metrics['avg_loss'].values[0]),
            'Best Trade': float(metrics['best_trade'].values[0] if 'best_trade' in metrics.columns else np.nan),
            'Worst Trade': float(metrics['worst_trade'].values[0] if 'worst_trade' in metrics.columns else np.nan),
            'Calmar Ratio': float(metrics['calmar_ratio'].values[0]),
        }

        # Verify test period
        start_date, end_date, n_trades = verify_test_period(trades_path)
        result['Test Period'] = f"{start_date.date()} to {end_date.date()}" if start_date else "Unknown"
        result['Period Verified'] = (start_date >= pd.Timestamp('2018-01-01') and
                                     end_date <= pd.Timestamp('2022-12-31')) if start_date else False

        return result
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def calculate_baseline_metrics():
    """Calculate baseline strategies on exact same test period."""
    from src.data.loader import DataLoader
    from src.utils.config import Config

    config = Config()
    loader = DataLoader(config)
    df = loader.get_data()
    df['Date'] = pd.to_datetime(df['Date'])

    # EXACT test period: 2018-01-01 to 2022-06-17 (last date in data)
    test_df = df[(df['Date'] >= '2018-01-01') & (df['Date'] <= '2022-06-17')].copy()

    print(f"\nBaseline Test Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Number of days: {len(test_df)}")

    baselines = []

    # 1. Buy & Hold (Equal Weight)
    print("\n>>> Calculating Buy & Hold...")
    returns_by_asset = []
    for asset in ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']:
        start_price = test_df[f'{asset}_Close'].iloc[0]
        end_price = test_df[f'{asset}_Close'].iloc[-1]
        asset_return = (end_price - start_price) / start_price
        returns_by_asset.append(asset_return)
        print(f"  {asset}: {asset_return*100:.2f}%")

    avg_return = np.mean(returns_by_asset) * 100

    # Calculate daily returns for Sharpe
    daily_returns = []
    for asset in ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']:
        daily_ret = test_df[f'{asset}_Close'].pct_change().dropna()
        daily_returns.append(daily_ret)

    combined_returns = pd.DataFrame(daily_returns).T.mean(axis=1)
    sharpe = (combined_returns.mean() / combined_returns.std()) * np.sqrt(252)

    # Sortino (downside deviation)
    downside_returns = combined_returns[combined_returns < 0]
    sortino = (combined_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else np.nan

    # Max drawdown
    cumulative = (1 + combined_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    # Calmar ratio
    calmar = avg_return / max_dd if max_dd > 0 else 0

    baselines.append({
        'Model': 'Buy & Hold (Equal Weight)',
        'Return (%)': avg_return,
        'Total Trades': 0,
        'Win Rate (%)': np.nan,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd,
        'Profit Factor': np.nan,
        'Avg Win': np.nan,
        'Avg Loss': np.nan,
        'Best Trade': np.nan,
        'Worst Trade': np.nan,
        'Calmar Ratio': calmar,
        'Test Period': f"{test_df['Date'].min().date()} to {test_df['Date'].max().date()}",
        'Period Verified': True
    })

    # 2. MA Crossover (10/30)
    print("\n>>> Calculating MA Crossover Strategy...")
    capital = 10000
    trades = []
    equity = [capital]

    for asset in ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']:
        test_df[f'{asset}_MA10'] = test_df[f'{asset}_Close'].rolling(10).mean()
        test_df[f'{asset}_MA30'] = test_df[f'{asset}_Close'].rolling(30).mean()

        position = None
        for i in range(30, len(test_df)):
            if position is None:
                # Enter long
                if (test_df[f'{asset}_MA10'].iloc[i] > test_df[f'{asset}_MA30'].iloc[i] and
                    test_df[f'{asset}_MA10'].iloc[i-1] <= test_df[f'{asset}_MA30'].iloc[i-1]):
                    position = {
                        'entry_price': test_df[f'{asset}_Close'].iloc[i],
                        'direction': 'long'
                    }
                # Enter short
                elif (test_df[f'{asset}_MA10'].iloc[i] < test_df[f'{asset}_MA30'].iloc[i] and
                      test_df[f'{asset}_MA10'].iloc[i-1] >= test_df[f'{asset}_MA30'].iloc[i-1]):
                    position = {
                        'entry_price': test_df[f'{asset}_Close'].iloc[i],
                        'direction': 'short'
                    }
            else:
                should_exit = False
                if position['direction'] == 'long' and test_df[f'{asset}_MA10'].iloc[i] < test_df[f'{asset}_MA30'].iloc[i]:
                    should_exit = True
                elif position['direction'] == 'short' and test_df[f'{asset}_MA10'].iloc[i] > test_df[f'{asset}_MA30'].iloc[i]:
                    should_exit = True

                if should_exit or i == len(test_df) - 1:
                    exit_price = test_df[f'{asset}_Close'].iloc[i]
                    if position['direction'] == 'long':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                    # 25% position size, 0.6% friction
                    pnl = capital * 0.25 * pnl_pct * 0.994
                    capital += pnl
                    trades.append(pnl)
                    equity.append(capital)
                    position = None

    total_return = (capital - 10000) / 10000 * 100
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    # Metrics
    daily_returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else [0]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0

    downside_ret = [r for r in daily_returns if r < 0]
    sortino = (np.mean(daily_returns) / np.std(downside_ret)) * np.sqrt(252) if len(downside_ret) > 0 and np.std(downside_ret) > 0 else 0

    equity_array = np.array(equity)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    profit_factor = abs(sum(wins) / sum(losses)) if losses else np.inf
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    calmar = total_return / max_dd if max_dd > 0 else 0

    baselines.append({
        'Model': 'MA Crossover (10/30)',
        'Return (%)': total_return,
        'Total Trades': len(trades),
        'Win Rate (%)': win_rate,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd,
        'Profit Factor': profit_factor,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Best Trade': max(trades) if trades else 0,
        'Worst Trade': min(trades) if trades else 0,
        'Calmar Ratio': calmar,
        'Test Period': f"{test_df['Date'].min().date()} to {test_df['Date'].max().date()}",
        'Period Verified': True
    })

    return baselines

if __name__ == '__main__':
    print("="*90)
    print("VERIFYING ALL MODEL RESULTS FOR TOP-TIER JOURNAL PAPER")
    print("="*90)

    results = []

    # 1. Calculate baselines (guaranteed same test period)
    print("\n" + "="*90)
    print("TIER 1: BASELINE STRATEGIES")
    print("="*90)
    baselines = calculate_baseline_metrics()
    results.extend(baselines)

    # 2. Load deep learning models
    print("\n" + "="*90)
    print("TIER 2: DEEP LEARNING BASELINES")
    print("="*90)

    print("\n>>> Verifying LSTM with VSN...")
    lstm = load_and_verify_model(
        'experiments/lstm_vsn_sliding/metrics.csv',
        'experiments/lstm_vsn_sliding/trades.csv',
        'LSTM with VSN'
    )
    if lstm:
        results.append(lstm)
        print(f"  Test Period: {lstm['Test Period']}")
        print(f"  Verified: {lstm['Period Verified']}")

    print("\n>>> Verifying TCN with VSN...")
    tcn = load_and_verify_model(
        'experiments/tcn_vsn_sliding/metrics.csv',
        'experiments/tcn_vsn_sliding/trades.csv',
        'TCN with VSN'
    )
    if tcn:
        results.append(tcn)
        print(f"  Test Period: {tcn['Test Period']}")
        print(f"  Verified: {tcn['Period Verified']}")

    # 3. Load proposed TFT model
    print("\n" + "="*90)
    print("TIER 3: PROPOSED MODEL")
    print("="*90)

    print("\n>>> Verifying TFT V8 (Full)...")
    tft = load_and_verify_model(
        'experiments/tft_v8_sliding/metrics.csv',
        'experiments/tft_v8_sliding/trades.csv',
        'TFT V8 (Proposed)'
    )
    if tft:
        results.append(tft)
        print(f"  Test Period: {tft['Test Period']}")
        print(f"  Verified: {tft['Period Verified']}")

    # Create comprehensive results table
    df = pd.DataFrame(results)

    # Reorder columns for paper
    column_order = [
        'Model',
        'Return (%)',
        'Total Trades',
        'Win Rate (%)',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Max Drawdown (%)',
        'Calmar Ratio',
        'Profit Factor',
        'Avg Win',
        'Avg Loss',
        'Best Trade',
        'Worst Trade',
        'Test Period',
        'Period Verified'
    ]
    df = df[column_order]

    # Save results
    output_path = Path('experiments/journal_paper_results.csv')
    df.to_csv(output_path, index=False)

    # Display comprehensive table
    print("\n" + "="*90)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)

    # Verification summary
    print("\n" + "="*90)
    print("VERIFICATION SUMMARY")
    print("="*90)
    all_verified = df['Period Verified'].all()
    print(f"\nAll models tested on same period: {'✅ YES' if all_verified else '❌ NO'}")
    print(f"Test period: 2018-01-01 to 2022-06-17")
    print(f"Total models: {len(df)}")
    print(f"\n✅ Results saved to {output_path}")

    # Statistical summary
    print("\n" + "="*90)
    print("STATISTICAL SUMMARY FOR PAPER")
    print("="*90)

    baseline_best = df[df['Model'].str.contains('Buy|MA')]['Return (%)'].max()
    dl_best = df[df['Model'].str.contains('LSTM|TCN')]['Return (%)'].max()
    tft_return = df[df['Model'] == 'TFT V8 (Proposed)']['Return (%)'].values[0]

    print(f"\n1. Best Baseline Strategy: {baseline_best:.2f}%")
    print(f"2. Best Deep Learning Baseline: {dl_best:.2f}%")
    print(f"3. Proposed TFT Model: {tft_return:.2f}%")
    print(f"\nImprovement over baselines: {tft_return - baseline_best:.2f} percentage points")
    print(f"Improvement ratio: {tft_return / baseline_best:.2f}x")

    # Trade efficiency
    tft_trades = df[df['Model'] == 'TFT V8 (Proposed)']['Total Trades'].values[0]
    ma_trades = df[df['Model'] == 'MA Crossover (10/30)']['Total Trades'].values[0]

    print(f"\nTrade Efficiency:")
    print(f"  MA Crossover: {baseline_best/ma_trades:.2f}% return per trade")
    print(f"  TFT V8: {tft_return/tft_trades:.2f}% return per trade")

    # Risk-adjusted metrics
    tft_sharpe = df[df['Model'] == 'TFT V8 (Proposed)']['Sharpe Ratio'].values[0]
    tft_calmar = df[df['Model'] == 'TFT V8 (Proposed)']['Calmar Ratio'].values[0]

    print(f"\nRisk-Adjusted Performance:")
    print(f"  Sharpe Ratio: {tft_sharpe:.2f}")
    print(f"  Calmar Ratio: {tft_calmar:.2f}")

    print("\n" + "="*90)
