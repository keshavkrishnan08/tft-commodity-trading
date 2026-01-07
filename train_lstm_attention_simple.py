#!/usr/bin/env python3
"""
Simple LSTM + Attention Model (NO SLIDING WINDOW)

Like your earlier successful models - single training run, clean architecture.

Architecture:
- Input: 219 calibrated features, 60-day sequences
- LSTM layers (2 layers, 128 hidden units)
- Multi-head attention (4 heads)
- Per-asset prediction heads
- Simple train/val/test split (no sliding window complexity)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.calibrated_features import CalibratedFeatureEngineer
from src.utils.config import Config


class AttentionLayer(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, hidden_size, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, _ = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(B, L, self.hidden_size)
        out = self.out_proj(out)

        return out


class SimpleLSTMAttention(nn.Module):
    """
    Simple LSTM + Attention model.

    No complexity - just works.
    """
    def __init__(self, input_dim, hidden_size=128, num_layers=2,
                 num_heads=4, dropout=0.4, num_assets=4):  # Higher dropout
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Input projection with dropout
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4,  # Higher dropout to prevent overfitting
            bidirectional=True
        )

        # Project bidirectional LSTM output
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Attention
        self.attention = AttentionLayer(hidden_size, num_heads, dropout)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # Per-asset prediction heads
        self.asset_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            for _ in range(num_assets)
        ])

    def forward(self, x):
        # x: [batch, seq_len, input_dim]

        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden]

        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        lstm_out = self.lstm_proj(lstm_out)  # [batch, seq_len, hidden]

        # Attention
        attn_out = self.attention(lstm_out)  # [batch, seq_len, hidden]

        # Residual + Norm
        x = self.norm(lstm_out + attn_out)

        # Take last timestep
        x = x[:, -1, :]  # [batch, hidden]

        # Per-asset predictions
        outputs = {}
        asset_names = ['WTI', 'Brent', 'NaturalGas', 'HeatingOil']
        for i, asset in enumerate(asset_names):
            outputs[asset] = self.asset_heads[i](x)

        return outputs


def prepare_sequences(df, feature_cols, label_cols, seq_length=60):
    """Create sequences for LSTM training."""
    sequences = []
    labels_list = {asset: [] for asset in label_cols.keys()}
    dates = []
    ohlc_data = {asset: [] for asset in label_cols.keys()}

    data = df[feature_cols].values

    for i in range(seq_length, len(df)):
        # Get sequence
        seq = data[i-seq_length:i]
        sequences.append(seq)

        # Get labels
        for asset, label_col in label_cols.items():
            labels_list[asset].append(df[label_col].iloc[i])

            # Store OHLC for realistic backtesting
            ohlc_data[asset].append({
                'open': df[f'{asset}_Open'].iloc[i],
                'high': df[f'{asset}_High'].iloc[i],
                'low': df[f'{asset}_Low'].iloc[i],
                'close': df[f'{asset}_Close'].iloc[i]
            })

        dates.append(df['Date'].iloc[i])

    return (np.array(sequences),
            {k: np.array(v) for k, v in labels_list.items()},
            dates,
            ohlc_data)


def train_simple_lstm():
    """Train simple LSTM + Attention model."""
    print("="*80)
    print("SIMPLE LSTM + ATTENTION MODEL")
    print("="*80)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n>>> Loading data...")
    loader = DataLoader(config)
    df_raw = loader.get_data()
    engineer = CalibratedFeatureEngineer(config, d=0.4)
    df = engineer.engineer_features(df_raw)

    # Feature columns
    exclude_cols = ['Date'] + [c for c in df.columns if 'Label' in c or 'FutureReturn' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    label_cols = {asset: f'{asset}_Label' for asset in config.TARGET_ASSETS}

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Simple train/val/test split
    # Train: 2001-2016, Val: 2017, Test: 2018-2022
    train_mask = df['Date'] < '2017-01-01'
    val_mask = (df['Date'] >= '2017-01-01') & (df['Date'] < '2018-01-01')
    test_mask = df['Date'] >= '2018-01-01'

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} samples ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Val:   {len(val_df)} samples ({val_df['Date'].min()} to {val_df['Date'].max()})")
    print(f"Test:  {len(test_df)} samples ({test_df['Date'].min()} to {test_df['Date'].max()})")

    # Create sequences
    seq_length = 60
    print(f"\nCreating sequences (length={seq_length})...")

    X_train, y_train, dates_train, ohlc_train = prepare_sequences(train_df, feature_cols, label_cols, seq_length)
    X_val, y_val, dates_val, ohlc_val = prepare_sequences(val_df, feature_cols, label_cols, seq_length)
    X_test, y_test, dates_test, ohlc_test = prepare_sequences(test_df, feature_cols, label_cols, seq_length)

    print(f"Train sequences: {X_train.shape}")
    print(f"Val sequences:   {X_val.shape}")
    print(f"Test sequences:  {X_test.shape}")

    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)

    y_train_t = {k: torch.FloatTensor(v) for k, v in y_train.items()}
    y_val_t = {k: torch.FloatTensor(v) for k, v in y_val.items()}
    y_test_t = {k: torch.FloatTensor(v) for k, v in y_test.items()}

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        X_train_t,
        *[y_train_t[asset] for asset in config.TARGET_ASSETS]
    )
    val_dataset = torch.utils.data.TensorDataset(
        X_val_t,
        *[y_val_t[asset] for asset in config.TARGET_ASSETS]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    print("\n>>> Initializing model...")
    model = SimpleLSTMAttention(
        input_dim=len(feature_cols),
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.4,  # Heavy dropout
        num_assets=len(config.TARGET_ASSETS)
    ).to(device)

    # Loss and optimizer with HEAVY regularization to prevent overfitting
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)  # Very low LR, heavy regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training with early stopping to prevent overfitting
    print("\n>>> Training...")
    best_val_loss = float('inf')
    patience = 3  # Very short patience - stop overfitting fast
    no_improve = 0

    for epoch in range(30):  # Even fewer epochs
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            X = batch[0].to(device)
            y = {asset: batch[i+1].to(device) for i, asset in enumerate(config.TARGET_ASSETS)}

            optimizer.zero_grad()
            outputs = model(X)

            loss = sum(criterion(outputs[asset].squeeze(), y[asset])
                      for asset in config.TARGET_ASSETS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0].to(device)
                y = {asset: batch[i+1].to(device) for i, asset in enumerate(config.TARGET_ASSETS)}

                outputs = model(X)
                loss = sum(criterion(outputs[asset].squeeze(), y[asset])
                          for asset in config.TARGET_ASSETS)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}", end='')

        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'experiments/simple_lstm_best.pt')
            print(" ✓")
        else:
            no_improve += 1
            print(f" ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load('experiments/simple_lstm_best.pt'))

    # Test predictions
    print("\n>>> Testing...")
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test_t.to(device)
        test_outputs = model(X_test_gpu)
        test_probs = {asset: torch.sigmoid(test_outputs[asset]).cpu().numpy().flatten()
                     for asset in config.TARGET_ASSETS}

    # Realistic backtest
    print("\n>>> Running backtest...")
    results = realistic_backtest(test_probs, y_test, dates_test, ohlc_test, config.TARGET_ASSETS)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total Return:   {results['return']:.2f}%")
    print(f"Win Rate:       {results['win_rate']*100:.1f}%")
    print(f"Total Trades:   {results['total_trades']}")
    print(f"Sharpe Ratio:   {results['sharpe']:.2f}")
    print(f"Max Drawdown:   {results['max_dd']*100:.1f}%")
    print("="*80)

    # Save
    pd.DataFrame([results]).to_csv('experiments/simple_lstm_results.csv', index=False)
    print(f"\n✅ Results saved to experiments/simple_lstm_results.csv")

    return model, results


def realistic_backtest(probs, labels, dates, ohlc_data, assets, threshold=0.50):
    """
    REALISTIC backtest with VOLATILITY and REGIME FILTERS.

    Key improvements:
    - DON'T trade in extreme volatility (insane market conditions)
    - DON'T trade during severe drawdowns (market crashes)
    - Holds trades until stop or target hit
    - Conservative 2:1 reward/risk
    - Trailing stop to lock in profits
    """
    capital = 10000
    equity = [capital]
    trades = []
    open_positions = {}  # Track open positions per asset

    # Simple ATR calculation (14-period average range)
    atr_window = 14

    # Calculate historical ATR percentiles for volatility filter
    all_atrs = {asset: [] for asset in assets}
    for i in range(atr_window, len(dates)):
        for asset in assets:
            recent_ranges = []
            for j in range(i - atr_window, i):
                ohlc = ohlc_data[asset][j]
                true_range = ohlc['high'] - ohlc['low']
                recent_ranges.append(true_range)
            all_atrs[asset].append(np.mean(recent_ranges))

    # Calculate 75th percentile (don't trade in top 25% volatility)
    atr_threshold = {asset: np.percentile(all_atrs[asset], 75) for asset in assets}

    for i in range(len(dates) - 1):
        # First, check exits for any open positions
        for asset in list(open_positions.keys()):
            pos = open_positions[asset]
            current_ohlc = ohlc_data[asset][i]

            exit_price = None
            hit_stop = False
            hit_target = False

            if pos['direction'] == 'long':
                # Update trailing stop (more conservative - trail by 0.75x stop distance)
                best_price = max(pos['best_price'], current_ohlc['high'])
                trail_distance = 0.75 * pos['stop_distance']  # Wider trail = less whipsaw
                new_trailing = best_price - trail_distance
                pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing)
                pos['best_price'] = best_price

                # Check exits
                if current_ohlc['low'] <= pos['trailing_stop']:
                    hit_stop = True
                    exit_price = pos['trailing_stop']
                elif current_ohlc['high'] >= pos['target_price']:
                    hit_target = True
                    exit_price = pos['target_price']

            else:  # short
                # Update trailing stop (more conservative - trail by 0.75x stop distance)
                best_price = min(pos['best_price'], current_ohlc['low'])
                trail_distance = 0.75 * pos['stop_distance']  # Wider trail = less whipsaw
                new_trailing = best_price + trail_distance
                pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing)
                pos['best_price'] = best_price

                # Check exits
                if current_ohlc['high'] >= pos['trailing_stop']:
                    hit_stop = True
                    exit_price = pos['trailing_stop']
                elif current_ohlc['low'] <= pos['target_price']:
                    hit_target = True
                    exit_price = pos['target_price']

            # Close position if stop or target hit
            if exit_price is not None:
                if pos['direction'] == 'long':
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']

                pnl = pos['position_size'] * pnl_pct
                capital += pnl
                equity.append(capital)

                trades.append({
                    'date': dates[i],
                    'asset': asset,
                    'direction': pos['direction'],
                    'prob': pos['prob'],
                    'entry': pos['entry_price'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'hit_stop': hit_stop,
                    'hit_target': hit_target,
                    'bars_held': i - pos['entry_bar']
                })

                # Remove from open positions
                del open_positions[asset]

        # Now check for new entries (only if no position in that asset)
        for asset in assets:
            if asset in open_positions:
                continue  # Already have position

            prob = probs[asset][i]

            # Take trade if moderate confidence
            if prob > threshold:
                direction = 'long'
            elif prob < (1 - threshold):
                direction = 'short'
            else:
                continue

            # Calculate ATR for stop/target
            if i >= atr_window:
                recent_ranges = []
                for j in range(max(0, i - atr_window), i):
                    ohlc = ohlc_data[asset][j]
                    true_range = ohlc['high'] - ohlc['low']
                    recent_ranges.append(true_range)
                atr = np.mean(recent_ranges)
            else:
                ohlc = ohlc_data[asset][i]
                atr = ohlc['high'] - ohlc['low']

            # VOLATILITY FILTER: Skip if volatility too high (insane market)
            if atr > atr_threshold[asset]:
                continue  # Don't trade in chaos

            # REGIME FILTER: Check if we're in severe drawdown
            if len(equity) > 50:
                recent_peak = max(equity[-50:])
                current_dd = (capital - recent_peak) / recent_peak
                if current_dd < -0.15:  # In 15%+ drawdown
                    continue  # Don't trade during severe losses

            # Entry: NEXT bar's open
            if i + 1 < len(dates):
                entry_price = ohlc_data[asset][i + 1]['open']
            else:
                continue

            # Position sizing: Smaller in higher vol
            if atr > atr_threshold[asset] * 0.8:
                position_size = capital * 0.03  # Reduce size in elevated vol
            else:
                position_size = capital * 0.05  # Normal size

            # BALANCED stops and 2:1 reward/risk (realistic for oil volatility)
            stop_distance = 1.5 * atr  # Not too tight, not too loose
            target_distance = 2.0 * stop_distance  # Achievable 2:1

            if direction == 'long':
                initial_stop = entry_price - stop_distance
                target_price = entry_price + target_distance
                trailing_stop = initial_stop
                best_price = entry_price
            else:
                initial_stop = entry_price + stop_distance
                target_price = entry_price - target_distance
                trailing_stop = initial_stop
                best_price = entry_price

            # Open position
            open_positions[asset] = {
                'entry_bar': i + 1,
                'entry_price': entry_price,
                'direction': direction,
                'prob': prob,
                'position_size': position_size,
                'stop_distance': stop_distance,
                'target_price': target_price,
                'trailing_stop': trailing_stop,
                'best_price': best_price
            }

    trades_df = pd.DataFrame(trades)

    if len(trades) == 0:
        return {
            'return': 0,
            'win_rate': 0,
            'total_trades': 0,
            'sharpe': 0,
            'max_dd': 0
        }

    # Count wins (positive P&L)
    wins = (trades_df['pnl'] > 0).sum()
    total = len(trades_df)

    equity_curve = np.array(equity)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = abs(drawdown.min())

    total_return = (capital - 10000) / 10000 * 100

    return {
        'return': total_return,
        'win_rate': wins / total,
        'total_trades': total,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


if __name__ == '__main__':
    train_simple_lstm()
