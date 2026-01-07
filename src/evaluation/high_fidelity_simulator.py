import pandas as pd
import numpy as np
from datetime import timedelta

class HighFidelitySimulator:
    """
    High-fidelity trading simulator that uses 1-hour bars to verify
    stop-loss and take-profit hits within the daily prediction horizon.
    """
    
    def __init__(self, config, intraday_data):
        """
        Args:
            config: Configuration object
            intraday_data: dict of {asset: DataFrame with 1h bars}
        """
        self.config = config
        self.data = intraday_data
        
    def simulate_trade(self, asset, entry_date, direction, position_dollars, 
                       stop_loss_pct, take_profit_pct, horizon_days=5):
        """
        Simulate a trade by stepping through 1-hour bars.
        
        Args:
            asset: Asset name
            entry_date: The date of the daily prediction
            direction: 'long' or 'short'
            position_dollars: Size of position
            stop_loss_pct: SL as decimal (e.g. 0.02)
            take_profit_pct: TP as decimal
            horizon_days: How many days to look forward
            
        Returns:
            trade_result: dict
        """
        if asset not in self.data:
            return None # Data missing
            
        df = self.data[asset]
        
        # Get entry price (Open of the bar following the prediction date)
        # We assume the prediction is made at the close of entry_date
        # so we enter at the next available 1h bar open.
        mask = df.index > pd.to_datetime(entry_date)
        entry_bars = df.loc[mask]
        
        if entry_bars.empty:
            return None
            
        entry_price = entry_bars.iloc[0]['Open']
        entry_timestamp = entry_bars.index[0]
        
        # Define exit levels
        if direction == 'long':
            sl_price = entry_price * (1.0 - stop_loss_pct)
            tp_price = entry_price * (1.0 + take_profit_pct)
        else: # short
            sl_price = entry_price * (1.0 + stop_loss_pct)
            tp_price = entry_price * (1.0 - take_profit_pct)
            
        # Determine horizon end
        horizon_end = entry_timestamp + timedelta(days=horizon_days)
        
        # Step through 1h bars within the horizon
        trade_bars = entry_bars.loc[entry_bars.index <= horizon_end]
        
        exit_price = None
        exit_date = None
        exit_type = 'horizon_expiry'
        trailing_sl = sl_price
        
        for idx, bar in trade_bars.iterrows():
            high = bar['High']
            low = bar['Low']
            
            # 1. Check Stop Loss
            if direction == 'long':
                if low <= trailing_sl:
                    exit_price = trailing_sl
                    exit_date = idx
                    exit_type = 'stop_loss'
                    break
            else: # short
                if high >= trailing_sl:
                    exit_price = trailing_sl
                    exit_date = idx
                    exit_type = 'stop_loss'
                    break
                    
            # 2. Check Take Profit
            if direction == 'long':
                if high >= tp_price:
                    exit_price = tp_price
                    exit_date = idx
                    exit_type = 'take_profit'
                    break
            else: # short
                if low <= tp_price:
                    exit_price = tp_price
                    exit_date = idx
                    exit_type = 'take_profit'
                    break
            
            # 3. Trailing Stop Logic (Simplified v2)
            # If in profit by half ATR, move SL to breakeven
            # (In high-fidelity, we can be much more granular)
            if self.config.ENABLE_TRAILING_STOP:
                current_price = bar['Close']
                if direction == 'long':
                    unrealized_pnl = (current_price - entry_price) / entry_price
                    if unrealized_pnl > (take_profit_pct * 0.5):
                        # Move SL to breakeven + buffer
                        trailing_sl = max(trailing_sl, entry_price)
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price
                    if unrealized_pnl > (take_profit_pct * 0.5):
                        trailing_sl = min(trailing_sl, entry_price)
                        
        # If no trigger, exit at horizon close
        if exit_price is None:
            exit_price = trade_bars.iloc[-1]['Close']
            exit_date = trade_bars.index[-1]
            exit_type = 'horizon_expiry'
            
        # P&L Calculation
        if direction == 'long':
            gross_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pnl_pct = (entry_price - exit_price) / entry_price
            
        gross_pnl = position_dollars * gross_pnl_pct
        
        # Friction
        entry_friction = position_dollars * (self.config.SLIPPAGE_PCT + self.config.COMMISSION_PCT)
        exit_friction = position_dollars * (self.config.SLIPPAGE_PCT + self.config.COMMISSION_PCT)
        total_friction = entry_friction + exit_friction
        
        net_pnl = gross_pnl - total_friction
        
        return {
            'entry_date': entry_timestamp,
            'exit_date': exit_date,
            'exit_type': exit_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'friction': total_friction,
            'net_pnl': net_pnl,
            'duration_hours': len(trade_bars.loc[trade_bars.index <= exit_date])
        }
