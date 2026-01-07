
import numpy as np
import pandas as pd
from src.utils.config import Config
from src.evaluation.advanced_backtest import AdvancedBacktest
from src.data.dataset import MultiAssetDataset

# Mock Dataset
class MockDataset:
    def __init__(self):
        self.sequence_length = 60
    
    def get_raw_prices(self, idx, asset):
        # Return logical prices based on index
        # We want to show that get_raw_prices(idx, asset) returns price at T+seq_len
        # Let's say price = idx + seq_len
        val = idx + self.sequence_length
        # Return [Open, High, Low, Close]
        return [val, val+1, val-1, val]

def test_volatility_indexing():
    print("Testing Volatility Indexing...")
    backtest = AdvancedBacktest()
    dataset = MockDataset()
    
    # We want to estimate volatility at current_idx = 100
    # The 'current' time T should be 100 + 59 = 159.
    # We expect volatility to be calculated using prices around 159.
    
    # Overwrite _estimate_volatility to log what it accesses
    indices_accessed = []
    original_get_raw = dataset.get_raw_prices
    
    def spy_get_raw_prices(idx, asset):
        indices_accessed.append(idx + dataset.sequence_length) # The actual time T being accessed
        return original_get_raw(idx, asset)
    
    dataset.get_raw_prices = spy_get_raw_prices
    
    backtest._estimate_volatility(dataset, 'WTI', current_idx=100)
    
    print(f"Indices accessed for Volatility Calculation at t=100 (SeqLen=60):")
    print(f"Min: {min(indices_accessed)}, Max: {max(indices_accessed)}")
    
    if max(indices_accessed) < 159:
        print("FAIL: Volatility uses OLD data! (Expected access around 159)")
    else:
        print("PASS: Volatility uses recent data.")

def test_gap_fill():
    print("\nTesting Gap Fill Logic...")
    backtest = AdvancedBacktest()
    backtest.config.TARGET_ASSETS = ['TEST']
    backtest.reset()
    
    # Setup specific scenario:
    # Entry: 100
    # SL: 98 (2%)
    # Next Day Open: 90 (Gap Down 10%)
    # Next Day Low: 85
    
    # We need to mock 'dataset' to return this specific sequence
    class ScenarioDataset(MockDataset):
        def get_raw_prices(self, idx, asset):
            # idx matches current_idx + d
            # If current_idx = 0.
            # d=0 (Day 1)
            return [90.0, 95.0, 85.0, 92.0] # Open, High, Low, Close
            
    dataset = ScenarioDataset()
    decision = {
        'direction': 'long',
        'position_dollars': 1000,
        'position_fraction': 0.1,
        'stop_loss_pct': 0.02, # SL at 98
        'take_profit_pct': 0.1, # TP at 110
        'risk_reward': 5.0,
        'raw_probability': 0.6,
        'calibrated_probability': 0.6
    }
    
    # Mock entry at 100
    # We need to hack dataset.get_raw_prices to returns 100 for entry_raw
    # The code calls get_raw_prices(current_idx) for entry.
    # Then get_raw_prices(current_idx + d) for loop.
    # Wait, code does: entry_raw = dataset.get_raw_prices(current_idx, asset)
    # Then loops d in range(horizon) starting with get_raw_prices(current_idx + d)
    # So d=0 is the SAME DAY as entry?!
    
    # Let's check AdvancedBacktest again.
    # entry_raw = dataset.get_raw_prices(current_idx)
    # for d in range(horizon): price = dataset.get_raw_prices(current_idx + d)
    # YES. d=0 is the entry day.
    
    # But usually we enter on T (Close) or T+1 (Open).
    # If we enter on T+1 Open, then the price action of T+1 involves T+1 Low.
    # So checking T+1 Low is correct.
    
    # In this mock:
    # We enter at 100.
    # Data returns Open=90.
    # Means we filled at 100 (which is weird if Open is 90, but let's assume we filled at 100 somehow, maybe previous Close).
    # But immediately the price is 90.
    
    # We want to verify EXIt PRICE.
    # Code calculates: sl_price = 100 * (1 - 0.02) = 98.
    # Low (85) <= SL (98). Hit.
    # Exit Price should be 90 (Open) or slightly worse, but definitely NOT 98.
    
    # Mock specific return for entry call vs loop call
    # We can't easily distinguish args (both are current_idx).
    # But let's assume entry is 100.
    
    # We can modify _fallback_sim to check if needed, but we rely on _simulate_trade logic which we can't easily override without subclassing or rewrite.
    # Better to run it and verify output pnl.
    
    res = backtest._simulate_trade(
        decision, 
        actual=1, 
        asset='TEST', 
        date='2022-01-01', 
        volatility=0.02, 
        current_idx=0, 
        dataset=dataset
    )
    
    # We forced Entry to be dataset[0].Open = 90
    # Wait, the code sets entry_price = float(entry_raw[0]).
    # So Entry Price is 90.
    # SL = 90 * 0.98 = 88.2.
    # Low is 85.
    # Stopped out.
    # Exit Price = SL Price = 88.2.
    # But Open was 90. So we entered at 90.
    # This doesn't reproduce the "Gap Past SL" scenario well because Entry and Day1 are same.
    
    # To repro Gap Past SL:
    # We need Entry Price >> Day1 Price.
    # But `entry_price` is read from `dataset`.
    
    # Scenario:
    # We entered YESTERDAY at 100.
    # Today Open is 90.
    # But `_simulate_trade` simulates the WHOLE trade starting TODAY.
    # It assumes we enter at `entry_raw[0]`.
    
    # If the logic is "Enter at Open of T+1", and "Stop Out based on Low of T+1".
    # Scenario: Open at 100. Low drops to 90. SL at 98.
    # We enter at 100.
    # Price drops. Crosses 98. Trigger SL.
    # We fill at 98. This is correct (intraday move).
    
    # Scenario Gap Down:
    # T+1 Open IS the entry.
    # So we enter at gap down price.
    # This avoids the Gap Loss on entry.
    
    # BUT! What if the Gap happens on T+2?
    # d=0 (Day 1): Open=100, Close=100. Safe.
    # d=1 (Day 2): Open=90, Low=85. SL=98.
    # Here, we carry position from Day 1 (100).
    # Day 2 opens at 90.
    # SL is 98.
    # 90 < 98.
    # We ARE gapped.
    # We should exit at Open (90).
    # Code sets `exit_price = sl_price` (98).
    # THIS IS THE BUG.
    
    class MultiDayDataset(MockDataset):
        def get_raw_prices(self, idx, asset):
            if idx == 0:
                return [100, 105, 99, 100] # Day 1: Enter 100. Safe. Low 99 > SL 98.
            if idx == 1:
                return [90, 92, 85, 88]    # Day 2: Gap Down to 90.
            return [100, 100, 100, 100]

    dataset_gap = MultiDayDataset()
    res = backtest._simulate_trade(
        decision, actual=1, asset='TEST', date='2022-01-01', volatility=0.02, current_idx=0, dataset=dataset_gap
    )
    
    pnl = res['pnl']
    algo_exit = (98 - 100)/100 # -2%
    real_exit = (90 - 100)/100 # -10%
    
    print(f"Trade Result PnL: ${pnl:.2f}")
    
    # If PnL reflects -2%, bug confirmed.
    expected_loss_dollars = 1000 * -0.02
    if abs(pnl - expected_loss_dollars) < 50: # Allow for friction
        print(f"FAIL: Gap Fill Bug Confirmed! Loss is small (-2%) despite 10% gap.")
    else:
        print(f"PASS: Loss reflects gap.")

if __name__ == "__main__":
    test_volatility_indexing()
    test_gap_fill()
