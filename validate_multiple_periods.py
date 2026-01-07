# validate_multiple_periods.py
"""
Test the trading system on MULTIPLE time periods to avoid cherry-picking.

This will show if the system works in normal markets or only during crashes.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Test periods (excluding 2020 crash)
TEST_PERIODS = {
    'Pre-Crisis 2010-2013': ('2010-01-01', '2013-12-31'),
    'Recovery 2013-2016': ('2013-01-01', '2016-12-31'),
    'Bull Market 2016-2019': ('2016-01-01', '2019-12-31'),
    'Post-Crash 2021-2022': ('2021-01-01', '2022-06-30'),
    'Full 2010-2019': ('2010-01-01', '2019-12-31'),  # Excludes crash
}

# Crash period (test separately)
CRASH_PERIOD = ('2020-01-01', '2020-12-31')

print("="*80)
print("MULTI-PERIOD VALIDATION")
print("="*80)
print()
print("This script tests the system on DIFFERENT time periods to ensure")
print("results aren't just lucky timing on the 2020 oil crash.")
print()
print("Expected results:")
print("  Normal markets: +15-25% annual return")
print("  2020 crash: +100-300% (extreme volatility)")
print()
print("="*80)
print()

# TODO: Implement time-period-specific backtesting
print("⚠️  NOT YET IMPLEMENTED")
print()
print("To implement:")
print("1. Modify Config to accept custom date ranges")
print("2. Run backtest on each period")
print("3. Compare results across periods")
print("4. Report if returns are consistent or crash-dependent")
print()
print("For now, manually test by changing data dates in config.py")
