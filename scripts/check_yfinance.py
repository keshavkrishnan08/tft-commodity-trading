import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_availability():
    tickers = {
        "WTI": "CL=F",
        "Brent": "BZ=F",
        "NaturalGas": "NG=F",
        "HeatingOil": "HO=F",
        "DXY": "DX-Y.NYB"
    }
    
    # Try different intervals and periods
    intervals = ["60m", "1h"] # yfinance uses "1h" or "60m"
    
    print("Checking intraday data availability (1h)...")
    results = {}
    
    for name, ticker in tickers.items():
        print(f"\nFetching {name} ({ticker})...")
        try:
            # Try last 730 days (max for hour data in yfinance)
            data = yf.download(ticker, period="730d", interval="60m")
            if not data.empty:
                start_date = data.index[0]
                end_date = data.index[-1]
                count = len(data)
                print(f"  ✓ Found {count} bars from {start_date} to {end_date}")
                results[name] = {"start": start_date, "end": end_date, "count": count}
            else:
                print(f"  ✗ No data found for {ticker}")
        except Exception as e:
            print(f"  ✗ Error for {ticker}: {e}")
            
    return results

if __name__ == "__main__":
    check_availability()
