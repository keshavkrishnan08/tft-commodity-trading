import yfinance as yf
import pandas as pd
import requests
import time
from pathlib import Path

# Set up a browser-like session for yfinance
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def fetch_intraday_robust():
    tickers = {
        "WTI": "CL=F",
        "Brent": "BZ=F",
        "NaturalGas": "NG=F",
        "HeatingOil": "HO=F",
        "DXY": "DX-Y.NYB"
    }
    
    save_dir = Path("/Users/keshavkrishnan/Oil_Project/data/intraday")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Fetching 1h Intraday Bars (2024-2025 OOS)...")
    
    for name, ticker_sym in tickers.items():
        print(f"\n  Processing {name} ({ticker_sym})...")
        try:
            # Using the session to avoid 429
            ticker = yf.Ticker(ticker_sym, session=session)
            
            # yfinance 1h data is limited to 730 days
            data = ticker.history(period="730d", interval="1h")
            
            if not data.empty:
                save_path = save_dir / f"{name}_1h_new.csv"
                data.to_csv(save_path)
                print(f"  ✓ Success! Saved {len(data)} bars to {save_path}")
                print(f"  Range: {data.index[0]} to {data.index[-1]}")
            else:
                print(f"  ✗ Failed: No data returned for {ticker_sym}")
        except Exception as e:
            print(f"  ✗ Error for {name}: {e}")
            
        time.sleep(2) # Politeness delay

if __name__ == "__main__":
    fetch_intraday_robust()
