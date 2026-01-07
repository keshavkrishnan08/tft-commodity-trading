import yfinance as yf
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

class IntradayDownloader:
    """Robust downloader for yfinance intraday data with retry logic."""
    
    def __init__(self, tickers=None):
        self.tickers = tickers if tickers else {
            "WTI": "CL=F",
            "Brent": "BZ=F",
            "NaturalGas": "NG=F",
            "HeatingOil": "HO=F",
            "DXY": "DX-Y.NYB"
        }
        self.max_retries = 5
        self.base_delay = 5 # seconds
        
    def download_asset(self, ticker_symbol, name):
        """Download single asset with exponential backoff."""
        retries = 0
        while retries < self.max_retries:
            try:
                print(f"  Downloading {name} ({ticker_symbol})...")
                # Interval 1h, period 730d (max for yfinance)
                data = yf.download(ticker_symbol, period="730d", interval="1h", progress=False)
                
                if not data.empty:
                    # Flatten columns if multi-index (sometimes happens with yfinance)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    return data
                
                print(f"  [!] No data returned for {name}, retrying...")
            except Exception as e:
                print(f"  [!] Error downloading {name}: {e}")
            
            retries += 1
            delay = self.base_delay * (2 ** retries) + random.uniform(0, 5)
            print(f"  [!] Retry {retries}/{self.max_retries} in {delay:.1f}s...")
            time.sleep(delay)
            
        return None

    def download_all(self, save_dir=None):
        """Download all tickers and return a dict of DataFrames."""
        all_data = {}
        for name, ticker in self.tickers.items():
            data = self.download_asset(ticker, name)
            if data is not None:
                all_data[name] = data
                if save_dir:
                    path = Path(save_dir) / f"{name}_1h.csv"
                    data.to_csv(path)
                    print(f"  ✓ Saved {name} to {path}")
            else:
                print(f"  ✗ Failed to download {name} after {self.max_retries} attempts.")
        
        return all_data

if __name__ == "__main__":
    # Test script
    import sys
    save_path = Path("/Users/keshavkrishnan/Oil_Project/data/intraday")
    save_path.mkdir(parents=True, exist_ok=True)
    
    downloader = IntradayDownloader()
    print("🚀 Starting Intraday Download (1h bars)...")
    downloader.download_all(save_dir=save_path)
