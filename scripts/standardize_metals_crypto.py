import pandas as pd
from pathlib import Path
import os

def standardize_data():
    project_root = Path(os.getcwd())
    data_dir = project_root / 'data'
    
    # Define source files and their format quirks
    sources = [
        {
            'file': 'gold prices.csv',
            'symbol': 'Gold',
            'format': 'MM/DD/YYYY',
            'cols': {'Close/Last': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume', 'Date': 'Date'}
        },
        {
            'file': 'silver prices.csv',
            'symbol': 'Silver',
            'format': 'MM/DD/YYYY',
            'cols': {'Close/Last': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume', 'Date': 'Date'}
        },
        {
            'file': 'BTC-USD.csv',
            'symbol': 'BTC',
            'format': 'YYYY-MM-DD',
            'cols': {'Close': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume', 'Date': 'Date'}
        }
    ]
    
    standardized_dfs = []
    
    for source in sources:
        file_path = project_root / source['file']
        if not file_path.exists():
            print(f"Skipping {source['symbol']}: File not found at {file_path}")
            continue
            
        print(f"Processing {source['symbol']}...")
        df = pd.read_csv(file_path)
        
        # Renaissance Columns
        df.rename(columns=source['cols'], inplace=True)
        
        # Standardize Date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add Symbol and Currency
        df['Symbol'] = source['symbol']
        df['Currency'] = 'USD'
        
        # Keep only standard columns
        keep_cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Currency']
        
        # Clean numeric columns (remove $ or commas if any)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[keep_cols]
        standardized_dfs.append(df)
        
    if not standardized_dfs:
        print("No data processed.")
        return

    # Merge all into one dataframe
    final_df = pd.concat(standardized_dfs, ignore_index=True)
    
    # Sort
    final_df.sort_values(['Symbol', 'Date'], inplace=True)
    
    # Save
    out_path = data_dir / 'metals_crypto.csv'
    final_df.to_csv(out_path, index=False)
    print(f"Successfully saved {len(final_df)} rows to {out_path}")
    print(final_df.head())
    print(final_df['Symbol'].value_counts())

if __name__ == "__main__":
    standardize_data()
