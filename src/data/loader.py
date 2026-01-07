# src/data/loader.py
"""
Data loader for multi-asset time series with DXY confluence
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class DataLoader:
    # Loads and merges data for all assets: WTI, Brent, Natural Gas, Heating Oil, Gold, Silver, Bitcoin

    def __init__(self, config=None):
        self.config = config if config else Config()

    def load_oil_gas_data(self):
        # Load energy commodities: WTI, Brent, Natural Gas, Heating Oil
        print(f"Loading oil & gas data from {self.config.OIL_GAS_FILE}...")

        df_raw = pd.read_csv(self.config.OIL_GAS_FILE)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        # Pivot to wide format
        df_pivot = pd.DataFrame()

        for csv_name, config_name in self.config.ASSET_NAME_MAP.items():
            if config_name == 'DXY':  # Skip DXY, load separately
                continue

            sub_df = df_raw[df_raw['Symbol'] == csv_name].copy()
            if len(sub_df) == 0:
                continue

            sub_df.set_index('Date', inplace=True)
            sub_df = sub_df.sort_index()

            # Keep OHLC
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in sub_df.columns:
                    sub_df.rename(columns={col: f'{config_name}_{col}'}, inplace=True)

            # Merge
            if df_pivot.empty:
                df_pivot = sub_df[[f'{config_name}_Open', f'{config_name}_High',
                                   f'{config_name}_Low', f'{config_name}_Close']]
            else:
                df_pivot = df_pivot.join(
                    sub_df[[f'{config_name}_Open', f'{config_name}_High',
                            f'{config_name}_Low', f'{config_name}_Close']],
                    how='inner'
                )

        df_pivot.reset_index(inplace=True)
        print(f"  Oil & Gas data shape: {df_pivot.shape}")
        return df_pivot

    def load_dxy_data(self):
        """Load DXY (US Dollar Index) data"""
        print(f"Loading DXY data from {self.config.DXY_FILE}...")

        df_dxy = pd.read_csv(self.config.DXY_FILE)
        df_dxy['Date'] = pd.to_datetime(df_dxy['Date'], format='%b %d, %Y')

        # Rename columns
        df_dxy.rename(columns={
            'Price': 'DXY_Close',
            'Open': 'DXY_Open',
            'High': 'DXY_High',
            'Low': 'DXY_Low'
        }, inplace=True)

        df_dxy = df_dxy[['Date', 'DXY_Open', 'DXY_High', 'DXY_Low', 'DXY_Close']]
        df_dxy.sort_values('Date', inplace=True)

        print(f"  DXY data shape: {df_dxy.shape}")
        return df_dxy

    def load_metals_crypto_data(self):
        # Load precious metals (Gold, Silver) and cryptocurrency (Bitcoin) for cross-asset validation
        if not self.config.METALS_CRYPTO_FILE.exists():
            print(f"Warning: {self.config.METALS_CRYPTO_FILE} not found.")
            return pd.DataFrame()

        print(f"Loading Metals & Crypto data from {self.config.METALS_CRYPTO_FILE}...")
        df_raw = pd.read_csv(self.config.METALS_CRYPTO_FILE)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        # Pivot
        df_pivot = pd.DataFrame()
        
        # Get unique symbols in the file
        symbols = df_raw['Symbol'].unique()
        
        for symbol in symbols:
            # Map symbol to config name if needed, but here they match
            config_name = symbol 
            
            sub_df = df_raw[df_raw['Symbol'] == symbol].copy()
            sub_df.set_index('Date', inplace=True)
            sub_df.sort_index(inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close']:
                sub_df.rename(columns={col: f'{config_name}_{col}'}, inplace=True)
                
            cols_to_merge = [f'{config_name}_{col}' for col in ['Open', 'High', 'Low', 'Close']]
            
            if df_pivot.empty:
                df_pivot = sub_df[cols_to_merge]
            else:
                df_pivot = df_pivot.join(sub_df[cols_to_merge], how='outer') # Outer join to preserve history if dates differ
                
        df_pivot.reset_index(inplace=True)
        print(f"  Metals & Crypto data shape: {df_pivot.shape}")
        return df_pivot

    def merge_all_data(self):
        """Merge oil/gas with DXY"""
        print("\nMerging all assets...")

        df_oil_gas = self.load_oil_gas_data()
        df_metals_crypto = self.load_metals_crypto_data()
        df_dxy = self.load_dxy_data()

        # Merge Oil/Gas and Metals/Crypto first
        if not df_metals_crypto.empty:
            # Outer join to maximize data availability, will dropna later
            df_assets = pd.merge(df_oil_gas, df_metals_crypto, on='Date', how='outer')
        else:
            df_assets = df_oil_gas

        # Merge with DXY (Inner join strictly limits to DXY availability)
        df_merged = pd.merge(df_assets, df_dxy, on='Date', how='inner')

        # Filter date range
        df_merged = df_merged[
            (df_merged['Date'] >= self.config.START_DATE) &
            (df_merged['Date'] <= self.config.END_DATE)
        ]

        df_merged.sort_values('Date', inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        # Handle missing values
        df_merged.fillna(method='ffill', inplace=True)
        df_merged.dropna(inplace=True)

        print(f"  Merged data shape: {df_merged.shape}")
        print(f"  Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")
        print(f"  Assets: {self.config.ALL_ASSETS}")

        return df_merged

    def get_data(self):
        """Main method to get clean merged data"""
        return self.merge_all_data()


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    df = loader.get_data()
    print("\nData loaded successfully!")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
