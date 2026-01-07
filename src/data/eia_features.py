import os
import pandas as pd
import numpy as np
try:
    import eia
except ImportError:
    eia = None

class EIAFeatureExtractor:
    """
    Fetches fundamental energy data from the EIA API to use as features.
    Requires an API key (https://www.eia.gov/opendata/register.php).
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('EIA_API_KEY')
        if eia and self.api_key:
            self.api = eia.API(self.api_key)
        else:
            self.api = None
            if not eia:
                print("⚠ eia-python not installed. Run 'pip install eia-python'")
            if not self.api_key:
                print("⚠ No EIA API key found. API calls will be mocked.")

    def fetch_crude_stocks(self):
        """Fetch US Weekly Crude Oil Stocks (Series ID: PET.WCESTUS1.W)"""
        if not self.api:
            return self._get_mock_fundamental("US_Crude_Stocks")
            
        try:
            # Series for US Weekly Crude Stocks
            series_id = 'PET.WCESTUS1.W'
            data = self.api.data_by_series(series=series_id)
            df = pd.DataFrame(data, columns=['Date', 'Crude_Stocks'])
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"Error fetching EIA data: {e}")
            return self._get_mock_fundamental("US_Crude_Stocks")

    def _get_mock_fundamental(self, name):
        """Returns dummy fundamental data for pipeline testing spanning 2000-2024."""
        dates = pd.date_range('2000-01-01', '2025-01-01', freq='W')
        n = len(dates)
        # Random walk with slight upward drift to simulate stocks
        values = 400 + np.cumsum(np.random.normal(0.5, 5.0, n))
        df = pd.DataFrame({'Date': dates, name: values})
        return df

if __name__ == "__main__":
    extractor = EIAFeatureExtractor()
    stocks = extractor.fetch_crude_stocks()
    print("Fetched Fundamental Data (Sample):")
    print(stocks.head())
