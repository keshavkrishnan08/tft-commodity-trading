# Data Files and Terms of Use

## Should Data Files Be Included in Repository?

**Recommendation: NO** - Do not include raw data files in the GitHub repository.

### Reasons:

1. **Terms of Use:**
   - **Yahoo Finance:** Terms of Service generally allow downloading for personal use but may restrict redistribution. Including data files could violate terms.
   - **EIA (Energy Information Administration):** Public data, but large files may have redistribution restrictions.
   - **FRED (Federal Reserve):** Generally allows redistribution, but attribution required.

2. **File Size:**
   - Data files can be large (several MB to GB)
   - GitHub has file size limits (100MB per file, 1GB per repo recommended)
   - Large files slow down repository cloning

3. **Data Freshness:**
   - Historical data should be downloaded fresh from original sources
   - Ensures users get complete, up-to-date datasets
   - Avoids version conflicts

4. **Best Practices:**
   - Reproducibility is achieved through code, not data
   - Users should download data themselves to ensure they have the correct version
   - Data download scripts can be provided (see `scripts/fetch_robust.py`)

## What Users Should Do:

1. **Download data from original sources:**
   - Energy commodities: EIA website
   - Precious metals and Bitcoin: Yahoo Finance (via `yfinance` Python library)
   - DXY: FRED or Yahoo Finance

2. **Use provided scripts:**
   - `scripts/fetch_robust.py` - Automated data download script (if available)
   - `scripts/standardize_metals_crypto.py` - Data standardization script

3. **Place data in correct location:**
   - `data/oil and gas.csv` - Energy commodities
   - `data/metals_crypto.csv` - Gold, Silver, Bitcoin
   - `data/dxy.csv` - US Dollar Index

## Data Format Requirements:

All CSV files should have columns:
- `Date` - Date in format YYYY-MM-DD or similar
- `Open` - Opening price
- `High` - High price
- `Low` - Low price
- `Close` - Closing price
- `Volume` - Trading volume (optional)

## Alternative: Sample Data

If you want to include a small sample dataset for testing:
- Include only a few months of data (e.g., 2022-01 to 2022-03)
- Clearly label as "sample data" in filename
- Add note that full dataset must be downloaded for full experiments

## Conclusion

**Do NOT include full data files.** Instead:
- Provide clear instructions in `EXPERIMENTAL_SETUP.md`
- Include data download scripts if possible
- Document data sources and formats
- Let users download fresh data from original sources

This approach:
- ✅ Respects terms of use
- ✅ Keeps repository size manageable
- ✅ Ensures data freshness
- ✅ Follows best practices for reproducible research

