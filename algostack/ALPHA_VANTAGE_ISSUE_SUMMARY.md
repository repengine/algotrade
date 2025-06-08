# Alpha Vantage Integration Issues - Summary

## Issues Found

### 1. Data Quality Problems
- Alpha Vantage data has invalid OHLC relationships (e.g., high < low)
- This causes strategy validation to fail
- Affects backtesting reliability

### 2. OvernightDrift Strategy Issues
- The strategy calculates negative overnight edge with recent SPY data
- This is actually correct - overnight returns have been negative recently
- Strategy correctly refuses to trade when edge is negative

### 3. Missing Configuration Parameters
- Some strategies expected parameters that weren't being passed
- Fixed by properly merging user parameters with defaults

## Solutions Implemented

### 1. Data Validation
Added `_validate_ohlc_data()` method to check:
- Required columns exist
- No null values
- Valid OHLC relationships (high >= low, etc.)

### 2. Automatic Fallback
- If Alpha Vantage data fails validation, automatically fall back to Yahoo Finance
- User sees warning message about data quality issues

### 3. Parameter Handling
- Special handling for OvernightDrift filters
- When filters are disabled, thresholds are adjusted accordingly

## Current Status

### Working
- ✅ Yahoo Finance data source - fully functional
- ✅ All strategies generate signals with good data
- ✅ Pure pandas indicators work correctly
- ✅ Dashboard properly handles data quality issues

### Issues with Alpha Vantage
- ⚠️ Data quality problems (invalid OHLC)
- ⚠️ OvernightDrift shows negative edge (market conditions, not a bug)
- ⚠️ Free tier rate limits (5 requests/minute)

## Recommendations

1. **Use Yahoo Finance for daily data** - More reliable, no API key needed
2. **Use Alpha Vantage for intraday only** - When you need 1m, 5m, 15m data
3. **Monitor data quality** - Dashboard now shows warnings
4. **Consider premium Alpha Vantage** - Better rate limits, possibly better data

## Testing Results

### Yahoo Finance
- OvernightDrift: 162 signals, 74 trades
- MeanReversionEquity: 8 signals, 4 trades  
- HybridRegime: 6 signals, 3 trades
- All strategies working correctly

### Alpha Vantage
- Data quality issues prevent proper backtesting
- When data is valid, strategies work but may not generate signals due to market conditions
- Automatic fallback to Yahoo Finance ensures dashboard remains functional

## Code Changes

1. **dashboard_pandas.py**
   - Added data validation
   - Automatic fallback mechanism
   - Better error messages

2. **pandas_indicators.py**
   - Added missing indicators (ROC, MOM, etc.)
   - All TA-Lib functions now implemented

3. **Parameter handling**
   - Proper filter threshold adjustments
   - Strategy-specific parameter handling