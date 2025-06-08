# AlgoStack Pandas Dashboard

A refactored version of the AlgoStack dashboard that uses pure pandas implementations of technical indicators, eliminating the TA-Lib dependency.

## Features

- **Pure Pandas Indicators**: All technical indicators implemented using only pandas and numpy
- **Alpha Vantage Integration**: Support for intraday market data via Alpha Vantage API
- **Yahoo Finance Fallback**: Automatic fallback to yfinance when Alpha Vantage is unavailable
- **Full Strategy Support**: Compatible with all existing AlgoStack strategies
- **No TA-Lib Required**: Easier installation and deployment

## Installation

1. Install minimal requirements:
```bash
pip install pandas numpy yfinance streamlit plotly pyyaml requests
```

2. Set up Alpha Vantage API key (optional):
   - Either set environment variable: `export ALPHA_VANTAGE_API_KEY=your_key`
   - Or configure in `config/secrets.yaml`:
   ```yaml
   data_providers:
     alphavantage:
       api_key: your_key_here
   ```

3. For premium Alpha Vantage accounts:
```bash
export ALPHA_VANTAGE_PREMIUM=true
```

## Running the Dashboard

```bash
./start_pandas_dashboard.sh
```

Or directly:
```bash
streamlit run dashboard_pandas.py
```

## Data Sources

### Yahoo Finance (Default)
- Daily data only
- No API key required
- Reliable for historical backtesting

### Alpha Vantage
- Intraday data (1m, 5m, 15m, 30m, 60m)
- Daily data with extended history
- Requires API key
- Rate limits:
  - Free tier: 5 requests/minute
  - Premium tier: 75 requests/minute

## Technical Indicators

The following indicators are implemented in pure pandas:

- **Trend**: SMA, EMA, MACD, SAR
- **Momentum**: RSI, STOCH, CCI, WILLR, TRIX
- **Volatility**: ATR, Bollinger Bands
- **Volume**: OBV, MFI
- **Directional**: ADX, PLUS_DI, MINUS_DI

## Advantages Over TA-Lib Version

1. **Easier Installation**: No C dependencies or compilation required
2. **Better Portability**: Works on any platform with Python
3. **Transparent Implementation**: All indicator logic is visible and modifiable
4. **Consistent Results**: No version-specific TA-Lib quirks
5. **Easy Debugging**: Pure Python code is easier to debug and understand

## Performance

The pandas implementations are optimized for vectorized operations and should provide comparable performance to TA-Lib for most use cases. For extremely large datasets or high-frequency applications, some indicators may be slightly slower than their TA-Lib counterparts.

## Compatibility

This dashboard is fully compatible with all existing AlgoStack strategies. The `pandas_indicators.py` module provides a TA-Lib compatible interface, so strategies don't need to be modified.

## Troubleshooting

### "No module named yaml"
Install PyYAML: `pip install pyyaml`

### Alpha Vantage rate limit errors
- Free tier is limited to 5 requests/minute
- Add delays between requests or upgrade to premium

### Missing data for symbols
- Some symbols may not be available on Alpha Vantage
- Dashboard automatically falls back to Yahoo Finance

## Development

To add new indicators:

1. Implement in `pandas_indicators.py` following the existing pattern
2. Add to the `PandasIndicators` class
3. Test against known values

Example:
```python
@staticmethod
def NEW_INDICATOR(close, timeperiod=14):
    """Your indicator description."""
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Your implementation here
    result = close.rolling(window=timeperiod).mean()  # Example
    
    return result
```