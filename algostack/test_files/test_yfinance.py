#!/usr/bin/env python3
"""Test yfinance data fetching"""

import yfinance as yf
from datetime import datetime, timedelta

# Test fetching SPY data
symbol = "SPY"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Fetching {symbol} from {start_date.date()} to {end_date.date()}")

# Try different approaches
print("\n1. Using datetime objects:")
data1 = yf.download(symbol, start=start_date, end=end_date, progress=False)
print(f"   Got {len(data1)} days of data")
if not data1.empty:
    print(f"   Date range: {data1.index[0].date()} to {data1.index[-1].date()}")

print("\n2. Using string dates:")
data2 = yf.download(symbol, start="2024-01-01", end="2024-12-31", progress=False)
print(f"   Got {len(data2)} days of data")
if not data2.empty:
    print(f"   Date range: {data2.index[0].date()} to {data2.index[-1].date()}")

print("\n3. Using Ticker object:")
ticker = yf.Ticker(symbol)
data3 = ticker.history(period="1y")
print(f"   Got {len(data3)} days of data")
if not data3.empty:
    print(f"   Date range: {data3.index[0].date()} to {data3.index[-1].date()}")

print("\n4. Using period parameter:")
data4 = yf.download(symbol, period="3mo", progress=False)
print(f"   Got {len(data4)} days of data")
if not data4.empty:
    print(f"   Date range: {data4.index[0].date()} to {data4.index[-1].date()}")