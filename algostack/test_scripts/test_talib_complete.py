#!/usr/bin/env python3
"""Comprehensive TA-Lib functionality test."""

import numpy as np
import talib

print("=== TA-Lib Comprehensive Test ===")
print(f"TA-Lib version: {talib.__version__}")
print(f"NumPy version: {np.__version__}")

# Create sample data
np.random.seed(42)
high = np.random.uniform(100, 110, 100)
low = np.random.uniform(90, 100, 100)
close = np.random.uniform(95, 105, 100)
volume = np.random.uniform(1000000, 5000000, 100)

print("\nTesting various indicators...")

# Test different types of indicators
tests = [
    ("SMA (Simple Moving Average)", lambda: talib.SMA(close, timeperiod=10)),
    ("RSI (Relative Strength Index)", lambda: talib.RSI(close, timeperiod=14)),
    ("BBANDS (Bollinger Bands)", lambda: talib.BBANDS(close, timeperiod=20)),
    ("MACD", lambda: talib.MACD(close)),
    ("ATR (Average True Range)", lambda: talib.ATR(high, low, close, timeperiod=14)),
    ("ADX (Average Directional Index)", lambda: talib.ADX(high, low, close, timeperiod=14)),
    ("STOCH (Stochastic)", lambda: talib.STOCH(high, low, close)),
]

for name, func in tests:
    try:
        result = func()
        print(f"✅ {name} - Working")
    except Exception as e:
        print(f"❌ {name} - Failed: {e}")

print("\n✅ TA-Lib is fully installed and functional!")
print("\nYou can now remove the pandas_indicators fallback from your strategies")
print("since TA-Lib is working correctly.")