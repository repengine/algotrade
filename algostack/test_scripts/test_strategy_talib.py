#!/usr/bin/env python3
"""Test talib functions used in strategies."""

import numpy as np
import pandas as pd

# Test both import methods
print("Testing talib import methods...")

# Method 1: Direct import
try:
    import talib
    print("✅ Direct talib import successful")
    direct_import_ok = True
except ImportError as e:
    print(f"❌ Direct talib import failed: {e}")
    direct_import_ok = False

# Method 2: Fallback import (as used in strategies)
try:
    try:
        import talib
    except ImportError:
        from pandas_indicators import create_talib_compatible_module
        talib = create_talib_compatible_module()
    print("✅ Strategy-style import successful")
    strategy_import_ok = True
except Exception as e:
    print(f"❌ Strategy-style import failed: {e}")
    strategy_import_ok = False

# Test functions used in overnight_drift.py
if strategy_import_ok:
    print("\nTesting functions used in overnight_drift.py...")
    
    # Create sample data
    data = {
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000000, 5000000, 100)
    }
    df = pd.DataFrame(data)
    
    tests = [
        ("SMA(close, 50)", lambda: talib.SMA(df['close'], timeperiod=50)),
        ("ATR(high, low, close, 14)", lambda: talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)),
        ("SMA(volume, 20)", lambda: talib.SMA(df['volume'], timeperiod=20)),
        ("RSI(close, 14)", lambda: talib.RSI(df['close'], timeperiod=14)),
    ]
    
    for name, func in tests:
        try:
            result = func()
            if hasattr(result, '__len__'):
                print(f"✅ {name} - Working (returned {len(result)} values)")
            else:
                print(f"✅ {name} - Working")
        except Exception as e:
            print(f"❌ {name} - Failed: {e}")

# Check if we're in a virtual environment
print("\nEnvironment check:")
import sys
print(f"Python executable: {sys.executable}")
print(f"Virtual environment: {'venv' in sys.executable}")

# Check talib location
if direct_import_ok:
    import talib
    print(f"TA-Lib location: {talib.__file__}")
    print(f"TA-Lib version: {talib.__version__}")