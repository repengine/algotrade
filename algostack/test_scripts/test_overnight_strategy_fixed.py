#!/usr/bin/env python3
"""Test the overnight drift strategy with correct configuration."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test importing the strategy
try:
    from strategies.overnight_drift import OvernightDrift
    print("✅ Successfully imported OvernightDrift strategy")
except Exception as e:
    print(f"❌ Failed to import OvernightDrift: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create test data
print("\nCreating test data...")
dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
data = {
    'open': np.random.uniform(100, 110, len(dates)),
    'high': np.random.uniform(105, 115, len(dates)),
    'low': np.random.uniform(95, 105, len(dates)),
    'close': np.random.uniform(100, 110, len(dates)),
    'volume': np.random.uniform(1000000, 5000000, len(dates))
}
df = pd.DataFrame(data, index=dates)
df.attrs['symbol'] = 'SPY'

# Initialize strategy with all required parameters
print("\nInitializing strategy...")
config = {
    'symbols': ['SPY'],
    'lookback_days': 252,  # Required by validator
    'min_edge': 0.01,      # Required by validator (1%)
    'min_win_rate': 0.50,  # Required by validator (50%)
    'entry_time': '15:45', # Required by validator
    # Optional parameters that are in the strategy's default_config
    'lookback_period': 252,
    'hold_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],
    'vix_threshold': 30,
    'volume_threshold': 0.8,
    'trend_filter': True,
    'sma_period': 50,
    'momentum_period': 20,
    'min_atr': 0.005,
    'max_atr': 0.03,
    'earnings_blackout_days': 2,
    'fomc_blackout_days': 1,
    'max_positions': 2,
}

try:
    strategy = OvernightDrift(config)
    print("✅ Strategy initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize strategy: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test strategy execution
print("\nTesting strategy execution...")
try:
    # Run strategy on a few data points
    signals_generated = 0
    for i in range(50, min(60, len(df))):
        window_data = df.iloc[:i+1].copy()
        signal = strategy.next(window_data)
        if signal:
            signals_generated += 1
            print(f"  Signal on {df.index[i].strftime('%Y-%m-%d')}: {signal.direction}")
    
    print(f"\n✅ Strategy executed successfully!")
    print(f"   Generated {signals_generated} signals")
    
except Exception as e:
    print(f"❌ Strategy execution failed: {e}")
    import traceback
    traceback.print_exc()

# Test calculate_indicators directly
print("\nTesting calculate_indicators method...")
try:
    indicators_df = strategy.calculate_indicators(df)
    print("✅ calculate_indicators successful")
    print(f"   Indicators calculated: {list(indicators_df.columns)}")
    
    # Check if talib indicators were calculated
    talib_indicators = ['sma', 'atr', 'volume_sma', 'rsi']
    for ind in talib_indicators:
        if ind in indicators_df.columns:
            non_nan = indicators_df[ind].notna().sum()
            print(f"   {ind}: {non_nan} non-NaN values")
            
except Exception as e:
    print(f"❌ calculate_indicators failed: {e}")
    import traceback
    traceback.print_exc()