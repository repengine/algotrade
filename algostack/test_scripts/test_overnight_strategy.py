#!/usr/bin/env python3
"""Test the overnight drift strategy."""


import numpy as np
import pandas as pd

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
dates = pd.date_range(start="2024-01-01", end="2024-06-01", freq="D")
data = {
    "open": np.random.uniform(100, 110, len(dates)),
    "high": np.random.uniform(105, 115, len(dates)),
    "low": np.random.uniform(95, 105, len(dates)),
    "close": np.random.uniform(100, 110, len(dates)),
    "volume": np.random.uniform(1000000, 5000000, len(dates)),
}
df = pd.DataFrame(data, index=dates)
df.attrs["symbol"] = "SPY"

# Initialize strategy
print("\nInitializing strategy...")
try:
    strategy = OvernightDrift({"symbols": ["SPY"]})
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
        window_data = df.iloc[: i + 1].copy()
        signal = strategy.next(window_data)
        if signal:
            signals_generated += 1
            print(f"  Signal on {df.index[i].strftime('%Y-%m-%d')}: {signal.direction}")

    print("\n✅ Strategy executed successfully!")
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
except Exception as e:
    print(f"❌ calculate_indicators failed: {e}")
    import traceback

    traceback.print_exc()
