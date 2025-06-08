#!/usr/bin/env python3
"""Debug walk-forward issue."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock talib
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import components
from core.backtest_engine import WalkForwardAnalyzer
from dashboard_pandas import PandasStrategyManager
from dashboard_enhanced import get_optimization_ranges

# Initialize
strategy_manager = PandasStrategyManager()

# Create small test data
dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
test_data = pd.DataFrame({
    'open': prices * 0.99,
    'high': prices * 1.01,
    'low': prices * 0.98,
    'close': prices,
    'Close': prices,
    'volume': np.random.randint(1000000, 2000000, len(dates))
}, index=dates)
test_data.attrs['symbol'] = 'TEST'

print(f"Test data: {len(test_data)} days")
print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")

# Get strategy
strategy_name = 'MeanReversionEquity'
strategy_class = strategy_manager.strategies[strategy_name]

# Get minimal parameter ranges
param_ranges = {
    'lookback_period': [20, 30],  # Just 2 values
    'zscore_threshold': [2.0, 2.5],  # Just 2 values
}
print(f"\nParameter ranges: {param_ranges}")
print(f"Total combinations: {2*2} = 4")

# Create simple backtest function
def debug_backtest_func(strategy_instance, data_slice, cost_model=None):
    print(f"  Backtest called with data slice: {len(data_slice)} days")
    
    # Get config from strategy
    if hasattr(strategy_instance, 'config'):
        config = strategy_instance.config
    else:
        config = {'lookback_period': 20, 'zscore_threshold': 2.0}
    
    # Just return dummy results for debugging
    return {
        'sharpe_ratio': np.random.random(),
        'total_return': np.random.random() * 10,
        'max_drawdown': np.random.random() * 20,
        'num_trades': np.random.randint(5, 20)
    }

# Test walk-forward with small window
print("\nRunning walk-forward test...")
wf_analyzer = WalkForwardAnalyzer(
    window_size=200,  # ~8 months
    step_size=30,     # 1 month steps
    min_window_size=100
)

start_time = time.time()

try:
    # Add timeout
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Walk-forward took too long")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    wf_results = wf_analyzer.run_analysis(
        strategy_class,
        test_data,
        param_ranges,
        debug_backtest_func,
        cost_model=None
    )
    
    signal.alarm(0)  # Cancel timeout
    
    elapsed = time.time() - start_time
    print(f"\n✅ Walk-forward completed in {elapsed:.1f} seconds")
    print(f"Results shape: {wf_results.shape}")
    print(f"Windows analyzed: {len(wf_results)}")
    
except TimeoutError:
    print("\n❌ Walk-forward timed out after 30 seconds!")
    print("This indicates an infinite loop or performance issue")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()