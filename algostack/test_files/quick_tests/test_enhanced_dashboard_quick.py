#!/usr/bin/env python3
"""Quick test of enhanced dashboard functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock talib
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import dashboard components
from algostack.dashboard import enhanced_backtest, get_optimization_ranges
from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager

print("Testing Enhanced Dashboard Components...\n")

# Initialize managers
strategy_manager = PandasStrategyManager()
data_manager = AlphaVantageDataManager()

# Create dummy data
print("1. Creating test data...")
dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
dummy_data = pd.DataFrame({
    'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
    'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.002)),
    'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.002)),
    'close': prices,
    'Close': prices,  # Both formats
    'volume': np.random.randint(1000000, 2000000, len(dates))
}, index=dates)
dummy_data.attrs['symbol'] = 'TEST'
print(f"   Created {len(dummy_data)} days of data")

# Test configuration
strategy_name = 'MeanReversionEquity'
strategy_class = strategy_manager.strategies.get(strategy_name)

if not strategy_class:
    print("❌ MeanReversionEquity strategy not found!")
    sys.exit(1)

user_params = {
    'symbol': 'TEST',
    'lookback_period': 30,
    'zscore_threshold': 2.0,
    'exit_zscore': 0.5,
    'position_size': 0.95
}

cost_config = {
    'enabled': True,
    'commission_per_share': 0.005,
    'commission_type': 'per_share',
    'spread_model': 'fixed',
    'base_spread_bps': 5,
    'slippage_model': 'linear',
    'market_impact_factor': 0.1
}

split_config = {
    'method': 'sequential',
    'oos_ratio': 0.2
}

validation_config = {
    'monte_carlo': True,
    'regime_analysis': True,
    'walk_forward': False  # Disabled for quick test
}

print("\n2. Running enhanced backtest...")
try:
    results = enhanced_backtest(
        strategy_manager,
        strategy_class,
        strategy_name,
        user_params,
        dummy_data,
        initial_capital=100000,
        cost_config=cost_config,
        split_config=split_config,
        validation_config=validation_config
    )
    
    print("\n3. Results Summary:")
    print(f"   IS Sharpe: {results['is_results'].get('sharpe_ratio', 0):.2f}")
    print(f"   OOS Sharpe: {results['oos_results'].get('sharpe_ratio', 0):.2f}")
    print(f"   Performance Decay: {results['performance_decay']['sharpe_decay']*100:.1f}%")
    
    if 'validation' in results:
        val = results['validation']
        print(f"\n   Statistical Validation:")
        print(f"     P-value: {val.get('p_value', 'N/A')}")
        print(f"     Significant: {val.get('significant', 'N/A')}")
        
    if 'regime_analysis' in results:
        regime = results['regime_analysis']
        print(f"\n   Regime Analysis:")
        print(f"     Consistency Score: {regime['consistency_score']*100:.1f}%")
        print(f"     All Regimes Positive: {regime['all_positive']}")
    
    print("\n✅ Enhanced backtest completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during backtest: {e}")
    import traceback
    traceback.print_exc()

# Test walk-forward parameter ranges
print("\n4. Testing walk-forward parameter ranges...")
ranges = get_optimization_ranges(strategy_name)
print(f"   Parameters: {list(ranges.keys())}")
total = 1
for param, values in ranges.items():
    total *= len(values)
print(f"   Total combinations: {total}")
print(f"   {'✅ PASS' if total < 50 else '❌ FAIL'} (target: <50)")

print("\n✅ All tests completed!")