#!/usr/bin/env python3
"""Test walk-forward optimization fix."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock talib before imports
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

from algostack.dashboard import get_optimization_ranges

# Test all strategies have reasonable parameter combinations
strategies = [
    'MeanReversionEquity',
    'TrendFollowingMulti', 
    'HybridRegime',
    'IntradayOrb',
    'OvernightDrift',
    'PairsStatArb'
]

print("Testing parameter combinations for walk-forward optimization:\n")

total_all = 0
for strategy in strategies:
    ranges = get_optimization_ranges(strategy)
    if ranges:
        total = 1
        for param, values in ranges.items():
            total *= len(values)
        total_all += total
        print(f"{strategy}:")
        print(f"  Parameters: {list(ranges.keys())}")
        print(f"  Total combinations: {total}")
        print()

print(f"Total combinations across all strategies: {total_all}")
print("\nTarget: <50 combinations per strategy, <200 total")
print(f"Result: {'✅ PASS' if total_all < 200 else '❌ FAIL'}")