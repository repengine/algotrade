#!/usr/bin/env python3
"""Test Monte Carlo validation fix."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from core.backtest_engine import MonteCarloValidator

print("Testing Monte Carlo validation fix...\n")

# Test 1: Empty returns
print("Test 1: Empty returns series")
empty_results = {"sharpe_ratio": 0, "returns_series": pd.Series([])}
validator = MonteCarloValidator(n_simulations=100)
result = validator.validate_strategy(empty_results)
print(f"Result: {result}")
print("✅ Passed - no crash\n")

# Test 2: Single return value
print("Test 2: Single return value")
single_results = {"sharpe_ratio": 1.0, "returns_series": pd.Series([0.01])}
result = validator.validate_strategy(single_results)
print(f"Result p-value: {result.get('p_value', 'N/A')}")
print("✅ Passed - no crash\n")

# Test 3: All zero returns
print("Test 3: All zero returns")
zero_results = {"sharpe_ratio": 0, "returns_series": pd.Series([0.0] * 100)}
result = validator.validate_strategy(zero_results)
print(f"Result p-value: {result.get('p_value', 'N/A')}")
print(f"Confidence interval: {result.get('confidence_interval', 'N/A')}")
print("✅ Passed - no crash\n")

# Test 4: Normal returns
print("Test 4: Normal returns (should work)")
normal_results = {
    "sharpe_ratio": 1.2,
    "returns_series": pd.Series(np.random.randn(252) * 0.01 + 0.0005),
}
result = validator.validate_strategy(normal_results)
print(f"Result p-value: {result.get('p_value', 'N/A')}")
print(f"Significant: {result.get('significant', 'N/A')}")
print(f"Confidence interval: {result.get('confidence_interval', 'N/A')}")
print("✅ Passed\n")

print("All Monte Carlo tests passed!")
