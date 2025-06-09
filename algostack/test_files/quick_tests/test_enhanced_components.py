#!/usr/bin/env python3
"""Test script to verify enhanced dashboard components."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import pandas as pd

# Test imports
try:
    from core.backtest_engine import (
        DataSplitter,
        MonteCarloValidator,
        RegimeAnalyzer,
        TransactionCostModel,
    )

    print("✅ Successfully imported backtest engine components")
except Exception as e:
    print(f"❌ Failed to import backtest engine: {e}")
    sys.exit(1)

try:
    from core.optimization import (
        PlateauDetector,
    )

    print("✅ Successfully imported optimization components")
except Exception as e:
    print(f"❌ Failed to import optimization: {e}")
    sys.exit(1)

# Test transaction cost model
print("\n📊 Testing Transaction Cost Model...")
cost_config = {
    "commission_per_share": 0.005,
    "commission_type": "per_share",
    "spread_model": "dynamic",
    "base_spread_bps": 5,
    "slippage_model": "linear",
    "market_impact_factor": 0.1,
}

try:
    cost_model = TransactionCostModel(cost_config)
    costs = cost_model.calculate_costs(
        price=100.0, shares=100, side="BUY", volatility=0.02, avg_daily_volume=1000000
    )
    print(f"  Commission: ${costs.commission:.2f}")
    print(f"  Spread cost: ${costs.spread_cost:.2f}")
    print(f"  Slippage: ${costs.slippage:.2f}")
    print(f"  Total cost: ${costs.total:.2f}")
    print("✅ Transaction cost model working")
except Exception as e:
    print(f"❌ Transaction cost model failed: {e}")

# Test data splitter
print("\n📊 Testing Data Splitter...")
try:
    # Create dummy data
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    dummy_data = pd.DataFrame(
        {
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000000, 2000000, len(dates)),
        },
        index=dates,
    )

    splitter = DataSplitter(method="sequential", oos_ratio=0.2)
    is_data, oos_data = splitter.split(dummy_data)

    print(f"  Total data: {len(dummy_data)} days")
    print(f"  IS data: {len(is_data)} days ({len(is_data)/len(dummy_data)*100:.1f}%)")
    print(
        f"  OOS data: {len(oos_data)} days ({len(oos_data)/len(dummy_data)*100:.1f}%)"
    )
    print("✅ Data splitter working")
except Exception as e:
    print(f"❌ Data splitter failed: {e}")

# Test Monte Carlo validator
print("\n📊 Testing Monte Carlo Validator...")
try:
    # Create dummy results
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # Daily returns
    test_results = {"sharpe_ratio": 1.2, "returns_series": returns}

    validator = MonteCarloValidator(n_simulations=100)  # Reduced for speed
    validation = validator.validate_strategy(test_results)

    print(f"  P-value: {validation['p_value']:.3f}")
    print(f"  Significant: {validation['significant']}")
    print(f"  Effect size: {validation['effect_size']:.2f}")
    print("✅ Monte Carlo validator working")
except Exception as e:
    print(f"❌ Monte Carlo validator failed: {e}")

# Test regime analyzer
print("\n📊 Testing Regime Analyzer...")
try:
    # Create dummy market data with clear regimes
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

    # Create synthetic data with different volatility regimes
    low_vol = np.random.randn(400) * 0.005
    high_vol = np.random.randn(400) * 0.02
    medium_vol = np.random.randn(len(dates) - 800) * 0.01

    returns = np.concatenate([low_vol, high_vol, medium_vol])
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    regime_data = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "Close": prices,  # Both cases for compatibility
            "volume": np.random.randint(1000000, 2000000, len(dates)),
        },
        index=dates,
    )

    analyzer = RegimeAnalyzer(regime_method="volatility")
    regimes = analyzer.identify_regimes(regime_data)

    print(f"  Found {len(regimes)} regimes:")
    for regime_name, regime_df in regimes.items():
        print(f"    {regime_name}: {len(regime_df)} days")
    print("✅ Regime analyzer working")
except Exception as e:
    print(f"❌ Regime analyzer failed: {e}")
    import traceback

    traceback.print_exc()

# Test plateau detector
print("\n📊 Testing Plateau Detector...")
try:
    # Create optimization results with a clear plateau
    param_values = np.linspace(10, 50, 20)
    sharpe_values = -((param_values - 30) ** 2) / 100 + 2  # Peak at 30

    results_df = pd.DataFrame(
        {
            "param": param_values,
            "sharpe": sharpe_values + np.random.randn(20) * 0.1,  # Add noise
        }
    )

    detector = PlateauDetector(min_plateau_size=3)
    plateaus = detector.find_plateaus(results_df, metric_col="sharpe")

    print(f"  Found {len(plateaus)} plateau(s)")
    if plateaus:
        print(f"  Best plateau center: param={plateaus[0]['center']['param']:.1f}")
        print(f"  Stability score: {plateaus[0]['stability']:.2f}")
    print("✅ Plateau detector working")
except Exception as e:
    print(f"❌ Plateau detector failed: {e}")

print("\n✅ All component tests completed!")
print("\nYou can now run the enhanced dashboard with confidence.")
print("If any components failed, please let me know the error messages.")
