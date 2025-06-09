#!/usr/bin/env python3
"""Test script to debug pandas dashboard backtest issues."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

import pandas as pd
import yfinance as yf
from dashboard_pandas import PandasStrategyManager


def test_backtest():
    """Test a simple backtest to debug issues."""

    # Initialize manager
    manager = PandasStrategyManager()

    print("Available strategies:", list(manager.strategies.keys()))

    # Fetch test data
    print("\nFetching SPY data...")
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="3mo", interval="1d")
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    print(f"First few rows:\n{data.head()}")

    # Test with Mean Reversion strategy
    strategy_name = "MeanReversionEquity"
    if strategy_name not in manager.strategies:
        print(f"\n{strategy_name} not found, trying first available strategy")
        strategy_name = list(manager.strategies.keys())[0]

    print(f"\nTesting with {strategy_name}")
    strategy_class = manager.strategies[strategy_name]

    # Get default parameters
    user_params = {
        "symbol": "SPY",
        "lookback_period": 20,  # Use shorter lookback for testing
        "position_size": 0.95,
    }

    # Run backtest
    results = manager.run_backtest(
        strategy_class, strategy_name, user_params, data, initial_capital=100000
    )

    # Check results
    if "error" in results:
        print(f"\nBacktest error: {results['error']}")
    else:
        print("\nBacktest completed successfully!")
        print(f"Total return: {results.get('total_return', 0):.2f}%")
        print(f"Number of trades: {results.get('num_trades', 0)}")
        print(f"Equity curve length: {len(results.get('equity_curve', []))}")

        # Check signals
        signals_df = results.get("signals", pd.DataFrame())
        if not signals_df.empty:
            print(f"\nSignals generated: {len(signals_df)}")
            print("First few signals:")
            print(signals_df.head())
        else:
            print("\nNo signals generated!")

    return results


if __name__ == "__main__":
    test_backtest()
