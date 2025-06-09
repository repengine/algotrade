#!/usr/bin/env python3
"""Debug why strategies show 0 signals in dashboard but work in tests."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

import pandas as pd
from dashboard_pandas import AlphaVantageDataManager, PandasStrategyManager
from strategy_defaults import merge_with_defaults


def debug_dashboard_backtest():
    """Debug the exact backtest process used by dashboard."""

    # Initialize managers like dashboard does
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()

    # Test with both data sources
    for data_source in ["alpha_vantage", "yfinance"]:
        print(f"\n{'='*60}")
        print(f"Testing with {data_source}")
        print(f"{'='*60}")

        # Fetch data
        data = data_manager.fetch_data("SPY", "1y", "1d", data_source)
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {list(data.columns)}")

        # Test OvernightDrift
        strategy_name = "OvernightDrift"
        strategy_class = strategy_manager.strategies[strategy_name]

        # Simulate dashboard parameters
        user_params = {
            "symbol": "SPY",
            "position_size": 0.95,
            "lookback_period": 60,
            "volume_filter": False,  # Try disabling filters
            "trend_filter": False,
            "volatility_filter": False,
        }

        # Run backtest exactly as dashboard does
        results = strategy_manager.run_backtest(
            strategy_class, strategy_name, user_params, data, initial_capital=100000
        )

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Total return: {results.get('total_return', 0):.2f}%")
            print(f"Number of trades: {results.get('num_trades', 0)}")

            # Check signals
            signals_df = results.get("signals", pd.DataFrame())
            if not signals_df.empty:
                print(f"Signals generated: {len(signals_df)}")
                if "direction" in signals_df.columns:
                    print(
                        f"Signal breakdown: {signals_df['direction'].value_counts().to_dict()}"
                    )

                    # Show some signal dates
                    print("\nFirst 5 signals:")
                    for _idx, row in signals_df.head().iterrows():
                        print(f"  {row['timestamp']}: {row['direction']}")
            else:
                print("No signals generated!")

        # Check the actual strategy instance to see what's happening
        print("\nDebugging strategy internals...")

        # Create strategy instance with merged config
        full_config = merge_with_defaults(strategy_name, user_params)
        print(f"Full config: {full_config}")

        # Check if the issue is with the symbol filtering
        if "symbols" in full_config:
            print(f"Strategy symbols: {full_config['symbols']}")
            if user_params["symbol"] not in full_config["symbols"]:
                print(f"WARNING: {user_params['symbol']} not in strategy symbols!")


def check_data_differences():
    """Check specific differences between data sources."""
    print("\nChecking data differences...")

    data_manager = AlphaVantageDataManager()

    # Get both datasets
    av_data = data_manager.fetch_data("SPY", "3mo", "1d", "alpha_vantage")
    yf_data = data_manager.fetch_data("SPY", "3mo", "1d", "yfinance")

    print("\nAlpha Vantage data:")
    print(f"  Shape: {av_data.shape}")
    print(f"  Date range: {av_data.index[0]} to {av_data.index[-1]}")
    print(f"  Timezone: {av_data.index.tz}")

    print("\nYahoo Finance data:")
    print(f"  Shape: {yf_data.shape}")
    print(f"  Date range: {yf_data.index[0]} to {yf_data.index[-1]}")
    print(f"  Timezone: {yf_data.index.tz}")

    # Check a specific date's data
    test_date = "2025-06-04"
    print(f"\nData for {test_date}:")

    if test_date in av_data.index.strftime("%Y-%m-%d"):
        av_row = av_data[av_data.index.strftime("%Y-%m-%d") == test_date].iloc[0]
        print(
            f"  Alpha Vantage: close={av_row['close']:.2f}, volume={av_row['volume']:.0f}"
        )

    if test_date in yf_data.index.strftime("%Y-%m-%d"):
        yf_row = yf_data[yf_data.index.strftime("%Y-%m-%d") == test_date].iloc[0]
        close_col = "Close" if "Close" in yf_data.columns else "close"
        vol_col = "Volume" if "Volume" in yf_data.columns else "volume"
        print(
            f"  Yahoo Finance: close={yf_row[close_col]:.2f}, volume={yf_row[vol_col]:.0f}"
        )


if __name__ == "__main__":
    debug_dashboard_backtest()
    print("\n" + "=" * 60)
    check_data_differences()
