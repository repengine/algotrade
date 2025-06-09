#!/usr/bin/env python3
"""Test RSI calculation issue."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager


def test_rsi_issue():
    """Test RSI calculation with Alpha Vantage data."""

    os.environ["ALPHA_VANTAGE_API_KEY"] = "991AR2LC298IGMX7"

    data_manager = AlphaVantageDataManager()

    # Get data
    av_data = data_manager.fetch_data("SPY", "3mo", "1d", "alpha_vantage")

    print(f"Data shape: {av_data.shape}")
    print("Close prices (first 20):")
    print(av_data["close"].head(20))

    # Test RSI calculation step by step
    close = av_data["close"]

    print("\nTesting RSI calculation...")

    # Import the talib replacement
    import talib

    # Calculate RSI with different periods
    for period in [2, 14]:
        print(f"\nRSI with period {period}:")
        rsi = talib.RSI(close, timeperiod=period)

        print(f"  First 10 values: {rsi.head(10).values}")
        print(f"  Last 10 values: {rsi.tail(10).values}")
        print(f"  Min: {rsi.min():.2f}, Max: {rsi.max():.2f}, Mean: {rsi.mean():.2f}")
        print(f"  Count of zeros: {(rsi == 0).sum()}")
        print(f"  Count of NaN: {rsi.isna().sum()}")

    # Check close price changes
    print("\nClose price changes:")
    price_changes = close.diff()
    print(f"  First 10 changes: {price_changes.head(10).values}")
    print(f"  Positive changes: {(price_changes > 0).sum()}")
    print(f"  Negative changes: {(price_changes < 0).sum()}")
    print(f"  Zero changes: {(price_changes == 0).sum()}")

    # Test with the actual strategy's RSI parameters
    from strategies.mean_reversion_equity import MeanReversionEquity

    # Check the default RSI period for MeanReversionEquity
    print("\nChecking MeanReversionEquity defaults...")
    strategy = MeanReversionEquity({"symbols": ["SPY"]})
    print(f"RSI period: {strategy.config.get('rsi_period', 'NOT SET')}")

    # Calculate with strategy's period
    rsi_period = strategy.config.get("rsi_period", 2)
    rsi = talib.RSI(close, timeperiod=rsi_period)

    print(f"\nRSI with strategy period {rsi_period}:")
    print(f"  Values where RSI < 10: {(rsi < 10).sum()}")
    print(f"  Values where RSI < 30: {(rsi < 30).sum()}")
    print("  Sample low RSI dates:")
    low_rsi_mask = (rsi < 30) & (rsi > 0)
    if low_rsi_mask.any():
        low_rsi_dates = av_data.index[low_rsi_mask]
        for date in low_rsi_dates[:5]:
            idx = av_data.index.get_loc(date)
            print(f"    {date}: RSI={rsi.iloc[idx]:.2f}, Close={close.iloc[idx]:.2f}")


if __name__ == "__main__":
    test_rsi_issue()
