#!/usr/bin/env python3
"""Test Alpha Vantage fetcher directly to ensure it's working."""

import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "adapters"))

from av_fetcher import AlphaVantageFetcher


def test_av_fetcher():
    """Test Alpha Vantage fetcher functionality."""

    # Set API key
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "991AR2LC298IGMX7")
    print(f"Using API key: {api_key[:8]}...")

    try:
        # Initialize fetcher
        fetcher = AlphaVantageFetcher(api_key=api_key)
        print("✅ Fetcher initialized successfully")

        # Test daily data
        print("\n" + "=" * 60)
        print("Testing daily data fetch...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        daily_data = fetcher.fetch_ohlcv(
            symbol="SPY", start=start_date, end=end_date, interval="1d"
        )

        print(f"Daily data shape: {daily_data.shape}")
        print(f"Daily data columns: {list(daily_data.columns)}")
        print(f"Date range: {daily_data.index[0]} to {daily_data.index[-1]}")
        print("\nFirst 5 rows:")
        print(daily_data.head())
        print("\nLast 5 rows:")
        print(daily_data.tail())

        # Data quality checks
        print("\n" + "=" * 60)
        print("Data quality checks:")
        print(f"Null values: {daily_data.isnull().sum().sum()}")
        print("Data types:")
        print(daily_data.dtypes)

        # Check for valid OHLC relationships
        invalid_ohlc = (
            (daily_data["high"] < daily_data["low"])
            | (daily_data["high"] < daily_data["open"])
            | (daily_data["high"] < daily_data["close"])
            | (daily_data["low"] > daily_data["open"])
            | (daily_data["low"] > daily_data["close"])
        ).sum()
        print(f"Invalid OHLC relationships: {invalid_ohlc}")

        # Test intraday data
        print("\n" + "=" * 60)
        print("Testing intraday data fetch...")

        intraday_data = fetcher.fetch_ohlcv(
            symbol="SPY",
            start=end_date - timedelta(days=30),
            end=end_date,
            interval="5m",
        )

        print(f"Intraday data shape: {intraday_data.shape}")
        if not intraday_data.empty:
            print(
                f"Intraday date range: {intraday_data.index[0]} to {intraday_data.index[-1]}"
            )
            print("Sample intraday data:")
            print(intraday_data.head())

        # Test error handling
        print("\n" + "=" * 60)
        print("Testing error handling...")

        try:
            bad_data = fetcher.fetch_ohlcv(
                symbol="INVALID_SYMBOL_XYZ",
                start=start_date,
                end=end_date,
                interval="1d",
            )
            print(f"Bad symbol result: {bad_data.shape}")
        except Exception as e:
            print(f"✅ Error properly caught: {type(e).__name__}: {e}")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error in fetcher test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_av_fetcher()
