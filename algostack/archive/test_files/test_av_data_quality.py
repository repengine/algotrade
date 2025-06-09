#!/usr/bin/env python3
"""Test Alpha Vantage data quality after fixes."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "adapters"))
from datetime import datetime, timedelta

from av_fetcher import AlphaVantageFetcher


def test_av_data_quality():
    """Test Alpha Vantage data quality."""

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "991AR2LC298IGMX7")
    fetcher = AlphaVantageFetcher(api_key=api_key)

    # Test daily data
    print("Testing daily data fetch...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    df = fetcher.fetch_ohlcv(
        symbol="SPY", start=start_date, end=end_date, interval="1d"
    )

    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3))

    # Check data quality
    print(f"\n{'='*60}")
    print("Data Quality Checks:")
    print(f"{'='*60}")

    # Check for nulls
    null_counts = df.isnull().sum()
    print("\nNull values:")
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count}")
    if null_counts.sum() == 0:
        print("  ✓ No null values")

    # Check OHLC relationships
    print("\nOHLC relationship checks:")

    # High >= Low
    invalid_high_low = (df["high"] < df["low"]).sum()
    print(f"  High < Low: {invalid_high_low} rows")

    # High >= Open
    invalid_high_open = (df["high"] < df["open"]).sum()
    print(f"  High < Open: {invalid_high_open} rows")

    # High >= Close
    invalid_high_close = (df["high"] < df["close"]).sum()
    print(f"  High < Close: {invalid_high_close} rows")

    # Low <= Open
    invalid_low_open = (df["low"] > df["open"]).sum()
    print(f"  Low > Open: {invalid_low_open} rows")

    # Low <= Close
    invalid_low_close = (df["low"] > df["close"]).sum()
    print(f"  Low > Close: {invalid_low_close} rows")

    total_invalid = (
        invalid_high_low
        + invalid_high_open
        + invalid_high_close
        + invalid_low_open
        + invalid_low_close
    )

    if total_invalid == 0:
        print("\n  ✓ All OHLC relationships are valid!")
    else:
        print(f"\n  ✗ Total invalid relationships: {total_invalid}")

        # Show some invalid rows
        invalid_mask = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if invalid_mask.any():
            print("\n  Sample invalid rows:")
            print(df[invalid_mask].head())

    # Test intraday data
    print(f"\n{'='*60}")
    print("Testing intraday data...")

    df_intraday = fetcher.fetch_ohlcv(
        symbol="SPY", start=start_date, end=end_date, interval="5m"
    )

    print(f"\nIntraday data shape: {df_intraday.shape}")
    print(f"Columns: {list(df_intraday.columns)}")
    print("\nFirst 3 rows:")
    print(df_intraday.head(3))

    # Check intraday data quality
    intraday_invalid = (
        (df_intraday["high"] < df_intraday["low"])
        | (df_intraday["high"] < df_intraday["open"])
        | (df_intraday["high"] < df_intraday["close"])
        | (df_intraday["low"] > df_intraday["open"])
        | (df_intraday["low"] > df_intraday["close"])
    ).sum()

    print(f"\nIntraday invalid OHLC relationships: {intraday_invalid}")

    return df, df_intraday


if __name__ == "__main__":
    test_av_data_quality()
