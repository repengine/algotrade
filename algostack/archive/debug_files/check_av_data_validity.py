#!/usr/bin/env python3
"""Check Alpha Vantage data validity issues."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager


def check_data_validity():
    """Check why validate_data fails on Alpha Vantage data."""

    os.environ["ALPHA_VANTAGE_API_KEY"] = "991AR2LC298IGMX7"

    data_manager = AlphaVantageDataManager()

    # Get data
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")

    print(f"Data shape: {av_data.shape}")
    print(f"Data columns: {list(av_data.columns)}")

    # Check for nulls
    print(f"\nNull values: {av_data.isnull().sum().sum()}")

    # Check OHLC relationships
    print("\nChecking OHLC relationships...")

    # Individual checks
    high_low_check = (av_data["high"] >= av_data["low"]).all()
    high_open_check = (av_data["high"] >= av_data["open"]).all()
    high_close_check = (av_data["high"] >= av_data["close"]).all()
    low_open_check = (av_data["low"] <= av_data["open"]).all()
    low_close_check = (av_data["low"] <= av_data["close"]).all()

    print(f"High >= Low: {high_low_check}")
    print(f"High >= Open: {high_open_check}")
    print(f"High >= Close: {high_close_check}")
    print(f"Low <= Open: {low_open_check}")
    print(f"Low <= Close: {low_close_check}")

    # Find any invalid rows
    invalid_mask = (
        (av_data["high"] < av_data["low"])
        | (av_data["high"] < av_data["open"])
        | (av_data["high"] < av_data["close"])
        | (av_data["low"] > av_data["open"])
        | (av_data["low"] > av_data["close"])
    )

    invalid_count = invalid_mask.sum()
    print(f"\nTotal invalid rows: {invalid_count}")

    if invalid_count > 0:
        print("\nInvalid rows:")
        print(av_data[invalid_mask])

    # Test on a subset (first 31 rows as in the debug)
    print("\n" + "=" * 60)
    print("Testing first 31 rows (as used in strategy test):")

    subset = av_data.iloc[:31]

    # Validate subset
    required_columns = ["open", "high", "low", "close", "volume"]
    has_columns = all(col in subset.columns for col in required_columns)
    print(f"Has required columns: {has_columns}")

    has_nulls = subset[required_columns].isnull().any().any()
    print(f"Has null values: {has_nulls}")

    # Check OHLC on subset
    subset_valid = (
        (subset["high"] >= subset["low"]).all()
        and (subset["high"] >= subset["open"]).all()
        and (subset["high"] >= subset["close"]).all()
        and (subset["low"] <= subset["open"]).all()
        and (subset["low"] <= subset["close"]).all()
    )
    print(f"Valid OHLC relationships: {subset_valid}")

    # Find the specific issue
    if not subset_valid:
        print("\nFinding specific issues in subset:")
        for idx, row in subset.iterrows():
            issues = []
            if row["high"] < row["low"]:
                issues.append(f"high({row['high']}) < low({row['low']})")
            if row["high"] < row["open"]:
                issues.append(f"high({row['high']}) < open({row['open']})")
            if row["high"] < row["close"]:
                issues.append(f"high({row['high']}) < close({row['close']})")
            if row["low"] > row["open"]:
                issues.append(f"low({row['low']}) > open({row['open']})")
            if row["low"] > row["close"]:
                issues.append(f"low({row['low']}) > close({row['close']})")

            if issues:
                print(f"{idx}: {', '.join(issues)}")


if __name__ == "__main__":
    check_data_validity()
