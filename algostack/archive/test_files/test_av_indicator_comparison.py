#!/usr/bin/env python3
"""Compare indicator calculations between Alpha Vantage and Yahoo Finance data."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager
from strategy_integration_helpers import DataFormatConverter

from strategies.mean_reversion_equity import MeanReversionEquity


def compare_indicators():
    """Compare indicators calculated on AV vs YF data."""

    os.environ["ALPHA_VANTAGE_API_KEY"] = "991AR2LC298IGMX7"

    data_manager = AlphaVantageDataManager()
    converter = DataFormatConverter()

    # Get both data sources for the same period
    print("Fetching data from both sources...")
    av_data = data_manager.fetch_data("SPY", "3mo", "1d", "alpha_vantage")
    yf_data = data_manager.fetch_data("SPY", "3mo", "1d", "yfinance")

    print(f"\nAlpha Vantage data shape: {av_data.shape}")
    print(f"Yahoo Finance data shape: {yf_data.shape}")

    # Convert both
    av_converted = converter.dashboard_to_strategy(av_data, symbol="SPY")
    yf_converted = converter.dashboard_to_strategy(yf_data, symbol="SPY")

    # Initialize strategy to calculate indicators
    config = {
        "symbol": "SPY",
        "symbols": ["SPY"],
        "lookback_period": 20,
        "rsi_period": 2,
        "rsi_oversold": 40,
        "rsi_overbought": 70,
        "position_size": 0.95,
        "volume_filter": False,
        "atr_period": 14,
        "atr_band_mult": 1.5,
        "stop_loss_atr": 3.0,
        "max_positions": 5,
    }

    strategy = MeanReversionEquity(config)
    strategy.init()

    # Calculate indicators on both datasets
    av_indicators = strategy.calculate_indicators(av_converted)
    yf_indicators = strategy.calculate_indicators(yf_converted)

    # Find common dates
    common_dates = av_indicators.index.intersection(yf_indicators.index)
    print(f"\nCommon dates: {len(common_dates)}")

    if len(common_dates) > 0:
        # Compare last 10 common dates
        compare_dates = common_dates[-10:]

        print("\nComparing indicators on common dates:")
        print("=" * 80)

        for date in compare_dates:
            av_row = av_indicators.loc[date]
            yf_row = yf_indicators.loc[date]

            print(f"\nDate: {date}")
            print("  Alpha Vantage:")
            print(
                f"    Close: {av_row['close']:.2f}, RSI: {av_row['rsi']:.2f}, Lower Band: {av_row['lower_band']:.2f}"
            )
            print(f"    Volume Ratio: {av_row['volume_ratio']:.2f}")
            print("  Yahoo Finance:")
            print(
                f"    Close: {yf_row['close']:.2f}, RSI: {yf_row['rsi']:.2f}, Lower Band: {yf_row['lower_band']:.2f}"
            )
            print(f"    Volume Ratio: {yf_row['volume_ratio']:.2f}")

            # Check if entry conditions would be met
            av_entry = (av_row["rsi"] < 40) and (av_row["close"] < av_row["lower_band"])
            yf_entry = (yf_row["rsi"] < 40) and (yf_row["close"] < yf_row["lower_band"])

            if av_entry or yf_entry:
                print(f"  *** Entry signal: AV={av_entry}, YF={yf_entry}")

    # Check for any entry conditions in the entire dataset
    print("\n" + "=" * 80)
    print("Checking entry conditions across entire datasets:")

    av_entries = av_indicators[
        (av_indicators["rsi"] < 40)
        & (av_indicators["close"] < av_indicators["lower_band"])
    ]
    yf_entries = yf_indicators[
        (yf_indicators["rsi"] < 40)
        & (yf_indicators["close"] < yf_indicators["lower_band"])
    ]

    print(f"\nAlpha Vantage - Days meeting entry conditions: {len(av_entries)}")
    if len(av_entries) > 0:
        print("Sample AV entry days:")
        print(av_entries[["close", "rsi", "lower_band", "volume_ratio"]].head())

    print(f"\nYahoo Finance - Days meeting entry conditions: {len(yf_entries)}")
    if len(yf_entries) > 0:
        print("Sample YF entry days:")
        print(yf_entries[["close", "rsi", "lower_band", "volume_ratio"]].head())


if __name__ == "__main__":
    compare_indicators()
