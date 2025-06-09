#!/usr/bin/env python3
"""Debug why strategies don't generate signals with Alpha Vantage data."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager, PandasStrategyManager


def debug_av_issues():
    """Debug Alpha Vantage data issues."""

    os.environ["ALPHA_VANTAGE_API_KEY"] = "991AR2LC298IGMX7"

    data_manager = AlphaVantageDataManager()
    PandasStrategyManager()

    # Get Alpha Vantage data
    print("Fetching Alpha Vantage daily data...")
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")

    print(f"\nAV Data shape: {av_data.shape}")
    print(f"AV Data columns: {list(av_data.columns)}")
    print(f"AV Data index type: {type(av_data.index)}")
    print(f"AV Data index timezone: {av_data.index.tz}")

    print("\nFirst 5 rows:")
    print(av_data.head())

    print("\nLast 5 rows:")
    print(av_data.tail())

    # Check data statistics
    print("\nData statistics:")
    print(av_data.describe())

    # Check for any issues with the data
    print("\nData quality checks:")
    print(f"Nulls: {av_data.isnull().sum().sum()}")
    print(f"Zeros in volume: {(av_data['volume'] == 0).sum()}")
    print(f"Duplicate indices: {av_data.index.duplicated().sum()}")

    # Compare with Yahoo Finance
    print("\n" + "=" * 60)
    print("Comparing with Yahoo Finance data...")
    yf_data = data_manager.fetch_data("SPY", "1y", "1d", "yfinance")

    print(f"\nYF Data shape: {yf_data.shape}")
    print(f"YF Data columns: {list(yf_data.columns)}")

    # Convert YF data for comparison
    from strategy_integration_helpers import DataFormatConverter

    converter = DataFormatConverter()

    av_converted = converter.dashboard_to_strategy(av_data, symbol="SPY")
    yf_converted = converter.dashboard_to_strategy(yf_data, symbol="SPY")

    print(f"\nConverted AV columns: {list(av_converted.columns)}")
    print(f"Converted YF columns: {list(yf_converted.columns)}")

    # Test a specific strategy
    print("\n" + "=" * 60)
    print("Testing MeanReversionEquity strategy...")

    from strategies.mean_reversion_equity import MeanReversionEquity

    config = {
        "symbol": "SPY",
        "symbols": ["SPY"],
        "lookback_period": 20,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "position_size": 0.95,
        "volume_filter": False,  # Disable filters
        "atr_period": 14,
        "atr_band_mult": 2.5,
        "stop_loss_atr": 3.0,
        "max_positions": 5,
    }

    strategy = MeanReversionEquity(config)
    strategy.init()

    # Test with a specific data window
    test_data = av_converted.iloc[-50:].copy()
    test_data.attrs["symbol"] = "SPY"

    # Calculate indicators
    df_with_indicators = strategy.calculate_indicators(test_data)

    print("\nIndicator values (last 5 rows):")
    print(
        df_with_indicators[
            ["close", "rsi", "atr", "sma_20", "lower_band", "volume_ratio"]
        ].tail()
    )

    # Check entry conditions
    latest = df_with_indicators.iloc[-1]
    print("\nLatest values:")
    print(f"  RSI: {latest['rsi']:.2f}")
    print(f"  Close: {latest['close']:.2f}")
    print(f"  Lower Band: {latest['lower_band']:.2f}")
    print(f"  Volume Ratio: {latest['volume_ratio']:.2f}")

    print("\nEntry conditions:")
    print(f"  RSI < 30 (oversold): {latest['rsi'] < 30}")
    print(f"  Close < Lower Band: {latest['close'] < latest['lower_band']}")
    print(f"  Volume > 1.2x avg: {latest['volume_ratio'] > 1.2}")

    # Try to generate a signal
    signal = strategy.next(test_data)
    print(f"\nSignal generated: {signal}")

    # Check with more historical data
    print("\n" + "=" * 60)
    print("Checking for any signals in the last 100 days...")

    signals_found = 0
    for i in range(50, min(len(av_converted), 150)):
        test_window = av_converted.iloc[: i + 1].copy()
        test_window.attrs["symbol"] = "SPY"

        signal = strategy.next(test_window)
        if signal:
            signals_found += 1
            print(f"Signal on {test_window.index[-1]}: {signal.direction}")
            if signals_found >= 5:
                break

    if signals_found == 0:
        print("No signals found in test period")

        # Check RSI distribution
        print("\nRSI distribution:")
        print(f"  Min: {df_with_indicators['rsi'].min():.2f}")
        print(f"  Max: {df_with_indicators['rsi'].max():.2f}")
        print(f"  Mean: {df_with_indicators['rsi'].mean():.2f}")
        print(f"  Days with RSI < 30: {(df_with_indicators['rsi'] < 30).sum()}")


if __name__ == "__main__":
    debug_av_issues()
