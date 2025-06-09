#!/usr/bin/env python3
"""Trace through OvernightDrift signal generation step by step."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager
from strategy_defaults import merge_with_defaults
from strategy_integration_helpers import DataFormatConverter

from strategies.overnight_drift import OvernightDrift


def trace_signal_generation():
    """Trace through signal generation for Alpha Vantage data."""

    data_manager = AlphaVantageDataManager()
    converter = DataFormatConverter()

    # Get Alpha Vantage data
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
    strategy_data = converter.dashboard_to_strategy(av_data, symbol="SPY")

    # Set up config
    user_params = {
        "symbol": "SPY",
        "position_size": 0.95,
        "lookback_period": 60,
        "volume_filter": False,
        "trend_filter": False,
        "volatility_filter": False,
    }

    full_config = merge_with_defaults("OvernightDrift", user_params)

    # Override filters
    full_config["volume_threshold"] = 0.0
    full_config["min_atr"] = 0.0
    full_config["max_atr"] = 1.0
    full_config["trend_filter"] = False
    full_config["vix_threshold"] = 100

    # Initialize strategy
    strategy = OvernightDrift(full_config)
    strategy.init()

    print("Strategy config:")
    print(f"  hold_days: {strategy.config.get('hold_days', 'NOT SET')}")
    print(f"  max_positions: {strategy.config.get('max_positions', 'NOT SET')}")
    print(f"  volume_threshold: {strategy.config.get('volume_threshold', 'NOT SET')}")
    print(f"  min_atr: {strategy.config.get('min_atr', 'NOT SET')}")
    print(f"  max_atr: {strategy.config.get('max_atr', 'NOT SET')}")

    # Process data day by day
    signals_generated = 0
    positions_opened = 0

    print(f"\nProcessing {len(strategy_data)} days of data...")
    print("Looking for signal generation on valid days...\n")

    # Start from day 61 (after lookback period)
    for i in range(60, min(len(strategy_data), 80)):  # Just check 20 days
        current_data = strategy_data.iloc[: i + 1].copy()
        current_data.attrs["symbol"] = "SPY"

        current_date = strategy_data.index[i]
        day_of_week = current_date.strftime("%A")

        # Calculate indicators for current day
        df_with_indicators = strategy.calculate_indicators(current_data)
        latest = df_with_indicators.iloc[-1]

        # Only print details for holding days
        if day_of_week in ["Monday", "Tuesday", "Wednesday", "Thursday"]:
            print(f"Date: {current_date.strftime('%Y-%m-%d')} ({day_of_week})")
            print(f"  Close: {latest['close']:.2f}")
            print(f"  ATR: {latest['atr']:.4f}")
            print(f"  Volume ratio: {latest.get('volume_ratio', 'N/A')}")

            # Check each filter
            print("  Filter checks:")
            print(
                f"    - Is holding day: {'Yes' if day_of_week in strategy.config.get('hold_days', []) else 'No'}"
            )
            print(
                f"    - Positions < max: {'Yes' if len(strategy.positions) < strategy.config['max_positions'] else 'No'}"
            )
            print(
                f"    - ATR in range: {'Yes' if latest['atr'] >= strategy.config['min_atr'] and latest['atr'] <= strategy.config['max_atr'] else 'No'}"
            )
            print(
                f"    - Volume OK: {'Yes' if latest.get('volume_ratio', 1) >= strategy.config['volume_threshold'] else 'No'}"
            )

            # Calculate edge
            edge = strategy.calculate_overnight_edge(current_data)
            print(f"  Overnight edge: {edge:.4f}")
            print(f"  Edge > 0.02: {'Yes' if edge > 0.02 else 'No'}")

            # Try to get signal
            signal = strategy.next(current_data)
            if signal:
                print(f"  ✓ SIGNAL GENERATED: {signal.direction}")
                signals_generated += 1
                if signal.direction == "LONG":
                    positions_opened += 1
            else:
                print("  ✗ No signal")

            print()

    print("\nSummary:")
    print(f"  Total signals: {signals_generated}")
    print(f"  Positions opened: {positions_opened}")


if __name__ == "__main__":
    trace_signal_generation()
