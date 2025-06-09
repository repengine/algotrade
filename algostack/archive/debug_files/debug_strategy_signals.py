#!/usr/bin/env python3
"""Debug why strategies aren't generating signals."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager
from strategy_defaults import get_strategy_defaults

from strategies.mean_reversion_equity import MeanReversionEquity


def debug_strategy():
    """Debug a single strategy to see why it's not generating signals."""

    # Get some test data
    data_manager = AlphaVantageDataManager()
    data = data_manager.fetch_data("SPY", "1mo", "15m", "alpha_vantage")

    if data.empty:
        print("Using Yahoo Finance instead...")
        data = data_manager.fetch_data("SPY", "1mo", "1d", "yfinance")

    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    print("Sample data:")
    print(data.tail())

    # Convert to strategy format
    from strategy_integration_helpers import DataFormatConverter

    converter = DataFormatConverter()
    strategy_data = converter.dashboard_to_strategy(data, symbol="SPY")

    print(f"\nConverted data columns: {list(strategy_data.columns)}")
    print(f"Data attrs: {strategy_data.attrs}")

    # Get strategy defaults
    defaults = get_strategy_defaults("MeanReversionEquity")
    print(f"\nStrategy defaults: {defaults}")

    # Initialize strategy
    config = {
        "symbol": "SPY",
        "symbols": ["SPY"],
        "lookback_period": 20,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "position_size": 0.95,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.05,
        "volume_filter": False,
        "atr_period": 14,
        "atr_multiplier": 1.5,
    }

    print(f"\nInitializing strategy with config: {config}")
    strategy = MeanReversionEquity(config)
    strategy.init()

    # Check if strategy has required attributes
    print("\nStrategy attributes:")
    print(f"- symbols: {getattr(strategy, 'symbols', 'NOT FOUND')}")
    print(f"- symbol: {getattr(strategy, 'symbol', 'NOT FOUND')}")
    print(f"- lookback_period: {getattr(strategy, 'lookback_period', 'NOT FOUND')}")

    # Try to generate a signal with sufficient data
    if len(strategy_data) > 50:
        test_data = strategy_data.iloc[:50].copy()
        test_data.attrs["symbol"] = "SPY"

        print(f"\nTrying to generate signal with {len(test_data)} bars...")
        try:
            signal = strategy.next(test_data)
            if signal:
                print(f"Signal generated: {signal}")
            else:
                print("No signal generated")

            # Check internal state
            if hasattr(strategy, "_last_signal"):
                print(f"Last signal: {strategy._last_signal}")
            if hasattr(strategy, "position"):
                print(f"Position: {getattr(strategy, 'position', 'NOT FOUND')}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    # Let's check what the strategy's next method is doing
    print(f"\nStrategy next method: {strategy.next}")
    print(f"Strategy class: {strategy.__class__.__name__}")


if __name__ == "__main__":
    debug_strategy()
