#!/usr/bin/env python3
"""Final debugging of Alpha Vantage signal generation."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

import pandas as pd
from dashboard_pandas import AlphaVantageDataManager, PandasStrategyManager
from strategy_integration_helpers import DataFormatConverter

from strategies.mean_reversion_equity import MeanReversionEquity


def debug_av_signals():
    """Debug why signals aren't being generated with Alpha Vantage data."""

    os.environ["ALPHA_VANTAGE_API_KEY"] = "991AR2LC298IGMX7"

    data_manager = AlphaVantageDataManager()
    strategy_manager = PandasStrategyManager()
    converter = DataFormatConverter()

    print("Fetching Alpha Vantage data...")
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")

    print(f"Raw AV data shape: {av_data.shape}")
    print(f"Raw AV data columns: {list(av_data.columns)}")
    print(f"Raw AV data index type: {type(av_data.index)}")
    print(f"Raw AV data attributes: {av_data.attrs}")

    # Convert data
    strategy_data = converter.dashboard_to_strategy(av_data, symbol="SPY")
    print(f"\nConverted data attributes: {strategy_data.attrs}")

    # Initialize strategy with very relaxed parameters
    config = {
        "symbol": "SPY",
        "symbols": ["SPY"],
        "lookback_period": 20,
        "rsi_period": 2,
        "rsi_oversold": 50,  # Very relaxed
        "rsi_overbought": 70,
        "position_size": 0.95,
        "volume_filter": False,
        "atr_period": 14,
        "atr_band_mult": 0.5,  # Very tight bands
        "stop_loss_atr": 3.0,
        "max_positions": 5,
    }

    strategy = MeanReversionEquity(config)
    strategy.init()

    # Clear positions
    strategy.positions = {}

    # Calculate indicators
    df_with_indicators = strategy.calculate_indicators(strategy_data)

    # Find potential entry points
    entry_conditions = df_with_indicators[
        (df_with_indicators["rsi"] < 50)
        & (df_with_indicators["close"] < df_with_indicators["lower_band"])
    ]

    print(f"\nPotential entry points: {len(entry_conditions)}")

    if len(entry_conditions) > 0:
        print("\nTesting signal generation at entry points...")

        # Test the first entry point
        first_entry_date = entry_conditions.index[0]
        date_idx = strategy_data.index.get_loc(first_entry_date)

        # Get data up to this point
        test_data = strategy_data.iloc[: date_idx + 1].copy()
        test_data.attrs["symbol"] = "SPY"

        print(f"\nTesting at date: {first_entry_date}")
        print(f"Data shape for test: {test_data.shape}")
        print(f"Data attributes: {test_data.attrs}")

        # Debug the next() method
        print("\nDebugging next() method...")

        # Check validate_data
        is_valid = strategy.validate_data(test_data)
        print(f"validate_data result: {is_valid}")

        if not is_valid:
            print("Data validation failed!")
            # Check what validate_data expects
            print(f"Data columns: {list(test_data.columns)}")
            print(f"Data has symbol attr: {'symbol' in test_data.attrs}")
            print(f"Symbol value: {test_data.attrs.get('symbol', 'NOT SET')}")

        # Try to generate signal
        try:
            signal = strategy.next(test_data)
            print(f"Signal generated: {signal}")

            if signal is None:
                # Manually check conditions
                latest = df_with_indicators.loc[first_entry_date]
                print("\nManual condition check:")
                print(f"  RSI: {latest['rsi']:.2f} < 50? {latest['rsi'] < 50}")
                print(f"  Close: {latest['close']:.2f}")
                print(f"  Lower Band: {latest['lower_band']:.2f}")
                print(f"  Close < Lower Band? {latest['close'] < latest['lower_band']}")
                print(f"  Current positions: {len(strategy.positions)}")
                print(f"  Max positions: {strategy.config['max_positions']}")

        except Exception as e:
            print(f"Error in next(): {e}")
            import traceback

            traceback.print_exc()

    # Run a full backtest to see if it's a backtest-specific issue
    print("\n" + "=" * 60)
    print("Running full backtest...")

    results = strategy_manager.run_backtest(
        MeanReversionEquity,
        "MeanReversionEquity",
        config,
        av_data,
        initial_capital=100000,
    )

    if "error" not in results:
        print("Backtest completed:")
        print(f"  Trades: {results.get('num_trades', 0)}")
        print(f"  Total return: {results.get('total_return', 0):.2f}%")

        signals_df = results.get("signals", pd.DataFrame())
        if not signals_df.empty:
            print(f"  Signals: {len(signals_df)}")
        else:
            print("  No signals generated in backtest")
    else:
        print(f"Backtest error: {results['error']}")


if __name__ == "__main__":
    debug_av_signals()
