#!/usr/bin/env python3
"""Debug version of OvernightDrift strategy."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module

sys.modules["talib"] = create_talib_compatible_module()

from typing import Optional

import pandas as pd

from strategies.base import Signal
from strategies.overnight_drift import OvernightDrift as BaseOvernightDrift


class DebugOvernightDrift(BaseOvernightDrift):
    """OvernightDrift with debug output."""

    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process daily data with debug output."""
        print("\n=== DEBUG next() called ===")
        print(f"Data length: {len(data)}")

        if not self.validate_data(data):
            print("❌ Data validation failed")
            # Debug what's wrong
            print(f"  Columns: {list(data.columns)}")
            print(f"  Has nulls: {data.isnull().any().any()}")
            if data.isnull().any().any():
                print(f"  Null columns: {data.columns[data.isnull().any()].tolist()}")
            # Check OHLC relationships
            if len(data) > 0:
                invalid_ohlc = (
                    (data["high"] < data["low"]).any()
                    or (data["high"] < data["open"]).any()
                    or (data["high"] < data["close"]).any()
                    or (data["low"] > data["open"]).any()
                    or (data["low"] > data["close"]).any()
                )
                if invalid_ohlc:
                    print("  Invalid OHLC relationships found")
            return None

        # Calculate indicators
        df = self.calculate_indicators(data)

        if len(df) < self.config["sma_period"]:
            print(f"❌ Not enough data: {len(df)} < {self.config['sma_period']}")
            return None

        latest = df.iloc[-1]
        symbol = data.attrs.get("symbol", "UNKNOWN")
        current_date = df.index[-1]
        day_of_week = current_date.strftime("%A")

        print(f"Date: {current_date} ({day_of_week})")
        print(f"Symbol: {symbol}")

        # Skip if not a holding day
        if day_of_week not in self.config["hold_days"]:
            print(f"❌ Not a holding day: {day_of_week}")
            # Exit if holding over weekend
            if symbol in self.positions and day_of_week == "Friday":
                print("✓ Exiting position for weekend")
                signal = Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    direction="FLAT",
                    strength=0.0,
                    strategy_id=self.name,
                    price=latest["close"],
                    metadata={"reason": "weekend_exit", "day": day_of_week},
                )
                del self.positions[symbol]
                return signal
            return None

        # Check for position exit
        if symbol in self.positions:
            print("✓ Have existing position")
            pos = self.positions[symbol]
            exit_price = latest["open"]
            pnl_pct = (
                ((exit_price - pos["entry_price"]) / pos["entry_price"] * 100)
                if pos["entry_price"] > 0
                else 0.0
            )

            signal = Signal(
                timestamp=current_date,
                symbol=symbol,
                direction="FLAT",
                strength=0.0,
                strategy_id=self.name,
                price=exit_price,
                metadata={
                    "reason": "overnight_exit",
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "held_overnight": True,
                },
            )
            del self.positions[symbol]
            print("✓ Returning EXIT signal")
            return signal

        # Check entry conditions
        print("Checking entry conditions...")
        print(f"  Positions: {len(self.positions)} / {self.config['max_positions']}")

        if len(self.positions) >= self.config["max_positions"]:
            print("❌ Max positions reached")
            return None

        # Skip if in blackout period
        if self.is_blackout_period(symbol, current_date):
            print("❌ In blackout period")
            return None

        # Volatility filter
        print(
            f"  ATR: {latest['atr']:.4f} (range: {self.config['min_atr']}-{self.config['max_atr']})"
        )
        if (
            latest["atr"] < self.config["min_atr"]
            or latest["atr"] > self.config["max_atr"]
        ):
            print("❌ ATR out of range")
            return None

        # Volume filter
        print(
            f"  Volume ratio: {latest['volume_ratio']:.2f} (threshold: {self.config['volume_threshold']})"
        )
        if latest["volume_ratio"] < self.config["volume_threshold"]:
            print("❌ Volume too low")
            return None

        # Trend filter
        if self.config["trend_filter"]:
            print(f"  Close: {latest['close']:.2f}, SMA: {latest['sma']:.2f}")
            if latest["close"] < latest["sma"]:
                print("❌ Below SMA (trend filter)")
                return None

        # VIX filter
        vix = latest.get("vix", 20)
        print(f"  VIX: {vix} (threshold: {self.config['vix_threshold']})")
        if vix > self.config["vix_threshold"]:
            print("❌ VIX too high")
            return None

        # Calculate overnight edge
        edge = self.calculate_overnight_edge(df)
        print(f"  Edge: {edge:.4f} (threshold: 0.02)")

        # Only trade if positive edge
        if edge > 0.02:
            print("✓ Edge is positive, creating signal...")
            strength = min(1.0, edge * 10)

            try:
                signal = Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    direction="LONG",
                    strength=strength,
                    strategy_id=self.name,
                    price=latest["close"],
                    atr=latest["atr"],
                    metadata={
                        "reason": "overnight_entry",
                        "day": day_of_week,
                        "edge": edge,
                        "momentum": latest.get("momentum", 0),
                        "volume_ratio": latest["volume_ratio"],
                        "atr": latest["atr"],
                        "entry_time": "close",
                    },
                )

                self.positions[symbol] = {
                    "entry_price": latest["close"],
                    "entry_date": current_date,
                    "expected_edge": edge,
                }

                print(f"✓ Signal created: {signal}")
                return signal

            except Exception as e:
                print(f"❌ Error creating signal: {e}")
                import traceback

                traceback.print_exc()
                return None
        else:
            print("❌ Edge too low")

        return None


def test_debug_strategy():
    """Test the debug strategy."""
    from dashboard_pandas import AlphaVantageDataManager
    from strategy_defaults import merge_with_defaults
    from strategy_integration_helpers import DataFormatConverter

    data_manager = AlphaVantageDataManager()
    converter = DataFormatConverter()

    # Get data
    av_data = data_manager.fetch_data("SPY", "3mo", "1d", "alpha_vantage")
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

    # Initialize debug strategy
    strategy = DebugOvernightDrift(full_config)
    strategy.init()

    # Test with multiple days
    print("Testing signal generation over multiple days...")

    signals_found = 0
    for i in range(61, min(len(strategy_data), 75)):
        test_data = strategy_data.iloc[: i + 1].copy()
        test_data.attrs["symbol"] = "SPY"

        signal = strategy.next(test_data)
        if signal:
            signals_found += 1
            print(f"\n✓ SIGNAL FOUND on day {i}: {signal.direction}")
            if signals_found >= 3:  # Just show first 3
                break

    if signals_found == 0:
        print("\nNo signals found in test period")

    # Also test with full year data
    print("\n" + "=" * 60)
    print("Testing with full year of data...")

    # Get full year
    av_data_year = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
    strategy_data_year = converter.dashboard_to_strategy(av_data_year, symbol="SPY")

    # Reinitialize strategy
    strategy2 = DebugOvernightDrift(full_config)
    strategy2.init()

    # Test a day in the middle
    test_idx = 150  # Day 150
    if test_idx < len(strategy_data_year):
        test_data = strategy_data_year.iloc[: test_idx + 1].copy()
        test_data.attrs["symbol"] = "SPY"

        print(f"\nTesting with {test_idx} days of data...")
        signal = strategy2.next(test_data)
        print(f"Result: {signal}")


if __name__ == "__main__":
    test_debug_strategy()
