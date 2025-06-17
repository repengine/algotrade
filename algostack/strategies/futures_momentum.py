#!/usr/bin/env python3
"""Futures Momentum Strategy - High-frequency momentum breakouts."""

from typing import Any, Optional

import pandas as pd

try:
    import talib
except ImportError:
    from pandas_indicators import create_talib_compatible_module

    talib = create_talib_compatible_module()

from algostack.strategies.base import BaseStrategy, RiskContext, Signal


class FuturesMomentum(BaseStrategy):
    """Futures momentum scalping strategy optimized for 5-minute bars.

    Trades momentum breakouts with tight risk management.
    Designed for ES/MES futures but tested on SPY as proxy.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        default_config = {
            "name": "FuturesMomentum",
            "symbols": ["SPY"],  # Use SPY as proxy for ES
            "lookback_period": 20,  # 20 bars = 100 minutes
            "breakout_threshold": 0.5,  # 0.5% above high
            "rsi_period": 14,
            "rsi_threshold": 60,  # Momentum confirmation
            "atr_period": 14,
            "stop_loss_atr": 2.0,  # 2 ATR stop
            "profit_target_atr": 3.0,  # 3 ATR target (1.5:1 R:R)
            "volume_multiplier": 1.2,  # Volume filter
            "position_size": 0.95,  # Use 95% of allocated capital
            "max_positions": 1,
            "trade_hours": {"start": 9.5, "end": 15.5},  # 9:30 AM  # 3:30 PM
        }

        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)

        self.positions = {}
        self.entry_prices = {}
        self.stop_prices = {}
        self.target_prices = {}
        self.entry_times = {}

    def init(self) -> None:
        """Initialize strategy state."""
        self.positions.clear()
        self.entry_prices.clear()
        self.stop_prices.clear()
        self.target_prices.clear()
        self.entry_times.clear()

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration parameters."""
        # For now, just return config as-is
        # Could add validation later
        return config

    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal based on momentum breakout."""

        symbol = data.attrs.get("symbol", "UNKNOWN")

        # Check if we have enough data
        if len(data) < self.config["lookback_period"]:
            return None

        # Get current values
        current_price = data["close"].iloc[-1]
        current_time = data.index[-1]

        # Check trading hours (if datetime index)
        if hasattr(current_time, "hour"):
            hour = current_time.hour + current_time.minute / 60
            if (
                hour < self.config["trade_hours"]["start"]
                or hour > self.config["trade_hours"]["end"]
            ):
                return None

        # Calculate indicators
        high_lookback = data["high"].iloc[-self.config["lookback_period"] :].max()
        data["low"].iloc[-self.config["lookback_period"] :].min()

        # RSI
        rsi = talib.RSI(data["close"], timeperiod=self.config["rsi_period"])
        current_rsi = rsi.iloc[-1]

        # ATR for position sizing and stops
        atr = talib.ATR(
            data["high"],
            data["low"],
            data["close"],
            timeperiod=self.config["atr_period"],
        )
        current_atr = atr.iloc[-1]

        # Volume analysis
        volume_sma = data["volume"].rolling(window=20).mean()
        current_volume_ratio = data["volume"].iloc[-1] / volume_sma.iloc[-1]

        # Check if we have a position
        if symbol in self.positions and self.positions[symbol] > 0:
            # Exit logic
            entry_price = self.entry_prices[symbol]
            stop_price = self.stop_prices[symbol]
            target_price = self.target_prices[symbol]

            # Check exit conditions
            if current_price <= stop_price:
                # Stop loss hit
                self.positions[symbol] = 0
                return Signal(
                    symbol=symbol,
                    direction="FLAT",
                    confidence=1.0,
                    reason="stop_loss",
                    metadata={
                        "exit_price": current_price,
                        "entry_price": entry_price,
                        "pnl_pct": (current_price - entry_price) / entry_price * 100,
                    },
                )
            elif current_price >= target_price:
                # Target hit
                self.positions[symbol] = 0
                return Signal(
                    symbol=symbol,
                    direction="FLAT",
                    confidence=1.0,
                    reason="profit_target",
                    metadata={
                        "exit_price": current_price,
                        "entry_price": entry_price,
                        "pnl_pct": (current_price - entry_price) / entry_price * 100,
                    },
                )
            elif (
                hasattr(current_time, "hour")
                and current_time.hour >= 15
                and current_time.minute >= 30
            ):
                # End of day exit
                self.positions[symbol] = 0
                return Signal(
                    symbol=symbol,
                    direction="FLAT",
                    confidence=0.8,
                    reason="eod_exit",
                    metadata={"exit_price": current_price},
                )

        else:
            # Entry logic - looking for breakouts
            breakout_level = high_lookback * (
                1 + self.config["breakout_threshold"] / 100
            )

            # Check for momentum breakout
            if (
                current_price > breakout_level
                and current_rsi > self.config["rsi_threshold"]
                and current_volume_ratio > self.config["volume_multiplier"]
                and not pd.isna(current_rsi)
                and not pd.isna(current_atr)
            ):

                # Calculate position details
                stop_distance = current_atr * self.config["stop_loss_atr"]
                stop_price = current_price - stop_distance
                target_price = current_price + (
                    current_atr * self.config["profit_target_atr"]
                )

                # Store position details
                self.positions[symbol] = 1
                self.entry_prices[symbol] = current_price
                self.stop_prices[symbol] = stop_price
                self.target_prices[symbol] = target_price
                self.entry_times[symbol] = current_time

                return Signal(
                    symbol=symbol,
                    direction="LONG",
                    confidence=0.8,
                    reason="momentum_breakout",
                    metadata={
                        "entry_price": current_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "rsi": current_rsi,
                        "atr": current_atr,
                        "volume_ratio": current_volume_ratio,
                        "risk_reward": self.config["profit_target_atr"]
                        / self.config["stop_loss_atr"],
                    },
                )

        return None

    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """Calculate position size based on risk management."""

        if signal.direction == "FLAT":
            return 0.0, 0.0

        # Get position sizing from metadata
        stop_price = signal.metadata.get("stop_price", 0)
        entry_price = signal.metadata.get("entry_price", 0)

        if stop_price > 0 and entry_price > 0:
            # Calculate position size based on 2% risk
            stop_distance_pct = abs(entry_price - stop_price) / entry_price
            max_risk = 0.02  # 2% risk per trade

            # Position size as percentage of equity
            position_size_pct = min(
                max_risk / stop_distance_pct, self.config["position_size"]
            )

            # In futures, this would be converted to contracts
            # For backtesting, we use percentage of equity
            position_value = risk_context.account_equity * position_size_pct

            return position_value, stop_price

        # Default sizing
        return risk_context.account_equity * self.config["position_size"], 0.0

    def update_state(self, symbol: str, data: pd.DataFrame) -> None:
        """Update internal state."""
        # State is updated in next() method
        pass

    def get_state(self) -> dict[str, Any]:
        """Get current strategy state."""
        return {
            "positions": self.positions.copy(),
            "entry_prices": self.entry_prices.copy(),
            "stop_prices": self.stop_prices.copy(),
            "target_prices": self.target_prices.copy(),
        }

    def backtest_metrics(self, trades_df: pd.DataFrame) -> dict[str, float]:
        """Calculate strategy-specific metrics."""

        if trades_df.empty:
            return {}

        # Calculate win rate by exit reason
        exit_reasons = trades_df["exit_reason"].value_counts()

        metrics = {
            "stop_loss_exits": exit_reasons.get("stop_loss", 0),
            "target_exits": exit_reasons.get("profit_target", 0),
            "eod_exits": exit_reasons.get("eod_exit", 0),
            "avg_bars_in_trade": (
                trades_df["bars_in_trade"].mean() if "bars_in_trade" in trades_df else 0
            ),
        }

        return metrics
