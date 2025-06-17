#!/usr/bin/env python3
"""Trend following strategy using Donchian channels across multiple assets."""

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    # Use pandas implementation if talib is not available
    from pandas_indicators import create_talib_compatible_module

    talib = create_talib_compatible_module()

from strategies.base import BaseStrategy, RiskContext, Signal
from utils.validators.strategy_validators import (
    validate_trend_following_config,
)


class TrendFollowingMulti(BaseStrategy):
    """Trend following strategy for futures and crypto.

    Entry: Price breaks above 20-day Donchian channel
    Exit: Trailing stop at 10-day low (for longs) or high (for shorts)
    Position sizing: Volatility-scaled across portfolio
    """

    def __init__(self, config: dict[str, Any]) -> None:
        default_config = {
            "name": "TrendFollowingMulti",
            "symbols": ["MES", "MNQ", "BTC-USD", "ETH-USD"],  # Micro futures + crypto
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25,  # Minimum ADX for trend
            "max_positions": 4,
            "use_volume_filter": True,
            "volume_threshold": 1.2,  # 20% above average
        }

        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)

        self.positions = {}  # Track open positions
        self.channel_breaks = {}  # Track recent breakouts

    def init(self) -> None:
        """Initialize strategy components."""
        self.positions.clear()
        self.channel_breaks.clear()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian channels, ATR, ADX, and volume metrics."""
        df = data.copy()

        # Donchian Channels
        df["channel_high"] = df["high"].rolling(self.config["channel_period"]).max()
        df["channel_low"] = df["low"].rolling(self.config["channel_period"]).min()
        df["channel_mid"] = (df["channel_high"] + df["channel_low"]) / 2

        # Trailing stop levels
        df["trail_high"] = df["high"].rolling(self.config["trail_period"]).max()
        df["trail_low"] = df["low"].rolling(self.config["trail_period"]).min()

        # ATR for volatility
        df["atr"] = talib.ATR(
            df["high"], df["low"], df["close"], timeperiod=self.config["atr_period"]
        )

        # ADX for trend strength
        df["adx"] = talib.ADX(
            df["high"], df["low"], df["close"], timeperiod=self.config["adx_period"]
        )

        # Directional indicators
        df["plus_di"] = talib.PLUS_DI(
            df["high"], df["low"], df["close"], timeperiod=self.config["adx_period"]
        )
        df["minus_di"] = talib.MINUS_DI(
            df["high"], df["low"], df["close"], timeperiod=self.config["adx_period"]
        )

        # Volume analysis
        df["volume_sma"] = talib.SMA(df["volume"], timeperiod=20)
        df["volume_ratio"] = np.where(
            df["volume_sma"] > 0, df["volume"] / df["volume_sma"], 1.0
        )

        # Price momentum
        df["roc"] = talib.ROC(df["close"], timeperiod=10)  # Rate of change

        # Channel position (0 = at low, 1 = at high)
        channel_width = df["channel_high"] - df["channel_low"]
        df["channel_position"] = np.where(
            channel_width > 0, (df["close"] - df["channel_low"]) / channel_width, 0.5
        )

        return df

    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process new data and generate trading signal."""
        if not self.validate_data(data):
            return None

        # Calculate indicators
        df = self.calculate_indicators(data)

        if len(df) < self.config["channel_period"]:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        symbol = data.attrs.get("symbol", "UNKNOWN")
        current_time = df.index[-1]

        # Check for exit signals first (for existing positions)
        if symbol in self.positions:
            pos = self.positions[symbol]
            exit_signal = False
            exit_reason = ""

            if pos["direction"] == "LONG":
                # Exit long if price breaks below trailing stop
                if latest["close"] < latest["trail_low"]:
                    exit_signal = True
                    exit_reason = "trailing_stop_long"
                # Or if trend weakens
                elif latest["adx"] < 20 or latest["minus_di"] > latest["plus_di"]:
                    exit_signal = True
                    exit_reason = "trend_weakness"

            elif pos["direction"] == "SHORT":
                # Exit short if price breaks above trailing stop
                if latest["close"] > latest["trail_high"]:
                    exit_signal = True
                    exit_reason = "trailing_stop_short"
                # Or if trend weakens
                elif latest["adx"] < 20 or latest["plus_di"] > latest["minus_di"]:
                    exit_signal = True
                    exit_reason = "trend_weakness"

            if exit_signal:
                pnl_pct = 0
                if pos["direction"] == "LONG":
                    pnl_pct = (
                        (
                            (latest["close"] - pos["entry_price"])
                            / pos["entry_price"]
                            * 100
                        )
                        if pos["entry_price"] > 0
                        else 0.0
                    )
                else:  # SHORT
                    pnl_pct = (
                        (
                            (pos["entry_price"] - latest["close"])
                            / pos["entry_price"]
                            * 100
                        )
                        if pos["entry_price"] > 0
                        else 0.0
                    )

                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction="FLAT",
                    strength=0.0,
                    strategy_id=self.name,
                    price=latest["close"],
                    atr=latest["atr"],
                    metadata={
                        "reason": exit_reason,
                        "adx": latest["adx"],
                        "pnl_pct": pnl_pct,
                        "holding_periods": (current_time - pos["entry_time"]).days,
                    },
                )
                del self.positions[symbol]
                return signal

        # Check for entry signals (if not in position)
        elif len(self.positions) < self.config["max_positions"]:
            # Strong trend filter
            strong_trend = latest["adx"] > self.config["adx_threshold"]

            # Volume confirmation (if enabled)
            volume_confirm = True
            if self.config["use_volume_filter"]:
                volume_confirm = (
                    latest["volume_ratio"] > self.config["volume_threshold"]
                )

            # Long entry: breakout above channel high
            if (
                latest["close"] > latest["channel_high"]
                and prev["close"] <= prev["channel_high"]
                and strong_trend
                and latest["plus_di"] > latest["minus_di"]
                and volume_confirm
            ):

                # Calculate signal strength based on breakout magnitude and trend strength
                breakout_strength = (
                    min(1.0, (latest["close"] - latest["channel_high"]) / latest["atr"])
                    if latest["atr"] > 0
                    else 0.5
                )
                trend_strength = min(1.0, latest["adx"] / 40)  # Normalize ADX
                strength = (breakout_strength + trend_strength) / 2

                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction="LONG",
                    strength=strength,
                    strategy_id=self.name,
                    price=latest["close"],
                    atr=latest["atr"],
                    metadata={
                        "reason": "channel_breakout_long",
                        "channel_high": latest["channel_high"],
                        "adx": latest["adx"],
                        "volume_ratio": latest["volume_ratio"],
                        "roc": latest["roc"],
                    },
                )

                # Track position
                self.positions[symbol] = {
                    "entry_price": latest["close"],
                    "entry_atr": latest["atr"],
                    "entry_time": current_time,
                    "direction": "LONG",
                }

                # Track breakout
                self.channel_breaks[symbol] = {
                    "time": current_time,
                    "direction": "LONG",
                    "strength": strength,
                }

                return signal

            # Short entry: breakout below channel low
            elif (
                latest["close"] < latest["channel_low"]
                and prev["close"] >= prev["channel_low"]
                and strong_trend
                and latest["minus_di"] > latest["plus_di"]
                and volume_confirm
            ):

                # Calculate signal strength
                breakout_strength = (
                    min(1.0, (latest["channel_low"] - latest["close"]) / latest["atr"])
                    if latest["atr"] > 0
                    else 0.5
                )
                trend_strength = min(1.0, latest["adx"] / 40)
                strength = (
                    -(breakout_strength + trend_strength) / 2
                )  # Negative for short

                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction="SHORT",
                    strength=strength,
                    strategy_id=self.name,
                    price=latest["close"],
                    atr=latest["atr"],
                    metadata={
                        "reason": "channel_breakout_short",
                        "channel_low": latest["channel_low"],
                        "adx": latest["adx"],
                        "volume_ratio": latest["volume_ratio"],
                        "roc": latest["roc"],
                    },
                )

                # Track position
                self.positions[symbol] = {
                    "entry_price": latest["close"],
                    "entry_atr": latest["atr"],
                    "entry_time": current_time,
                    "direction": "SHORT",
                }

                # Track breakout
                self.channel_breaks[symbol] = {
                    "time": current_time,
                    "direction": "SHORT",
                    "strength": abs(strength),
                }

                return signal

        return None

    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """Calculate position size using volatility scaling."""
        if signal.direction == "FLAT":
            return 0.0, 0.0

        # Get current ATR for volatility scaling
        atr = signal.atr or 0.02 * signal.price

        # Volatility-scaled position size
        daily_vol = atr / signal.price if signal.price > 0 else 0.01
        annualized_vol = daily_vol * np.sqrt(252)

        # Base volatility weight
        vol_weight = (
            min(1.0, risk_context.volatility_target / annualized_vol)
            if annualized_vol > 0
            else 0.5
        )

        # Adjust for signal strength
        vol_weight *= abs(signal.strength)

        # Apply Kelly fraction if available
        kelly_fraction = self.calculate_kelly_fraction()
        if kelly_fraction > 0:
            vol_weight *= kelly_fraction

        # Apply maximum position size constraint
        max_position_value = (
            risk_context.account_equity * risk_context.max_position_size
        )
        position_value = risk_context.account_equity * vol_weight
        position_value = min(position_value, max_position_value)

        # Calculate shares/contracts
        position_size = position_value / signal.price if signal.price > 0 else 0

        # Calculate stop loss based on trailing stop
        if signal.direction == "LONG":
            # For micro futures, use wider stop
            stop_multiplier = 2.0 if "M" in signal.symbol else 1.5
            stop_loss = signal.price - (stop_multiplier * atr)
        else:  # SHORT
            stop_multiplier = 2.0 if "M" in signal.symbol else 1.5
            stop_loss = signal.price + (stop_multiplier * atr)

        return position_size, stop_loss

    def backtest_metrics(self, trades: pd.DataFrame) -> dict:
        """Calculate trend-following specific metrics."""
        if trades.empty:
            return {}

        # Basic metrics
        metrics = {
            "total_trades": len(trades),
            "avg_holding_days": (
                trades["holding_periods"].mean() if "holding_periods" in trades else 0
            ),
        }

        # Separate long and short trades
        long_trades = trades[trades["size"] > 0]
        short_trades = trades[trades["size"] < 0]

        # Long metrics
        if not long_trades.empty:
            long_wins = long_trades[long_trades["pnl"] > 0]
            metrics["long_trades"] = len(long_trades)
            metrics["long_win_rate"] = (
                len(long_wins) / len(long_trades) if len(long_trades) > 0 else 0.0
            )
            metrics["avg_long_pnl"] = long_trades["pnl"].mean()

        # Short metrics
        if not short_trades.empty:
            short_wins = short_trades[short_trades["pnl"] > 0]
            metrics["short_trades"] = len(short_trades)
            metrics["short_win_rate"] = (
                len(short_wins) / len(short_trades) if len(short_trades) > 0 else 0.0
            )
            metrics["avg_short_pnl"] = short_trades["pnl"].mean()

        # Calculate expectancy
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]

        if len(wins) > 0 and len(losses) > 0:
            avg_win = wins["pnl"].mean()
            avg_loss = abs(losses["pnl"].mean())
            win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0
            metrics["expectancy"] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return metrics

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate trend following strategy configuration."""
        return validate_trend_following_config(config)


# Alias for backward compatibility
TrendFollowingMultiStrategy = TrendFollowingMulti
