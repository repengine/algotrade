"""Mean reversion strategy using RSI and ATR bands."""

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
from utils.constants import (
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_SCALAR,
)
from utils.validators.strategy_validators import (
    validate_mean_reversion_config,
)


class MeanReversionEquity(BaseStrategy):
    """Mean reversion strategy for liquid equities.

    Entry: RSI(2) < 10 AND Close < Lower Band (2.5 ATR)
    Exit: Close > MA(10) OR Stop Loss at 3 ATR
    """

    def __init__(self, config: dict[str, Any]) -> None:
        default_config = {
            "name": "MeanReversionEquity",
            "symbols": ["SPY", "QQQ", "IWM", "DIA"],
            "lookback_period": 252,
            "rsi_period": 2,
            "rsi_oversold": 10,
            "rsi_overbought": 90,
            "atr_period": 14,
            "atr_band_mult": 2.5,
            "ma_exit_period": 10,
            "stop_loss_atr": 3.0,
            "max_positions": 5,
        }

        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)

        self.positions = {}  # Track open positions
        self.indicators = {}  # Cache indicators

    def init(self) -> None:
        """Initialize strategy components."""
        self.positions.clear()
        self.indicators.clear()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, ATR, and bands."""
        df = data.copy()

        # RSI
        df["rsi"] = talib.RSI(df["close"], timeperiod=self.config["rsi_period"])

        # ATR
        df["atr"] = talib.ATR(
            df["high"], df["low"], df["close"], timeperiod=self.config["atr_period"]
        )

        # Moving averages
        df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
        df["sma_exit"] = talib.SMA(
            df["close"], timeperiod=self.config["ma_exit_period"]
        )

        # ATR Bands
        df["upper_band"] = df["sma_20"] + (df["atr"] * self.config["atr_band_mult"])
        df["lower_band"] = df["sma_20"] - (df["atr"] * self.config["atr_band_mult"])

        # Volume filter
        df["volume_sma"] = talib.SMA(df["volume"], timeperiod=20)
        df["volume_ratio"] = np.where(
            df["volume_sma"] > 0, df["volume"] / df["volume_sma"], 1.0
        )

        return df

    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process new data and generate trading signal."""
        if not self.validate_data(data):
            return None

        # Calculate indicators
        df = self.calculate_indicators(data)

        if len(df) < self.config["atr_period"]:  # Need enough data
            return None

        latest = df.iloc[-1]
        symbol = data.attrs.get("symbol", "UNKNOWN")
        current_time = df.index[-1]

        # Check for exit signals first
        if symbol in self.positions:
            pos = self.positions[symbol]

            # Exit conditions
            exit_signal = False

            # 1. Price above exit MA
            if latest["close"] > latest["sma_exit"]:
                exit_signal = True

            # 2. Stop loss
            stop_price = pos["entry_price"] - (
                self.config["stop_loss_atr"] * pos["entry_atr"]
            )
            if latest["close"] < stop_price:
                exit_signal = True

            # 3. Overbought RSI (profit taking)
            if latest["rsi"] > self.config["rsi_overbought"]:
                exit_signal = True

            if exit_signal:
                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction="FLAT",
                    strength=0.0,
                    strategy_id=self.name,
                    price=latest["close"],
                    atr=latest["atr"],
                    metadata={
                        "reason": "mean_reversion_exit",
                        "exit_trigger": (
                            "price_above_ma"
                            if latest["close"] > latest["sma_exit"]
                            else "stop_loss"
                        ),
                        "pnl_pct": (
                            (
                                (latest["close"] - pos["entry_price"])
                                / pos["entry_price"]
                                * 100
                            )
                            if pos["entry_price"] > 0
                            else 0.0
                        ),
                    },
                )
                del self.positions[symbol]
                return signal

        # Check for entry signals
        elif len(self.positions) < self.config["max_positions"]:
            # Entry conditions
            oversold = latest["rsi"] < self.config["rsi_oversold"]
            below_band = latest["close"] < latest["lower_band"]

            # Apply volume filter if enabled
            if self.config.get("volume_filter", True):
                volume_confirm = latest["volume_ratio"] > 1.2  # 20% above average
            else:
                volume_confirm = True  # Always true if filter disabled

            if oversold and below_band and volume_confirm:
                # Calculate signal strength based on oversold magnitude
                strength = (
                    min(
                        1.0,
                        (self.config["rsi_oversold"] - latest["rsi"])
                        / self.config["rsi_oversold"],
                    )
                    if self.config["rsi_oversold"] > 0
                    else 0.5
                )

                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction="LONG",
                    strength=strength,
                    strategy_id=self.name,
                    price=latest["close"],
                    atr=latest["atr"],
                    metadata={
                        "reason": "mean_reversion_entry",
                        "rsi": latest["rsi"],
                        "band_distance": (latest["lower_band"] - latest["close"])
                        / latest["atr"],
                        "volume_ratio": latest["volume_ratio"],
                    },
                )

                # Track position
                self.positions[symbol] = {
                    "entry_price": latest["close"],
                    "entry_atr": latest["atr"],
                    "entry_time": current_time,
                }

                return signal

        return None

    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """Calculate position size using volatility scaling and Kelly criterion."""
        if signal.direction == "FLAT":
            return 0.0, 0.0

        # Get current ATR for volatility scaling
        atr = signal.atr or 0.02 * signal.price  # Default 2% if no ATR

        # Volatility-scaled position size
        # weight_i = min(1, σ_target / σ_i)
        # Assuming ATR/price approximates daily volatility
        daily_vol = atr / signal.price if signal.price > 0 else VOLATILITY_SCALAR
        annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        vol_weight = (
            min(1.0, risk_context.volatility_target / annualized_vol)
            if annualized_vol > 0
            else 0.5
        )

        # Apply Kelly fraction if we have enough history
        kelly_fraction = self.calculate_kelly_fraction()
        if kelly_fraction > 0:
            vol_weight *= kelly_fraction

        # Apply maximum position size constraint
        max_position_value = (
            risk_context.account_equity * risk_context.max_position_size
        )
        position_value = risk_context.account_equity * vol_weight * signal.strength
        position_value = min(position_value, max_position_value)

        # Calculate shares (will be rounded by execution layer)
        position_size = position_value / signal.price if signal.price > 0 else 0

        # Calculate stop loss price
        stop_loss = signal.price - (self.config["stop_loss_atr"] * atr)

        return position_size, stop_loss

    def backtest_metrics(self, trades: pd.DataFrame) -> dict:
        """Calculate strategy-specific metrics."""
        if trades.empty:
            return {}

        # Win rate
        wins = trades[trades["pnl"] > 0]
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0

        # Average win/loss
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        losses = trades[trades["pnl"] < 0]
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1

        # Profit factor
        total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
        total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = (
            total_wins / total_losses
            if total_losses > 0
            else (float("inf") if total_wins > 0 else 0.0)
        )

        # Calculate holding period if we have entry and exit times
        avg_holding_period = 0
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            trades["holding_days"] = (
                trades["exit_time"] - trades["entry_time"]
            ).dt.days
            avg_holding_period = trades["holding_days"].mean()
        elif "holding_days" in trades.columns:
            avg_holding_period = trades["holding_days"].mean()

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_holding_period": avg_holding_period,
            "total_trades": len(trades),
        }

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate mean reversion strategy configuration."""
        return validate_mean_reversion_config(config)


# Alias for backward compatibility
MeanReversionEquityStrategy = MeanReversionEquity
