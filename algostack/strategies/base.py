"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class Signal(BaseModel):
    """Trading signal with metadata."""

    timestamp: datetime
    symbol: str
    direction: str = Field(..., pattern="^(LONG|SHORT|FLAT)$")
    strength: float = Field(..., ge=-1.0, le=1.0)
    strategy_id: str
    price: float
    atr: Optional[float] = None
    metadata: dict[str, Any] = {}

    @field_validator("strength")
    @classmethod
    def validate_strength(cls, v: float, info) -> float:
        direction = info.data.get("direction") if info.data else None
        if direction == "FLAT" and v != 0:
            raise ValueError("FLAT signals must have strength=0")
        if direction == "LONG" and v < 0:
            raise ValueError("LONG signals must have positive strength")
        if direction == "SHORT" and v > 0:
            raise ValueError("SHORT signals must have negative strength")
        return v


@dataclass
class RiskContext:
    """Risk parameters passed to strategy sizing."""

    account_equity: float
    open_positions: int
    daily_pnl: float
    max_drawdown_pct: float
    volatility_target: float = 0.10  # 10% annualized
    max_position_size: float = 0.20  # 20% of equity
    current_regime: str = "NORMAL"  # NORMAL, HIGH_VOL, RISK_OFF


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: dict[str, Any]):
        # Validate configuration before using it
        self.config = self.validate_config(config)
        self.name = self.config.get("name", self.__class__.__name__)
        self.symbols = self.config.get("symbols", [])
        self.lookback_period = self.config.get("lookback_period", 252)
        self.enabled = self.config.get("enabled", True)
        self._last_signal: Optional[Signal] = None
        self._performance_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "gross_pnl": 0.0,
            "max_drawdown": 0.0,
        }

    @abstractmethod
    def init(self) -> None:
        """Initialize strategy state and indicators."""
        pass

    @abstractmethod
    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Process new data and generate trading signal.

        Args:
            data: DataFrame with OHLCV data, indexed by datetime

        Returns:
            Signal object or None if no action
        """
        pass

    @abstractmethod
    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """
        Calculate position size given signal and risk context.

        Args:
            signal: Trading signal from next()
            risk_context: Current portfolio risk parameters

        Returns:
            Tuple of (position_size, stop_loss_price)
        """
        pass

    def update_performance(self, trade_result: dict[str, Any]) -> None:
        """Update internal performance tracking."""
        self._performance_stats["trades"] += 1
        if trade_result["pnl"] > 0:
            self._performance_stats["wins"] += 1
        else:
            self._performance_stats["losses"] += 1
        self._performance_stats["gross_pnl"] += trade_result["pnl"]

    @property
    def hit_rate(self) -> float:
        """Calculate win rate of the strategy."""
        if self._performance_stats["trades"] == 0:
            return 0.0
        return self._performance_stats["wins"] / self._performance_stats["trades"]

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)."""
        wins = self._performance_stats["wins"]
        losses = self._performance_stats["losses"]
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    def calculate_kelly_fraction(self) -> float:
        """Calculate Kelly fraction for position sizing."""
        if self._performance_stats["trades"] < 30:  # Need statistical significance
            return 0.0

        p = self.hit_rate
        if p == 0 or p == 1:
            return 0.0

        # Simplified Kelly assuming equal win/loss sizes
        # f = p - (1-p) = 2p - 1
        kelly = 2 * p - 1

        # Half-Kelly for safety
        return max(0, kelly * 0.5)

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate incoming data quality."""
        if data.empty:
            return False

        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_columns):
            return False

        # Check for NaN values
        if data[required_columns].isnull().any().any():
            return False

        # Check for valid OHLC relationships
        valid_ohlc = (
            (data["high"] >= data["low"]).all()
            and (data["high"] >= data["open"]).all()
            and (data["high"] >= data["close"]).all()
            and (data["low"] <= data["open"]).all()
            and (data["low"] <= data["close"]).all()
        )

        return valid_ohlc

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Validate strategy configuration parameters.

        Override this method in subclasses to implement specific validation.

        Args:
            config: Raw configuration dictionary

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Base validation - ensure required fields exist
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Validate common parameters
        if "lookback_period" in config:
            if (
                not isinstance(config["lookback_period"], int)
                or config["lookback_period"] <= 0
            ):
                raise ValueError("lookback_period must be a positive integer")

        if "enabled" in config:
            if not isinstance(config["enabled"], bool):
                raise ValueError("enabled must be a boolean")

        return config

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate trading signals from market data.
        
        This is a default implementation that can be overridden by subclasses.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of trading signals
        """
        # Default implementation returns no signals
        # Subclasses should override this method
        return []
