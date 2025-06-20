#!/usr/bin/env python3
"""Portfolio management with volatility budgeting and risk controls."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategies.base import RiskContext, Signal
from utils.constants import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_KELLY_FRACTION,
    MAX_KELLY_FRACTION,
    MIN_TRADES_FOR_KELLY,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    strategy_id: str = ""
    direction: str = "LONG"  # LONG or SHORT
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    realized_pnl: float = 0.0
    position_type: Optional[str] = None  # Backward compatibility

    def __post_init__(self):
        """Initialize current price if not set."""
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        # Handle position_type backward compatibility
        if self.position_type is not None:
            self.direction = self.position_type
        elif self.direction:
            self.position_type = self.direction

        # If quantity is negative, it's a SHORT position
        if self.quantity < 0 and self.direction == "LONG":
            self.direction = "SHORT"
            self.position_type = "SHORT"

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        if self.direction == "LONG":
            return self.quantity * (self.current_price - self.entry_price)
        else:  # SHORT
            return abs(self.quantity) * (self.entry_price - self.current_price)

    @property
    def pnl_percentage(self) -> float:
        """P&L as percentage of entry value."""
        entry_value = abs(self.quantity) * self.entry_price
        return (self.unrealized_pnl / entry_value) * 100 if entry_value > 0 else 0

    @property
    def avg_price(self) -> float:
        """Average price (backward compatibility alias for entry_price)."""
        return self.entry_price

    @avg_price.setter
    def avg_price(self, value: float) -> None:
        """Set average price (backward compatibility)."""
        self.entry_price = value

    def update_price(self, price: float) -> None:
        """Update current price of position."""
        self.current_price = price

    def reduce_position(self, quantity: float, price: float) -> float:
        """Reduce position by given quantity and calculate realized P&L."""
        if abs(quantity) > abs(self.quantity):
            raise ValueError("Cannot reduce position by more than current quantity")

        # Calculate realized P&L for the portion being closed
        if self.direction == "LONG":
            realized = quantity * (price - self.entry_price)
        else:  # SHORT
            realized = abs(quantity) * (self.entry_price - price)

        # Update position
        self.quantity -= quantity
        self.realized_pnl += realized
        self.current_price = price

        return realized

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_time": self.entry_time.isoformat()
            if isinstance(self.entry_time, datetime)
            else str(self.entry_time),
            "direction": self.direction,
            "position_type": self.position_type,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.market_value,
            "pnl_percentage": self.pnl_percentage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
        }


class PortfolioEngine:
    """Manages portfolio allocation, risk, and position sizing across strategies."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        # Backward compatibility parameters
        initial_capital: Optional[float] = None,
        max_positions: Optional[int] = None,
        position_size_method: Optional[str] = None,
        risk_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Handle backward compatibility
        if config is None and initial_capital is not None:
            # Old API - build config from parameters
            config = {
                "initial_capital": initial_capital,
                "max_positions": max_positions or 10,
                "position_size_method": position_size_method or "equal_weight",
                **kwargs,
            }
        elif config is None:
            config = {}

        self.config = config
        # Ensure initial_capital is set properly
        if initial_capital is not None and "initial_capital" not in config:
            config["initial_capital"] = initial_capital
        self.initial_capital = config.get(
            "initial_capital", initial_capital or DEFAULT_INITIAL_CAPITAL
        )
        self.current_equity = self.initial_capital
        self._cash = self.initial_capital  # Actual cash balance
        self.max_positions = config.get("max_positions", 10)  # Backward compatibility
        self.position_size_method = config.get(
            "position_size_method", "equal_weight"
        )  # Backward compatibility
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.strategy_allocations: dict[str, float] = {}  # strategy -> allocation
        self.performance_history: list[dict[str, Any]] = []
        self.correlation_matrix = pd.DataFrame()
        self.volatility_targets = config.get("volatility_targets", {})

        # Handle risk_config parameter
        self.risk_config = risk_config or {}
        self.risk_limits = {
            "max_portfolio_volatility": config.get("target_vol", 0.10),
            "max_position_size": self.risk_config.get(
                "max_position_size", config.get("max_position_size", 0.20)
            ),
            "max_sector_exposure": config.get("max_sector_exposure", 0.40),
            "max_drawdown": config.get("max_drawdown", 0.15),
            "max_correlation": config.get("max_correlation", 0.70),
            "margin_buffer": config.get("margin_buffer", 0.25),
            "max_leverage": self.risk_config.get("max_leverage", 1.0),
            "risk_per_trade": self.risk_config.get("risk_per_trade", 0.02),
            "stop_loss_pct": self.risk_config.get("stop_loss_pct", 0.02),
        }

        # Backward compatibility attributes
        self.trades: list[dict[str, Any]] = []  # Trade history
        self.equity_curve: list[Any] = []  # Equity history (can be float or dict)
        self.total_commission = 0.0  # Track total commissions paid

        # Risk tracking
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl: list[float] = []
        self.is_risk_off = False
        self.risk_off_until: Optional[datetime] = None
        self._realized_pnl = 0.0  # Private attribute for realized P&L

        # Kelly tracking
        self.strategy_kelly_fractions: dict[str, float] = {}
        self.strategy_performance: dict[
            str, dict[str, float]
        ] = {}  # Track per-strategy metrics

    @property
    def total_value(self) -> float:
        """Total portfolio value (backward compatibility alias for current_equity)."""
        return self.current_equity

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (method version for compatibility)."""
        # Calculate total value from cash and positions
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self._cash + positions_value

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

    def update_positions(self, prices: dict[str, float]) -> None:
        """Update all position prices (alias for update_market_prices)."""
        self.update_market_prices(prices)

    def update_position(
        self, symbol: str, quantity: float, avg_price: float, current_price: float
    ) -> None:
        """Update or create a position with given parameters."""
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            position.quantity = quantity
            position.entry_price = avg_price
            position.current_price = current_price
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                strategy_id="UNKNOWN",  # Will be set by actual trades
                direction="LONG" if quantity > 0 else "SHORT",
                quantity=quantity,
                entry_price=avg_price,
                entry_time=datetime.now(),
                current_price=current_price,
            )

    def calculate_portfolio_metrics(self) -> dict[str, float]:
        """Calculate current portfolio metrics."""
        # Calculate total market value
        total_value = self.current_equity
        positions_value = sum(pos.market_value for pos in self.positions.values())
        cash = total_value - positions_value

        # Calculate unrealized P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        # Update peak and drawdown
        if total_value > self.peak_equity:
            self.peak_equity = total_value
        self.current_drawdown = (
            (self.peak_equity - total_value) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )

        # Count positions efficiently in a single pass
        long_positions = short_positions = 0
        for pos in self.positions.values():
            if pos.direction == "LONG":
                long_positions += 1
            elif pos.direction == "SHORT":
                short_positions += 1

        return {
            "total_equity": total_value,
            "cash": cash,
            "positions_value": positions_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": total_value - self.initial_capital - unrealized_pnl,
            "current_drawdown": self.current_drawdown,
            "position_count": len(self.positions),
            "long_positions": long_positions,
            "short_positions": short_positions,
            "margin_usage": positions_value / total_value if total_value > 0 else 0,
        }

    def calculate_portfolio_volatility(self, returns_data: pd.DataFrame) -> float:
        """Calculate current portfolio volatility."""
        if returns_data.empty or len(self.positions) == 0:
            return 0.0

        # Get position weights
        total_value = self.current_equity
        weights = {}

        for symbol, position in self.positions.items():
            weight = position.market_value / total_value
            if position.direction == "SHORT":
                weight = -weight
            weights[symbol] = weight

        # Calculate portfolio variance
        portfolio_variance = 0.0
        symbols = list(weights.keys())

        # Covariance calculation
        for sym1 in symbols:
            for sym2 in symbols:
                if sym1 in returns_data.columns and sym2 in returns_data.columns:
                    cov = returns_data[sym1].cov(returns_data[sym2])
                    portfolio_variance += weights[sym1] * weights[sym2] * cov

        # Annualized volatility
        portfolio_vol = np.sqrt(portfolio_variance * TRADING_DAYS_PER_YEAR)

        return float(portfolio_vol)

    def update_correlation_matrix(self, returns_data: pd.DataFrame) -> None:
        """Update correlation matrix for portfolio assets."""
        if len(returns_data.columns) > 1:
            self.correlation_matrix = returns_data.corr()

    def check_risk_limits(self) -> tuple[bool, list[str]]:
        """Check if portfolio is within risk limits."""
        violations = []

        # Check drawdown limit (applies even without positions)
        if self.current_drawdown > self.risk_limits["max_drawdown"]:
            violations.append(
                f"Drawdown {self.current_drawdown:.1%} exceeds limit {self.risk_limits['max_drawdown']:.1%}"
            )

        # Only check position-related limits if we have positions
        if self.positions:
            # Check position concentration
            violations.extend(self._check_position_concentration())

            # Check correlation limits
            violations.extend(self._check_correlation_violations())

        return len(violations) == 0, violations

    def _check_position_concentration(self) -> list[str]:
        """Check position concentration limits."""
        violations = []

        # Batch calculate all position weights at once
        if self.current_equity <= 0:
            if self.positions:
                violations.append(
                    "Cannot calculate position concentration: total equity is zero or negative"
                )
            return violations

        # Calculate all weights in one pass
        max_limit = self.risk_limits["max_position_size"]

        for symbol, position in self.positions.items():
            try:
                position_weight = float(position.market_value) / self.current_equity
                if position_weight > max_limit:
                    violations.append(
                        f"{symbol} weight {position_weight:.1%} exceeds limit {max_limit:.1%}"
                    )
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Error calculating weight for {symbol}: {e}")
                violations.append(f"Invalid market value for {symbol}")

        return violations

    def _check_correlation_violations(self) -> list[str]:
        """Check correlation limit violations between positions."""
        violations = []

        if self.correlation_matrix.empty or len(self.positions) < 2:
            return violations

        # Get positions that exist in correlation matrix
        position_symbols = set(self.positions.keys())
        matrix_symbols = set(self.correlation_matrix.index) & set(
            self.correlation_matrix.columns
        )
        relevant_symbols = list(position_symbols & matrix_symbols)

        if len(relevant_symbols) < 2:
            return violations

        # Use numpy for efficient correlation checking
        max_corr = self.risk_limits["max_correlation"]

        try:
            # Get correlation submatrix for relevant symbols only
            corr_subset = self.correlation_matrix.loc[
                relevant_symbols, relevant_symbols
            ]

            # Use numpy operations for efficiency
            corr_values = corr_subset.values
            mask = np.abs(corr_values) > max_corr

            # Only check upper triangle (avoid duplicates)
            mask = np.triu(mask, k=1)

            # Find violations
            violations_idx = np.where(mask)

            for i, j in zip(violations_idx[0], violations_idx[1]):
                sym1, sym2 = relevant_symbols[i], relevant_symbols[j]
                corr = float(corr_values[i, j])
                violations.append(
                    f"{sym1}-{sym2} correlation {corr:.2f} exceeds limit {max_corr:.2f}"
                )

        except Exception as e:
            logger.error(f"Error checking correlations: {e}")
            violations.append("Error processing correlation matrix")

        return violations

    def allocate_capital(
        self, signals: list[Signal], market_data: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Allocate capital across strategies using volatility budgeting."""
        if not signals:
            return {}

        allocations = {}

        # Group signals by strategy
        strategy_signals: dict[str, list[Any]] = {}
        for signal in signals:
            if signal.direction != "FLAT":
                strategy = signal.strategy_id
                if strategy not in strategy_signals:
                    strategy_signals[strategy] = []
                strategy_signals[strategy].append(signal)

        # Calculate volatility for each strategy
        strategy_vols = {}
        for strategy, strat_signals in strategy_signals.items():
            # Estimate strategy volatility from signals
            symbols = [s.symbol for s in strat_signals]
            if symbols and all(sym in market_data for sym in symbols):
                returns = pd.DataFrame()
                for sym in symbols:
                    if "returns" in market_data[sym].columns:
                        returns[sym] = market_data[sym]["returns"]
                    else:
                        returns[sym] = market_data[sym]["close"].pct_change()

                # Strategy volatility as average of constituents
                if not returns.empty:
                    strategy_vols[strategy] = returns.std().mean() * np.sqrt(
                        TRADING_DAYS_PER_YEAR
                    )
                else:
                    strategy_vols[strategy] = 0.10  # Default 10%
            else:
                strategy_vols[strategy] = 0.10

        # Volatility budget allocation
        total_risk_budget = self.risk_limits["max_portfolio_volatility"]

        if self.config.get("use_equal_risk", True):
            # Equal risk contribution
            n_strategies = len(strategy_vols)
            risk_per_strategy = total_risk_budget / np.sqrt(n_strategies)

            for strategy, vol in strategy_vols.items():
                if vol > 0:
                    # Allocation = risk_budget / strategy_vol
                    allocations[strategy] = risk_per_strategy / vol
                else:
                    allocations[strategy] = 0.0
        else:
            # Use configured weights
            for strategy in strategy_vols:
                base_weight = self.strategy_allocations.get(
                    strategy, 1.0 / len(strategy_vols)
                )
                allocations[strategy] = base_weight

        # Normalize allocations
        total_alloc = sum(allocations.values())
        if total_alloc > 0:
            for strategy in allocations:
                allocations[strategy] /= total_alloc

        # Apply Kelly fractions if available
        for strategy in allocations:
            if strategy in self.strategy_kelly_fractions:
                kelly = self.strategy_kelly_fractions[strategy]
                allocations[strategy] *= kelly

        return allocations

    def size_position(self, signal: Signal, allocation: float) -> tuple[float, float]:
        """Size individual position within strategy allocation."""
        # Get risk context
        RiskContext(
            account_equity=self.current_equity,
            open_positions=len(self.positions),
            daily_pnl=sum(self.daily_pnl[-5:]) if self.daily_pnl else 0,
            max_drawdown_pct=self.current_drawdown,
            volatility_target=self.risk_limits["max_portfolio_volatility"],
            max_position_size=self.risk_limits["max_position_size"],
            current_regime="RISK_OFF" if self.is_risk_off else "NORMAL",
        )

        # Base position value from allocation
        position_value = self.current_equity * allocation * abs(signal.strength)

        # Apply position limits
        max_position = self.current_equity * self.risk_limits["max_position_size"]
        position_value = min(position_value, max_position)

        # Check margin requirements
        margin_used = sum(pos.market_value for pos in self.positions.values())
        available_margin = self.current_equity * (1 - self.risk_limits["margin_buffer"])

        if margin_used + position_value > available_margin:
            # Scale down to fit margin
            position_value = max(0, available_margin - margin_used)

        # Calculate shares
        position_size = position_value / signal.price if signal.price > 0 else 0

        # Stop loss from signal metadata or default
        stop_loss = signal.metadata.get("stop_loss", 0.0)
        if stop_loss == 0 and signal.atr:
            # Default stop at 2 ATR
            if signal.direction == "LONG":
                stop_loss = signal.price - (2 * signal.atr)
            else:
                stop_loss = signal.price + (2 * signal.atr)

        return position_size, stop_loss

    def execute_signal(
        self, signal: Signal, position_size: float, stop_loss: float
    ) -> Optional[Position]:
        """Execute a trading signal and create position."""
        if position_size <= 0:
            return None

        # Check if we already have a position
        if signal.symbol in self.positions:
            existing = self.positions[signal.symbol]
            if existing.direction != signal.direction:
                # Close existing position first
                self.close_position(signal.symbol, signal.price)
            else:
                # Already have position in same direction
                return None

        # Create new position
        position = Position(
            symbol=signal.symbol,
            strategy_id=signal.strategy_id,
            direction=signal.direction,
            quantity=position_size if signal.direction == "LONG" else -position_size,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            current_price=signal.price,
            stop_loss=stop_loss,
            metadata=signal.metadata,
        )

        self.positions[signal.symbol] = position

        logger.info(
            f"Opened {position.direction} position in {position.symbol}: "
            f"{position.quantity:.2f} @ ${position.entry_price:.2f}"
        )

        return position

    def close_position_simple(
        self, symbol: str, exit_price: float
    ) -> Optional[dict[str, Any]]:
        """Close a position and calculate P&L."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        position.current_price = exit_price

        # Calculate P&L
        pnl = position.unrealized_pnl
        pnl_pct = position.pnl_percentage

        # Update strategy performance
        strategy = position.strategy_id
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "win_pnl": 0.0,
                "loss_pnl": 0.0,
            }

        perf = self.strategy_performance[strategy]
        perf["trades"] += 1
        perf["total_pnl"] += pnl

        if pnl > 0:
            perf["wins"] += 1
            perf["win_pnl"] += pnl
        else:
            perf["loss_pnl"] += abs(pnl)

        # Update equity
        self.current_equity += pnl

        # Remove position
        del self.positions[symbol]

        logger.info(
            f"Closed {position.direction} position in {symbol}: "
            f"P&L ${pnl:.2f} ({pnl_pct:.1f}%)"
        )

        return {
            "symbol": symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_period": (datetime.now() - position.entry_time).days,
        }

    def update_strategy_kelly_fractions(self) -> None:
        """Update Kelly fractions based on strategy performance."""
        for strategy, perf in self.strategy_performance.items():
            if perf["trades"] < MIN_TRADES_FOR_KELLY:  # Need sufficient history
                self.strategy_kelly_fractions[strategy] = (
                    DEFAULT_KELLY_FRACTION  # Default half-Kelly
                )
                continue

            # Calculate win rate and win/loss ratio
            win_rate = perf["wins"] / perf["trades"]
            avg_win = perf["win_pnl"] / perf["wins"] if perf["wins"] > 0 else 0
            avg_loss = (
                perf["loss_pnl"] / (perf["trades"] - perf["wins"])
                if perf["trades"] > perf["wins"]
                else 1
            )

            if avg_loss == 0:
                self.strategy_kelly_fractions[strategy] = 0.5
                continue

            # Kelly formula: f = p - q/b
            # where p = win_rate, q = 1-p, b = avg_win/avg_loss
            b = avg_win / avg_loss

            # Handle edge case where b is 0 (no average win)
            if b == 0:
                kelly = 0  # No edge, no allocation
            else:
                kelly = win_rate - (1 - win_rate) / b

            # Apply half-Kelly for safety
            kelly = max(
                0, min(MAX_KELLY_FRACTION * 0.25, kelly * DEFAULT_KELLY_FRACTION)
            )  # Cap at 25%

            self.strategy_kelly_fractions[strategy] = kelly

    def process_fill(self, fill_data: dict[str, Any]) -> None:
        """Process a fill event and update positions accordingly."""
        symbol = fill_data.get("symbol")
        quantity = fill_data.get("quantity", 0)
        price = fill_data.get("price", 0)
        side = fill_data.get("side", "BUY")
        commission = fill_data.get("commission", 0.0)
        timestamp = fill_data.get("timestamp", datetime.now())

        if symbol and quantity != 0 and price > 0:
            # Record trade
            self.trades.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "order_id": fill_data.get("order_id", "")
            })
            
            # Update commission tracking
            self.total_commission += commission
            
            # For BUY orders, add to position
            if side == "BUY":
                # Update cash
                total_cost = (quantity * price) + commission
                self._cash -= total_cost
                
                if symbol in self.positions:
                    # Add to existing position
                    position = self.positions[symbol]
                    total_value = (position.quantity * position.entry_price) + (
                        quantity * price
                    )
                    position.quantity += quantity
                    position.entry_price = (
                        total_value / position.quantity
                        if position.quantity != 0
                        else price
                    )
                    position.current_price = price
                else:
                    # Create new position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        strategy_id=fill_data.get("strategy_id", "UNKNOWN"),
                        direction="LONG",
                        quantity=quantity,
                        entry_price=price,
                        entry_time=timestamp,
                        current_price=price,
                    )
            else:  # SELL
                # Update cash
                total_proceeds = (quantity * price) - commission
                self._cash += total_proceeds
                
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # Calculate realized P&L
                    realized_pnl = quantity * (price - position.entry_price)
                    position.realized_pnl += realized_pnl
                    self._realized_pnl += realized_pnl - commission
                    
                    position.quantity -= quantity
                    position.current_price = price

                    # Close position if quantity reaches zero
                    if (
                        abs(position.quantity) < 0.0001
                    ):  # Small epsilon for float comparison
                        del self.positions[symbol]
                else:
                    # Short sell - create new short position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        strategy_id=fill_data.get("strategy_id", "UNKNOWN"),
                        direction="SHORT",
                        quantity=-quantity,
                        entry_price=price,
                        entry_time=timestamp,
                        current_price=price,
                    )
            
            # Update equity
            self._update_equity()

    def check_stops_and_targets(self, current_prices: dict[str, float]) -> list[Signal]:
        """Check stop losses and take profits, generate exit signals."""
        exit_signals = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            position.current_price = current_price

            # Check stop loss
            if position.stop_loss > 0:
                if (
                    position.direction == "LONG" and current_price <= position.stop_loss
                ) or (
                    position.direction == "SHORT"
                    and current_price >= position.stop_loss
                ):
                    exit_signals.append(
                        Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            direction="FLAT",
                            strength=0.0,
                            strategy_id=position.strategy_id,
                            price=current_price,
                            metadata={
                                "reason": "stop_loss",
                                "entry_price": position.entry_price,
                                "stop_price": position.stop_loss,
                            },
                        )
                    )

            # Check take profit
            if position.take_profit > 0:
                if (
                    position.direction == "LONG"
                    and current_price >= position.take_profit
                ) or (
                    position.direction == "SHORT"
                    and current_price <= position.take_profit
                ):
                    exit_signals.append(
                        Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            direction="FLAT",
                            strength=0.0,
                            strategy_id=position.strategy_id,
                            price=current_price,
                            metadata={
                                "reason": "take_profit",
                                "entry_price": position.entry_price,
                                "target_price": position.take_profit,
                            },
                        )
                    )

        return exit_signals

    def global_risk_check(self) -> tuple[bool, list[Signal]]:
        """Perform global risk check and generate risk-off signals if needed."""
        exit_signals = []

        # Check if we're in risk-off mode
        if self.is_risk_off and self.risk_off_until:
            if datetime.now() < self.risk_off_until:
                return False, []  # Still in risk-off period
            else:
                # Risk-off period ended
                self.is_risk_off = False
                self.risk_off_until = None
                logger.info("Risk-off period ended, resuming normal trading")

        # Check drawdown limit
        if self.current_drawdown > self.risk_limits["max_drawdown"]:
            logger.warning(
                f"RISK ALERT: Drawdown {self.current_drawdown:.1%} exceeds limit"
            )

            # Generate exit signals for all positions
            for symbol, position in self.positions.items():
                exit_signals.append(
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        direction="FLAT",
                        strength=0.0,
                        strategy_id=position.strategy_id,
                        price=position.current_price,
                        metadata={
                            "reason": "global_risk_off",
                            "drawdown": self.current_drawdown,
                        },
                    )
                )

            # Enter risk-off mode
            self.is_risk_off = True
            self.risk_off_until = datetime.now() + timedelta(days=2)
            logger.warning(f"Entering risk-off mode until {self.risk_off_until}")

            return False, exit_signals

        # Check position correlations
        is_compliant, violations = self.check_risk_limits()
        if not is_compliant:
            logger.warning(f"Risk violations: {violations}")
            # Could generate selective exits here

        return True, exit_signals

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get comprehensive portfolio summary."""
        metrics = self.calculate_portfolio_metrics()

        # Add strategy breakdown
        strategy_exposure = {}
        for position in self.positions.values():
            strategy = position.strategy_id
            if strategy not in strategy_exposure:
                strategy_exposure[strategy] = 0.0
            strategy_exposure[strategy] += position.market_value

        # Performance by strategy
        strategy_metrics = {}
        for strategy, perf in self.strategy_performance.items():
            if perf["trades"] > 0:
                strategy_metrics[strategy] = {
                    "trades": perf["trades"],
                    "win_rate": perf["wins"] / perf["trades"],
                    "total_pnl": perf["total_pnl"],
                    "kelly_fraction": self.strategy_kelly_fractions.get(strategy, 0.5),
                }

        return {
            "portfolio_metrics": metrics,
            "strategy_exposure": strategy_exposure,
            "strategy_performance": strategy_metrics,
            "risk_status": {
                "is_risk_off": self.is_risk_off,
                "current_drawdown": self.current_drawdown,
                "violations": self.check_risk_limits()[1],
            },
            "positions": {
                symbol: {
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_pct": pos.pnl_percentage,
                }
                for symbol, pos in self.positions.items()
            },
        }

    # Removed duplicate total_value property - already defined at line 210

    @property
    def cash(self) -> float:
        """Available cash in the portfolio."""
        return self._cash

    @cash.setter
    def cash(self, value: float) -> None:
        """Set available cash (backward compatibility)."""
        self._cash = value

    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L from closed trades (backward compatibility)."""
        # If _realized_pnl has been set directly, return that
        if self._realized_pnl != 0.0:
            return self._realized_pnl

        total_pnl = 0.0

        # Group trades by symbol to calculate P&L
        symbol_trades: dict[str, list] = {}
        for trade in self.trades:
            symbol = trade.get("symbol") or getattr(trade, "symbol", None)
            if symbol:
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)

        # Calculate P&L for each symbol - only count closed positions
        for _symbol, trades in symbol_trades.items():
            buy_quantity = 0.0
            sell_quantity = 0.0
            buy_value = 0.0
            sell_value = 0.0

            for trade in trades:
                side = trade.get("side") or getattr(trade, "side", "BUY")
                quantity = trade.get("quantity") or getattr(trade, "quantity", 0)
                price = trade.get("price") or getattr(trade, "price", 0)
                commission = trade.get("commission") or getattr(trade, "commission", 0)

                if side == "BUY":
                    buy_quantity += quantity
                    buy_value += quantity * price + commission
                else:
                    sell_quantity += quantity
                    sell_value += quantity * price - commission

            # Only calculate P&L on the closed portion (minimum of buy and sell quantities)
            closed_quantity = min(buy_quantity, sell_quantity)
            if closed_quantity > 0 and buy_quantity > 0:
                avg_buy_price = buy_value / buy_quantity
                avg_sell_price = sell_value / sell_quantity if sell_quantity > 0 else 0
                total_pnl += closed_quantity * (avg_sell_price - avg_buy_price)

        return total_pnl

    @realized_pnl.setter
    def realized_pnl(self, value: float) -> None:
        """Set realized P&L (backward compatibility)."""
        self._realized_pnl = value

    # Removed duplicate update_position - already defined at line 228

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate and return portfolio performance metrics."""
        # Get basic portfolio metrics first
        base_metrics = self.calculate_portfolio_metrics()

        # Calculate performance metrics from equity curve and trades
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                **base_metrics,
            }

        # Calculate returns from equity curve
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (
            self.equity_curve[-1] - self.initial_capital
        ) / self.initial_capital

        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Trade statistics
        if self.trades:
            # Calculate P&L for each trade
            wins = []
            losses = []

            # Group trades by symbol to match buys and sells
            symbol_trades = {}
            for trade in self.trades:
                symbol = trade.symbol
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)

            # Calculate P&L for each symbol
            for _symbol, trades in symbol_trades.items():
                buy_value = 0
                sell_value = 0
                for trade in trades:
                    if trade.side == "BUY":
                        buy_value += trade.quantity * trade.price + trade.commission
                    else:  # SELL
                        sell_value += trade.quantity * trade.price - trade.commission

                pnl = sell_value - buy_value
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))

            # Calculate trade metrics
            total_trades = len(wins) + len(losses)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            gross_profit = sum(wins)
            gross_loss = sum(losses)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            **base_metrics,
        }

    def calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        # Simple implementation - in production this would track intraday changes
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return total_pnl

    # Backward compatibility methods
    def add_position(self, trade: "Trade") -> None:
        """Add a new position or add to existing position (backward compatibility)."""
        symbol = trade.get("symbol") or getattr(trade, "symbol", None)
        quantity = trade.get("quantity") or getattr(trade, "quantity", 0)
        price = trade.get("price") or getattr(trade, "price", 0)
        side = trade.get("side") or getattr(trade, "side", "BUY")
        commission = trade.get("commission") or getattr(trade, "commission", 0)
        slippage = trade.get("slippage") or getattr(trade, "slippage", 0)

        # Deduct commission and slippage from cash
        self.cash -= commission + slippage

        # Process based on side
        if side == "BUY":
            # Deduct cost from cash
            cost = quantity * price
            self.cash -= cost

            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                old_value = position.quantity * position.entry_price
                new_value = quantity * price
                total_quantity = position.quantity + quantity
                position.quantity = total_quantity
                position.entry_price = (
                    (old_value + new_value) / total_quantity
                    if total_quantity != 0
                    else 0
                )
                position.current_price = price
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    strategy_id="unknown",
                    direction="LONG",
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                )
        else:  # SELL
            if symbol in self.positions:
                # Reduce existing long position
                position = self.positions[symbol]
                if position.direction == "LONG":
                    # Add proceeds to cash
                    proceeds = quantity * price
                    self.cash += proceeds

                    position.quantity -= quantity
                    if position.quantity <= 0:
                        del self.positions[symbol]
                else:  # SHORT position
                    # Adding to short position (selling more)
                    cost = quantity * price
                    self.cash += cost  # Receive cash from short sale

                    old_value = abs(position.quantity) * position.entry_price
                    new_value = quantity * price
                    total_quantity = abs(position.quantity) + quantity
                    position.quantity = -total_quantity  # Negative for short
                    position.entry_price = (
                        (old_value + new_value) / total_quantity
                        if total_quantity != 0
                        else 0
                    )
                    position.current_price = price
            else:
                # Opening a new short position
                proceeds = quantity * price
                self.cash += proceeds  # Receive cash from short sale

                self.positions[symbol] = Position(
                    symbol=symbol,
                    strategy_id="unknown",
                    direction="SHORT",
                    quantity=-quantity,  # Negative for short
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                )

        # Add to trades history
        self.trades.append(trade)

        # Update equity curve
        self.equity_curve.append(self.get_portfolio_value())

    # Removed duplicate get_portfolio_value - already defined at line 214

    def update_market_values(self, prices: dict[str, float]) -> None:
        """Update market values of positions (backward compatibility)."""
        self.update_market_prices(prices)

    def close_position(
        self,
        symbol_or_trade,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Optional["Trade"]:
        """Close position or reduce position size (backward compatibility)."""
        # Handle Trade object input
        if hasattr(symbol_or_trade, "symbol") or isinstance(symbol_or_trade, dict):
            trade_obj = symbol_or_trade
            symbol = trade_obj.get("symbol") or getattr(trade_obj, "symbol", None)
            quantity = trade_obj.get("quantity") or getattr(trade_obj, "quantity", 0)
            price = trade_obj.get("price") or getattr(trade_obj, "price", 0)
            commission = trade_obj.get("commission") or getattr(
                trade_obj, "commission", 0
            )
            timestamp = trade_obj.get("timestamp") or getattr(
                trade_obj, "timestamp", None
            )

            # Process as a sell trade
            self.add_position(trade_obj)
            return trade_obj
        else:
            # Handle individual parameters
            symbol = symbol_or_trade

        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Create trade record
        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            side="SELL" if position.direction == "LONG" else "BUY",
            timestamp=timestamp or datetime.now(),
            commission=commission,
        )

        # Calculate P&L
        if position.direction == "LONG":
            quantity * (price - position.entry_price) - commission
        else:
            quantity * (position.entry_price - price) - commission

        # Update cash
        self.cash += quantity * price - commission

        # Update position
        position.quantity -= quantity
        if position.quantity <= 0:
            del self.positions[symbol]

        # Add to trades
        self.trades.append(trade)

        # Update equity
        self.equity_curve.append(self.get_portfolio_value())

        return trade

    def get_open_positions(
        self, side: Optional[str] = None
    ) -> dict[str, "PositionInfo"]:
        """Get all open positions (backward compatibility)."""
        result = {}
        for symbol, pos in self.positions.items():
            # Filter by side if specified
            if side is not None:
                if side == "LONG" and pos.direction != "LONG":
                    continue
                elif side == "SHORT" and pos.direction != "SHORT":
                    continue

            result[symbol] = PositionInfo(
                symbol=symbol,
                quantity=pos.quantity,
                avg_price=pos.entry_price,
                current_price=pos.current_price,
                market_value=pos.market_value,
                unrealized_pnl=pos.unrealized_pnl,
                pnl_pct=pos.pnl_percentage,
            )
        return result

    def reset(self) -> None:
        """Reset portfolio to initial state (backward compatibility)."""
        self._cash = self.initial_capital
        self.current_equity = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve = []
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl.clear()
        self.is_risk_off = False
        self.risk_off_until = None
        self.strategy_kelly_fractions.clear()
        self.strategy_performance.clear()
        self.total_commission = 0.0

    # Removed duplicate process_fill - already defined at line 677

    def _update_equity(self) -> None:
        """Update current equity based on cash and positions."""
        # Calculate total position value
        position_value = sum(pos.market_value for pos in self.positions.values())

        # Update current equity
        self.current_equity = self._cash + position_value

        # Update equity curve only if it changed
        if (
            not self.equity_curve
            or (
                isinstance(self.equity_curve[-1], dict)
                and self.current_equity != self.equity_curve[-1].get("value", 0)
            )
            or (
                isinstance(self.equity_curve[-1], (int, float))
                and self.current_equity != self.equity_curve[-1]
            )
        ):
            # Support both dict and numeric formats for backward compatibility
            self.equity_curve.append(
                {"timestamp": datetime.now(), "value": self.current_equity}
            )

        # Update peak and drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        self.current_drawdown = (
            (self.peak_equity - self.current_equity) / self.peak_equity
            if self.peak_equity > 0
            else 0.0
        )

    def get_performance_metrics(self) -> dict[str, float]:
        """Calculate portfolio performance metrics (backward compatibility)."""
        # Calculate total value
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_value = self._cash + positions_value

        # Calculate returns
        total_return = total_value - self.initial_capital
        total_return_pct = (
            (total_return / self.initial_capital * 100)
            if self.initial_capital > 0
            else 0.0
        )

        # Calculate P&L components
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl = self._realized_pnl + unrealized_pnl
        net_pnl = total_pnl - self.total_commission

        # Trade count
        trade_count = len(self.trades)

        # Return comprehensive metrics
        return {
            "total_value": total_value,
            "total_return_pct": total_return_pct,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "commission_paid": self.total_commission,
            "net_pnl": net_pnl,
            "trade_count": trade_count,
            "total_return": total_return,
            "annualized_return": 0.0,  # TODO: Calculate properly
            "sharpe_ratio": 0.0,  # TODO: Calculate properly
            "max_drawdown": self.calculate_max_drawdown(),
            "win_rate": 0.0,  # TODO: Calculate properly
            "profit_factor": 0.0,  # TODO: Calculate properly
        }

        # Old implementation for reference
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        # Calculate returns
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (
            self.equity_curve[-1] - self.initial_capital
        ) / self.initial_capital

        # Annualized return (simplified)
        days = len(self.equity_curve)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        max_drawdown = self.current_drawdown

        # Win rate from trades
        if self.trades:
            winning_trades = sum(
                1 for t in self.trades if self._calculate_trade_pnl(t) > 0
            )
            win_rate = winning_trades / len(self.trades)
        else:
            win_rate = 0.0

        # Profit factor
        gross_profit = sum(
            self._calculate_trade_pnl(t)
            for t in self.trades
            if self._calculate_trade_pnl(t) > 0
        )
        gross_loss = abs(
            sum(
                self._calculate_trade_pnl(t)
                for t in self.trades
                if self._calculate_trade_pnl(t) < 0
            )
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def _calculate_trade_pnl(self, trade: "Trade") -> float:
        """Calculate P&L for a trade (helper method)."""
        # Simplified - would need more context in production
        return 0.0  # Placeholder

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float = 1.0,
        volatility: float = 0.02,
        win_rate: float = None,
        avg_win_loss_ratio: float = None,
        risk_per_trade: float = 0.01,
        requested_allocation: float = None,
        **kwargs,
    ) -> int:
        """Calculate position size based on method (backward compatibility)."""
        # Use total portfolio value for position sizing, not just cash
        available_capital = self.get_portfolio_value()

        if self.position_size_method == "equal_weight":
            position_value = available_capital / self.max_positions
        elif self.position_size_method == "kelly":
            # Kelly sizing with provided or calculated parameters
            if win_rate is not None and avg_win_loss_ratio is not None:
                # Kelly formula: f = p - q/b
                # where p = win_rate, q = 1-p, b = avg_win_loss_ratio
                if avg_win_loss_ratio > 0:
                    kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
                    # Apply safety factor (half-Kelly)
                    kelly_fraction = max(0, min(0.25, kelly_fraction * 0.5))
                else:
                    kelly_fraction = 0.1
            else:
                # Use default or calculate from trade history
                kelly_fraction = self.strategy_kelly_fractions.get(
                    symbol, self.strategy_kelly_fractions.get("default", 0.25)
                )
            position_value = available_capital * kelly_fraction * abs(signal_strength)
        elif self.position_size_method == "volatility":
            # Volatility-based sizing
            risk_amount = available_capital * risk_per_trade
            # Position value = risk_amount / volatility, but we need shares
            # So shares = risk_amount / (price * volatility)
            shares = (
                int(risk_amount / (price * volatility))
                if price > 0 and volatility > 0
                else 0
            )
            return shares  # Return early since we already calculated shares
        else:
            position_value = available_capital / self.max_positions

        # Apply requested allocation if provided
        if requested_allocation is not None:
            requested_value = available_capital * requested_allocation
            position_value = min(position_value, requested_value)

        # Apply position limit if explicitly set (not for equal weight method)
        if self.position_size_method != "equal_weight":
            if (
                hasattr(self, "max_position_size")
                and self.max_position_size is not None
            ):
                max_position_value = available_capital * self.max_position_size
                position_value = min(position_value, max_position_value)
            elif "max_position_size" in self.risk_limits:
                max_position_value = (
                    available_capital * self.risk_limits["max_position_size"]
                )
                position_value = min(position_value, max_position_value)

        # Calculate shares
        shares = int(position_value / price) if price > 0 else 0

        return shares

    def check_margin_requirements(self, required_margin: float) -> bool:
        """Check if margin requirements are met (backward compatibility)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        used_margin = positions_value
        available_margin = self.current_equity * (1 - self.risk_limits["margin_buffer"])

        return used_margin + required_margin <= available_margin

    def get_state_snapshot(self) -> dict:
        """Get portfolio state snapshot (backward compatibility)."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        return {
            "timestamp": datetime.now().isoformat(),
            "cash": self.cash,
            "total_value": self.get_portfolio_value(),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            "metrics": self.calculate_portfolio_metrics(),
            "trades_count": len(self.trades),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
        }

    def update_position_price(self, symbol: str, price: float) -> None:
        """Update position price (backward compatibility)."""
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    def calculate_total_pnl(self) -> float:
        """Calculate total P&L (backward compatibility)."""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized = self.realized_pnl
        return unrealized + realized

    def add_position_method(self, symbol: str, quantity: float, price: float) -> None:
        """Add position (backward compatibility alias)."""
        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            side="BUY",
            timestamp=datetime.now(),
        )
        self.add_position(trade)

    def can_add_position(self, symbol: str) -> bool:
        """Check if we can add a new position for the symbol."""
        # Can always add to existing position
        if symbol in self.positions:
            return True

        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            return False

        return True

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        # Extract numeric values from equity curve (handle both dict and numeric formats)
        numeric_curve = []
        for item in self.equity_curve:
            if isinstance(item, dict):
                numeric_curve.append(float(item.get("value", 0)))
            elif isinstance(item, (int, float)):
                numeric_curve.append(float(item))

        if len(numeric_curve) < 2:
            return 0.0

        equity_series = pd.Series(numeric_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        return float(drawdown.min())  # Return negative value

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        if len(returns) < 1 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return float(excess_returns.mean() / returns.std() * np.sqrt(252))

    def calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        if len(returns) < 1:
            return 0.0

        # Calculate downside deviation
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return float("inf")  # No downside

        downside_std = downside_returns.std()

        if downside_std == 0:
            return float("inf")

        return float((returns.mean() - target_return) / downside_std * np.sqrt(252))

    def calculate_margin_requirements(self) -> float:
        """Calculate total margin requirements for all positions."""
        margin_req = 0.0

        for pos in self.positions.values():
            if pos.direction == "SHORT":
                # Short positions require margin (typically 30-50%)
                margin_req += pos.market_value * 0.3
            # Long positions may also require margin in leveraged accounts
            # but we'll assume cash account for now

        return margin_req

    def get_performance_attribution(self) -> dict[str, dict[str, float]]:
        """Get performance attribution by symbol."""
        attribution = {}

        for trade in self.trades:
            symbol = trade.symbol
            pnl = getattr(trade, "pnl", 0.0)

            if symbol not in attribution:
                attribution[symbol] = {"total_pnl": 0.0, "trade_count": 0}

            attribution[symbol]["total_pnl"] += pnl
            attribution[symbol]["trade_count"] += 1

        return attribution

    def get_positions_summary(self) -> list[dict[str, Any]]:
        """Get summary of all positions (backward compatibility)."""
        summary = []
        for symbol, pos in self.positions.items():
            summary.append(
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_pct": pos.pnl_percentage,
                    "position_type": pos.position_type or pos.direction,
                }
            )
        return summary

    def update_equity_curve(self) -> None:
        """Update equity curve with current portfolio value (backward compatibility)."""
        self._update_equity()

    def calculate_drawdown(self) -> dict[str, float]:
        """Calculate current and maximum drawdown (backward compatibility)."""
        max_dd_pct = self.calculate_max_drawdown()

        # Calculate the actual peak from equity curve for max drawdown value
        if self.equity_curve:
            numeric_curve = []
            for item in self.equity_curve:
                if isinstance(item, dict):
                    numeric_curve.append(float(item.get("value", 0)))
                elif isinstance(item, (int, float)):
                    numeric_curve.append(float(item))

            if numeric_curve:
                peak = max(numeric_curve)
                max_dd_value = peak * abs(max_dd_pct) if max_dd_pct < 0 else 0
            else:
                max_dd_value = (
                    self.peak_equity * abs(max_dd_pct) if max_dd_pct < 0 else 0
                )
        else:
            max_dd_value = self.peak_equity * abs(max_dd_pct) if max_dd_pct < 0 else 0

        return {
            "current_drawdown": self.current_drawdown,
            "current_drawdown_pct": self.current_drawdown * 100,
            "max_drawdown": max_dd_pct,
            "max_drawdown_pct": max_dd_pct * 100,
            "max_drawdown_value": -max_dd_value if max_dd_value > 0 else 0,
            "peak_equity": self.peak_equity,
        }

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Get trade history (backward compatibility)."""
        trades = self.trades.copy()

        # Filter by symbol if provided
        if symbol is not None:
            trades = [trade for trade in trades if trade.get("symbol") == symbol]

        # Filter by date range if provided
        if start_date is not None or end_date is not None:
            filtered_trades = []
            for trade in trades:
                trade_time = trade.get("timestamp")
                if isinstance(trade_time, str):
                    trade_time = datetime.fromisoformat(trade_time)

                # Convert dates to datetime for comparison if needed
                if start_date:
                    start_dt = (
                        start_date
                        if isinstance(start_date, datetime)
                        else datetime.combine(start_date, datetime.min.time())
                    )
                    if trade_time < start_dt:
                        continue
                if end_date:
                    end_dt = (
                        end_date
                        if isinstance(end_date, datetime)
                        else datetime.combine(
                            end_date, datetime.max.time().replace(microsecond=0)
                        )
                    )
                    if trade_time > end_dt:
                        continue

                filtered_trades.append(trade)
            trades = filtered_trades

        return trades

    def export_state(self) -> dict[str, Any]:
        """Export current portfolio state for persistence."""
        positions_data = {}
        for symbol, position in self.positions.items():
            positions_data[symbol] = {
                "symbol": position.symbol,
                "strategy_id": position.strategy_id,
                "direction": position.direction,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time.isoformat()
                if hasattr(position.entry_time, "isoformat")
                else str(position.entry_time),
                "current_price": position.current_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "metadata": position.metadata,
            }

        return {
            "cash": self._cash,
            "initial_capital": self.initial_capital,
            "positions": positions_data,
            "trades": self.trades[-100:],  # Last 100 trades
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "strategy_allocations": self.strategy_allocations,
            "strategy_performance": self.strategy_performance,
            "performance_metrics": self.get_performance_metrics(),
            "timestamp": datetime.now().isoformat(),
        }


# Backward compatibility classes and aliases
@dataclass
class Trade:
    """Trade record for backward compatibility."""

    symbol: str
    quantity: float
    price: float
    side: str
    timestamp: datetime
    commission: float = 0.0
    pnl: float = 0.0  # Add pnl parameter for backward compatibility
    slippage: float = 0.0  # Add slippage for transaction costs

    def __getitem__(self, key):
        """Dict-like access for backward compatibility."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Dict-like get for backward compatibility."""
        return getattr(self, key, default)


@dataclass
class PositionInfo:
    """Position information for backward compatibility."""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    pnl_pct: float


# Aliases for backward compatibility
Portfolio = PortfolioEngine
PortfolioMetrics = dict  # Placeholder type alias
PortfolioState = dict  # Placeholder type alias
