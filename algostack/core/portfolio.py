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
    strategy_id: str
    direction: str  # LONG or SHORT
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

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


class PortfolioEngine:
    """Manages portfolio allocation, risk, and position sizing across strategies."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.initial_capital = config.get("initial_capital", DEFAULT_INITIAL_CAPITAL)
        self.current_equity = self.initial_capital
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.strategy_allocations: dict[str, float] = {}  # strategy -> allocation
        self.performance_history: list[dict[str, Any]] = []
        self.correlation_matrix = pd.DataFrame()
        self.volatility_targets = config.get("volatility_targets", {})
        self.risk_limits = {
            "max_portfolio_volatility": config.get("target_vol", 0.10),
            "max_position_size": config.get("max_position_size", 0.20),
            "max_sector_exposure": config.get("max_sector_exposure", 0.40),
            "max_drawdown": config.get("max_drawdown", 0.15),
            "max_correlation": config.get("max_correlation", 0.70),
            "margin_buffer": config.get("margin_buffer", 0.25),
        }

        # Risk tracking
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl: list[float] = []
        self.is_risk_off = False
        self.risk_off_until: Optional[datetime] = None

        # Kelly tracking
        self.strategy_kelly_fractions: dict[str, float] = {}
        self.strategy_performance: dict[str, dict[str, float]] = {}  # Track per-strategy metrics

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

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
        matrix_symbols = set(self.correlation_matrix.index) & set(self.correlation_matrix.columns)
        relevant_symbols = list(position_symbols & matrix_symbols)

        if len(relevant_symbols) < 2:
            return violations

        # Use numpy for efficient correlation checking
        max_corr = self.risk_limits["max_correlation"]

        try:
            # Get correlation submatrix for relevant symbols only
            corr_subset = self.correlation_matrix.loc[relevant_symbols, relevant_symbols]

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

    def close_position(
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

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value including cash and positions."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_equity + positions_value

    @property
    def cash(self) -> float:
        """Available cash in the portfolio."""
        return self.current_equity

    def update_position(self, symbol: str, quantity: float, avg_price: float, current_price: float) -> None:
        """Update or create a position with current market data."""
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            position.quantity = quantity
            position.current_price = current_price
            # Update entry price if quantity changed significantly
            if abs(position.quantity - quantity) > 0.01:
                position.entry_price = avg_price
        else:
            # Create new position
            if quantity != 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    strategy_id="unknown",
                    direction="LONG" if quantity > 0 else "SHORT",
                    quantity=quantity,
                    entry_price=avg_price,
                    entry_time=datetime.now(),
                    current_price=current_price,
                )

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate and return portfolio performance metrics."""
        return self.calculate_portfolio_metrics()

    def calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        # Simple implementation - in production this would track intraday changes
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return total_pnl


# Aliases for backward compatibility
Portfolio = PortfolioEngine
PortfolioMetrics = dict  # Placeholder type alias
PortfolioState = dict  # Placeholder type alias
Trade = dict  # Placeholder type alias
