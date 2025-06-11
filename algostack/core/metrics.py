"""
Performance Metrics Collection and Calculation.

This module provides comprehensive metrics tracking for the trading system.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Completed trade record."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # "long" or "short"
    pnl: float
    pnl_percentage: float
    commission: float
    strategy_id: Optional[str] = None


@dataclass
class DailyMetrics:
    """Daily performance metrics."""

    date: datetime
    starting_value: float
    ending_value: float
    high_water_mark: float
    low_water_mark: float
    daily_return: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    gross_profit: float
    gross_loss: float
    commission_paid: float
    net_pnl: float


class MetricsCollector:
    """
    Collects and calculates performance metrics.

    Tracks trades, portfolio values, and calculates various
    performance indicators.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize metrics collector.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital

        # Trade history
        self.trades: list[Trade] = []
        self.open_trades: dict[str, dict] = {}  # symbol -> trade info

        # Portfolio value tracking
        self.value_history: deque[tuple[datetime, float]] = deque(maxlen=10000)
        self.daily_metrics: dict[datetime, DailyMetrics] = {}

        # Current state
        self.current_value = initial_capital
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Performance metrics cache
        self._metrics_cache: dict[str, float] = {}
        self._cache_timestamp: Optional[datetime] = None

    def record_trade_entry(
        self,
        symbol: str,
        price: float,
        quantity: int,
        side: str,
        timestamp: datetime,
        strategy_id: Optional[str] = None,
    ) -> None:
        """Record trade entry."""
        self.open_trades[symbol] = {
            "entry_time": timestamp,
            "entry_price": price,
            "quantity": quantity,
            "side": side,
            "strategy_id": strategy_id,
        }

    def record_trade_exit(
        self,
        symbol: str,
        price: float,
        quantity: int,
        timestamp: datetime,
        commission: float = 0.0,
    ) -> Optional[Trade]:
        """Record trade exit and calculate P&L."""
        if symbol not in self.open_trades:
            logger.warning(f"No open trade found for {symbol}")
            return None

        open_trade = self.open_trades[symbol]

        # Handle partial exits
        if quantity < open_trade["quantity"]:
            # Partial exit - adjust open trade
            trade_quantity = quantity
            open_trade["quantity"] -= quantity
        else:
            # Full exit
            trade_quantity = open_trade["quantity"]
            del self.open_trades[symbol]

        # Calculate P&L
        if open_trade["side"] == "long":
            pnl = (price - open_trade["entry_price"]) * trade_quantity - commission
        else:  # short
            pnl = (open_trade["entry_price"] - price) * trade_quantity - commission

        pnl_percentage = (pnl / (open_trade["entry_price"] * trade_quantity)) * 100

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=open_trade["entry_time"],
            exit_time=timestamp,
            entry_price=open_trade["entry_price"],
            exit_price=price,
            quantity=trade_quantity,
            side=open_trade["side"],
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            commission=commission,
            strategy_id=open_trade.get("strategy_id"),
        )

        self.trades.append(trade)

        # Invalidate cache
        self._metrics_cache = {}

        return trade

    def update_portfolio_value(
        self, value: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Update current portfolio value."""
        if timestamp is None:
            timestamp = datetime.now()

        self.current_value = value
        self.value_history.append((timestamp, value))

        # Update high water mark
        if value > self.high_water_mark:
            self.high_water_mark = value

        # Update drawdown
        self.current_drawdown = (self.high_water_mark - value) / self.high_water_mark
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def record_daily_metrics(self, date: datetime) -> DailyMetrics:
        """Calculate and record metrics for a specific day."""
        # Get trades for the day
        daily_trades = [t for t in self.trades if t.exit_time.date() == date.date()]

        # Calculate daily statistics
        winning_trades = [t for t in daily_trades if t.pnl > 0]
        losing_trades = [t for t in daily_trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = sum(t.pnl for t in losing_trades)
        commission_paid = sum(t.commission for t in daily_trades)
        net_pnl = gross_profit + gross_loss

        # Get portfolio values for the day
        day_values = [
            (ts, val) for ts, val in self.value_history if ts.date() == date.date()
        ]

        if day_values:
            starting_value = day_values[0][1]
            ending_value = day_values[-1][1]
            high_water = max(val for _, val in day_values)
            low_water = min(val for _, val in day_values)
        else:
            # No data for the day
            starting_value = ending_value = self.current_value
            high_water = low_water = self.current_value

        daily_return = ((ending_value - starting_value) / starting_value) * 100

        metrics = DailyMetrics(
            date=date,
            starting_value=starting_value,
            ending_value=ending_value,
            high_water_mark=high_water,
            low_water_mark=low_water,
            daily_return=daily_return,
            trades_count=len(daily_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            commission_paid=commission_paid,
            net_pnl=net_pnl,
        )

        self.daily_metrics[date] = metrics
        return metrics

    def get_performance_metrics(self) -> dict[str, float]:
        """Calculate comprehensive performance metrics."""
        # Check cache
        if self._metrics_cache and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < timedelta(seconds=60):
                return self._metrics_cache

        metrics: dict[str, float] = {}

        # Basic metrics
        metrics["total_trades"] = float(len(self.trades))
        metrics["open_trades"] = float(len(self.open_trades))

        if not self.trades:
            # No trades yet
            metrics.update(
                {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "average_win": 0.0,
                    "average_loss": 0.0,
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                    "average_trade_pnl": 0.0,
                    "total_pnl": 0.0,
                    "total_commission": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "max_drawdown": self.max_drawdown,
                    "current_drawdown": self.current_drawdown,
                }
            )
            return metrics

        # Win/Loss statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        metrics["winning_trades"] = len(winning_trades)
        metrics["losing_trades"] = len(losing_trades)
        metrics["win_rate"] = (
            len(winning_trades) / len(self.trades) if self.trades else 0
        )

        # P&L statistics
        if winning_trades:
            metrics["average_win"] = float(np.mean([t.pnl for t in winning_trades]))
            metrics["largest_win"] = max(t.pnl for t in winning_trades)
        else:
            metrics["average_win"] = 0.0
            metrics["largest_win"] = 0.0

        if losing_trades:
            metrics["average_loss"] = float(np.mean([t.pnl for t in losing_trades]))
            metrics["largest_loss"] = min(t.pnl for t in losing_trades)
        else:
            metrics["average_loss"] = 0.0
            metrics["largest_loss"] = 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Overall statistics
        metrics["average_trade_pnl"] = float(np.mean([t.pnl for t in self.trades]))
        metrics["total_pnl"] = sum(t.pnl for t in self.trades)
        metrics["total_commission"] = sum(t.commission for t in self.trades)

        # Risk-adjusted returns
        metrics["sharpe_ratio"] = self._calculate_sharpe_ratio()
        metrics["sortino_ratio"] = self._calculate_sortino_ratio()
        metrics["calmar_ratio"] = self._calculate_calmar_ratio()

        # Drawdown
        metrics["max_drawdown"] = self.max_drawdown
        metrics["current_drawdown"] = self.current_drawdown

        # Strategy breakdown - don't include in metrics dict since it's not a float
        # self._calculate_strategy_performance() returns dict[str, dict[str, float]]

        # Cache results
        self._metrics_cache = metrics
        self._cache_timestamp = datetime.now()

        return metrics

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.value_history) < 2:
            return 0.0

        # Calculate returns
        values = [val for _, val in self.value_history]
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) < 2:
            return 0.0

        # Annualized Sharpe ratio
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

        if excess_returns.std() == 0:
            return 0.0

        return float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())

    def _calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(self.value_history) < 2:
            return 0.0

        # Calculate returns
        values = [val for _, val in self.value_history]
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) < 2:
            return 0.0

        # Downside returns only
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return float(np.sqrt(252) * (returns.mean() - target_return) / downside_returns.std())

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if self.max_drawdown == 0:
            return 0.0

        # Annualized return
        total_return = (
            self.current_value - self.initial_capital
        ) / self.initial_capital

        # Estimate annualized return based on trading period
        if self.trades:
            first_trade = min(t.entry_time for t in self.trades)
            last_trade = max(t.exit_time for t in self.trades)
            trading_days = (last_trade - first_trade).days

            if trading_days > 0:
                annualized_return = total_return * (365 / trading_days)
            else:
                annualized_return = total_return
        else:
            annualized_return = total_return

        return annualized_return / self.max_drawdown

    def _calculate_strategy_performance(self) -> dict[str, dict[str, float]]:
        """Calculate performance breakdown by strategy."""
        strategy_metrics = {}

        # Group trades by strategy
        from collections import defaultdict

        strategy_trades = defaultdict(list)

        for trade in self.trades:
            strategy_id = trade.strategy_id or "unknown"
            strategy_trades[strategy_id].append(trade)

        # Calculate metrics for each strategy
        for strategy_id, trades in strategy_trades.items():
            winning = [t for t in trades if t.pnl > 0]

            strategy_metrics[strategy_id] = {
                "trades": float(len(trades)),
                "win_rate": float(len(winning)) / float(len(trades)) if trades else 0.0,
                "total_pnl": float(sum(t.pnl for t in trades)),
                "average_pnl": float(np.mean([t.pnl for t in trades])),
            }

        return strategy_metrics

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.value_history:
            return pd.DataFrame()

        timestamps, values = zip(*self.value_history)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": values,
            }
        )

        df["returns"] = df["value"].pct_change()
        df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1
        df["drawdown"] = df["value"] / df["value"].cummax() - 1

        return df


# Alias for backward compatibility
BacktestMetrics = MetricsCollector
