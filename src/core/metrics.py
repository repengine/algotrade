"""
Performance Metrics Collection and Calculation.

This module provides comprehensive metrics tracking for the trading system.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """
    Represents a single trade record.

    Attributes:
        timestamp: When the trade was executed
        symbol: Trading symbol (e.g., 'AAPL')
        side: Trade side ('BUY' or 'SELL')
        quantity: Number of shares/units
        price: Execution price
        commission: Trading commission/fees
        entry_time: Optional entry time for completed trades
        exit_time: Optional exit time for completed trades
        entry_price: Optional entry price for completed trades
        exit_price: Optional exit price for completed trades
        pnl: Optional profit/loss for completed trades
        pnl_percentage: Optional P&L percentage for completed trades
        strategy_id: Optional strategy identifier

    Example:
        trade = Trade(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            commission=1.0
        )
    """
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
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
    portfolio_value: Optional[float] = None
    positions_held: Optional[int] = None
    total_trades: Optional[int] = None


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

        # Invalidate cache
        self._metrics_cache = {}
        self._cache_timestamp = None

    def record_trade_exit(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        quantity: Optional[int] = None,
    ) -> Optional[Trade]:
        """Record trade exit and calculate P&L."""
        if symbol not in self.open_trades:
            logger.warning(f"No open trade found for {symbol}")
            raise KeyError(f"No open trade found for {symbol}")

        open_trade = self.open_trades[symbol]

        # If quantity not specified, exit full position
        if quantity is None:
            quantity = open_trade["quantity"]

        # Handle partial exits
        if quantity < open_trade["quantity"]:
            # Partial exit - adjust open trade
            trade_quantity = quantity
            open_trade["quantity"] -= quantity
        else:
            # Full exit
            trade_quantity = open_trade["quantity"]
            del self.open_trades[symbol]

        # Calculate P&L - handle both "BUY"/"SELL" and "long"/"short" formats
        side = open_trade["side"].lower()
        if side in ["buy", "long"]:
            pnl = (price - open_trade["entry_price"]) * trade_quantity - commission
        else:  # sell/short
            pnl = (open_trade["entry_price"] - price) * trade_quantity - commission

        pnl_percentage = (pnl / (open_trade["entry_price"] * trade_quantity)) * 100

        # Create trade record
        trade = Trade(
            timestamp=timestamp,  # Exit timestamp
            symbol=symbol,
            side=open_trade["side"],
            quantity=trade_quantity,
            price=price,  # Exit price
            commission=commission,
            entry_time=open_trade["entry_time"],
            exit_time=timestamp,
            entry_price=open_trade["entry_price"],
            exit_price=price,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
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

    def record_daily_metrics(self, date: Union[datetime, datetime.date]) -> DailyMetrics:
        """Calculate and record metrics for a specific day."""
        # Get trades for the day
        if hasattr(date, 'date'):
            target_date = date.date()
        else:
            target_date = date
        daily_trades = [t for t in self.trades if t.exit_time and t.exit_time.date() == target_date]

        # Calculate daily statistics
        winning_trades = [t for t in daily_trades if t.pnl is not None and t.pnl > 0]
        losing_trades = [t for t in daily_trades if t.pnl is not None and t.pnl <= 0]

        gross_profit = sum(t.pnl for t in winning_trades if t.pnl is not None)
        gross_loss = sum(t.pnl for t in losing_trades if t.pnl is not None)
        commission_paid = sum(t.commission for t in daily_trades if t.commission is not None)
        net_pnl = gross_profit + gross_loss

        # Get portfolio values for the day
        day_values = [
            (ts, val) for ts, val in self.value_history if ts.date() == target_date
        ]

        # Get previous day's ending value or initial capital
        all_previous_values = [
            (ts, val) for ts, val in self.value_history if ts.date() < target_date
        ]

        if all_previous_values:
            previous_day_value = all_previous_values[-1][1]
        else:
            previous_day_value = self.initial_capital

        if day_values:
            starting_value = previous_day_value  # Start of day is previous day's end
            ending_value = day_values[-1][1]
            high_water = max(val for _, val in day_values)
            low_water = min(val for _, val in day_values)
        else:
            # No data for the day
            starting_value = previous_day_value
            ending_value = self.current_value
            high_water = low_water = self.current_value

        daily_return = (ending_value - starting_value) / starting_value if starting_value != 0 else 0.0

        metrics = DailyMetrics(
            date=target_date,
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
            portfolio_value=ending_value,
            positions_held=len(self.open_trades),
            total_trades=len(daily_trades),
        )

        self.daily_metrics[target_date] = metrics
        return metrics

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        # Check cache
        if self._metrics_cache and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < timedelta(seconds=60):
                return self._metrics_cache

        metrics: dict[str, Any] = {}

        # Basic metrics
        metrics["total_trades"] = float(len(self.trades))
        metrics["open_trades"] = float(len(self.open_trades))

        # Total return calculation
        total_return = (self.current_value - self.initial_capital) / self.initial_capital
        metrics["total_return"] = total_return

        # Calculate risk-adjusted returns even without trades if we have value history
        metrics["sharpe_ratio"] = self._calculate_sharpe_ratio()
        metrics["sortino_ratio"] = self._calculate_sortino_ratio()
        metrics["calmar_ratio"] = self._calculate_calmar_ratio()
        metrics["max_drawdown"] = self.max_drawdown
        metrics["current_drawdown"] = self.current_drawdown

        if not self.trades:
            # No trades yet - set trade-related metrics to zero
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
                    "strategy_performance": {},
                }
            )
            # Cache results even with no trades
            self._metrics_cache = metrics
            self._cache_timestamp = datetime.now()
            return metrics

        # Win/Loss statistics
        winning_trades = [t for t in self.trades if t.pnl is not None and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl is not None and t.pnl <= 0]

        metrics["winning_trades"] = len(winning_trades)
        metrics["losing_trades"] = len(losing_trades)
        metrics["win_rate"] = (
            len(winning_trades) / len(self.trades) if self.trades else 0
        )

        # P&L statistics
        if winning_trades:
            wins = [t.pnl for t in winning_trades if t.pnl is not None]
            metrics["average_win"] = float(np.mean(wins)) if wins else 0.0
            metrics["largest_win"] = max(t.pnl for t in winning_trades)
        else:
            metrics["average_win"] = 0.0
            metrics["largest_win"] = 0.0

        if losing_trades:
            losses = [t.pnl for t in losing_trades if t.pnl is not None]
            metrics["average_loss"] = float(np.mean(losses)) if losses else 0.0
            metrics["largest_loss"] = min(t.pnl for t in losing_trades)
        else:
            metrics["average_loss"] = 0.0
            metrics["largest_loss"] = 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades if t.pnl is not None)
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl is not None))
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Overall statistics
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        metrics["average_trade_pnl"] = float(np.mean(pnls)) if pnls else 0.0
        metrics["total_pnl"] = sum(t.pnl for t in self.trades if t.pnl is not None)
        metrics["total_commission"] = sum(t.commission for t in self.trades if t.commission is not None)

        # Strategy breakdown
        metrics["strategy_performance"] = self._calculate_strategy_performance()

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
        if self.value_history and len(self.value_history) > 1:
            first_time = self.value_history[0][0]
            last_time = self.value_history[-1][0]
            trading_days = (last_time - first_time).days

            if trading_days > 0:
                annualized_return = total_return * (365 / trading_days)
            else:
                annualized_return = total_return
        elif self.trades:
            first_trade = min(t.entry_time for t in self.trades if t.entry_time is not None)
            last_trade = max(t.exit_time for t in self.trades if t.exit_time is not None)
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
            winning = [t for t in trades if t.pnl is not None and t.pnl > 0]

            strategy_metrics[strategy_id] = {
                "total_trades": float(len(trades)),
                "win_rate": float(len(winning)) / float(len(trades)) if trades else 0.0,
                "total_pnl": float(sum(t.pnl for t in trades if t.pnl is not None)),
                "average_pnl": float(np.mean([t.pnl for t in trades if t.pnl is not None])) if trades else 0.0,
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


# Import BacktestMetrics for convenience
try:
    from .backtest_metrics import BacktestMetrics
except ImportError:
    # Fallback alias for backward compatibility
    BacktestMetrics = MetricsCollector  # type: ignore
