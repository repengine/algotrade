"""
Backtest Metrics Module.

Provides comprehensive metrics calculation for backtesting results.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .metrics import Trade

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """
    Comprehensive metrics calculator for backtest results.

    This class provides detailed performance metrics including returns,
    risk metrics, and trade statistics for backtesting analysis.

    Attributes:
        initial_capital: Starting capital amount
        risk_free_rate: Risk-free rate for Sharpe/Sortino calculations
        periods_per_year: Trading periods per year (252 for daily, 365 for calendar)
        trades: List of completed trades
        equity_curve: List of portfolio values over time

    Example:
        metrics = BacktestMetrics(initial_capital=100000)
        metrics.add_trade(trade)
        metrics.update_equity(101000)

        print(f"Total Return: {metrics.total_return():.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio():.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown():.2f}%")
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> None:
        """
        Initialize BacktestMetrics.

        Args:
            initial_capital: Starting capital (default: 100000)
            risk_free_rate: Annual risk-free rate (default: 0.02)
            periods_per_year: Trading periods per year (default: 252)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        self.trades: List[Trade] = []
        self._equity_curve = [initial_capital]  # Internal storage
        self.timestamps: List[datetime] = [datetime.now()]
        self._equity_index = None  # Store index if equity_curve is set as Series

        self._metrics_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None

    def add_trade(self, trade: Trade) -> None:
        """
        Add a completed trade.

        Args:
            trade: Trade object to add

        Example:
            trade = Trade(
                timestamp=datetime.now(),
                symbol='AAPL',
                side='BUY',
                quantity=100,
                price=150.0,
                commission=1.0
            )
            metrics.add_trade(trade)
        """
        self.trades.append(trade)
        self._invalidate_cache()

    def update_equity(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update equity curve with new portfolio value.

        Args:
            value: Current portfolio value
            timestamp: Timestamp for the update (default: now)

        Example:
            metrics.update_equity(105000)  # Portfolio worth $105,000
        """
        self._equity_curve.append(value)
        self.timestamps.append(timestamp or datetime.now())
        self._invalidate_cache()

    def calculate_returns(self) -> pd.Series:
        """
        Calculate returns series from equity curve.

        Returns:
            pd.Series: Returns indexed by timestamp

        Example:
            returns = metrics.calculate_returns()
            print(f"Mean return: {returns.mean():.4f}")
        """
        if isinstance(self.equity_curve, pd.Series):
            # If it's already a Series, use it directly
            return self.equity_curve.pct_change().dropna()

        # Convert to numpy array to get length
        equity_array = np.array(self.equity_curve)

        if len(equity_array) < 2:
            return pd.Series(dtype=float)

        # Check if we have a saved index
        if self._equity_index is not None:
            equity_series = pd.Series(equity_array, index=self._equity_index)
        elif len(self.timestamps) >= len(equity_array):
            # Use timestamps if available
            equity_series = pd.Series(
                equity_array,
                index=pd.DatetimeIndex(self.timestamps[:len(equity_array)])
            )
        else:
            # Create default index
            equity_series = pd.Series(equity_array)

        # Calculate returns - pct_change naturally shifts the index
        returns = equity_series.pct_change()
        # Remove the first NaN value but keep the index
        return returns.dropna()

    @property
    def equity_curve(self) -> List[float]:
        """Get equity curve."""
        return self._equity_curve

    @equity_curve.setter
    def equity_curve(self, value: Union[List[float], pd.Series]) -> None:
        """Set equity curve, preserving index if it's a Series."""
        if isinstance(value, pd.Series):
            self._equity_index = value.index
            self._equity_curve = value.values.tolist()
        else:
            self._equity_curve = value
            self._equity_index = None

    def total_return(self) -> float:
        """
        Calculate total return percentage.

        Returns:
            float: Total return as percentage

        Example:
            total = metrics.total_return()
            print(f"Total return: {total:.2f}%")
        """
        # Use caching mechanism
        return self._calculate_metric('total_return')

    def _calculate_total_return(self) -> float:
        """Internal method to calculate total return."""
        if not self.equity_curve:
            return 0.0

        final_value = self.equity_curve[-1]
        return float(((final_value - self.initial_capital) / self.initial_capital) * 100)

    def annual_return(self, periods: Optional[int] = None) -> float:
        """
        Calculate annualized return.

        Args:
            periods: Number of periods (default: from equity curve)

        Returns:
            float: Annualized return percentage

        Example:
            annual = metrics.annual_return()
            print(f"Annual return: {annual:.2f}%")
        """
        total = self.total_return() / 100  # Convert to decimal

        if periods is None:
            if len(self.timestamps) < 2:
                return total * 100

            time_diff = self.timestamps[-1] - self.timestamps[0]
            years = time_diff.days / 365.25
        else:
            years = periods / self.periods_per_year

        if years <= 0:
            return total * 100

        # Compound annual growth rate
        annual_return = (np.power(1 + total, 1 / years) - 1) * 100
        return float(annual_return)

    def sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            risk_free_rate: Override default risk-free rate

        Returns:
            float: Sharpe ratio

        Example:
            sharpe = metrics.sharpe_ratio()
            print(f"Sharpe ratio: {sharpe:.2f}")
        """
        returns = self.calculate_returns()

        if len(returns) < 2:
            return 0.0

        rf_rate = risk_free_rate or self.risk_free_rate
        rf_daily = rf_rate / self.periods_per_year

        excess_returns = returns - rf_daily

        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(self.periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return float(sharpe)

    def sortino_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation).

        Args:
            risk_free_rate: Override default risk-free rate

        Returns:
            float: Sortino ratio

        Example:
            sortino = metrics.sortino_ratio()
            print(f"Sortino ratio: {sortino:.2f}")
        """
        returns = self.calculate_returns()

        if len(returns) < 2:
            return 0.0

        rf_rate = risk_free_rate or self.risk_free_rate
        rf_daily = rf_rate / self.periods_per_year

        excess_returns = returns - rf_daily
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_std = np.sqrt((downside_returns ** 2).mean())
        sortino = np.sqrt(self.periods_per_year) * (excess_returns.mean() / downside_std)

        return float(sortino)

    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown percentage.

        Returns:
            float: Maximum drawdown as negative percentage

        Example:
            mdd = metrics.max_drawdown()
            print(f"Max drawdown: {mdd:.2f}%")
        """
        if len(self.equity_curve) < 2:
            return 0.0

        equity_array = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax * 100

        return float(np.min(drawdowns))

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Returns:
            float: Calmar ratio

        Example:
            calmar = metrics.calmar_ratio()
            print(f"Calmar ratio: {calmar:.2f}")
        """
        annual_ret = self.annual_return()
        max_dd = abs(self.max_drawdown())

        if max_dd == 0:
            return float('inf') if annual_ret > 0 else 0.0

        return annual_ret / max_dd

    def win_rate(self) -> float:
        """
        Calculate win rate from completed trades.

        Returns:
            float: Win rate (0 to 1)

        Example:
            wr = metrics.win_rate()
            print(f"Win rate: {wr:.2%}")
        """
        if not self.trades:
            return 0.0

        completed_trades = [t for t in self.trades if t.pnl is not None]

        if not completed_trades:
            return 0.0

        winning_trades = [t for t in completed_trades if t.pnl > 0]
        return len(winning_trades) / len(completed_trades)

    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            float: Profit factor

        Example:
            pf = metrics.profit_factor()
            print(f"Profit factor: {pf:.2f}")
        """
        if not self.trades:
            return 0.0

        completed_trades = [t for t in self.trades if t.pnl is not None]

        if not completed_trades:
            return 0.0

        gross_profit = sum(t.pnl for t in completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def average_win(self) -> float:
        """
        Calculate average winning trade.

        Returns:
            float: Average profit per winning trade

        Example:
            avg_win = metrics.average_win()
            print(f"Average win: ${avg_win:.2f}")
        """
        if not self.trades:
            return 0.0

        winning_trades = [t for t in self.trades if t.pnl is not None and t.pnl > 0]

        if not winning_trades:
            return 0.0

        return sum(t.pnl for t in winning_trades) / len(winning_trades)

    def average_loss(self) -> float:
        """
        Calculate average losing trade.

        Returns:
            float: Average loss per losing trade (positive number)

        Example:
            avg_loss = metrics.average_loss()
            print(f"Average loss: ${avg_loss:.2f}")
        """
        if not self.trades:
            return 0.0

        losing_trades = [t for t in self.trades if t.pnl is not None and t.pnl < 0]

        if not losing_trades:
            return 0.0

        return abs(sum(t.pnl for t in losing_trades) / len(losing_trades))

    def expectancy(self) -> float:
        """
        Calculate trade expectancy.

        Returns:
            float: Expected profit per trade

        Example:
            exp = metrics.expectancy()
            print(f"Expectancy: ${exp:.2f}")
        """
        win_rate_val = self.win_rate()
        avg_win_val = self.average_win()
        avg_loss_val = self.average_loss()

        return (win_rate_val * avg_win_val) - ((1 - win_rate_val) * avg_loss_val)

    def recovery_factor(self) -> float:
        """
        Calculate recovery factor (net profit / max drawdown).

        Returns:
            float: Recovery factor

        Example:
            rf = metrics.recovery_factor()
            print(f"Recovery factor: {rf:.2f}")
        """
        net_profit = self.equity_curve[-1] - self.initial_capital if self.equity_curve else 0
        max_dd_value = abs(self.max_drawdown()) * self.initial_capital / 100

        if max_dd_value == 0:
            return float('inf') if net_profit > 0 else 0.0

        return net_profit / max_dd_value

    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index (volatility of drawdowns).

        Returns:
            float: Ulcer Index

        Example:
            ui = metrics.ulcer_index()
            print(f"Ulcer Index: {ui:.2f}")
        """
        if len(self.equity_curve) < 2:
            return 0.0

        equity_array = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_array)
        drawdown_pct = ((equity_array - cummax) / cummax) * 100

        ulcer = np.sqrt(np.mean(drawdown_pct ** 2))
        return float(ulcer)

    def total_trades(self) -> int:
        """
        Get total number of completed trades.

        Only counts trades that have been closed (i.e., have a PnL).

        Returns:
            int: Total completed trade count

        Example:
            count = metrics.total_trades()
            print(f"Total trades: {count}")
        """
        # Only count trades with PnL (completed trades)
        completed_trades = [t for t in self.trades if hasattr(t, 'pnl') and t.pnl is not None]
        return len(completed_trades)

    def average_trade_duration(self) -> float:
        """
        Calculate average trade duration in days.

        Returns:
            float: Average duration in days

        Example:
            duration = metrics.average_trade_duration()
            print(f"Average hold time: {duration:.1f} days")
        """
        if not self.trades:
            return 0.0

        durations = []
        for trade in self.trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                if trade.entry_time and trade.exit_time:
                    duration = (trade.exit_time - trade.entry_time).days
                    durations.append(duration)

        return float(np.mean(durations)) if durations else 0.0

    def drawdown_analysis(self) -> Dict[str, Any]:
        """
        Analyze drawdown periods and recovery.

        Returns:
            dict: Drawdown analysis including max DD, duration, recovery time

        Example:
            dd_info = metrics.drawdown_analysis()
            print(f"Max drawdown: {dd_info['max_drawdown']:.2f}%")
            print(f"Recovery days: {dd_info['recovery_days']}")
        """
        if len(self.equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_days': 0,
                'recovery_days': 0,
                'current_drawdown': 0.0,
                'drawdown_periods': []
            }

        equity_array = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax * 100

        # Find max drawdown index
        max_dd_idx = np.argmin(drawdowns)
        max_dd = drawdowns[max_dd_idx]

        # Find drawdown start (last peak before max DD)
        dd_start = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdowns[i] == 0:
                dd_start = i
                break

        # Find recovery point (next peak after max DD)
        recovery_idx = len(drawdowns) - 1
        for i in range(max_dd_idx, len(drawdowns)):
            if drawdowns[i] >= -0.01:  # Nearly recovered
                recovery_idx = i
                break

        recovery_time = recovery_idx - max_dd_idx if recovery_idx > max_dd_idx else None
        underwater_time = max_dd_idx - dd_start

        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_days': underwater_time,
            'recovery_days': recovery_time,
            'recovery_time': recovery_time,  # Alias for backward compatibility
            'underwater_time': underwater_time,  # Time spent in drawdown
            'current_drawdown': float(drawdowns[-1]),
            'drawdown_periods': self._find_drawdown_periods(drawdowns)
        }

    def _find_drawdown_periods(self, drawdowns: np.ndarray) -> List[Dict[str, Any]]:
        """Find all significant drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = 0

        for i, dd in enumerate(drawdowns):
            if dd < -1.0 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= -0.1 and in_drawdown:  # Recovery
                periods.append({
                    'start': start_idx,
                    'end': i,
                    'depth': float(np.min(drawdowns[start_idx:i+1])),
                    'duration': i - start_idx
                })
                in_drawdown = False

        return periods

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio vs benchmark.

        Args:
            benchmark_returns: Benchmark return series

        Returns:
            float: Information ratio

        Example:
            ir = metrics.information_ratio(spy_returns)
            print(f"Information Ratio: {ir:.2f}")
        """
        strategy_returns = self.calculate_returns()

        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align the series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) < 2:
            return 0.0

        excess_returns = aligned['strategy'] - aligned['benchmark']

        if excess_returns.std() == 0:
            return 0.0

        return float(np.sqrt(self.periods_per_year) * (excess_returns.mean() / excess_returns.std()))

    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get comprehensive metrics summary (alias for get_summary).

        Returns:
            dict: All calculated metrics
        """
        return self.get_summary()

    def _calculate_metric(self, metric_name: str) -> float:
        """
        Calculate a specific metric with caching.

        Args:
            metric_name: Name of metric to calculate

        Returns:
            float: Calculated metric value
        """
        # Check cache first
        if metric_name in self._metrics_cache:
            return float(self._metrics_cache[metric_name])

        # Calculate based on metric name
        metric_map = {
            'total_return': self._calculate_total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': lambda: self.win_rate() * 100,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'expectancy': self.expectancy,
            'total_trades': self.total_trades,
        }

        if metric_name in metric_map:
            result = metric_map[metric_name]()
            # total_trades returns int, others return float
            value = float(result) if metric_name != 'total_trades' else result
            self._metrics_cache[metric_name] = value
            self._cache_timestamp = datetime.now()
            return float(value)

        raise ValueError(f"Unknown metric: {metric_name}")

    def get_summary(self) -> Dict[str, float]:
        """
        Get comprehensive metrics summary.

        Returns:
            dict: All calculated metrics

        Example:
            summary = metrics.get_summary()
            for metric, value in summary.items():
                print(f"{metric}: {value:.2f}")
        """
        return {
            'total_return': self.total_return(),
            'annual_return': self.annual_return(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
            'win_rate': self.win_rate() * 100,
            'profit_factor': self.profit_factor(),
            'average_win': self.average_win(),
            'average_loss': self.average_loss(),
            'expectancy': self.expectancy(),
            'recovery_factor': self.recovery_factor(),
            'ulcer_index': self.ulcer_index(),
            'total_trades': len(self.trades),
            'final_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital
        }

    def _invalidate_cache(self) -> None:
        """Invalidate metrics cache."""
        self._metrics_cache = {}
        self._cache_timestamp = None

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        The Omega ratio is the probability weighted ratio of gains versus losses
        for a given threshold return level.

        Args:
            threshold: Minimum acceptable return threshold (default: 0)

        Returns:
            Omega ratio
        """
        returns = self.calculate_returns()
        if len(returns) == 0:
            return 0.0

        # Calculate gains and losses relative to threshold
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            return float('inf') if gains > 0 else 1.0

        return gains / losses

    def tail_ratio(self, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio.

        The tail ratio compares the size of the right tail (gains) to the left tail (losses)
        of the returns distribution.

        Args:
            percentile: Percentile for tail calculation (default: 5%)

        Returns:
            Tail ratio
        """
        returns = self.calculate_returns()
        if len(returns) < 20:  # Need sufficient data
            return 0.0

        # Calculate percentiles
        right_tail = np.percentile(returns, 100 * (1 - percentile))
        left_tail = np.percentile(returns, 100 * percentile)

        if abs(left_tail) < 1e-10:
            return float('inf') if right_tail > 0 else 0.0

        return abs(right_tail / left_tail)
