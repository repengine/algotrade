"""Comprehensive test suite for metrics module."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from core.backtest_metrics import BacktestMetrics
from core.metrics import Trade


class TestBacktestMetrics:
    """Test suite for BacktestMetrics class."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return [
            Trade(
                timestamp=datetime(2023, 1, 1),
                symbol='AAPL',
                side='BUY',
                quantity=100,
                price=150.0,
                commission=1.0
            ),
            Trade(
                timestamp=datetime(2023, 1, 5),
                symbol='AAPL',
                side='SELL',
                quantity=100,
                price=155.0,
                commission=1.0,
                pnl=498.0  # (155-150)*100 - 2 commission
            ),
            Trade(
                timestamp=datetime(2023, 1, 10),
                symbol='GOOGL',
                side='BUY',
                quantity=10,
                price=2800.0,
                commission=2.0
            ),
            Trade(
                timestamp=datetime(2023, 1, 15),
                symbol='GOOGL',
                side='SELL',
                quantity=10,
                price=2750.0,
                commission=2.0,
                pnl=-504.0  # (2750-2800)*10 - 4 commission
            ),
            Trade(
                timestamp=datetime(2023, 1, 20),
                symbol='MSFT',
                side='BUY',
                quantity=50,
                price=300.0,
                commission=1.5
            ),
            Trade(
                timestamp=datetime(2023, 1, 25),
                symbol='MSFT',
                side='SELL',
                quantity=50,
                price=310.0,
                commission=1.5,
                pnl=497.0  # (310-300)*50 - 3 commission
            )
        ]

    @pytest.fixture
    def equity_curve(self):
        """Create sample equity curve."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        # Start at 100k, add some noise
        values = 100000 + np.cumsum(np.random.randn(len(dates)) * 500)
        return pd.Series(values, index=dates)

    @pytest.fixture
    def metrics(self):
        """Create BacktestMetrics instance."""
        return BacktestMetrics(initial_capital=100000)

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = BacktestMetrics(
            initial_capital=50000,
            risk_free_rate=0.03,
            periods_per_year=365
        )

        assert metrics.initial_capital == 50000
        assert metrics.risk_free_rate == 0.03
        assert metrics.periods_per_year == 365
        assert metrics._metrics_cache == {}
        assert metrics._cache_timestamp is None

    def test_add_trade(self, metrics):
        """Test adding trades."""
        trade = Trade(
            timestamp=datetime.now(),
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            commission=1.0
        )

        metrics.add_trade(trade)

        assert len(metrics.trades) == 1
        assert metrics.trades[0] == trade
        assert metrics._cache_timestamp is None  # Cache invalidated

    def test_update_equity(self, metrics):
        """Test updating equity curve."""
        metrics.update_equity(101000)
        metrics.update_equity(102000)
        metrics.update_equity(99000)

        assert len(metrics.equity_curve) == 4  # Initial + 3 updates
        assert metrics.equity_curve[-1] == 99000

    def test_calculate_returns(self, metrics, equity_curve):
        """Test returns calculation."""
        # Test with Series (preserves index)
        metrics.equity_curve = equity_curve

        returns = metrics.calculate_returns()

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(equity_curve) - 1
        assert returns.index[0] == equity_curve.index[1]

        # Test with list (no index preservation)
        metrics.equity_curve = equity_curve.tolist()
        returns_list = metrics.calculate_returns()

        assert isinstance(returns_list, pd.Series)
        assert len(returns_list) == len(equity_curve) - 1

    def test_total_return(self, metrics):
        """Test total return calculation."""
        metrics.equity_curve = [100000, 110000, 120000]

        total_return = metrics.total_return()

        assert total_return == 20.0  # 20% return

    def test_annual_return(self, metrics):
        """Test annualized return calculation."""
        # Set up 2-year period
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 1, 1)
        _ = (end_date - start_date).days  # Calculate days for reference

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        metrics.equity_curve = [100000] + [120000] * len(dates)  # 20% total over 2 years

        # Pass trading days (252 per year) not calendar days
        trading_days = 2 * 252  # 2 years of trading days
        annual_return = metrics.annual_return(periods=trading_days)

        # Should be approximately 9.5% per year (sqrt(1.2) - 1)
        assert 9 < annual_return < 10

    def test_sharpe_ratio(self, metrics):
        """Test Sharpe ratio calculation."""
        # Create returns with known properties
        returns = pd.Series([0.01, -0.005, 0.015, -0.002, 0.008, 0.012, -0.003, 0.01])
        metrics.calculate_returns = Mock(return_value=returns)

        sharpe = metrics.sharpe_ratio()

        # Should be positive (positive returns, reasonable volatility)
        assert sharpe > 0
        assert sharpe < 15  # With these returns, Sharpe should be around 10-12

    def test_sortino_ratio(self, metrics):
        """Test Sortino ratio calculation."""
        # Create returns with known downside
        returns = pd.Series([0.01, -0.005, 0.015, -0.002, 0.008, 0.012, -0.003, 0.01])
        metrics.calculate_returns = Mock(return_value=returns)

        sortino = metrics.sortino_ratio()

        # Sortino should be higher than Sharpe (only penalizes downside)
        sharpe = metrics.sharpe_ratio()
        assert sortino > sharpe

    def test_max_drawdown(self, metrics):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        metrics.equity_curve = [100000, 105000, 110000, 102000, 98000, 103000, 108000]

        max_dd = metrics.max_drawdown()

        # Max drawdown from 110k to 98k = -10.91%
        assert max_dd == pytest.approx(-10.91, rel=0.01)

    def test_calmar_ratio(self, metrics):
        """Test Calmar ratio calculation."""
        metrics.annual_return = Mock(return_value=15.0)
        metrics.max_drawdown = Mock(return_value=-10.0)

        calmar = metrics.calmar_ratio()

        assert calmar == 1.5  # 15% return / 10% drawdown

    def test_win_rate(self, metrics, sample_trades):
        """Test win rate calculation."""
        metrics.trades = sample_trades

        win_rate = metrics.win_rate()

        # 2 wins out of 3 completed trades
        assert win_rate == pytest.approx(0.667, rel=0.01)

    def test_profit_factor(self, metrics, sample_trades):
        """Test profit factor calculation."""
        metrics.trades = sample_trades

        pf = metrics.profit_factor()

        # Gross profit: 498 + 497 = 995
        # Gross loss: 504
        # PF = 995 / 504 = 1.97
        assert pf == pytest.approx(1.97, rel=0.01)

    def test_average_win_loss(self, metrics, sample_trades):
        """Test average win/loss calculation."""
        metrics.trades = sample_trades

        avg_win = metrics.average_win()
        avg_loss = metrics.average_loss()

        # Average win: (498 + 497) / 2 = 497.5
        # Average loss: 504
        assert avg_win == pytest.approx(497.5, rel=0.01)
        assert avg_loss == pytest.approx(504.0, rel=0.01)

    def test_expectancy(self, metrics, sample_trades):
        """Test expectancy calculation."""
        metrics.trades = sample_trades

        expectancy = metrics.expectancy()

        # (Win% * Avg Win) - (Loss% * Avg Loss)
        # (0.667 * 497.5) - (0.333 * 504) = 163.5
        assert expectancy == pytest.approx(163.5, rel=0.1)

    def test_total_trades(self, metrics, sample_trades):
        """Test total trades count."""
        metrics.trades = sample_trades

        total = metrics.total_trades()

        # Only completed trades (with pnl) are counted: 3 trades
        assert total == 3

    def test_average_trade_duration(self, metrics, sample_trades):
        """Test average trade duration calculation."""
        metrics.trades = sample_trades

        avg_duration = metrics.average_trade_duration()

        # Since our sample trades don't have entry_time/exit_time attributes
        # the average duration will be 0
        assert avg_duration == 0.0

    def test_get_metrics_summary(self, metrics, sample_trades):
        """Test getting complete metrics summary."""
        metrics.trades = sample_trades
        metrics.equity_curve = [100000, 101000, 99000, 102000, 100500]

        summary = metrics.get_metrics_summary()

        # Check all expected metrics present
        expected_metrics = [
            'total_return', 'annual_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'calmar_ratio', 'win_rate', 'profit_factor',
            'average_win', 'average_loss', 'expectancy', 'total_trades'
        ]

        for metric in expected_metrics:
            assert metric in summary
            assert isinstance(summary[metric], (int, float))

    def test_metrics_caching(self, metrics):
        """Test metrics caching functionality."""
        metrics.equity_curve = [100000, 101000, 102000]

        # First call should calculate
        with patch.object(metrics, '_calculate_total_return') as mock_calc:
            mock_calc.return_value = 2.0
            result1 = metrics.total_return()
            assert mock_calc.called
            assert result1 == 2.0

        # Second call should use cache (not call _calculate_total_return)
        with patch.object(metrics, '_calculate_total_return') as mock_calc:
            result2 = metrics.total_return()
            assert not mock_calc.called
            assert result2 == 2.0  # Should get cached value

    def test_cache_invalidation(self, metrics):
        """Test cache invalidation on new data."""
        metrics.equity_curve = [100000, 101000]

        # Calculate and cache
        _ = metrics.total_return()
        assert len(metrics._metrics_cache) > 0

        # Add new data
        metrics.update_equity(102000)

        # Cache should be cleared
        assert len(metrics._metrics_cache) == 0

    def test_empty_trades_handling(self, metrics):
        """Test handling of empty trade list."""
        metrics.trades = []

        assert metrics.win_rate() == 0
        assert metrics.profit_factor() == 0
        assert metrics.average_win() == 0
        assert metrics.average_loss() == 0
        assert metrics.total_trades() == 0

    def test_single_trade_handling(self, metrics):
        """Test handling of single trade."""
        metrics.trades = [
            Trade(
                timestamp=datetime.now(),
                symbol='AAPL',
                side='BUY',
                quantity=100,
                price=150.0,
                commission=1.0
            )
        ]

        # Should handle incomplete trades gracefully
        assert metrics.total_trades() == 0
        assert metrics.win_rate() == 0

    def test_drawdown_recovery_time(self, metrics):
        """Test drawdown recovery time calculation."""
        # Create equity curve with drawdown and recovery
        metrics.equity_curve = [
            100000,  # Start
            105000,  # Peak
            102000,  # Drawdown start
            98000,   # Bottom
            100000,  # Partial recovery
            105000,  # Full recovery
            108000   # New high
        ]

        dd_info = metrics.drawdown_analysis()

        assert 'max_drawdown' in dd_info
        assert 'recovery_time' in dd_info
        assert 'underwater_time' in dd_info
        assert dd_info['recovery_time'] == 2  # 2 periods to recover (from bottom at index 3 to recovery at index 5)

    def test_risk_adjusted_returns(self, metrics):
        """Test various risk-adjusted return metrics."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        metrics.calculate_returns = Mock(return_value=returns)

        # Test Information Ratio
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        ir = metrics.information_ratio(benchmark_returns)
        assert isinstance(ir, float)

        # Test Omega Ratio
        omega = metrics.omega_ratio(threshold=0)
        assert omega > 0

        # Test Tail Ratio
        tail_ratio = metrics.tail_ratio()
        assert tail_ratio > 0
