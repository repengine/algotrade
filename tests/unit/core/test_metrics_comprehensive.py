"""
Comprehensive test suite for MetricsCollector.

CRITICAL FOR PROFIT GENERATION: These tests ensure accurate tracking of:
- Trade performance (win rate, profit factor)
- Portfolio returns (total, annualized, risk-adjusted)
- Drawdowns (for capital preservation)
- Strategy-specific metrics
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from core.metrics import DailyMetrics, MetricsCollector


class TestMetricsCollector:
    """Test suite for MetricsCollector - CRITICAL for tracking profits."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a MetricsCollector with 100k initial capital."""
        return MetricsCollector(initial_capital=100000.0)

    @pytest.fixture
    def sample_trades(self):
        """Create realistic trade scenarios for testing."""
        base_time = datetime(2024, 1, 1, 9, 30)
        trades = []

        # Winning trade - AAPL
        trades.append({
            'entry': {
                'symbol': 'AAPL',
                'price': 150.0,
                'quantity': 100,
                'side': 'BUY',
                'timestamp': base_time,
                'strategy_id': 'momentum'
            },
            'exit': {
                'price': 155.0,
                'timestamp': base_time + timedelta(days=5),
                'commission': 2.0
            }
        })

        # Losing trade - GOOGL
        trades.append({
            'entry': {
                'symbol': 'GOOGL',
                'price': 2800.0,
                'quantity': 10,
                'side': 'BUY',
                'timestamp': base_time + timedelta(days=10),
                'strategy_id': 'mean_reversion'
            },
            'exit': {
                'price': 2750.0,
                'timestamp': base_time + timedelta(days=15),
                'commission': 4.0
            }
        })

        # Another winning trade - MSFT
        trades.append({
            'entry': {
                'symbol': 'MSFT',
                'price': 350.0,
                'quantity': 50,
                'side': 'BUY',
                'timestamp': base_time + timedelta(days=20),
                'strategy_id': 'momentum'
            },
            'exit': {
                'price': 360.0,
                'timestamp': base_time + timedelta(days=25),
                'commission': 2.0
            }
        })

        return trades

    def test_initialization(self, metrics_collector):
        """Test proper initialization of MetricsCollector."""
        assert metrics_collector.initial_capital == 100000.0
        assert metrics_collector.current_value == 100000.0
        assert metrics_collector.high_water_mark == 100000.0
        assert metrics_collector.max_drawdown == 0.0
        assert metrics_collector.current_drawdown == 0.0
        assert len(metrics_collector.trades) == 0
        assert len(metrics_collector.open_trades) == 0

    def test_record_trade_entry(self, metrics_collector):
        """Test recording trade entry - CRITICAL for position tracking."""
        timestamp = datetime.now()

        metrics_collector.record_trade_entry(
            symbol='AAPL',
            price=150.0,
            quantity=100,
            side='BUY',
            timestamp=timestamp,
            strategy_id='momentum'
        )

        assert 'AAPL' in metrics_collector.open_trades
        open_trade = metrics_collector.open_trades['AAPL']
        assert open_trade['entry_price'] == 150.0
        assert open_trade['quantity'] == 100
        assert open_trade['side'] == 'BUY'
        assert open_trade['entry_time'] == timestamp
        assert open_trade['strategy_id'] == 'momentum'

    def test_record_trade_exit_profit(self, metrics_collector):
        """Test recording profitable trade exit - CRITICAL for P&L tracking."""
        entry_time = datetime(2024, 1, 1)
        exit_time = datetime(2024, 1, 5)

        # Record entry
        metrics_collector.record_trade_entry(
            symbol='AAPL',
            price=150.0,
            quantity=100,
            side='BUY',
            timestamp=entry_time,
            strategy_id='momentum'
        )

        # Record exit with profit
        metrics_collector.record_trade_exit(
            symbol='AAPL',
            price=155.0,
            timestamp=exit_time,
            commission=2.0
        )

        # Verify trade was closed
        assert 'AAPL' not in metrics_collector.open_trades

        # Verify trade was recorded
        assert len(metrics_collector.trades) == 1
        trade = metrics_collector.trades[0]
        assert trade.symbol == 'AAPL'
        assert trade.pnl == 498.0  # (155-150)*100 - 2
        assert trade.pnl_percentage == pytest.approx(3.32, rel=0.01)  # 498/(150*100)

    def test_record_trade_exit_loss(self, metrics_collector):
        """Test recording losing trade exit - CRITICAL for risk tracking."""
        entry_time = datetime(2024, 1, 1)
        exit_time = datetime(2024, 1, 5)

        # Record entry
        metrics_collector.record_trade_entry(
            symbol='GOOGL',
            price=2800.0,
            quantity=10,
            side='BUY',
            timestamp=entry_time
        )

        # Record exit with loss
        metrics_collector.record_trade_exit(
            symbol='GOOGL',
            price=2750.0,
            timestamp=exit_time,
            commission=4.0
        )

        # Verify trade recorded with loss
        trade = metrics_collector.trades[0]
        assert trade.pnl == -504.0  # (2750-2800)*10 - 4
        assert trade.pnl_percentage == pytest.approx(-1.8, rel=0.01)

    def test_update_portfolio_value(self, metrics_collector):
        """Test portfolio value updates - CRITICAL for equity curve."""
        timestamps = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ]
        values = [100000, 102000, 101000]

        for ts, val in zip(timestamps, values):
            metrics_collector.update_portfolio_value(val, ts)

        assert metrics_collector.current_value == 101000
        assert metrics_collector.high_water_mark == 102000
        assert len(metrics_collector.value_history) == 3

    def test_drawdown_calculation(self, metrics_collector):
        """Test drawdown calculation - CRITICAL for capital preservation."""
        # Simulate portfolio values with drawdown
        values = [100000, 105000, 110000, 108000, 106000, 109000]
        base_time = datetime(2024, 1, 1)

        for i, val in enumerate(values):
            metrics_collector.update_portfolio_value(
                val,
                base_time + timedelta(days=i)
            )

        # Peak was 110000, current is 109000
        expected_drawdown = (110000 - 109000) / 110000
        assert metrics_collector.current_drawdown == pytest.approx(expected_drawdown, rel=0.0001)

        # Max drawdown was from 110000 to 106000
        expected_max_dd = (110000 - 106000) / 110000
        assert metrics_collector.max_drawdown == pytest.approx(expected_max_dd, rel=0.0001)

    def test_get_performance_metrics(self, metrics_collector, sample_trades):
        """Test comprehensive performance metrics - CRITICAL for strategy evaluation."""
        # Execute sample trades
        for trade_data in sample_trades:
            entry = trade_data['entry']
            metrics_collector.record_trade_entry(**entry)

            exit_data = trade_data['exit']
            metrics_collector.record_trade_exit(
                symbol=entry['symbol'],
                price=exit_data['price'],
                timestamp=exit_data['timestamp'],
                commission=exit_data.get('commission', 0)
            )

        # Update final portfolio value
        metrics_collector.update_portfolio_value(100948.0, datetime(2024, 1, 30))

        # Get metrics
        metrics = metrics_collector.get_performance_metrics()

        # Verify key metrics
        assert 'total_return' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_trades' in metrics

        # Verify calculations
        assert metrics['total_trades'] == 3
        assert metrics['winning_trades'] == 2
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == pytest.approx(0.667, rel=0.01)

    def test_sharpe_ratio_calculation(self, metrics_collector):
        """Test Sharpe ratio calculation - CRITICAL for risk-adjusted returns."""
        # Simulate daily returns
        base_time = datetime(2024, 1, 1)
        values = [100000]

        # Generate returns with some volatility
        np.random.seed(42)
        for i in range(252):  # One year of trading days
            daily_return = np.random.normal(0.0005, 0.01)  # 0.05% mean, 1% vol
            values.append(values[-1] * (1 + daily_return))
            metrics_collector.update_portfolio_value(
                values[-1],
                base_time + timedelta(days=i)
            )

        metrics = metrics_collector.get_performance_metrics()

        # Sharpe should be positive with positive returns
        assert metrics['sharpe_ratio'] > 0
        # With these parameters, Sharpe should be reasonable
        assert 0 < metrics['sharpe_ratio'] < 3

    def test_sortino_ratio_calculation(self, metrics_collector):
        """Test Sortino ratio - focuses on downside risk."""
        # Create returns with asymmetric risk
        base_time = datetime(2024, 1, 1)
        values = [100000]

        for i in range(100):
            # More small gains than large losses
            if i % 10 == 0:
                daily_return = -0.02  # Occasional 2% loss
            else:
                daily_return = 0.003  # Regular 0.3% gain

            values.append(values[-1] * (1 + daily_return))
            metrics_collector.update_portfolio_value(
                values[-1],
                base_time + timedelta(days=i)
            )

        metrics = metrics_collector.get_performance_metrics()

        # Sortino should be higher than Sharpe for this return profile
        assert metrics['sortino_ratio'] > metrics['sharpe_ratio']

    def test_calmar_ratio_calculation(self, metrics_collector):
        """Test Calmar ratio - return vs max drawdown."""
        # Create scenario with known drawdown
        base_time = datetime(2024, 1, 1)

        # Portfolio grows to 110k then drops to 105k
        metrics_collector.update_portfolio_value(100000, base_time)
        metrics_collector.update_portfolio_value(110000, base_time + timedelta(days=30))
        metrics_collector.update_portfolio_value(105000, base_time + timedelta(days=60))
        metrics_collector.update_portfolio_value(108000, base_time + timedelta(days=365))

        metrics = metrics_collector.get_performance_metrics()

        # Annual return: 8%, Max DD: 4.5%
        # Calmar should be around 1.77
        assert metrics['calmar_ratio'] > 0
        assert 1 < metrics['calmar_ratio'] < 3

    def test_strategy_performance_breakdown(self, metrics_collector, sample_trades):
        """Test per-strategy metrics - CRITICAL for strategy comparison."""
        # Execute trades with different strategies
        for trade_data in sample_trades:
            entry = trade_data['entry']
            metrics_collector.record_trade_entry(**entry)

            exit_data = trade_data['exit']
            metrics_collector.record_trade_exit(
                symbol=entry['symbol'],
                price=exit_data['price'],
                timestamp=exit_data['timestamp'],
                commission=exit_data.get('commission', 0)
            )

        metrics = metrics_collector.get_performance_metrics()

        # Check strategy breakdown exists
        assert 'strategy_performance' in metrics
        strat_perf = metrics['strategy_performance']

        # Momentum strategy should have 2 winning trades
        assert 'momentum' in strat_perf
        assert strat_perf['momentum']['total_trades'] == 2
        assert strat_perf['momentum']['win_rate'] == 1.0

        # Mean reversion should have 1 losing trade
        assert 'mean_reversion' in strat_perf
        assert strat_perf['mean_reversion']['total_trades'] == 1
        assert strat_perf['mean_reversion']['win_rate'] == 0.0

    def test_get_equity_curve(self, metrics_collector):
        """Test equity curve generation - CRITICAL for visualization."""
        # Add some portfolio values
        base_time = datetime(2024, 1, 1)
        values = [100000, 101000, 99500, 102000, 101500]

        for i, val in enumerate(values):
            metrics_collector.update_portfolio_value(
                val,
                base_time + timedelta(days=i)
            )

        # Get equity curve
        equity_curve = metrics_collector.get_equity_curve()

        assert isinstance(equity_curve, pd.DataFrame)
        assert len(equity_curve) == 5
        assert 'value' in equity_curve.columns
        assert 'returns' in equity_curve.columns
        assert 'cumulative_returns' in equity_curve.columns
        assert 'drawdown' in equity_curve.columns

    def test_daily_metrics_recording(self, metrics_collector):
        """Test daily metrics snapshots."""
        date = datetime(2024, 1, 1)

        # Set up some state
        metrics_collector.update_portfolio_value(105000, date)
        metrics_collector.record_trade_entry(
            symbol='AAPL',
            price=150.0,
            quantity=100,
            side='BUY',
            timestamp=date
        )

        # Record daily metrics
        daily = metrics_collector.record_daily_metrics(date)

        assert isinstance(daily, DailyMetrics)
        assert daily.date == date.date()
        assert daily.portfolio_value == 105000
        assert daily.daily_return == 0.05  # 5% from initial 100k
        assert daily.positions_held == 1
        assert daily.total_trades == 0  # No completed trades yet

    def test_cache_invalidation(self, metrics_collector):
        """Test metrics cache is invalidated on updates."""
        # Get initial metrics
        metrics_collector.get_performance_metrics()

        # Cache should be set
        assert metrics_collector._cache_timestamp is not None

        # Add a trade
        metrics_collector.record_trade_entry(
            symbol='AAPL',
            price=150.0,
            quantity=100,
            side='BUY',
            timestamp=datetime.now()
        )

        # Cache should be invalidated
        assert metrics_collector._cache_timestamp is None

        # Get metrics again - should recalculate
        metrics2 = metrics_collector.get_performance_metrics()
        assert metrics2['total_trades'] == 0  # Still no completed trades

    def test_profit_factor_calculation(self, metrics_collector):
        """Test profit factor - CRITICAL for strategy viability."""
        # Create trades with known P&L
        base_time = datetime(2024, 1, 1)

        # Winning trades: +500, +300
        # Losing trades: -200, -100
        trades_data = [
            ('AAPL', 100, 150, 155, 500),  # +500
            ('GOOGL', 10, 2800, 2780, -200),  # -200
            ('MSFT', 50, 350, 356, 300),  # +300
            ('AMZN', 20, 180, 175, -100),  # -100
        ]

        for i, (symbol, qty, entry_price, exit_price, _) in enumerate(trades_data):
            metrics_collector.record_trade_entry(
                symbol=symbol,
                price=entry_price,
                quantity=qty,
                side='BUY',
                timestamp=base_time + timedelta(days=i*2)
            )
            metrics_collector.record_trade_exit(
                symbol=symbol,
                price=exit_price,
                timestamp=base_time + timedelta(days=i*2+1),
                commission=0  # Ignore commission for simple calculation
            )

        metrics = metrics_collector.get_performance_metrics()

        # Profit factor = 800 / 300 = 2.67
        assert metrics['profit_factor'] == pytest.approx(2.67, rel=0.01)

    def test_empty_metrics(self, metrics_collector):
        """Test metrics with no trades."""
        metrics = metrics_collector.get_performance_metrics()

        assert metrics['total_return'] == 0.0
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['profit_factor'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0

    def test_edge_cases(self, metrics_collector):
        """Test edge cases and error handling."""
        # Try to exit non-existent position
        with pytest.raises(KeyError):
            metrics_collector.record_trade_exit(
                symbol='NONEXISTENT',
                price=100.0,
                timestamp=datetime.now()
            )

        # Record entry with zero quantity (should work)
        metrics_collector.record_trade_entry(
            symbol='AAPL',
            price=150.0,
            quantity=0,
            side='BUY',
            timestamp=datetime.now()
        )

        # Update with same portfolio value (no return)
        metrics_collector.update_portfolio_value(100000, datetime.now())
        metrics_collector.update_portfolio_value(100000, datetime.now())

        metrics = metrics_collector.get_performance_metrics()
        assert metrics['total_return'] == 0.0
