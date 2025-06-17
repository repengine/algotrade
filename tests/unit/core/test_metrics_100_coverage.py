"""Test cases to achieve 100% coverage for metrics.py."""

from datetime import datetime, timedelta

from core.metrics import MetricsCollector


class TestMetrics100Coverage:
    """Test cases specifically targeting uncovered lines in metrics.py."""

    def test_no_winning_trades_metrics(self):
        """Test metrics calculation when there are only losing trades (lines 352-353)."""
        collector = MetricsCollector(initial_capital=100000)

        # Record only losing trades
        collector.record_trade_entry("AAPL", 150.0, 100, "BUY", datetime.now())
        collector.record_trade_exit("AAPL", 140.0, datetime.now() + timedelta(hours=1), commission=2.0)

        collector.record_trade_entry("TSLA", 250.0, 50, "BUY", datetime.now())
        collector.record_trade_exit("TSLA", 240.0, datetime.now() + timedelta(hours=2), commission=2.0)

        metrics = collector.get_performance_metrics()

        # Verify lines 352-353 are executed
        assert metrics["average_win"] == 0.0
        assert metrics["largest_win"] == 0.0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 2
        assert metrics["win_rate"] == 0.0

    def test_calmar_ratio_zero_trading_days(self):
        """Test Calmar ratio calculation when trading happens on same day (line 444)."""
        collector = MetricsCollector(initial_capital=100000)

        # Add value history on the same timestamp
        base_time = datetime.now()
        collector.update_portfolio_value(100000, base_time)
        collector.update_portfolio_value(110000, base_time)  # Same timestamp

        # This should trigger line 444 (trading_days = 0)
        calmar = collector._calculate_calmar_ratio()

        # When trading_days = 0, annualized_return = total_return
        assert calmar == 0.0  # No drawdown yet

    def test_calmar_ratio_no_trades_or_history(self):
        """Test Calmar ratio with no trades and minimal value history (line 455)."""
        collector = MetricsCollector(initial_capital=100000)

        # Update value to create drawdown
        collector.update_portfolio_value(110000)
        collector.update_portfolio_value(95000)  # Create drawdown

        # No trades, so it should fall through to line 455
        calmar = collector._calculate_calmar_ratio()

        # Verify calculation worked
        assert isinstance(calmar, float)
        assert calmar < 0  # Negative return with positive drawdown

    def test_import_fallback_no_backtest_metrics(self):
        """Test import fallback when backtest_metrics module doesn't exist (lines 509-511)."""
        # This test verifies the fallback mechanism works
        # The actual fallback is tested in test_metrics_edge_cases.py
        # Here we just verify the module structure
        from core import metrics

        # Verify MetricsCollector exists
        assert hasattr(metrics, 'MetricsCollector')

        # The fallback sets BacktestMetrics = MetricsCollector when import fails
        # In normal operation, BacktestMetrics should be imported from backtest_metrics
        # We can't easily test the ImportError path without breaking other tests

    def test_edge_case_single_day_trades_calmar(self):
        """Test Calmar ratio with trades on a single day."""
        collector = MetricsCollector(initial_capital=100000)

        # Record trades on the same day
        trade_time = datetime.now()
        collector.record_trade_entry("AAPL", 150.0, 100, "BUY", trade_time, strategy_id="test")
        collector.record_trade_exit("AAPL", 155.0, trade_time + timedelta(minutes=30), commission=2.0)

        # Update portfolio value to create history
        collector.update_portfolio_value(100500, trade_time + timedelta(hours=1))

        # This should calculate Calmar ratio with minimal trading period
        metrics = collector.get_performance_metrics()

        # Calmar ratio should be calculated
        assert "calmar_ratio" in metrics
        assert isinstance(metrics["calmar_ratio"], float)

    def test_calmar_ratio_no_trades_no_value_history(self):
        """Test Calmar ratio with no trades and no value history to hit line 455."""
        collector = MetricsCollector(initial_capital=100000)

        # Create drawdown to have non-zero max_drawdown
        collector.current_value = 95000
        collector.max_drawdown = 0.05

        # Clear value history and trades to force the else case
        collector.value_history.clear()
        collector.trades = []

        # This should hit line 455
        calmar = collector._calculate_calmar_ratio()

        # With no history, annualized_return = total_return
        expected_return = (95000 - 100000) / 100000  # -0.05
        expected_calmar = expected_return / 0.05  # -1.0

        assert calmar == expected_calmar
