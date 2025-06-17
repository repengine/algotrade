"""Additional tests to achieve 100% coverage for metrics.py."""

from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np

from algostack.core.metrics import MetricsCollector, Trade, DailyMetrics


class TestMetricsCoverage:
    """Tests for missing coverage in metrics.py."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance."""
        return MetricsCollector(initial_capital=100000)
    
    def test_partial_exit(self, metrics_collector):
        """Test partial position exit - lines 165-166."""
        # Enter a position
        metrics_collector.record_trade_entry(
            symbol='AAPL',
            price=100.0,
            quantity=200,  # Buy 200 shares
            side='BUY',
            timestamp=datetime(2024, 1, 1)
        )
        
        # Partial exit of 100 shares
        trade = metrics_collector.record_trade_exit(
            symbol='AAPL',
            price=110.0,
            timestamp=datetime(2024, 1, 2),
            quantity=100  # Exit only 100 shares
        )
        
        # Check trade recorded correctly
        assert trade.quantity == 100  # Only 100 shares exited
        assert trade.pnl == 1000.0  # (110-100) * 100
        
        # Check remaining position
        assert 'AAPL' in metrics_collector.open_trades
        assert metrics_collector.open_trades['AAPL']['quantity'] == 100  # 100 shares remaining
    
    def test_short_trade(self, metrics_collector):
        """Test short trade P&L calculation - line 177."""
        # Enter a short position
        metrics_collector.record_trade_entry(
            symbol='TSLA',
            price=200.0,
            quantity=50,
            side='SHORT',
            timestamp=datetime(2024, 1, 1)
        )
        
        # Exit short (buy back at lower price for profit)
        trade = metrics_collector.record_trade_exit(
            symbol='TSLA',
            price=180.0,
            timestamp=datetime(2024, 1, 2),
            commission=5.0
        )
        
        # Check P&L calculation for short
        # Profit = (entry_price - exit_price) * quantity - commission
        # = (200 - 180) * 50 - 5 = 995
        assert trade.pnl == 995.0
        assert trade.side == 'SHORT'
    
    def test_sell_side_trade(self, metrics_collector):
        """Test SELL side trade (alternative to SHORT)."""
        # Enter a sell position
        metrics_collector.record_trade_entry(
            symbol='GME',
            price=50.0,
            quantity=100,
            side='SELL',
            timestamp=datetime(2024, 1, 1)
        )
        
        # Cover sell position
        trade = metrics_collector.record_trade_exit(
            symbol='GME',
            price=45.0,
            timestamp=datetime(2024, 1, 2)
        )
        
        # Check P&L
        assert trade.pnl == 500.0  # (50-45) * 100
    
    def test_timestamp_as_none(self, metrics_collector):
        """Test update_portfolio_value with None timestamp - line 210."""
        # Update without timestamp
        metrics_collector.update_portfolio_value(105000)
        
        # Check it was recorded
        assert len(metrics_collector.value_history) == 1
        assert metrics_collector.value_history[0][1] == 105000
        # Timestamp should be auto-generated
        assert isinstance(metrics_collector.value_history[0][0], datetime)
    
    def test_daily_return_zero_starting_value(self, metrics_collector):
        """Test daily return calculation with zero starting value - line 229."""
        # This is an edge case that shouldn't happen in practice
        # but we handle it to prevent division by zero
        metrics_collector.value_history = []  # Clear history
        
        # Record a daily metric for today
        daily = metrics_collector.record_daily_metrics(datetime.now().date())
        
        # Should handle gracefully
        assert daily.daily_return == 0.0
    
    def test_previous_day_value_lookup(self, metrics_collector):
        """Test previous day value calculation - line 252."""
        # Add value history across multiple days
        base_date = datetime(2024, 1, 1)
        
        # Day 1
        metrics_collector.update_portfolio_value(100000, base_date)
        
        # Day 2
        metrics_collector.update_portfolio_value(105000, base_date + timedelta(days=1))
        
        # Day 3 - no values yet
        daily = metrics_collector.record_daily_metrics((base_date + timedelta(days=2)).date())
        
        # Should use Day 2's ending value as starting value
        assert daily.starting_value == 105000
        assert daily.ending_value == 105000  # No trades, so same
        assert daily.daily_return == 0.0
    
    def test_no_data_for_day(self, metrics_collector):
        """Test daily metrics with no data for the day - lines 263-265."""
        # Set current value
        metrics_collector.current_value = 110000
        
        # Get metrics for a day with no data
        daily = metrics_collector.record_daily_metrics(datetime(2024, 1, 1).date())
        
        # Should use current value
        assert daily.starting_value == 100000  # initial capital
        assert daily.ending_value == 110000
        assert daily.high_water_mark == 110000
        assert daily.low_water_mark == 110000
    
    def test_cache_check_within_timeout(self, metrics_collector):
        """Test cache is returned within timeout - lines 295-296."""
        # Get metrics to populate cache
        metrics1 = metrics_collector.get_performance_metrics()
        
        # Get again immediately (within 60 second timeout)
        metrics2 = metrics_collector.get_performance_metrics()
        
        # Should be the same object (from cache)
        assert metrics1 is metrics2
    
    def test_profit_factor_with_trades(self, metrics_collector):
        """Test profit factor calculation - lines 352-353, 360-361."""
        # Create winning trade
        metrics_collector.record_trade_entry(
            symbol='WIN',
            price=100.0,
            quantity=100,
            side='BUY',
            timestamp=datetime(2024, 1, 1)
        )
        metrics_collector.record_trade_exit(
            symbol='WIN',
            price=110.0,
            timestamp=datetime(2024, 1, 2)
        )
        
        # Create losing trade
        metrics_collector.record_trade_entry(
            symbol='LOSE',
            price=100.0,
            quantity=100,
            side='BUY',
            timestamp=datetime(2024, 1, 3)
        )
        metrics_collector.record_trade_exit(
            symbol='LOSE',
            price=90.0,
            timestamp=datetime(2024, 1, 4)
        )
        
        metrics = metrics_collector.get_performance_metrics()
        
        # Profit factor = gross_profit / gross_loss = 1000 / 1000 = 1.0
        assert metrics['profit_factor'] == 1.0
        assert metrics['average_loss'] == -1000.0
        assert metrics['largest_loss'] == -1000.0
    
    def test_sharpe_with_zero_std(self, metrics_collector):
        """Test Sharpe ratio with zero standard deviation - line 401."""
        # Add constant values (zero volatility)
        for i in range(10):
            metrics_collector.update_portfolio_value(
                100000,  # Same value
                datetime(2024, 1, 1) + timedelta(days=i)
            )
        
        metrics = metrics_collector.get_performance_metrics()
        
        # Should return 0 when std is 0
        assert metrics['sharpe_ratio'] == 0.0
    
    def test_sortino_with_no_downside(self, metrics_collector):
        """Test Sortino ratio with no downside returns - line 421."""
        # Add only positive returns
        value = 100000
        for i in range(10):
            value *= 1.01  # 1% daily gain
            metrics_collector.update_portfolio_value(
                value,
                datetime(2024, 1, 1) + timedelta(days=i)
            )
        
        metrics = metrics_collector.get_performance_metrics()
        
        # Should return 0 when no downside returns
        assert metrics['sortino_ratio'] == 0.0
    
    def test_calmar_ratio_with_value_history(self, metrics_collector):
        """Test Calmar ratio using value history - lines 444-455."""
        # Create value history with drawdown
        base_date = datetime(2024, 1, 1)
        
        # Rise
        for i in range(10):
            metrics_collector.update_portfolio_value(
                100000 + i * 1000,
                base_date + timedelta(days=i)
            )
        
        # Fall (create drawdown)
        for i in range(5):
            metrics_collector.update_portfolio_value(
                109000 - i * 2000,
                base_date + timedelta(days=10+i)
            )
        
        # Partial recovery
        for i in range(5):
            metrics_collector.update_portfolio_value(
                99000 + i * 1000,
                base_date + timedelta(days=15+i)
            )
        
        metrics = metrics_collector.get_performance_metrics()
        
        # Should calculate based on value history
        assert metrics['calmar_ratio'] != 0.0
        assert metrics['max_drawdown'] > 0.0
    
    def test_empty_equity_curve(self, metrics_collector):
        """Test get_equity_curve with no data - line 488."""
        # Get equity curve with no value history
        df = metrics_collector.get_equity_curve()
        
        # Should return empty DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_equity_curve_with_data(self, metrics_collector):
        """Test get_equity_curve with data - lines 509-511."""
        # Add some value history
        for i in range(5):
            metrics_collector.update_portfolio_value(
                100000 + i * 1000,
                datetime(2024, 1, 1) + timedelta(days=i)
            )
        
        # Get equity curve
        df = metrics_collector.get_equity_curve()
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'timestamp' in df.columns
        assert 'value' in df.columns
        assert 'drawdown' in df.columns
        assert 'returns' in df.columns
        assert 'cumulative_returns' in df.columns
        
        # Check values
        assert df['value'].iloc[-1] == 104000
        assert (df['drawdown'] <= 0).all()  # Drawdown should be negative or zero