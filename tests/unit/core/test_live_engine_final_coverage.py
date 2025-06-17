"""
Final test coverage for LiveTradingEngine focusing on missing methods.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Mock apscheduler before importing
sys.modules['apscheduler'] = Mock()
sys.modules['apscheduler.schedulers'] = Mock()
sys.modules['apscheduler.schedulers.asyncio'] = Mock()
sys.modules['apscheduler.triggers'] = Mock()
sys.modules['apscheduler.triggers.cron'] = Mock()
sys.modules['apscheduler.triggers.interval'] = Mock()

# Create mock scheduler that doesn't hang
class MockScheduler:
    def __init__(self):
        self.jobs = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, func, trigger, **kwargs):
        job = Mock()
        job.id = f"job_{len(self.jobs)}"
        job.func = func
        job.trigger = trigger
        job.kwargs = kwargs
        self.jobs.append(job)
        return job

    def start(self):
        self.started = True

    def shutdown(self):
        self.shutdown_called = True

# Install mock
mock_scheduler_class = Mock(return_value=MockScheduler())
sys.modules['apscheduler.schedulers.asyncio'].AsyncIOScheduler = mock_scheduler_class

# Now import LiveTradingEngine
from core.live_engine import LiveTradingEngine, TradingMode


class TestLiveTradingEngineFinalCoverage:
    """Tests to achieve 100% coverage for LiveTradingEngine."""

    def test_initialization_modes(self):
        """Test initialization with different trading modes."""
        # Test PAPER mode
        config1 = {'mode': TradingMode.PAPER}
        engine1 = LiveTradingEngine(config1)
        assert engine1.mode == TradingMode.PAPER

        # Test LIVE mode
        config2 = {'mode': TradingMode.LIVE}
        engine2 = LiveTradingEngine(config2)
        assert engine2.mode == TradingMode.LIVE

        # Test SIMULATION mode
        config3 = {'mode': TradingMode.SIMULATION}
        engine3 = LiveTradingEngine(config3)
        assert engine3.mode == TradingMode.SIMULATION

    def test_state_management_methods(self):
        """Test state management methods."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Test set_state
        engine.set_state('custom_key', 'custom_value')
        assert 'custom_key' in engine.stats
        assert engine.stats['custom_key'] == 'custom_value'

        # Test get_state
        value = engine.get_state('custom_key')
        assert value == 'custom_value'

        # Test get_state with default
        value = engine.get_state('missing_key', default='default_value')
        assert value == 'default_value'

    def test_scheduler_methods(self):
        """Test scheduler-related methods."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Initialize scheduler
        engine._initialize_scheduler()
        assert engine.scheduler is not None
        assert isinstance(engine.scheduler, MockScheduler)

        # Initialize reporting
        engine._initialize_reporting()
        assert len(engine.scheduler.jobs) > 0

        # Check scheduled jobs
        job_funcs = [job.func for job in engine.scheduler.jobs]
        assert any('_generate_daily_report' in str(func) for func in job_funcs)
        assert any('_save_state' in str(func) for func in job_funcs)

    def test_market_hours_checking(self):
        """Test market hours checking."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Test during market hours
        with patch('core.live_engine.datetime') as mock_dt:
            # Tuesday 10:30 AM EST
            mock_dt.now.return_value = datetime(2024, 1, 2, 10, 30)
            mock_dt.today.return_value = datetime(2024, 1, 2)
            result = engine._check_market_hours()
            assert result is True

        # Test outside market hours
        with patch('core.live_engine.datetime') as mock_dt:
            # Tuesday 8 PM EST
            mock_dt.now.return_value = datetime(2024, 1, 2, 20, 0)
            mock_dt.today.return_value = datetime(2024, 1, 2)
            result = engine._check_market_hours()
            assert result is False

        # Test weekend
        with patch('core.live_engine.datetime') as mock_dt:
            # Saturday
            mock_dt.now.return_value = datetime(2024, 1, 6, 10, 30)
            mock_dt.today.return_value = datetime(2024, 1, 6)
            result = engine._check_market_hours()
            assert result is False

    def test_state_persistence(self):
        """Test state saving and loading."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Create temp file for state
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            engine.state_file = f.name

        # Save state
        engine.stats['test_stat'] = 12345
        engine._save_state()

        # Load state
        engine.stats = {}  # Clear stats
        engine._load_state()

        # Verify loaded
        assert engine.stats.get('test_stat') == 12345

        # Clean up
        import os
        os.unlink(engine.state_file)

    def test_log_statistics(self):
        """Test statistics logging."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Set up some stats
        engine.stats['signals_generated'] = 100
        engine.stats['orders_placed'] = 50
        engine.stats['orders_filled'] = 45
        engine.stats['total_pnl'] = 5000.0
        engine.stats['engine_start'] = datetime.now() - timedelta(hours=2)

        # Should not raise
        engine._log_statistics()

    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid config
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)
        engine._validate_configuration()  # Should not raise

        # Invalid mode
        engine.mode = "INVALID"
        with pytest.raises(ValueError):
            engine._validate_configuration()

    def test_market_event_handlers(self):
        """Test market open/close handlers."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Mock async execution
        async def run_handler(handler):
            await handler()

        # Test market open
        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_handler(engine._handle_market_open))
        assert 'market_open_time' in engine.stats

        # Test market close
        loop.run_until_complete(run_handler(engine._handle_market_close))
        loop.close()
        assert 'market_close_time' in engine.stats

    def test_data_error_handling(self):
        """Test data error handler."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Should log error without raising
        error = Exception("Test data error")
        engine._handle_data_error(error)

    def test_portfolio_health_check(self):
        """Test portfolio health checking."""
        # Mock components
        portfolio = Mock()
        portfolio.total_equity = 70000.0  # 30% loss
        portfolio.cash = 70000.0

        config = {'mode': TradingMode.PAPER, 'portfolio_config': {'initial_capital': 100000}}

        with patch('core.live_engine.PortfolioEngine', return_value=portfolio):
            engine = LiveTradingEngine(config)
            engine.portfolio_engine = portfolio

            # Run health check
            loop = asyncio.new_event_loop()
            loop.run_until_complete(engine._check_portfolio_health())
            loop.close()

    def test_generate_daily_report(self):
        """Test daily report generation."""
        # Mock components
        portfolio = Mock()
        portfolio.to_dict.return_value = {
            'cash': 95000.0,
            'total_equity': 105000.0,
            'positions': {'AAPL': {'quantity': 100, 'value': 15000.0}}
        }

        risk_manager = Mock()
        risk_manager.to_dict.return_value = {'max_position_size': 10000}

        order_manager = Mock()
        order_manager.get_order_statistics.return_value = {
            'total_orders': 50,
            'filled_orders': 45,
            'rejected_orders': 5
        }

        config = {'mode': TradingMode.PAPER}

        with patch('core.live_engine.PortfolioEngine', return_value=portfolio):
            with patch('core.live_engine.RiskManager', return_value=risk_manager):
                with patch('core.live_engine.EnhancedOrderManager', return_value=order_manager):
                    engine = LiveTradingEngine(config)
                    engine.portfolio_engine = portfolio
                    engine.risk_manager = risk_manager
                    engine.order_manager = order_manager

                    # Generate report
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(engine._generate_daily_report())
                    loop.close()

                    assert 'daily_pnl' in engine.stats

    def test_get_portfolio_summary(self):
        """Test getting portfolio summary."""
        portfolio = Mock()
        portfolio.to_dict.return_value = {
            'cash': 95000.0,
            'total_equity': 105000.0,
            'positions': {'AAPL': {'quantity': 100}}
        }

        config = {'mode': TradingMode.PAPER}

        with patch('core.live_engine.PortfolioEngine', return_value=portfolio):
            engine = LiveTradingEngine(config)
            engine.portfolio_engine = portfolio

            loop = asyncio.new_event_loop()
            summary = loop.run_until_complete(engine.get_portfolio_summary())
            loop.close()

            assert summary['cash'] == 95000.0
            assert summary['total_equity'] == 105000.0
            assert 'AAPL' in summary['positions']

    def test_reconnect_data_feed(self):
        """Test data feed reconnection."""
        data_handler = Mock()
        data_handler.subscribe = AsyncMock()

        strategy = Mock()
        strategy.symbols = ['AAPL', 'GOOGL']

        config = {'mode': TradingMode.PAPER}

        with patch('core.live_engine.DataHandler', return_value=data_handler):
            engine = LiveTradingEngine(config)
            engine.data_handler = data_handler
            engine.strategies = {'test': strategy}

            loop = asyncio.new_event_loop()
            loop.run_until_complete(engine._reconnect_data_feed())
            loop.close()

            # Should resubscribe to all symbols
            assert data_handler.subscribe.call_count >= 2

    def test_close_position(self):
        """Test closing a position."""
        portfolio = Mock()
        position = Mock()
        position.symbol = 'AAPL'
        position.quantity = 100
        portfolio.get_position.return_value = position

        order_manager = Mock()
        order = Mock()
        order.order_id = 'close_123'
        order_manager.create_order = AsyncMock(return_value=order)
        order_manager.submit_order = AsyncMock()

        config = {'mode': TradingMode.PAPER}

        with patch('core.live_engine.PortfolioEngine', return_value=portfolio):
            with patch('core.live_engine.EnhancedOrderManager', return_value=order_manager):
                engine = LiveTradingEngine(config)
                engine.portfolio_engine = portfolio
                engine.order_manager = order_manager

                loop = asyncio.new_event_loop()
                loop.run_until_complete(engine._close_position('AAPL'))
                loop.close()

                # Verify sell order created
                order_manager.create_order.assert_called_once()
                call_args = order_manager.create_order.call_args[1]
                assert call_args['side'] == 'SELL'
                assert call_args['quantity'] == 100

    def test_handle_order_event(self):
        """Test order event handling."""
        order = Mock()
        order.order_id = '123'
        order.symbol = 'AAPL'
        order.filled_quantity = 100
        order.average_fill_price = 150.0

        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Initialize stats
        engine.stats['orders_placed'] = 10
        engine.stats['orders_filled'] = 5

        # Handle filled order
        event_data = {'order': order, 'status': 'FILLED'}
        engine._handle_order_event(order, 'FILLED', event_data)

        assert engine.stats['orders_filled'] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
