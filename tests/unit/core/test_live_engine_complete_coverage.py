"""
Complete test coverage for LiveTradingEngine.

This test file aims to achieve 100% coverage by testing all remaining
uncovered methods and edge cases.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
import pytest_asyncio

# Mock all dependencies before importing
sys.modules['apscheduler'] = Mock()
sys.modules['apscheduler.schedulers'] = Mock()
sys.modules['apscheduler.schedulers.asyncio'] = Mock()
sys.modules['apscheduler.triggers'] = Mock()
sys.modules['apscheduler.triggers.cron'] = Mock()
sys.modules['apscheduler.triggers.interval'] = Mock()

class MockScheduler:
    """Mock AsyncIOScheduler."""
    def __init__(self):
        self.jobs = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, func, trigger, **kwargs):
        """Mock add_job method."""
        job = Mock()
        job.id = f"job_{len(self.jobs)}"
        job.func = func
        job.trigger = trigger
        job.kwargs = kwargs
        self.jobs.append(job)
        return job

    def start(self):
        """Mock start method."""
        self.started = True

    def shutdown(self):
        """Mock shutdown method."""
        self.shutdown_called = True

# Mock the scheduler
mock_scheduler_class = Mock(return_value=MockScheduler())
sys.modules['apscheduler.schedulers.asyncio'].AsyncIOScheduler = mock_scheduler_class

# Mock triggers
sys.modules['apscheduler.triggers.cron'].CronTrigger = Mock
sys.modules['apscheduler.triggers.interval'].IntervalTrigger = Mock

# Now we can import LiveTradingEngine
from core.live_engine import LiveTradingEngine, TradingMode
from strategies.base import BaseStrategy, Signal


# Removed run_async helper - using pytest-asyncio instead


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    def __init__(self, name="test_strategy"):
        config = {'name': name, 'symbols': ["AAPL", "GOOGL"]}
        super().__init__(config)
        self.name = name
        self.symbols = ["AAPL", "GOOGL"]
        self.enabled = True
        self.next_called = False

    def init(self):
        """Initialize strategy."""
        pass

    def size(self, symbol):
        """Get position size."""
        return 100

    def next(self, data):
        """Generate test signal."""
        self.next_called = True
        return Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id=self.name,
            price=150.0
        )


class TestLiveTradingEngineComplete:
    """Complete test coverage for LiveTradingEngine."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        # Data handler
        data_handler = Mock()
        data_handler.subscribe = AsyncMock()
        data_handler.unsubscribe = AsyncMock()
        data_handler.get_current_price = Mock(return_value=150.0)
        data_handler.get_latest_data = Mock(return_value=pd.DataFrame({
            'close': [150.0],
            'volume': [1000000]
        }))

        # Portfolio engine
        portfolio = Mock()
        portfolio.cash = 100000.0
        portfolio.total_equity = 100000.0
        portfolio.current_equity = 100000.0
        portfolio.positions = {}
        portfolio.update_price = Mock()
        portfolio.update_portfolio_value = Mock()
        portfolio.get_position = Mock(return_value=None)
        portfolio.add_position = Mock()
        portfolio.to_dict = Mock(return_value={'cash': 100000.0, 'positions': {}})

        # Risk manager
        risk_manager = Mock()
        risk_manager.check_pre_trade = Mock(return_value=True)
        risk_manager.check_position_risk = Mock(return_value=True)
        risk_manager.check_portfolio_risk = Mock(return_value=True)
        risk_manager.calculate_position_size = Mock(return_value=100)
        risk_manager.to_dict = Mock(return_value={'max_position_size': 10000})

        # Order manager
        order_manager = Mock()
        order_manager.executors = {}
        order_manager.create_order = AsyncMock()
        order_manager.submit_order = AsyncMock(return_value=True)
        order_manager.cancel_order = AsyncMock(return_value=True)
        order_manager.get_active_orders = Mock(return_value=[])
        order_manager.get_order_statistics = Mock(return_value={
            'total_orders': 10,
            'filled_orders': 8,
            'rejected_orders': 2
        })

        # Executor
        executor = Mock()
        executor.connect = AsyncMock(return_value=True)
        executor.disconnect = AsyncMock()
        executor.place_order = AsyncMock(return_value={'order_id': '123', 'status': 'FILLED'})

        # Add executor to order manager
        order_manager.executors = {'test_executor': executor}

        # Memory manager
        memory_manager = Mock()
        memory_manager.check_memory = Mock()
        memory_manager.cleanup = Mock()
        memory_manager.get_usage = Mock(return_value={'used': 100, 'available': 1000})

        return {
            'data_handler': data_handler,
            'portfolio_engine': portfolio,
            'risk_manager': risk_manager,
            'order_manager': order_manager,
            'memory_manager': memory_manager,
            'executor': executor
        }

    @pytest.fixture
    def engine(self, mock_components):
        """Create LiveTradingEngine with mocked components."""
        # Mock the initialization to use our components
        with patch('core.live_engine.DataHandler', return_value=mock_components['data_handler']):
            with patch('core.live_engine.PortfolioEngine', return_value=mock_components['portfolio_engine']):
                with patch('core.live_engine.RiskManager', return_value=mock_components['risk_manager']):
                    with patch('core.live_engine.EnhancedOrderManager', return_value=mock_components['order_manager']):
                        with patch('core.live_engine.MemoryManager', return_value=mock_components['memory_manager']):
                            with patch('core.live_engine.MetricsCollector'):
                                config = {
                                    'mode': TradingMode.PAPER,
                                    'portfolio_config': {'initial_capital': 100000}
                                }
                                engine = LiveTradingEngine(config)

        # Add test strategy
        engine.strategies['test_strategy'] = MockStrategy()
        return engine

    @pytest.mark.asyncio
    async def test_start_stop_engine(self, engine, mock_components):
        """Test starting and stopping the engine."""
        # Mock main loop to exit immediately
        async def mock_main_loop():
            engine.is_running = False

        engine._main_loop = mock_main_loop

        # Start engine
        await engine.start()

        # Verify connections
        mock_components['executor'].connect.assert_called_once()
        assert engine.scheduler.started

        # Stop engine
        await engine.stop()

        # Verify disconnections
        mock_components['executor'].disconnect.assert_called_once()
        assert hasattr(engine.scheduler, 'shutdown_called') and engine.scheduler.shutdown_called
        assert not engine.is_running

    @pytest.mark.asyncio
    async def test_main_loop_execution(self, engine, mock_components):
        """Test main loop execution."""
        # Track iterations
        iterations = 0

        async def mock_update_market_data():
            nonlocal iterations
            iterations += 1
            if iterations > 2:
                engine.is_running = False

        engine._update_market_data = mock_update_market_data
        engine._update_positions = AsyncMock()
        engine._run_strategies = AsyncMock()
        engine._check_risk_limits = AsyncMock()
        engine._check_memory_health = AsyncMock()
        engine.is_running = True
        engine.is_trading_hours = True

        # Run main loop with mocked sleep
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await engine._main_loop()

        # Verify execution
        assert iterations > 2
        assert engine._check_risk_limits.call_count >= 2
        assert engine._update_positions.call_count >= 2

    @pytest.mark.asyncio
    async def test_process_market_data(self, engine, mock_components):
        """Test market data processing."""
        # Setup market data
        market_data = pd.DataFrame({
            'close': [150.0, 151.0],
            'volume': [1000000, 1100000]
        })
        mock_components['data_handler'].get_latest_data.return_value = market_data

        # Process data
        await engine._process_market_data()

        # Verify strategy was called
        assert engine.strategies['test_strategy'].next_called

    @pytest.mark.asyncio
    async def test_check_risk_limits(self, engine, mock_components):
        """Test risk limit checking."""
        # Add a position
        position = Mock()
        position.symbol = "AAPL"
        position.quantity = 100
        position.current_price = 150.0
        position.unrealized_pnl = -500.0  # Loss position
        mock_components['portfolio_engine'].positions = {"AAPL": position}

        # Check risk limits
        await engine._check_risk_limits()

        # Verify risk checks
        mock_components['risk_manager'].check_position_risk.assert_called()
        mock_components['risk_manager'].check_portfolio_risk.assert_called()

    @pytest.mark.asyncio
    async def test_update_portfolio(self, engine, mock_components):
        """Test portfolio updates."""
        # Setup price data
        mock_components['data_handler'].get_current_price.return_value = 155.0

        # Add position
        position = Mock()
        position.symbol = "AAPL"
        mock_components['portfolio_engine'].positions = {"AAPL": position}

        # Update portfolio
        await engine._update_portfolio()

        # Verify updates
        mock_components['portfolio_engine'].update_price.assert_called_with("AAPL", 155.0)
        mock_components['portfolio_engine'].update_portfolio_value.assert_called()

    @pytest.mark.asyncio
    async def test_process_signal(self, engine, mock_components):
        """Test signal processing."""
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )

        # Mock order creation
        order = Mock()
        order.order_id = "123"
        mock_components['order_manager'].create_order.return_value = order

        # Process signal
        await engine._process_signal(signal)

        # Verify order creation and submission
        mock_components['risk_manager'].check_pre_trade.assert_called()
        mock_components['risk_manager'].calculate_position_size.assert_called()
        mock_components['order_manager'].create_order.assert_called()
        mock_components['order_manager'].submit_order.assert_called()

    @pytest.mark.asyncio
    async def test_process_signal_rejected_by_risk(self, engine, mock_components):
        """Test signal rejected by risk manager."""
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )

        # Mock risk rejection
        mock_components['risk_manager'].check_pre_trade.return_value = False

        # Process signal
        await engine._process_signal(signal)

        # Verify order not created
        mock_components['order_manager'].create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_data_feeds(self, engine, mock_components):
        """Test data feed initialization."""
        await engine._initialize_data_feeds()

        # Verify subscriptions
        expected_symbols = ["AAPL", "GOOGL"]
        for symbol in expected_symbols:
            mock_components['data_handler'].subscribe.assert_any_call(symbol)

    def test_check_market_hours(self, engine):
        """Test market hours checking."""
        # During market hours (mock)
        with patch('core.live_engine.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 2, 10, 30)  # Tuesday 10:30 AM
            mock_dt.today.return_value = datetime(2024, 1, 2)
            assert engine._check_market_hours() is True

        # Outside market hours
        with patch('core.live_engine.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 2, 20, 0)  # Tuesday 8 PM
            mock_dt.today.return_value = datetime(2024, 1, 2)
            assert engine._check_market_hours() is False

        # Weekend
        with patch('core.live_engine.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 6, 10, 30)  # Saturday
            mock_dt.today.return_value = datetime(2024, 1, 6)
            assert engine._check_market_hours() is False

    def test_initialize_reporting(self, engine):
        """Test reporting initialization."""
        # Initialize reporting
        engine._initialize_reporting()

        # Check that jobs were scheduled
        assert len(engine.scheduler.jobs) > 0

        # Find reporting jobs
        reporting_jobs = [job for job in engine.scheduler.jobs if 'report' in str(job.func)]
        assert len(reporting_jobs) > 0

    @pytest.mark.asyncio
    async def test_generate_daily_report(self, engine, mock_components):
        """Test daily report generation."""
        # Setup portfolio state
        mock_components['portfolio_engine'].to_dict.return_value = {
            'cash': 95000.0,
            'total_equity': 105000.0,
            'positions': {
                'AAPL': {'quantity': 100, 'value': 15000.0}
            }
        }

        # Generate report
        await engine._generate_daily_report()

        # Report should be created (check stats)
        assert 'daily_pnl' in engine.stats

    def test_log_statistics(self, engine, mock_components):
        """Test statistics logging."""
        # Setup statistics
        engine.stats = {
            'engine_start': datetime.now() - timedelta(hours=2),
            'signals_generated': 50,
            'orders_placed': 45,
            'orders_filled': 40,
            'total_pnl': 5000.0
        }

        # Log statistics (should not raise)
        engine._log_statistics()

    def test_save_state(self, engine, mock_components, tmp_path):
        """Test state saving."""
        # Setup state file
        state_file = tmp_path / "test_state.json"
        engine.state_file = str(state_file)

        # Add some state
        engine.stats['test_value'] = 123

        # Save state
        engine._save_state()

        # Verify file created
        assert state_file.exists()

        # Load and verify content
        with open(state_file) as f:
            state = json.load(f)
        assert 'portfolio' in state
        assert 'stats' in state
        assert state['stats']['test_value'] == 123

    def test_load_state(self, engine, tmp_path):
        """Test state loading."""
        # Create state file
        state_file = tmp_path / "test_state.json"
        state_data = {
            'stats': {'loaded_value': 456},
            'portfolio': {'cash': 90000.0}
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        engine.state_file = str(state_file)

        # Load state
        engine._load_state()

        # Verify loaded
        assert engine.stats.get('loaded_value') == 456

    def test_load_state_no_file(self, engine):
        """Test loading state when file doesn't exist."""
        engine.state_file = "nonexistent.json"

        # Should not raise
        engine._load_state()

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, engine, mock_components):
        """Test cancelling all orders."""
        # Mock active orders
        orders = [
            Mock(order_id="order1"),
            Mock(order_id="order2"),
            Mock(order_id="order3")
        ]
        mock_components['order_manager'].get_active_orders.return_value = orders

        # Cancel all
        await engine._cancel_all_orders()

        # Verify all cancelled
        assert mock_components['order_manager'].cancel_order.call_count == 3
        mock_components['order_manager'].cancel_order.assert_any_call("order1")
        mock_components['order_manager'].cancel_order.assert_any_call("order2")
        mock_components['order_manager'].cancel_order.assert_any_call("order3")

    @pytest.mark.asyncio
    async def test_emergency_stop(self, engine, mock_components):
        """Test emergency stop functionality."""
        # Add position
        position = Mock()
        position.symbol = "AAPL"
        position.quantity = 100
        mock_components['portfolio_engine'].positions = {"AAPL": position}

        # Mock order creation
        order = Mock()
        order.order_id = "emergency_123"
        mock_components['order_manager'].create_order.return_value = order

        # Emergency stop
        await engine.emergency_stop("Test emergency")

        # Verify all positions closed
        mock_components['order_manager'].create_order.assert_called()
        mock_components['order_manager'].submit_order.assert_called()

        # Verify engine stopped
        assert not engine.is_running

    def test_add_remove_strategy(self, engine):
        """Test adding and removing strategies."""
        # Add strategy
        new_strategy = MockStrategy("new_strategy")
        engine.add_strategy("new_strategy", new_strategy)

        assert "new_strategy" in engine.strategies

        # Remove strategy
        engine.remove_strategy("new_strategy")

        assert "new_strategy" not in engine.strategies

    def test_enable_disable_strategies(self, engine):
        """Test enabling/disabling strategies."""
        strategy = engine.strategies['test_strategy']

        # Disable
        engine.disable_strategy('test_strategy')
        assert not strategy.enabled

        # Enable
        engine.enable_strategy('test_strategy')
        assert strategy.enabled

    def test_disable_all_strategies(self, engine):
        """Test disabling all strategies."""
        # Add another strategy
        engine.strategies['another'] = MockStrategy('another')

        # Disable all
        engine.disable_all_strategies()

        # Verify all disabled
        for strategy in engine.strategies.values():
            assert not strategy.enabled

    @pytest.mark.asyncio
    async def test_check_portfolio_health(self, engine, mock_components):
        """Test portfolio health checking."""
        # Setup unhealthy portfolio (big loss)
        mock_components['portfolio_engine'].total_equity = 70000.0  # 30% loss
        mock_components['portfolio_engine'].cash = 70000.0

        # Check health
        await engine._check_portfolio_health()

        # Should trigger some action (logged warning at minimum)
        # In real implementation might trigger emergency stop

    @pytest.mark.asyncio
    async def test_close_position(self, engine, mock_components):
        """Test closing a position."""
        # Mock position
        position = Mock()
        position.symbol = "AAPL"
        position.quantity = 100
        mock_components['portfolio_engine'].get_position.return_value = position

        # Mock order
        order = Mock()
        order.order_id = "close_123"
        mock_components['order_manager'].create_order.return_value = order

        # Close position
        await engine._close_position("AAPL")

        # Verify sell order created
        mock_components['order_manager'].create_order.assert_called()
        call_args = mock_components['order_manager'].create_order.call_args[1]
        assert call_args['side'] == 'SELL'
        assert call_args['quantity'] == 100

    @pytest.mark.asyncio
    async def test_reconnect_data_feed(self, engine, mock_components):
        """Test data feed reconnection."""
        # Simulate disconnection and reconnection
        mock_components['data_handler'].subscribe.reset_mock()

        # Reconnect
        await engine._reconnect_data_feed()

        # Verify resubscription
        assert mock_components['data_handler'].subscribe.call_count > 0

    def test_handle_order_event(self, engine, mock_components):
        """Test order event handling."""
        # Mock order event
        order = Mock()
        order.order_id = "123"
        order.symbol = "AAPL"
        order.filled_quantity = 100
        order.average_fill_price = 150.0

        event_data = {
            'order': order,
            'status': 'FILLED'
        }

        # Handle filled order
        engine._handle_order_event(order, 'FILLED', event_data)

        # Stats should be updated
        assert engine.stats.get('orders_filled', 0) > 0

    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, engine, mock_components):
        """Test getting portfolio summary."""
        # Setup portfolio
        mock_components['portfolio_engine'].to_dict.return_value = {
            'cash': 95000.0,
            'total_equity': 105000.0,
            'positions': {'AAPL': {'quantity': 100}}
        }

        summary = await engine.get_portfolio_summary()

        assert summary['cash'] == 95000.0
        assert summary['total_equity'] == 105000.0
        assert 'AAPL' in summary['positions']

    def test_get_active_strategies(self, engine):
        """Test getting active strategies."""
        # Add disabled strategy
        disabled = MockStrategy('disabled')
        disabled.enabled = False
        engine.strategies['disabled'] = disabled

        active = engine.get_active_strategies()

        assert len(active) == 1
        assert active[0].name == 'test_strategy'

    def test_validate_signal(self, engine):
        """Test signal validation."""
        # Valid signal
        valid_signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )
        assert engine._validate_signal(valid_signal) is True

        # Invalid signal (wrong symbol)
        invalid_signal = Signal(
            symbol="INVALID",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )
        assert engine._validate_signal(invalid_signal) is False

    def test_calculate_position_size(self, engine, mock_components):
        """Test position size calculation."""
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )

        mock_components['risk_manager'].calculate_position_size.return_value = 100

        size = engine._calculate_position_size(signal)

        assert size == 100
        mock_components['risk_manager'].calculate_position_size.assert_called()

    @pytest.mark.asyncio
    async def test_market_open_close_handlers(self, engine):
        """Test market open/close event handlers."""
        # Market open
        await engine._handle_market_open()
        assert 'market_open_time' in engine.stats

        # Market close
        await engine._handle_market_close()
        assert 'market_close_time' in engine.stats

    def test_handle_data_error(self, engine, mock_components):
        """Test data error handling."""
        # Simulate data error
        error = Exception("Data feed disconnected")

        # Handle error
        engine._handle_data_error(error)

        # Should attempt reconnection
        # In real implementation would trigger reconnection logic

    def test_validate_configuration(self, engine):
        """Test configuration validation."""
        # Should not raise for valid config
        engine._validate_configuration()

        # Test with invalid config
        engine.mode = "INVALID_MODE"
        with pytest.raises(ValueError):
            engine._validate_configuration()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
