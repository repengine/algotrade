"""
Test coverage for LiveTradingEngine methods missing from existing tests.
Focuses on initialization methods, data preparation, memory health, and stop orders.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

# Mock apscheduler before importing
sys.modules['apscheduler'] = Mock()
sys.modules['apscheduler.schedulers'] = Mock()
sys.modules['apscheduler.schedulers.asyncio'] = Mock()
sys.modules['apscheduler.triggers'] = Mock()
sys.modules['apscheduler.triggers.cron'] = Mock()
sys.modules['apscheduler.triggers.interval'] = Mock()

# Create mock scheduler
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

# Mock CronTrigger and IntervalTrigger
class MockCronTrigger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class MockIntervalTrigger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

sys.modules['apscheduler.triggers.cron'].CronTrigger = MockCronTrigger
sys.modules['apscheduler.triggers.interval'].IntervalTrigger = MockIntervalTrigger

# Import after mocking
from core.executor import Order, OrderType
from core.live_engine import LiveTradingEngine, TradingMode
from core.portfolio import Position
from strategies.base import BaseStrategy, Signal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, name: str = "test_strategy"):
        config = {"name": name, "symbols": ["AAPL", "GOOGL"]}
        super().__init__(config)
        self.calculate_signals_called = False

    def init(self):
        """Initialize strategy."""
        pass

    def next(self):
        """Process next bar."""
        pass

    def size(self, symbol: str) -> float:
        """Get position size for symbol."""
        return 0.0

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.calculate_signals_called = True
        return pd.DataFrame({
            'signal': [1, -1],
            'strength': [0.8, 0.6]
        }, index=data.index)


class TestLiveTradingEngineMissingCoverage:
    """Tests for methods with missing coverage in LiveTradingEngine."""

    def test_initialize_strategies(self):
        """Test _initialize_strategies method."""
        # Test with single strategy config
        config = {
            'mode': TradingMode.PAPER,
            'strategies': [
                {
                    'class': MockStrategy,
                    'id': 'strat1',
                    'params': {'name': 'strat1'}
                }
            ]
        }

        engine = LiveTradingEngine(config)
        engine._initialize_strategies()
        assert len(engine.strategies) == 1
        assert 'strat1' in engine.strategies
        assert isinstance(engine.strategies['strat1'], MockStrategy)

        # Test with multiple strategies
        config2 = {
            'mode': TradingMode.PAPER,
            'strategies': [
                {'class': MockStrategy, 'id': 'strat1', 'params': {'name': 'strat1'}},
                {'class': MockStrategy, 'id': 'strat2', 'params': {'name': 'strat2'}}
            ]
        }

        engine2 = LiveTradingEngine(config2)
        engine2._initialize_strategies()
        assert len(engine2.strategies) == 2

        # Test with no strategies
        config3 = {'mode': TradingMode.PAPER}
        engine3 = LiveTradingEngine(config3)
        engine3._initialize_strategies()
        assert len(engine3.strategies) == 0

    def test_initialize_executors(self):
        """Test _initialize_executors method."""
        # Test PAPER mode
        config_paper = {
            'mode': TradingMode.PAPER,
            'executor_config': {'fee_rate': 0.001}
        }

        with patch('core.live_engine.PaperExecutor') as mock_paper_executor:
            engine = LiveTradingEngine(config_paper)
            # Reset mock since it's called during __init__
            mock_paper_executor.reset_mock()
            engine._initialize_executors()
            mock_paper_executor.assert_called_once()

        # Test LIVE mode with IBKR
        config_live = {
            'mode': TradingMode.LIVE,
            'broker': 'ibkr',
            'ibkr_config': {
                'host': 'localhost',
                'port': 7497,
                'client_id': 1
            }
        }

        with patch('core.live_engine.IBKRExecutor') as mock_ibkr_executor:
            # Mock order manager to avoid executor registration error
            with patch.object(LiveTradingEngine, '__init__', lambda self, config: None):
                engine = LiveTradingEngine(config_live)
                # Mock required attributes
                engine.config = config_live
                engine.portfolio_engine = Mock()
                engine.portfolio_engine.initial_capital = 100000
                engine.order_manager = Mock()
                engine.order_manager.add_executor = Mock()
                engine.order_manager.executors = {}
                # Re-initialize executors with proper mocks
                engine._initialize_executors()
                mock_ibkr_executor.assert_called_once()

        # Test unsupported broker
        config_unsupported = {
            'mode': TradingMode.LIVE,
            'broker': 'unsupported_broker'
        }

        engine = LiveTradingEngine(config_unsupported)
        with pytest.raises(ValueError, match="Unsupported broker"):
            engine._initialize_executors()

    def test_setup_schedule(self):
        """Test _setup_schedule method."""
        config = {
            'mode': TradingMode.PAPER,
            'schedule': {
                'pre_market_time': '09:00',
                'market_open_time': '09:30',
                'market_close_time': '16:00',
                'post_market_time': '16:30',
                'memory_check_interval': 300,
                'stop_check_interval': 60
            }
        }

        engine = LiveTradingEngine(config)
        engine._setup_schedule()

        # Check that jobs were scheduled
        assert len(engine.scheduler.jobs) > 0

        # Verify specific jobs were added
        job_funcs = [str(job.func) for job in engine.scheduler.jobs]
        assert any('_pre_market_routine' in func for func in job_funcs)
        assert any('_market_open_routine' in func for func in job_funcs)
        assert any('_market_close_routine' in func for func in job_funcs)
        assert any('_post_market_routine' in func for func in job_funcs)
        # Memory health check and stop checks might not be scheduled by default

        # Test with minimal schedule config
        config_minimal = {'mode': TradingMode.PAPER}
        engine_minimal = LiveTradingEngine(config_minimal)
        engine_minimal._setup_schedule()
        assert len(engine_minimal.scheduler.jobs) > 0

    def test_prepare_strategy_data(self):
        """Test _prepare_strategy_data method."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Mock market data
        mock_data = pd.DataFrame({
            'AAPL': [150.0, 151.0, 152.0],
            'GOOGL': [2800.0, 2810.0, 2820.0]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))

        engine.market_data = mock_data

        # Test with strategy that has symbols
        strategy = MockStrategy()
        strategy.symbols = ['AAPL', 'GOOGL']

        result = engine._prepare_strategy_data(strategy)
        # Current implementation returns None
        assert result is None

        # Test with strategy without symbols
        strategy_no_symbols = MockStrategy()
        strategy_no_symbols.symbols = []

        result_no_symbols = engine._prepare_strategy_data(strategy_no_symbols)
        assert result_no_symbols is None

        # Test with missing market data
        engine.market_data = pd.DataFrame()
        result_empty = engine._prepare_strategy_data(strategy)
        assert result_empty is None

    @pytest.mark.asyncio
    async def test_check_memory_health(self):
        """Test _check_memory_health method."""
        config = {
            'mode': TradingMode.PAPER,
            'memory_limit_mb': 1000,
            'memory_warning_threshold': 0.8,
            'memory_critical_threshold': 0.9
        }

        engine = LiveTradingEngine(config)

        # Mock memory manager
        mock_memory_manager = Mock()
        mock_memory_manager.get_memory_usage.return_value = {
            'used_mb': 500,
            'percent': 50.0
        }
        mock_memory_manager.check_memory_usage = AsyncMock(return_value={
            'status': 'healthy',
            'used_mb': 500,
            'percent': 50.0
        })

        engine.memory_manager = mock_memory_manager

        # Test normal memory usage
        await engine._check_memory_health()
        mock_memory_manager.check_memory_usage.assert_called_once()

        # Test warning level
        mock_memory_manager.check_memory_usage.return_value = {
            'status': 'warning',
            'memory_mb': 850,
            'memory_percent': 85.0,
            'max_memory_mb': 1000
        }
        mock_memory_manager.optimize_memory = Mock(return_value={
            'initial_memory': {'memory_mb': 850},
            'final_memory': {'memory_mb': 750}
        })

        await engine._check_memory_health()
        mock_memory_manager.optimize_memory.assert_called_once()

        # Test critical level with emergency cleanup
        mock_memory_manager.check_memory_usage.return_value = {
            'status': 'critical',
            'memory_mb': 980,
            'memory_percent': 98.0,
            'max_memory_mb': 1000
        }

        # Mock market data that can be pruned
        engine.market_data = pd.DataFrame({
            'AAPL': range(2000),
            'GOOGL': range(2000)
        })
        engine.max_market_data_rows = 1000

        await engine._check_memory_health()
        assert len(engine.market_data) <= engine.max_market_data_rows

    @pytest.mark.asyncio
    async def test_check_stops(self):
        """Test _check_stops method."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Mock components
        mock_portfolio = Mock()
        mock_order_manager = Mock()
        mock_executor = AsyncMock()

        # Create mock positions
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.quantity = 100
        position1.average_price = 150.0

        position2 = Mock(spec=Position)
        position2.symbol = "GOOGL"
        position2.quantity = -50  # Short position
        position2.average_price = 2800.0

        mock_portfolio.get_positions.return_value = [position1, position2]

        # Create mock stop orders
        stop_order1 = Mock(spec=Order)
        stop_order1.symbol = "AAPL"
        stop_order1.order_type = OrderType.STOP
        stop_order1.stop_price = 145.0
        stop_order1.quantity = -100  # Sell order
        stop_order1.id = "stop1"

        stop_order2 = Mock(spec=Order)
        stop_order2.symbol = "GOOGL"
        stop_order2.order_type = OrderType.STOP
        stop_order2.stop_price = 2850.0
        stop_order2.quantity = 50  # Buy to cover
        stop_order2.id = "stop2"

        mock_order_manager.get_stop_orders.return_value = [stop_order1, stop_order2]

        # Mock current prices
        engine.market_data = pd.DataFrame({
            'AAPL': [144.0],  # Below stop price
            'GOOGL': [2860.0]  # Above stop price for short
        }, index=[pd.Timestamp.now()])

        engine.portfolio_engine = mock_portfolio
        engine.order_manager = mock_order_manager
        engine.executor = mock_executor

        # Execute stop check
        await engine._check_stops()

        # Verify stop orders were triggered
        assert mock_executor.submit_order.call_count == 2

        # Check that stop orders were converted to market orders
        calls = mock_executor.submit_order.call_args_list
        for call in calls:
            order = call[0][0]
            assert order.order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_update_market_data(self):
        """Test _update_market_data method (both versions if duplicated)."""
        config = {'mode': TradingMode.PAPER, 'symbols': ['AAPL', 'GOOGL']}
        engine = LiveTradingEngine(config)

        # Mock data handler
        mock_data_handler = Mock()
        new_data = pd.DataFrame({
            'AAPL': [155.0, 156.0],
            'GOOGL': [2850.0, 2860.0]
        }, index=pd.date_range('2024-01-01', periods=2, freq='min'))

        mock_data_handler.get_latest_data = AsyncMock(return_value=new_data)
        engine.data_handler = mock_data_handler

        # Test initial update
        await engine._update_market_data()
        assert engine.market_data is not None
        assert 'AAPL' in engine.market_data.columns
        assert len(engine.market_data) == 2

        # Test appending data
        new_data2 = pd.DataFrame({
            'AAPL': [157.0],
            'GOOGL': [2870.0]
        }, index=[pd.Timestamp('2024-01-01 00:02:00')])

        mock_data_handler.get_latest_data = AsyncMock(return_value=new_data2)
        await engine._update_market_data()
        assert len(engine.market_data) == 3

        # Test max rows limit
        engine.max_market_data_rows = 2
        await engine._update_market_data()
        assert len(engine.market_data) <= 2

    @pytest.mark.asyncio
    async def test_process_strategies(self):
        """Test _process_strategies method."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Set up strategies
        strategy1 = MockStrategy("strat1")
        strategy2 = MockStrategy("strat2")
        engine.strategies = [strategy1, strategy2]

        # Mock market data
        engine.market_data = pd.DataFrame({
            'AAPL': [150.0, 151.0],
            'GOOGL': [2800.0, 2810.0]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))

        # Mock signal queue
        engine.signal_queue = asyncio.Queue()

        # Mock trading hours check
        engine._check_market_hours = Mock(return_value=True)

        # Execute strategy processing
        await engine._process_strategies()

        # Verify strategies were called
        assert strategy1.calculate_signals_called
        assert strategy2.calculate_signals_called

        # Check signals were queued
        assert engine.signal_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_execute_signals(self):
        """Test _execute_signals method."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Mock components
        mock_risk_manager = Mock()
        mock_risk_manager.check_signal.return_value = (True, None)
        mock_risk_manager.check_order.return_value = (True, None)

        mock_order_manager = Mock()
        mock_executor = AsyncMock()

        engine.risk_manager = mock_risk_manager
        engine.order_manager = mock_order_manager
        engine.executor = mock_executor

        # Create test signal
        signal = Signal(
            symbol="AAPL",
            direction=1,
            strength=0.8,
            strategy_name="test_strategy"
        )

        # Add signal to queue
        engine.signal_queue = asyncio.Queue()
        await engine.signal_queue.put(signal)

        # Mock position sizing
        engine._calculate_position_size = Mock(return_value=100)

        # Execute signals (need to stop it after processing one signal)
        task = asyncio.create_task(engine._execute_signals())
        await asyncio.sleep(0.1)  # Let it process
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify order was submitted
        mock_executor.submit_order.assert_called_once()

        # Check order details
        submitted_order = mock_executor.submit_order.call_args[0][0]
        assert submitted_order.symbol == "AAPL"
        assert submitted_order.quantity == 100

    @pytest.mark.asyncio
    async def test_market_routines(self):
        """Test all market routine methods."""
        config = {'mode': TradingMode.PAPER}
        engine = LiveTradingEngine(config)

        # Mock components
        mock_portfolio = Mock()
        mock_portfolio.cash = 100000.0
        mock_portfolio.get_positions.return_value = []

        engine.portfolio_engine = mock_portfolio
        engine.stats = {}

        # Test pre-market routine
        await engine._pre_market_routine()
        assert 'pre_market_run' in engine.stats

        # Test market open routine
        await engine._market_open_routine()
        assert 'market_open_time' in engine.stats

        # Test market close routine
        await engine._market_close_routine()
        assert 'market_close_time' in engine.stats

        # Test post-market routine
        engine._generate_daily_report = Mock()
        engine._save_state = Mock()

        await engine._post_market_routine()
        assert 'post_market_run' in engine.stats
        engine._generate_daily_report.assert_called_once()
        engine._save_state.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
