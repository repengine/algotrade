"""
Core functionality tests for LiveTradingEngine without external dependencies.

This test file focuses on testing the actual implementation logic of LiveTradingEngine
by mocking external dependencies like apscheduler.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

# Import Signal before patching


# Mock the scheduler before importing live_engine
class MockScheduler:
    def __init__(self):
        self.jobs = []
        self.started = False

    def add_job(self, func, trigger, **kwargs):
        self.jobs.append({'func': func, 'trigger': trigger, 'kwargs': kwargs})

    def start(self):
        self.started = True

    def shutdown(self):
        self.started = False

    def get_jobs(self):
        return [Mock(id=job['kwargs'].get('id', 'unknown')) for job in self.jobs]


def test_imports():
    """Test that we can import with mocked dependencies."""
    with patch('apscheduler.schedulers.asyncio.AsyncIOScheduler', MockScheduler), \
         patch('apscheduler.triggers.cron.CronTrigger', Mock):
        from core.live_engine import LiveTradingEngine, TradingMode
        assert LiveTradingEngine is not None
        assert TradingMode is not None


class TestLiveTradingEngineCore:
    """Test core LiveTradingEngine functionality."""

    @pytest.fixture
    def mock_modules(self):
        """Mock all external modules."""
        with patch('apscheduler.schedulers.asyncio.AsyncIOScheduler', MockScheduler), \
             patch('apscheduler.triggers.cron.CronTrigger', Mock), \
             patch('core.data_handler.DataHandler') as mock_dh, \
             patch('core.portfolio.PortfolioEngine') as mock_pe, \
             patch('core.risk.EnhancedRiskManager') as mock_rm, \
             patch('core.engine.enhanced_order_manager.EnhancedOrderManager') as mock_om, \
             patch('core.metrics.MetricsCollector') as mock_mc, \
             patch('core.memory_manager.MemoryManager') as mock_mm, \
             patch('adapters.paper_executor.PaperExecutor') as mock_paper, \
             patch('adapters.ibkr_executor.IBKRExecutor') as mock_ibkr:

            # Configure mocks
            mock_pe.return_value.initial_capital = 100000
            mock_pe.return_value.current_equity = 100000
            mock_pe.return_value.get_position = Mock(return_value=None)
            mock_pe.return_value.get_daily_pnl = Mock(return_value=0)

            mock_om.return_value.executors = {}
            mock_om.return_value.add_executor = Mock()
            mock_om.return_value.set_active_executor = Mock()
            mock_om.return_value.register_event_callback = Mock()
            mock_om.return_value._orders = {}
            mock_om.return_value.create_order = AsyncMock()

            mock_rm.return_value.check_position_size = Mock(return_value=True)
            mock_rm.return_value.check_risk_limits = Mock(return_value=None)

            yield {
                'DataHandler': mock_dh,
                'PortfolioEngine': mock_pe,
                'RiskManager': mock_rm,
                'OrderManager': mock_om,
                'MetricsCollector': mock_mc,
                'MemoryManager': mock_mm,
                'PaperExecutor': mock_paper,
                'IBKRExecutor': mock_ibkr
            }

    def test_initialization_with_defaults(self, mock_modules):
        """Test engine initialization with default configuration."""
        # Import inside the patch context
        with patch('apscheduler.schedulers.asyncio.AsyncIOScheduler', MockScheduler), \
             patch('apscheduler.triggers.cron.CronTrigger', Mock):
            from core.live_engine import LiveTradingEngine, TradingMode

            # Arrange
            config = {
                'mode': TradingMode.PAPER,
                'strategies': []
            }

            # Act
            engine = LiveTradingEngine(config)

            # Assert
            assert engine.mode == TradingMode.PAPER
            assert engine.is_running is False
            assert engine.running is False
            assert engine.emergency_shutdown is False
            assert isinstance(engine.stats, dict)
            assert engine.stats['engine_start'] is None
            assert engine.stats['total_signals'] == 0
            assert engine.stats['errors'] == 0

            # Check components were initialized
            mock_modules['DataHandler'].assert_called_once()
            mock_modules['PortfolioEngine'].assert_called_once()
            mock_modules['RiskManager'].assert_called_once()
            mock_modules['OrderManager'].assert_called_once()
            mock_modules['MetricsCollector'].assert_called_once()
            mock_modules['MemoryManager'].assert_called_once()

    def test_strategy_initialization(self, mock_modules):
        """Test strategy initialization and symbol collection."""
        from core.live_engine import LiveTradingEngine, TradingMode

        # Mock strategy class
        mock_strategy_class = Mock()
        mock_strategy_instance = Mock()
        mock_strategy_instance.symbols = ['AAPL', 'GOOGL']
        mock_strategy_class.return_value = mock_strategy_instance

        # Arrange
        config = {
            'mode': TradingMode.PAPER,
            'strategies': [{
                'class': mock_strategy_class,
                'id': 'test_strategy',
                'params': {'param1': 'value1'}
            }]
        }

        # Act
        engine = LiveTradingEngine(config)

        # Assert
        assert 'test_strategy' in engine.strategies
        assert engine.strategies['test_strategy'] == mock_strategy_instance
        assert 'AAPL' in engine._active_symbols
        assert 'GOOGL' in engine._active_symbols
        mock_strategy_class.assert_called_once_with({'param1': 'value1'})

    def test_position_sizing_calculation(self, mock_modules):
        """Test position size calculation respects limits."""
        from core.live_engine import LiveTradingEngine, TradingMode
        from strategies.base import Signal

        # Arrange
        config = {
            'mode': TradingMode.PAPER,
            'portfolio_config': {
                'initial_capital': 100000,
                'max_position_size': 0.1  # 10% max
            }
        }

        engine = LiveTradingEngine(config)
        engine.current_prices['AAPL'] = 150.0

        signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.8,
            timestamp=datetime.now()
        )

        # Act
        position_size = engine._calculate_position_size(signal)

        # Assert
        # With 10% max and $100k capital at $150/share
        # Max position = $10k / $150 = 66.67 shares
        assert position_size >= 0
        assert position_size <= 67  # Should respect limit

    def test_signal_validation(self, mock_modules):
        """Test signal validation logic."""
        from core.live_engine import LiveTradingEngine, TradingMode
        from strategies.base import Signal

        # Arrange
        engine = LiveTradingEngine({'mode': TradingMode.PAPER})

        # Valid signal
        valid_signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.8,
            timestamp=datetime.now()
        )

        # Invalid signals
        no_direction = Signal('AAPL', direction=0, strength=0.8, timestamp=datetime.now())
        no_strength = Signal('AAPL', direction=1, strength=0, timestamp=datetime.now())
        bad_strength = Signal('AAPL', direction=1, strength=1.5, timestamp=datetime.now())

        # Act & Assert
        assert engine._is_valid_signal(valid_signal) is True
        assert engine._is_valid_signal(no_direction) is False
        assert engine._is_valid_signal(no_strength) is False
        assert engine._is_valid_signal(bad_strength) is False

    def test_should_trade_signal_logic(self, mock_modules):
        """Test signal trading decision logic."""
        from core.live_engine import LiveTradingEngine, TradingMode
        from strategies.base import Signal

        # Arrange
        engine = LiveTradingEngine({'mode': TradingMode.PAPER})
        engine.is_running = True
        engine.is_trading_hours = True

        signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.8,
            timestamp=datetime.now()
        )

        # Test 1: Normal conditions - should trade
        assert engine._should_trade_signal(signal) is True

        # Test 2: Not running - shouldn't trade
        engine.is_running = False
        assert engine._should_trade_signal(signal) is False

        # Test 3: Outside trading hours - shouldn't trade
        engine.is_running = True
        engine.is_trading_hours = False
        assert engine._should_trade_signal(signal) is False

        # Test 4: Risk check fails - shouldn't trade
        engine.is_trading_hours = True
        mock_modules['RiskManager'].return_value.check_position_size.return_value = False
        assert engine._should_trade_signal(signal) is False

    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self, mock_modules):
        """Test emergency stop cancels orders and sets flags."""
        from core.live_engine import LiveTradingEngine, TradingMode

        # Arrange
        engine = LiveTradingEngine({'mode': TradingMode.PAPER})
        engine.is_running = True
        engine.running = True

        # Mock executor
        mock_executor = Mock()
        mock_executor.cancel_all_orders = AsyncMock(return_value=True)
        mock_executor.close_all_positions = AsyncMock(return_value=True)
        engine.executor = mock_executor
        engine.order_manager.executors['paper'] = mock_executor

        # Act
        await engine.emergency_stop()

        # Assert
        assert engine.emergency_shutdown is True
        assert engine.is_running is False
        assert engine.running is False
        mock_executor.cancel_all_orders.assert_called_once()
        mock_executor.close_all_positions.assert_called_once()

    def test_handle_order_event_updates_stats(self, mock_modules):
        """Test order event handling updates statistics."""
        from core.live_engine import LiveTradingEngine, TradingMode

        # Arrange
        engine = LiveTradingEngine({'mode': TradingMode.PAPER})

        # Create fill event
        fill_event = {
            'event_type': 'FILL',
            'order_id': 'order123',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'timestamp': datetime.now()
        }

        # Act
        engine._handle_order_event(fill_event)

        # Assert
        assert engine.stats['total_fills'] == 1
        assert 'AAPL' in engine._last_prices
        assert engine._last_prices['AAPL'] == 150.0

    @pytest.mark.asyncio
    async def test_market_data_update(self, mock_modules):
        """Test market data update functionality."""
        from core.live_engine import LiveTradingEngine, TradingMode

        # Arrange
        engine = LiveTradingEngine({'mode': TradingMode.PAPER})
        engine._active_symbols = {'AAPL', 'GOOGL'}

        # Mock data
        mock_data = pd.DataFrame({
            'close': [150, 151, 152],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(end=datetime.now(), periods=3, freq='1min'))

        mock_modules['DataHandler'].return_value.get_latest_bars = Mock(return_value=mock_data)

        # Act
        await engine._update_market_data()

        # Assert
        assert 'AAPL' in engine.market_data
        assert 'GOOGL' in engine.market_data
        assert engine.current_prices['AAPL'] == 152  # Latest close
        assert engine.current_prices['GOOGL'] == 152
        assert engine.stats['data_updates'] == 2  # One for each symbol

    @pytest.mark.asyncio
    async def test_risk_limit_check(self, mock_modules):
        """Test risk limit checking triggers appropriate actions."""
        from core.live_engine import LiveTradingEngine, TradingMode

        # Arrange
        engine = LiveTradingEngine({'mode': TradingMode.PAPER})
        engine.is_running = True

        # Mock risk violation
        mock_modules['RiskManager'].return_value.check_risk_limits.return_value = {
            'daily_loss_exceeded': True,
            'current_loss': -0.03,
            'limit': -0.02
        }

        # Mock emergency liquidation
        engine._emergency_liquidation = AsyncMock()

        # Act
        await engine._check_risk_limits()

        # Assert
        engine._emergency_liquidation.assert_called_once()
        violation = engine._emergency_liquidation.call_args[0][0]
        assert violation['daily_loss_exceeded'] is True

    def test_scheduler_setup(self, mock_modules):
        """Test scheduler is properly configured."""
        from core.live_engine import LiveTradingEngine, TradingMode

        # Arrange
        config = {
            'mode': TradingMode.PAPER,
            'schedule': {
                'pre_market': '09:00',
                'market_open': '09:30',
                'market_close': '16:00',
                'post_market': '16:30'
            }
        }

        # Act
        engine = LiveTradingEngine(config)

        # Assert
        jobs = engine.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert 'pre_market_routine' in job_ids
        assert 'market_open_routine' in job_ids
        assert 'market_close_routine' in job_ids
        assert 'post_market_routine' in job_ids
        assert 'update_market_data' in job_ids
        assert 'check_risk_limits' in job_ids
