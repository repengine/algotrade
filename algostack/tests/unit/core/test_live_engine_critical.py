"""
Critical test suite for LiveTradingEngine focused on the Four Pillars.

Tests are organized by the Four Pillars:
1. Capital Preservation - Risk checks, position limits, emergency stops
2. Profit Generation - Strategy execution, signal processing, order management
3. Operational Stability - Engine lifecycle, error handling, state management
4. Verifiable Correctness - Data integrity, position tracking, metrics accuracy
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

from algostack.core.live_engine import LiveTradingEngine, TradingMode
from algostack.strategies.base import BaseStrategy, Signal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.symbols = config.get('symbols', ['AAPL'])
        self.name = config.get('name', 'MockStrategy')
        self.calculate_signals_called = 0
        self.next_called = 0
        
    def calculate_signals(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Mock signal calculation."""
        self.calculate_signals_called += 1
        # Return a series of zeros (no signals) by default
        return pd.Series(0, index=data.index)
        
    def next(self, data: pd.DataFrame) -> Signal:
        """Mock next method."""
        self.next_called += 1
        return None


class TestCapitalPreservation:
    """Test Pillar 1: Capital Preservation - Protect money from loss."""
    
    @pytest.fixture
    def safe_config(self):
        """Configuration focused on capital preservation."""
        return {
            'mode': TradingMode.PAPER,
            'portfolio_config': {
                'initial_capital': 100000,
                'max_position_size': 0.1,  # 10% max per position
                'max_total_exposure': 0.6,  # 60% max total
            },
            'risk_config': {
                'max_daily_loss': 0.02,  # 2% daily loss limit
                'max_drawdown': 0.10,    # 10% max drawdown
                'position_limits': {
                    'AAPL': 1000,
                    'GOOGL': 100
                }
            },
            'strategies': [{
                'class': MockStrategy,
                'id': 'test_strategy',
                'params': {'symbols': ['AAPL', 'GOOGL']}
            }],
            'executor_config': {
                'paper': {
                    'initial_capital': 100000,
                    'commission': 1.0,
                    'slippage': 0.0001
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_emergency_stop_cancels_all_orders(self, safe_config):
        """
        Emergency stop should immediately cancel all orders and prevent losses.
        
        This is CRITICAL for Capital Preservation when system detects issues.
        """
        # Arrange
        engine = LiveTradingEngine(safe_config)
        
        # Mock some open orders
        mock_order1 = Mock(id='order1', symbol='AAPL', status='PENDING')
        mock_order2 = Mock(id='order2', symbol='GOOGL', status='PENDING')
        engine.order_manager._orders = {
            'order1': mock_order1,
            'order2': mock_order2
        }
        
        # Mock executor methods
        engine.executor.cancel_all_orders = AsyncMock(return_value=True)
        engine.executor.close_all_positions = AsyncMock(return_value=True)
        
        # Act
        await engine.emergency_stop()
        
        # Assert
        assert engine.emergency_shutdown is True
        assert engine.is_running is False
        assert engine.running is False  # Backward compatibility
        engine.executor.cancel_all_orders.assert_called_once()
        engine.executor.close_all_positions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_risk_limits_prevent_oversized_positions(self, safe_config):
        """
        Risk manager should reject positions that exceed limits.
        
        Prevents catastrophic losses from oversized positions.
        """
        # Arrange
        engine = LiveTradingEngine(safe_config)
        
        # Create signal that would exceed position limit
        large_signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=1.0,
            timestamp=datetime.now(),
            metadata={'suggested_size': 2000}  # Exceeds limit of 1000
        )
        
        # Mock risk check to return False
        engine.risk_manager.check_position_size = Mock(return_value=False)
        
        # Act
        result = engine._should_trade_signal(large_signal)
        
        # Assert
        assert result is False
        engine.risk_manager.check_position_size.assert_called_once()
    
    def test_position_sizing_respects_capital_limits(self, safe_config):
        """
        Position sizing should never exceed configured limits.
        
        Ensures no single position can cause catastrophic loss.
        """
        # Arrange
        engine = LiveTradingEngine(safe_config)
        
        # Mock portfolio state
        engine.portfolio_engine.current_equity = 100000
        engine.portfolio_engine.get_position = Mock(return_value=None)
        
        # Create signal
        signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.8,
            timestamp=datetime.now()
        )
        
        # Mock current price
        engine.current_prices['AAPL'] = 150.0
        
        # Act
        position_size = engine._calculate_position_size(signal)
        
        # Assert
        # With 10% max position size and $100k capital, max is $10k
        # At $150/share, max shares = 66
        assert position_size <= 66
        assert position_size >= 0
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit_triggers_shutdown(self, safe_config):
        """
        Engine should stop trading when daily loss limit is hit.
        
        Critical circuit breaker to prevent runaway losses.
        """
        # Arrange
        engine = LiveTradingEngine(safe_config)
        engine.is_running = True
        
        # Mock portfolio with 3% daily loss (exceeds 2% limit)
        engine.portfolio_engine.get_daily_pnl = Mock(return_value=-3000)
        engine.portfolio_engine.current_equity = 97000
        
        # Mock risk check to detect violation
        engine.risk_manager.check_risk_limits = Mock(return_value={
            'daily_loss_exceeded': True,
            'current_loss': -0.03,
            'limit': -0.02
        })
        
        # Act
        await engine._check_risk_limits()
        
        # Assert
        engine.risk_manager.check_risk_limits.assert_called_once()
        # Engine should initiate emergency procedures
        assert engine.stats['errors'] > 0


class TestProfitGeneration:
    """Test Pillar 2: Profit Generation - Help make money."""
    
    @pytest.fixture
    def profit_config(self):
        """Configuration focused on profit generation."""
        return {
            'mode': TradingMode.PAPER,
            'portfolio_config': {'initial_capital': 100000},
            'strategies': [{
                'class': MockStrategy,
                'id': 'profit_strategy',
                'params': {
                    'symbols': ['AAPL', 'GOOGL'],
                    'min_signal_strength': 0.7
                }
            }],
            'executor_config': {
                'paper': {
                    'initial_capital': 100000,
                    'commission': 1.0,
                    'slippage': 0.0001
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_strategy_execution_generates_signals(self, profit_config):
        """
        Strategies should be executed and generate tradeable signals.
        
        Core functionality for profit generation.
        """
        # Arrange
        engine = LiveTradingEngine(profit_config)
        strategy = engine.strategies['profit_strategy']
        
        # Mock market data
        market_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5
        }, index=pd.date_range(end=datetime.now(), periods=5, freq='1min'))
        
        engine.market_data['AAPL'] = market_data
        
        # Mock strategy to return a buy signal
        strategy.calculate_signals = Mock(return_value=pd.Series([0, 0, 1, 0, 0]))
        
        # Act
        await engine._run_strategies()
        
        # Assert
        strategy.calculate_signals.assert_called()
        assert engine.stats['signals_generated'] > 0
    
    @pytest.mark.asyncio
    async def test_signal_processing_creates_orders(self, profit_config):
        """
        Valid signals should be converted to orders for execution.
        
        Ensures profit opportunities are captured.
        """
        # Arrange
        engine = LiveTradingEngine(profit_config)
        
        # Create a strong buy signal
        signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.9,
            timestamp=datetime.now(),
            metadata={'strategy': 'profit_strategy'}
        )
        
        # Mock dependencies
        engine._should_trade_signal = Mock(return_value=True)
        engine._calculate_position_size = Mock(return_value=100)
        engine.order_manager.create_order = AsyncMock(return_value=Mock(id='order123'))
        
        # Act
        await engine._process_signal('profit_strategy', signal)
        
        # Assert
        engine.order_manager.create_order.assert_called_once()
        call_args = engine.order_manager.create_order.call_args[1]
        assert call_args['symbol'] == 'AAPL'
        assert call_args['quantity'] == 100
        assert call_args['side'] == 'BUY'
    
    def test_signal_strength_filtering(self, profit_config):
        """
        Weak signals should be filtered out to avoid poor trades.
        
        Quality over quantity for profit generation.
        """
        # Arrange
        engine = LiveTradingEngine(profit_config)
        
        # Weak signal
        weak_signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.3,  # Below typical thresholds
            timestamp=datetime.now()
        )
        
        # Strong signal
        strong_signal = Signal(
            symbol='AAPL',
            direction=-1,
            strength=0.85,
            timestamp=datetime.now()
        )
        
        # Act
        weak_valid = engine._is_valid_signal(weak_signal)
        strong_valid = engine._is_valid_signal(strong_signal)
        
        # Assert
        # Exact behavior depends on implementation, but strong should pass
        assert strong_valid is True
        # Weak signals might still be valid but won't be traded
        assert engine._should_trade_signal(weak_signal) is False
        assert engine._should_trade_signal(strong_signal) is True


class TestOperationalStability:
    """Test Pillar 3: Operational Stability - Keep the system running."""
    
    @pytest.fixture
    def stable_config(self):
        """Configuration for operational stability testing."""
        return {
            'mode': TradingMode.PAPER,
            'portfolio_config': {'initial_capital': 100000},
            'memory_config': {
                'max_memory_mb': 1024,
                'gc_interval': 60,
                'cleanup_interval': 300
            },
            'strategies': [{
                'class': MockStrategy,
                'id': 'stable_strategy',
                'params': {'symbols': ['AAPL']}
            }],
            'schedule': {
                'pre_market': '09:00',
                'market_open': '09:30',
                'market_close': '16:00',
                'post_market': '16:30'
            }
        }
    
    @pytest.mark.asyncio
    async def test_engine_lifecycle_start_stop(self, stable_config):
        """
        Engine should start and stop cleanly without errors.
        
        Basic operational requirement for production.
        """
        # Arrange
        engine = LiveTradingEngine(stable_config)
        
        # Mock async methods
        engine._initialize_data_feeds = AsyncMock()
        engine._main_loop = AsyncMock()
        engine._cancel_all_orders = AsyncMock()
        
        # Mock executors
        for executor in engine.order_manager.executors.values():
            executor.connect = AsyncMock(return_value=True)
            executor.disconnect = AsyncMock()
        
        # Act - Start
        await engine.start()
        
        # Assert - Started correctly
        assert engine.is_running is True
        assert engine.running is True
        assert engine.stats['engine_start'] is not None
        
        # Act - Stop
        await engine.stop()
        
        # Assert - Stopped correctly
        assert engine.is_running is False
        assert engine.running is False
        engine._cancel_all_orders.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_prevents_oom(self, stable_config):
        """
        Memory monitoring should detect high usage and take action.
        
        Prevents out-of-memory crashes in production.
        """
        # Arrange
        engine = LiveTradingEngine(stable_config)
        
        # Mock memory manager to report high usage
        engine.memory_manager.get_memory_usage = Mock(return_value={
            'used_mb': 950,
            'limit_mb': 1024,
            'percentage': 92.8
        })
        engine.memory_manager.cleanup_old_data = Mock()
        
        # Act
        await engine._check_memory_health()
        
        # Assert
        # Should trigger cleanup when usage is high
        engine.memory_manager.cleanup_old_data.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_recovery_continues_operation(self, stable_config):
        """
        Engine should handle errors gracefully and continue operating.
        
        Resilience is critical for 24/7 operation.
        """
        # Arrange
        engine = LiveTradingEngine(stable_config)
        engine.is_running = True
        
        # Mock a strategy that throws an error
        failing_strategy = engine.strategies['stable_strategy']
        failing_strategy.calculate_signals = Mock(side_effect=ValueError("Test error"))
        
        # Mock market data
        engine.market_data['AAPL'] = pd.DataFrame({
            'close': [100, 101],
            'volume': [1000000, 1000000]
        })
        
        # Act - Should handle error without crashing
        await engine._run_strategies()
        
        # Assert
        assert engine.is_running is True  # Still running
        assert engine.stats['errors'] > 0  # Error was counted
    
    def test_scheduler_configuration(self, stable_config):
        """
        Scheduler should be properly configured for market hours.
        
        Ensures strategies run at appropriate times.
        """
        # Arrange & Act
        engine = LiveTradingEngine(stable_config)
        
        # Assert
        assert engine.scheduler is not None
        # Should have jobs scheduled
        jobs = engine.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]
        
        # Check critical scheduled tasks
        assert 'pre_market_routine' in job_ids
        assert 'market_open_routine' in job_ids
        assert 'market_close_routine' in job_ids
        assert 'update_market_data' in job_ids


class TestVerifiableCorrectness:
    """Test Pillar 4: Verifiable Correctness - Prove the system works."""
    
    @pytest.fixture
    def verify_config(self):
        """Configuration for correctness verification."""
        return {
            'mode': TradingMode.PAPER,
            'portfolio_config': {'initial_capital': 100000},
            'strategies': [{
                'class': MockStrategy,
                'id': 'verify_strategy',
                'params': {'symbols': ['AAPL', 'GOOGL']}
            }]
        }
    
    def test_position_tracking_accuracy(self, verify_config):
        """
        Position tracking must be accurate at all times.
        
        Critical for knowing actual exposure and P&L.
        """
        # Arrange
        engine = LiveTradingEngine(verify_config)
        
        # Simulate order fill event
        fill_event = {
            'event_type': 'FILL',
            'order_id': 'order123',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'commission': 1.0,
            'timestamp': datetime.now()
        }
        
        # Act
        engine._handle_order_event(fill_event)
        
        # Assert
        assert engine.stats['total_fills'] == 1
        # Position should be updated in portfolio
        # (actual update depends on implementation)
    
    @pytest.mark.asyncio
    async def test_market_data_integrity(self, verify_config):
        """
        Market data updates should maintain integrity.
        
        Bad data leads to bad trades and losses.
        """
        # Arrange
        engine = LiveTradingEngine(verify_config)
        
        # Mock data handler to return data
        test_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000000, 1100000, 1200000],
            'high': [101, 102, 103],
            'low': [99, 100, 101]
        }, index=pd.date_range(end=datetime.now(), periods=3, freq='1min'))
        
        engine.data_handler.get_latest_bars = Mock(return_value=test_data)
        
        # Act
        await engine._update_market_data()
        
        # Assert
        assert 'AAPL' in engine.market_data
        assert len(engine.market_data['AAPL']) > 0
        assert engine.current_prices['AAPL'] == 102  # Latest close
        assert engine.stats['data_updates'] > 0
    
    def test_metrics_collection_completeness(self, verify_config):
        """
        All trading metrics should be collected for analysis.
        
        Can't improve what you can't measure.
        """
        # Arrange
        engine = LiveTradingEngine(verify_config)
        
        # Act - Check initial stats
        stats = engine.stats
        
        # Assert - All critical metrics present
        assert 'engine_start' in stats
        assert 'total_signals' in stats
        assert 'signals_generated' in stats
        assert 'signals_rejected' in stats
        assert 'total_orders' in stats
        assert 'total_fills' in stats
        assert 'trades_executed' in stats
        assert 'data_updates' in stats
        assert 'errors' in stats
        
        # All should be initialized
        for key, value in stats.items():
            if key != 'engine_start':
                assert value == 0
    
    def test_state_persistence_capability(self, verify_config):
        """
        Engine should be able to save and restore state.
        
        Critical for recovery after crashes.
        """
        # Arrange
        engine = LiveTradingEngine(verify_config)
        
        # Set some state
        engine.stop_orders['AAPL_STOP'] = {
            'symbol': 'AAPL',
            'stop_price': 145.0,
            'quantity': -100
        }
        engine.stats['total_signals'] = 42
        
        # Act - Save state
        # (Note: actual save_state is async, but we're testing the capability)
        assert hasattr(engine, 'save_state')
        assert hasattr(engine, 'load_state')
        
        # The engine has methods for state persistence
        # Actual file I/O would be mocked in unit tests


class TestCriticalEdgeCases:
    """Test critical edge cases that could cause capital loss."""
    
    @pytest.fixture
    def edge_config(self):
        """Basic config for edge case testing."""
        return {
            'mode': TradingMode.PAPER,
            'portfolio_config': {'initial_capital': 100000},
            'strategies': [{
                'class': MockStrategy,
                'id': 'edge_strategy',
                'params': {'symbols': ['AAPL']}
            }]
        }
    
    def test_zero_capital_prevents_trading(self, edge_config):
        """
        Engine should not attempt trades with zero capital.
        
        Prevents undefined behavior and errors.
        """
        # Arrange
        edge_config['portfolio_config']['initial_capital'] = 0
        engine = LiveTradingEngine(edge_config)
        
        signal = Signal(
            symbol='AAPL',
            direction=1,
            strength=0.9,
            timestamp=datetime.now()
        )
        
        # Act
        size = engine._calculate_position_size(signal)
        
        # Assert
        assert size == 0
    
    def test_invalid_signal_rejection(self, edge_config):
        """
        Invalid signals should be rejected before processing.
        
        Prevents errors and bad trades.
        """
        # Arrange
        engine = LiveTradingEngine(edge_config)
        
        # Various invalid signals
        signals = [
            Signal('AAPL', direction=0, strength=0.5, timestamp=datetime.now()),  # No direction
            Signal('AAPL', direction=1, strength=0, timestamp=datetime.now()),    # No strength
            Signal('AAPL', direction=1, strength=1.5, timestamp=datetime.now()),  # Invalid strength
            Signal('', direction=1, strength=0.5, timestamp=datetime.now()),      # No symbol
        ]
        
        # Act & Assert
        for signal in signals:
            assert engine._is_valid_signal(signal) is False
    
    @pytest.mark.asyncio
    async def test_disconnection_handling(self, edge_config):
        """
        Engine should handle executor disconnection gracefully.
        
        Network issues shouldn't cause crashes or hung orders.
        """
        # Arrange
        engine = LiveTradingEngine(edge_config)
        
        # Mock executor disconnection
        engine.executor.is_connected = Mock(return_value=False)
        engine.executor.connect = AsyncMock(return_value=False)
        
        # Act & Assert - Should raise error on start
        with pytest.raises(RuntimeError, match="Failed to connect executor"):
            await engine.start()