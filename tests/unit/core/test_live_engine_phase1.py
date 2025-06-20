"""Test suite for Phase 1 LiveTradingEngine implementations."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from core.live_engine import LiveTradingEngine
from strategies.base import Signal


class TestPhase1LiveTradingEngine:
    """Test suite for Phase 1 critical safety features."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'mode': 'PAPER',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'strategies': [],
            'risk_config': {
                'max_position_size': 0.20,
                'max_portfolio_risk': 0.06,
                'stop_loss_pct': 0.02
            },
            'max_price_change_pct': 0.10,  # 10% circuit breaker
            'max_data_age_seconds': 30,
            'signal_timeout_ms': 100,
        }

    @pytest.fixture
    def live_engine(self, config):
        """Create LiveTradingEngine instance with mocked dependencies."""
        with patch('core.live_engine.PortfolioEngine'):
            with patch('core.live_engine.RiskManager'):
                with patch('core.live_engine.DataHandler'):
                    with patch('core.live_engine.EnhancedOrderManager'):
                        with patch('core.live_engine.MetricsCollector'):
                            with patch('core.live_engine.PaperExecutor'):
                                engine = LiveTradingEngine(config)
                                # Initialize some test data
                                engine._active_symbols = {'AAPL', 'GOOGL', 'MSFT'}
                                engine._last_prices = {'AAPL': 150.0, 'GOOGL': 2800.0}
                                return engine

    @pytest.mark.asyncio
    async def test_process_market_data_valid(self, live_engine):
        """Test process_market_data with valid data."""
        market_data = {
            'AAPL': {
                'close': 151.0,
                'volume': 1000000,
                'timestamp': datetime.now()
            },
            'GOOGL': pd.DataFrame({
                'close': [2800, 2805, 2810],
                'volume': [50000, 51000, 52000]
            }, index=pd.date_range(end=datetime.now(), periods=3, freq='1min'))
        }

        # Mock data handler update
        live_engine.data_handler.update_data = AsyncMock()

        await live_engine.process_market_data(market_data)

        # Check prices were updated
        assert live_engine.current_prices['AAPL'] == 151.0
        assert live_engine.current_prices['GOOGL'] == 2810.0
        assert live_engine._last_prices['AAPL'] == 151.0
        assert live_engine._last_prices['GOOGL'] == 2810.0

        # Check data handler was called
        live_engine.data_handler.update_data.assert_called_once()

        # Check stats
        assert live_engine.stats['data_updates'] == 1

    @pytest.mark.asyncio
    async def test_process_market_data_stale(self, live_engine):
        """Test process_market_data rejects stale data."""
        market_data = {
            'AAPL': {
                'close': 151.0,
                'volume': 1000000,
                'timestamp': datetime.now() - timedelta(minutes=5)  # 5 minutes old
            }
        }

        await live_engine.process_market_data(market_data)

        # Check price was not updated
        assert 'AAPL' not in live_engine.current_prices
        assert live_engine.stats.get('stale_data_rejected', 0) == 1

    @pytest.mark.asyncio
    async def test_process_market_data_circuit_breaker(self, live_engine):
        """Test process_market_data circuit breaker."""
        market_data = {
            'AAPL': {
                'close': 200.0,  # 33% jump from 150
                'volume': 1000000,
                'timestamp': datetime.now()
            }
        }

        live_engine.data_handler.update_data = AsyncMock()

        await live_engine.process_market_data(market_data)

        # Price should be updated but circuit breaker triggered
        assert live_engine.current_prices['AAPL'] == 200.0
        assert live_engine.stats.get('circuit_breaker_triggered', 0) == 1

        # Check the data was flagged
        call_args = live_engine.data_handler.update_data.call_args[0][0]
        assert 'circuit_breaker' in call_args['AAPL']

    @pytest.mark.asyncio
    async def test_process_market_data_invalid_price(self, live_engine):
        """Test process_market_data rejects invalid prices."""
        market_data = {
            'AAPL': {
                'close': -10.0,  # Invalid negative price
                'volume': 1000000,
                'timestamp': datetime.now()
            },
            'GOOGL': {
                'close': 0.0,  # Invalid zero price
                'volume': 50000,
                'timestamp': datetime.now()
            }
        }

        await live_engine.process_market_data(market_data)

        # Check prices were not updated
        assert 'AAPL' not in live_engine.current_prices
        assert 'GOOGL' not in live_engine.current_prices
        assert live_engine.stats.get('invalid_price_rejected', 0) == 2

    @pytest.mark.asyncio
    async def test_collect_signals_single_strategy(self, live_engine):
        """Test collect_signals with a single strategy."""
        # Create mock strategy
        mock_strategy = Mock()
        mock_strategy.enabled = True
        mock_strategy.symbols = ['AAPL']

        # Mock generate_signals as an async function
        async def mock_generate_signals(data):
            return [Signal(
                symbol='AAPL',
                direction='LONG',
                strength=0.8,
                timestamp=datetime.now(),
                strategy_id='test_strategy',
                price=151.0
            )]

        mock_strategy.generate_signals = mock_generate_signals
        live_engine.strategies = {'test_strategy': mock_strategy}
        live_engine.market_data = {'AAPL': pd.DataFrame({'close': [150, 151]})}

        signals = await live_engine.collect_signals()

        assert len(signals) == 1
        assert signals[0].symbol == 'AAPL'
        assert signals[0].direction == 'LONG'
        assert signals[0].strength == 0.8
        assert live_engine.stats['total_signals'] == 1

    @pytest.mark.asyncio
    async def test_collect_signals_multiple_strategies(self, live_engine):
        """Test collect_signals with multiple strategies."""
        # Create mock strategies
        strategies = {}
        for i in range(3):
            strategy = Mock()
            strategy.enabled = True
            strategy.symbols = ['AAPL']

            # Each strategy returns a signal with different strength
            def make_generate_signals(strength):
                async def generate_signals(data):
                    return [Signal(
                        symbol='AAPL',
                        direction='LONG',
                        strength=strength,
                        timestamp=datetime.now(),
                        strategy_id=f'strategy_{strength}',
                        price=151.0
                    )]
                return generate_signals

            strategy.generate_signals = make_generate_signals(0.5 + i * 0.1)
            strategies[f'strategy_{i}'] = strategy

        live_engine.strategies = strategies
        live_engine.market_data = {'AAPL': pd.DataFrame({'close': [150, 151]})}

        signals = await live_engine.collect_signals()

        # Should have one aggregated signal
        assert len(signals) == 1
        assert signals[0].symbol == 'AAPL'
        assert signals[0].metadata['aggregated'] is True
        assert signals[0].metadata['source_count'] == 3
        # Average strength should be (0.5 + 0.6 + 0.7) / 3 = 0.6
        assert abs(signals[0].strength - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_collect_signals_timeout(self, live_engine):
        """Test collect_signals handles strategy timeout."""
        # Create slow strategy
        mock_strategy = Mock()
        mock_strategy.enabled = True
        mock_strategy.symbols = ['AAPL']

        async def slow_generate_signals(data):
            await asyncio.sleep(1)  # Sleep longer than timeout
            return [Signal(symbol='AAPL', direction='LONG', strength=0.8,
                          timestamp=datetime.now(), strategy_id='slow_strategy', price=151.0)]

        mock_strategy.generate_signals = slow_generate_signals
        live_engine.strategies = {'slow_strategy': mock_strategy}
        live_engine.market_data = {'AAPL': pd.DataFrame({'close': [150, 151]})}

        signals = await live_engine.collect_signals()

        assert len(signals) == 0
        assert live_engine.stats.get('strategy_timeouts', {}).get('slow_strategy', 0) == 1

    @pytest.mark.asyncio
    async def test_collect_signals_deduplication(self, live_engine):
        """Test signal deduplication for conflicting directions."""
        # Create strategies with conflicting signals
        buy_strategy = Mock()
        buy_strategy.enabled = True
        buy_strategy.symbols = ['AAPL']

        async def buy_signals(data):
            return [Signal(symbol='AAPL', direction='LONG', strength=0.8,
                          timestamp=datetime.now(), strategy_id='buy_strategy', price=151.0)]

        buy_strategy.generate_signals = buy_signals

        sell_strategy = Mock()
        sell_strategy.enabled = True
        sell_strategy.symbols = ['AAPL']

        async def sell_signals(data):
            return [Signal(symbol='AAPL', direction='SHORT', strength=-0.6,
                          timestamp=datetime.now(), strategy_id='sell_strategy', price=151.0)]

        sell_strategy.generate_signals = sell_signals

        live_engine.strategies = {
            'buy_strategy': buy_strategy,
            'sell_strategy': sell_strategy
        }
        live_engine.market_data = {'AAPL': pd.DataFrame({'close': [150, 151]})}

        signals = await live_engine.collect_signals()

        # Should have two signals (one buy, one sell)
        assert len(signals) == 2
        directions = [s.direction for s in signals]
        assert 'LONG' in directions and 'SHORT' in directions

    @pytest.mark.asyncio
    async def test_update_portfolio_reconciliation(self, live_engine):
        """Test _update_portfolio with position reconciliation."""
        # Mock broker positions
        broker_positions = {
            'AAPL': Mock(quantity=100, average_cost=150, current_price=155),
            'GOOGL': Mock(quantity=50, average_cost=2800, current_price=2850)
        }
        live_engine.order_manager.get_positions = AsyncMock(return_value=broker_positions)

        # Mock internal positions (with discrepancy)
        internal_positions = {
            'AAPL': Mock(quantity=95, current_price=150),  # Quantity mismatch
            'MSFT': Mock(quantity=30, current_price=350)   # Not in broker
        }
        live_engine.portfolio_engine.positions = internal_positions
        live_engine.portfolio_engine.update_position = Mock()

        # Remove reconcile_positions to force manual reconciliation
        if hasattr(live_engine.portfolio_engine, 'reconcile_positions'):
            delattr(live_engine.portfolio_engine, 'reconcile_positions')

        # Set current prices
        live_engine.current_prices = {'AAPL': 155, 'GOOGL': 2850, 'MSFT': 350}

        await live_engine._update_portfolio()

        # Check reconciliation happened
        assert live_engine.portfolio_engine.update_position.call_count >= 2

        # Check stats updated
        assert live_engine.stats.get('portfolio_updates', 0) == 1

    @pytest.mark.asyncio
    async def test_update_portfolio_atomic_locking(self, live_engine):
        """Test _update_portfolio uses atomic locking."""
        # Mock slow broker call
        async def slow_get_positions():
            await asyncio.sleep(0.1)
            return {}

        live_engine.order_manager.get_positions = slow_get_positions

        # Run multiple updates concurrently
        tasks = [live_engine._update_portfolio() for _ in range(5)]
        await asyncio.gather(*tasks)

        # All should complete without error
        assert live_engine.stats.get('portfolio_updates', 0) == 5

    @pytest.mark.asyncio
    async def test_update_portfolio_error_handling(self, live_engine):
        """Test _update_portfolio handles errors gracefully."""
        # Mock broker error
        live_engine.order_manager.get_positions = AsyncMock(
            side_effect=Exception("Broker connection error")
        )

        # Should not raise
        await live_engine._update_portfolio()

        # Check error was logged
        assert live_engine.stats.get('errors', 0) == 1

    @pytest.mark.asyncio
    async def test_signal_validation(self, live_engine):
        """Test signal validation logic."""
        # Valid signal
        valid_signal = Signal(
            symbol='AAPL',
            direction='LONG',
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id='test',
            price=151.0
        )
        assert live_engine._validate_signal(valid_signal) is True

        # Invalid symbol
        invalid_symbol = Signal(
            symbol='INVALID',
            direction='LONG',
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id='test',
            price=151.0
        )
        assert live_engine._validate_signal(invalid_symbol) is False

        # Test zero strength for non-FLAT signal
        zero_strength = Signal(
            symbol='AAPL',
            direction='FLAT',
            strength=0,  # Valid for FLAT
            timestamp=datetime.now(),
            strategy_id='test',
            price=151.0
        )
        assert live_engine._validate_signal(zero_strength) is True

        # Stale signal
        stale_signal = Signal(
            symbol='AAPL',
            direction='LONG',
            strength=0.8,
            timestamp=datetime.now() - timedelta(seconds=10),  # 10 seconds old
            strategy_id='test',
            price=151.0
        )
        assert live_engine._validate_signal(stale_signal) is False

    @pytest.mark.asyncio
    async def test_integration_data_to_signals(self, live_engine):
        """Test full integration from market data to signals."""
        # Setup strategy
        mock_strategy = Mock()
        mock_strategy.enabled = True
        mock_strategy.symbols = ['AAPL']

        async def generate_signals_from_data(data):
            # Strategy that generates signal based on price increase
            if 'AAPL' in data and len(data['AAPL']) > 0:
                return [Signal(
                    symbol='AAPL',
                    direction='LONG',
                    strength=0.7,
                    timestamp=datetime.now(),
                    strategy_id='momentum',
                    price=154.0
                )]
            return []

        mock_strategy.generate_signals = generate_signals_from_data
        # Add mock for on_market_data if it gets called
        mock_strategy.on_market_data = AsyncMock()
        live_engine.strategies = {'momentum': mock_strategy}

        # Process market data
        market_data = {
            'AAPL': pd.DataFrame({
                'close': [150, 152, 154],  # Upward trend
                'volume': [100000, 110000, 120000]
            }, index=pd.date_range(end=datetime.now(), periods=3, freq='1min'))
        }

        live_engine.data_handler.update_data = AsyncMock()
        await live_engine.process_market_data(market_data)

        # Collect signals
        signals = await live_engine.collect_signals()

        # Verify full pipeline
        assert len(signals) == 1
        assert signals[0].symbol == 'AAPL'
        assert signals[0].direction == 'LONG'
        assert live_engine.current_prices['AAPL'] == 154.0
        assert live_engine.stats['data_updates'] == 1
        assert live_engine.stats['total_signals'] == 1
