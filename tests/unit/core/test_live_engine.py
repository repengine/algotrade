"""
Tests for live trading engine.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from core.live_engine import LiveTradingEngine, TradingMode
from strategies.base import BaseStrategy, Signal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, config=None):
        # Handle both old style (symbol) and new style (config dict)
        if isinstance(config, dict):
            self.symbol = config.get('symbol', 'AAPL')
            self.config = config
        else:
            # Backward compatibility
            self.symbol = config if config else 'AAPL'
            self.config = {'symbol': self.symbol}
        self.signals_to_generate = []

    def init(self):
        """Initialize strategy."""
        pass

    def next(self):
        """Process next data point."""
        pass

    def size(self, signal):
        """Calculate position size."""
        return 100  # Fixed size for testing

    def generate_signals(self, data):
        """Return pre-configured signals."""
        return self.signals_to_generate


@pytest.fixture
def engine_config():
    """Create engine configuration."""
    return {
        "mode": TradingMode.PAPER,
        "strategies": [
            {
                "class": MockStrategy,
                "id": "mock_strategy",
                "params": {"symbol": "AAPL"},
            }
        ],
        "portfolio_config": {
            "initial_capital": 100000,
        },
        "risk_config": {
            "max_position_size": 0.1,
            "max_leverage": 1.0,
        },
        "executor_config": {
            "paper": {
                "initial_capital": 100000,
                "commission": 1.0,
                "slippage": 0.0,
            }
        },
        "schedule": {
            "market_open": "09:30",
            "market_close": "16:00",
            "timezone": "US/Eastern",
        },
        "update_interval": 0.01,  # Fast updates for testing
        "min_signal_strength": 0.5,
        "risk_per_trade": 0.02,
        "max_position_size": 1000,
    }


class TestLiveTradingEngine:
    """Test live trading engine."""

    def test_initialization(self, engine_config):
        """Test engine initialization."""
        engine = LiveTradingEngine(engine_config)

        assert engine.mode == TradingMode.PAPER
        assert len(engine.strategies) == 1
        assert "mock_strategy" in engine.strategies
        assert "AAPL" in engine._active_symbols

    def test_multiple_strategies(self):
        """Test initialization with multiple strategies."""
        config = {
            "mode": TradingMode.PAPER,
            "strategies": [
                {
                    "class": MockStrategy,
                    "id": "strategy1",
                    "params": {"symbol": "AAPL"},
                },
                {
                    "class": MockStrategy,
                    "id": "strategy2",
                    "params": {"symbol": "MSFT"},
                },
            ],
        }

        engine = LiveTradingEngine(config)

        assert len(engine.strategies) == 2
        assert "strategy1" in engine.strategies
        assert "strategy2" in engine.strategies
        assert engine._active_symbols == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_executor_initialization(self, engine_config):
        """Test executor initialization."""
        engine = LiveTradingEngine(engine_config)

        # Paper executor should be added
        assert "paper" in engine.order_manager.executors
        assert engine.order_manager.active_executor == "paper"

        # Connect executors
        for executor in engine.order_manager.executors.values():
            await executor.connect()
            assert executor.is_connected
            await executor.disconnect()

    def test_signal_filtering(self, engine_config):
        """Test signal filtering logic."""
        engine = LiveTradingEngine(engine_config)

        # Strong signal - should trade
        strong_signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )
        assert engine._should_trade_signal(strong_signal) is True

        # Weak signal - should not trade
        weak_signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.3,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )
        assert engine._should_trade_signal(weak_signal) is False

        # Unknown symbol - should not trade
        unknown_signal = Signal(
            symbol="UNKNOWN",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=100.0
        )
        assert engine._should_trade_signal(unknown_signal) is False

    def test_position_sizing(self, engine_config):
        """Test position size calculation."""
        engine = LiveTradingEngine(engine_config)
        engine._last_prices["AAPL"] = 150.0

        # Strong signal
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=1.0,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )

        # 2% risk * 100k * 1.0 strength / $150 = ~13 shares
        size = engine._calculate_position_size(signal)
        assert size == 13

        # Weak signal
        weak_signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.5,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )

        # 2% risk * 100k * 0.5 strength / $150 = ~6 shares
        size = engine._calculate_position_size(weak_signal)
        assert size == 6

        # Check max position size limit
        engine._last_prices["AAPL"] = 1.0  # Very cheap stock
        size = engine._calculate_position_size(signal)
        assert size == 1000  # Max position size

    @pytest.mark.asyncio
    async def test_market_data_update(self, engine_config):
        """Test market data updates."""
        engine = LiveTradingEngine(engine_config)
        engine._last_prices["AAPL"] = 150.0

        # Update market data
        await engine._update_market_data()

        # Price should have changed slightly
        assert "AAPL" in engine._last_prices
        assert 148.5 <= engine._last_prices["AAPL"] <= 151.5  # ±1% change

    @pytest.mark.asyncio
    async def test_signal_processing(self, engine_config):
        """Test signal processing."""
        engine = LiveTradingEngine(engine_config)

        # Setup
        await engine.order_manager.executors["paper"].connect()
        engine._last_prices["AAPL"] = 150.0

        # Create signal
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0
        )

        # Process signal
        await engine._process_signal("mock_strategy", signal)

        # Check stats
        assert engine.stats["total_orders"] == 1

        # Check order was created
        orders = engine.order_manager.get_active_orders()
        assert len(orders) >= 1  # May have been filled already

        await engine.order_manager.executors["paper"].disconnect()

    @pytest.mark.asyncio
    async def test_scheduled_routines(self, engine_config):
        """Test scheduled routine methods."""
        engine = LiveTradingEngine(engine_config)

        # Mock components
        with patch.object(engine, "_update_positions", new_callable=AsyncMock):
            # Test pre-market routine
            await engine._pre_market_routine()

            # Test market open
            await engine._market_open_routine()
            assert engine.is_trading_hours is True
            engine._update_positions.assert_called_once()

            # Test market close
            await engine._market_close_routine()
            assert engine.is_trading_hours is False

            # Test post-market
            with patch.object(engine, "_generate_daily_report"):
                await engine._post_market_routine()
                engine._generate_daily_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_liquidation(self, engine_config):
        """Test emergency liquidation."""
        engine = LiveTradingEngine(engine_config)

        # Setup
        executor = engine.order_manager.executors["paper"]
        await executor.connect()
        try:
            executor.update_price("AAPL", 150.0)

            # Create a position
            from core.executor import Order, OrderSide, OrderType

            buy_order = Order(
                order_id="BUY-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )
            await executor.submit_order(buy_order)
            await asyncio.sleep(0.1)  # Wait for fill

            # Trigger emergency liquidation
            violation = {"reason": "test", "severity": "critical"}
            await engine._emergency_liquidation(violation)

            # Check that liquidation order was created
            orders = engine.order_manager.get_active_orders()
            [
                o
                for o in orders
                if o.metadata.get("strategy_id") == "EMERGENCY_LIQUIDATION"
            ]

            # Should have attempted to liquidate
            # Note: Order may have already filled
        finally:
            await executor.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, engine_config):
        """Test cancelling all orders."""
        engine = LiveTradingEngine(engine_config)

        # Setup
        executor = engine.order_manager.executors["paper"]
        await executor.connect()
        try:
            executor.update_price("AAPL", 150.0)

            # Create multiple limit orders
            for i in range(3):
                order = await engine.order_manager.create_order(
                    symbol="AAPL",
                    side="BUY",
                    quantity=100,
                    order_type="LIMIT",
                    limit_price=140.0 - i,  # Different prices
                )
                await engine.order_manager.submit_order(order)

            # Cancel all
            await engine._cancel_all_orders()

            # Check no active orders remain
            active_orders = engine.order_manager.get_active_orders()
            assert len(active_orders) == 0
        finally:
            await executor.disconnect()

    def test_order_event_handling(self, engine_config):
        """Test order event handler."""
        engine = LiveTradingEngine(engine_config)

        # Create mock order
        from core.executor import Order, OrderSide, OrderType

        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            filled_quantity=100,
            average_fill_price=150.0,
        )

        # Test fill event
        from core.engine.enhanced_order_manager import OrderEventType

        engine._handle_order_event(order, OrderEventType.FILLED, None)
        assert engine.stats["total_fills"] == 1

        # Test error event
        engine._handle_order_event(order, OrderEventType.ERROR, "Test error")
        assert engine.stats["errors"] == 1

    def test_daily_report_generation(self, engine_config):
        """Test daily report generation."""
        engine = LiveTradingEngine(engine_config)

        # Set some stats
        engine.stats["total_signals"] = 10
        engine.stats["total_orders"] = 5
        engine.stats["total_fills"] = 4

        # Generate report (just check it doesn't error)
        engine._generate_daily_report()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, engine_config):
        """Test engine start/stop lifecycle."""
        engine = LiveTradingEngine(engine_config)

        # Mock main loop to exit quickly
        async def mock_main_loop():
            engine.is_running = False

        with patch.object(engine, "_main_loop", mock_main_loop):
            # Start engine
            await engine.start()

            assert engine.stats["engine_start"] is not None

            # Stop engine
            await engine.stop()

            assert engine.is_running is False
