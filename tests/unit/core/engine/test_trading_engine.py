"""Comprehensive test suite for trading engine module."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from core.engine.execution_handler import ExecutionHandler
from core.engine.order_manager import OrderManager
from core.engine.trading_engine import TradingEngine


class TestTradingEngine:
    """Test suite for TradingEngine class."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for trading engine."""
        return {
            "portfolio": Mock(),
            "risk_manager": Mock(),
            "data_handler": Mock(),
            "executor": Mock(),
            "strategies": {"test_strategy": Mock()},
        }

    @pytest.fixture
    def trading_engine(self, mock_components):
        """Create TradingEngine instance."""
        engine = TradingEngine(
            portfolio=mock_components["portfolio"],
            risk_manager=mock_components["risk_manager"],
            data_handler=mock_components["data_handler"],
            executor=mock_components["executor"],
            strategies=mock_components["strategies"],
        )
        return engine

    def test_initialization(self, mock_components):
        """Test TradingEngine initialization."""
        engine = TradingEngine(**mock_components)

        assert engine.portfolio == mock_components["portfolio"]
        assert engine.risk_manager == mock_components["risk_manager"]
        assert engine.data_handler == mock_components["data_handler"]
        assert engine.executor == mock_components["executor"]
        assert engine.strategies == mock_components["strategies"]
        assert engine.is_running is False
        assert engine.order_queue.empty()

    @pytest.mark.asyncio
    async def test_start_stop(self, trading_engine):
        """Test starting and stopping the engine."""
        # Mock executor connection
        trading_engine.executor.connect = AsyncMock(return_value=True)
        trading_engine.executor.disconnect = AsyncMock()

        # Start engine
        await trading_engine.start()
        assert trading_engine.is_running is True
        trading_engine.executor.connect.assert_called_once()

        # Stop engine
        await trading_engine.stop()
        assert trading_engine.is_running is False
        trading_engine.executor.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_signals(self, trading_engine):
        """Test signal processing."""
        # Setup mock strategy
        mock_strategy = trading_engine.strategies["test_strategy"]
        mock_signal = {
            "symbol": "AAPL",
            "direction": "LONG",
            "strength": 0.8,
            "price": 150.0,
            "timestamp": datetime.now(),
        }
        mock_strategy.next = Mock(return_value=mock_signal)

        # Setup mock data
        mock_data = pd.DataFrame({"close": [150.0], "volume": [1000000]})
        trading_engine.data_handler.get_latest = AsyncMock(
            return_value={"AAPL": mock_data}
        )

        # Setup risk approval
        trading_engine.risk_manager.pre_trade_check = Mock(
            return_value={"approved": True}
        )

        # Process signals
        await trading_engine._process_signals()

        # Verify signal was processed
        mock_strategy.next.assert_called()
        assert not trading_engine.order_queue.empty()

    @pytest.mark.asyncio
    async def test_execute_orders(self, trading_engine):
        """Test order execution."""
        # Add order to queue
        order = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "BUY",
            "order_type": "MARKET",
        }
        trading_engine.order_queue.put(order)

        # Mock executor
        trading_engine.executor.place_order = AsyncMock(
            return_value={"order_id": "123", "status": "FILLED"}
        )

        # Mock portfolio update
        trading_engine.portfolio.process_fill = Mock()

        # Execute orders
        await trading_engine._execute_orders()

        # Verify execution
        trading_engine.executor.place_order.assert_called_with(order)
        trading_engine.portfolio.process_fill.assert_called()
        assert trading_engine.order_queue.empty()

    @pytest.mark.asyncio
    async def test_update_positions(self, trading_engine):
        """Test position updates."""
        # Mock current prices
        prices = {"AAPL": 155.0, "GOOGL": 2850.0}
        trading_engine.data_handler.get_latest = AsyncMock(
            return_value={"AAPL": {"close": 155.0}, "GOOGL": {"close": 2850.0}}
        )

        # Mock portfolio positions
        trading_engine.portfolio.positions = {"AAPL": Mock(), "GOOGL": Mock()}
        trading_engine.portfolio.update_positions = Mock()

        # Update positions
        await trading_engine._update_positions()

        # Verify updates
        trading_engine.portfolio.update_positions.assert_called_with(prices)

    def test_add_strategy(self, trading_engine):
        """Test adding a new strategy."""
        new_strategy = Mock()
        trading_engine.add_strategy("new_strategy", new_strategy)

        assert "new_strategy" in trading_engine.strategies
        assert trading_engine.strategies["new_strategy"] == new_strategy

    def test_remove_strategy(self, trading_engine):
        """Test removing a strategy."""
        trading_engine.remove_strategy("test_strategy")

        assert "test_strategy" not in trading_engine.strategies

    def test_enable_disable_strategy(self, trading_engine):
        """Test enabling/disabling strategies."""
        strategy = trading_engine.strategies["test_strategy"]
        strategy.enabled = True

        # Disable
        trading_engine.disable_strategy("test_strategy")
        assert strategy.enabled is False

        # Enable
        trading_engine.enable_strategy("test_strategy")
        assert strategy.enabled is True

    def test_get_status(self, trading_engine):
        """Test getting engine status."""
        trading_engine.is_running = True
        trading_engine.portfolio.get_portfolio_value = Mock(return_value=150000)
        trading_engine.portfolio.positions = {"AAPL": Mock(), "GOOGL": Mock()}

        status = trading_engine.get_status()

        assert status["is_running"] is True
        assert status["portfolio_value"] == 150000
        assert status["position_count"] == 2
        assert status["active_strategies"] == 1
        assert "timestamp" in status

    @pytest.mark.asyncio
    async def test_risk_rejection(self, trading_engine):
        """Test order rejection by risk manager."""
        # Setup signal
        signal = {"symbol": "AAPL", "direction": "LONG", "strength": 0.8}
        trading_engine.strategies["test_strategy"].next = Mock(return_value=signal)

        # Setup risk rejection
        trading_engine.risk_manager.pre_trade_check = Mock(
            return_value={"approved": False, "reason": "Position limit exceeded"}
        )

        # Mock data
        trading_engine.data_handler.get_latest = AsyncMock(return_value={"AAPL": {}})

        # Process signals
        await trading_engine._process_signals()

        # Verify order was not queued
        assert trading_engine.order_queue.empty()

    @pytest.mark.asyncio
    async def test_error_handling(self, trading_engine):
        """Test error handling in main loop."""
        # Mock an error in signal processing
        trading_engine.strategies["test_strategy"].next = Mock(
            side_effect=Exception("Strategy error")
        )

        trading_engine.data_handler.get_latest = AsyncMock(return_value={"AAPL": {}})

        # Should handle error gracefully
        await trading_engine._process_signals()

        # Engine should continue running
        assert trading_engine.is_running is False  # Not started yet

    def test_get_performance_metrics(self, trading_engine):
        """Test performance metrics retrieval."""
        mock_metrics = {
            "total_return": 15.5,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.12,
        }
        trading_engine.portfolio.get_performance_metrics = Mock(
            return_value=mock_metrics
        )

        metrics = trading_engine.get_performance_metrics()

        assert metrics == mock_metrics
        trading_engine.portfolio.get_performance_metrics.assert_called_once()


class TestOrderManager:
    """Test suite for OrderManager class."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager instance."""
        return OrderManager()

    async def test_create_order(self, order_manager):
        """Test order creation."""
        from core.engine.order_manager import OrderSide, OrderType

        order = await order_manager.create_order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=150.0,
        )

        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.price == 150.0
        assert order.order_id is not None
        assert order.created_at is not None
        assert order.status.value == "pending"

    async def test_update_order_status(self, order_manager):
        """Test order status update."""
        from core.engine.order_manager import OrderSide, OrderType, OrderStatus

        # Create order
        order = await order_manager.create_order(
            symbol="AAPL", quantity=100, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order_id = order.order_id

        # Update status
        order_manager.update_order_status(
            order_id, OrderStatus.FILLED, {"filled_quantity": 100}
        )

        # Verify update
        updated_order = order_manager.get_order(order_id)
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == 100

    async def test_cancel_order(self, order_manager):
        """Test order cancellation."""
        from core.engine.order_manager import OrderSide, OrderType, OrderStatus

        order = await order_manager.create_order(
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=150.0,
        )
        order_id = order.order_id

        # Cancel order
        result = await order_manager.cancel_order(order_id)

        assert result is True
        assert order_manager.get_order(order_id).status == OrderStatus.CANCELLED

    async def test_get_pending_orders(self, order_manager):
        """Test getting pending orders."""
        from core.engine.order_manager import OrderSide, OrderType, OrderStatus

        # Create multiple orders
        order1 = await order_manager.create_order(
            symbol="AAPL", quantity=100, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        await order_manager.create_order(
            symbol="GOOGL",
            quantity=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=2800,
        )
        await order_manager.create_order(
            symbol="MSFT", quantity=50, side=OrderSide.SELL, order_type=OrderType.MARKET
        )

        # Update one to filled
        order_manager.update_order_status(order1.order_id, OrderStatus.FILLED)

        # Get active orders
        active = order_manager.get_active_orders()

        assert len(active) == 2
        assert all(o.is_active() for o in active)

    async def test_get_orders_by_symbol(self, order_manager):
        """Test getting orders by symbol."""
        from core.engine.order_manager import OrderSide, OrderType

        # Create orders for different symbols
        await order_manager.create_order(
            symbol="AAPL", quantity=100, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        await order_manager.create_order(
            symbol="AAPL",
            quantity=50,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=155,
        )
        await order_manager.create_order(
            symbol="GOOGL", quantity=10, side=OrderSide.BUY, order_type=OrderType.MARKET
        )

        # Get AAPL orders
        aapl_orders = order_manager.get_active_orders(symbol="AAPL")

        assert len(aapl_orders) == 2
        assert all(o.symbol == "AAPL" for o in aapl_orders)

    async def test_order_validation(self, order_manager):
        """Test order validation through create_order."""
        from core.engine.order_manager import OrderSide, OrderType

        # Test invalid order - negative quantity should raise
        with pytest.raises(ValueError, match="quantity must be positive"):
            await order_manager.create_order(
                symbol="AAPL",
                quantity=-100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )

        # Test invalid order - limit order without price should raise
        with pytest.raises(ValueError, match="limit order requires price"):
            await order_manager.create_order(
                symbol="AAPL",
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
            )


class TestExecutionHandler:
    """Test suite for ExecutionHandler class."""

    @pytest.fixture
    def execution_handler(self):
        """Create ExecutionHandler instance."""
        mock_executor = Mock()
        return ExecutionHandler(executor=mock_executor)

    @pytest.mark.asyncio
    async def test_execute_order(self, execution_handler):
        """Test order execution."""
        order = {
            "order_id": "123",
            "symbol": "AAPL",
            "quantity": 100,
            "side": "BUY",
            "order_type": "MARKET",
        }

        # Mock executor response
        execution_handler.executor.place_order = AsyncMock(
            return_value={
                "order_id": "123",
                "status": "FILLED",
                "filled_quantity": 100,
                "avg_fill_price": 150.50,
            }
        )

        # Execute
        result = await execution_handler.execute_order(order)

        assert result["status"] == "FILLED"
        assert result["filled_quantity"] == 100
        execution_handler.executor.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, execution_handler):
        """Test order execution with retry logic."""
        order = {"order_id": "123", "symbol": "AAPL", "quantity": 100}

        # Mock failure then success
        execution_handler.executor.place_order = AsyncMock(
            side_effect=[
                Exception("Connection error"),
                {"order_id": "123", "status": "FILLED"},
            ]
        )

        execution_handler.max_retries = 2
        result = await execution_handler.execute_order(order)

        assert result["status"] == "FILLED"
        assert execution_handler.executor.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_smart_routing(self, execution_handler):
        """Test smart order routing."""
        order = {
            "symbol": "AAPL",
            "quantity": 10000,  # Large order
            "side": "BUY",
            "order_type": "MARKET",
        }

        # Mock routing decision
        execution_handler.route_order = Mock(
            return_value={"venue": "SMART", "slice_size": 1000}
        )

        routing = execution_handler.route_order(order)

        assert routing["venue"] == "SMART"
        assert routing["slice_size"] == 1000

    def test_calculate_slippage(self, execution_handler):
        """Test slippage calculation."""
        order = {
            "symbol": "AAPL",
            "quantity": 1000,
            "side": "BUY",
            "expected_price": 150.00,
        }

        fill = {"avg_fill_price": 150.25, "filled_quantity": 1000}

        slippage = execution_handler.calculate_slippage(order, fill)

        # Buy order filled at higher price = negative slippage
        assert slippage["amount"] == -0.25
        assert slippage["percentage"] == pytest.approx(-0.167, rel=0.01)
        assert slippage["cost"] == -250.0
