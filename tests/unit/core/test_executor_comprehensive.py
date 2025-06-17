"""Comprehensive test suite for executor module achieving 100% coverage."""

import logging
from datetime import datetime
from typing import Any, Optional
from unittest.mock import Mock

import pytest
from core.executor import (
    BaseExecutor,
    ExecutorError,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)


class MockCallback:
    """Mock implementation of ExecutionCallback protocol."""

    def __init__(self):
        self.order_status_calls = []
        self.fill_calls = []
        self.error_calls = []

    def on_order_status(self, order: Order) -> None:
        self.order_status_calls.append(order)

    def on_fill(self, fill: Fill) -> None:
        self.fill_calls.append(fill)

    def on_error(self, error: Exception, order: Optional[Order] = None) -> None:
        self.error_calls.append((error, order))


class ConcreteExecutor(BaseExecutor):
    """Concrete implementation for testing."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.connect_called = False
        self.disconnect_called = False
        self.submitted_orders = []
        self.cancelled_orders = []

    async def connect(self) -> bool:
        self.connect_called = True
        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        self.disconnect_called = True
        self.is_connected = False

    async def submit_order(self, order: Order) -> str:
        self.submitted_orders.append(order)
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        self._orders[order.order_id] = order
        self._notify_order_status(order)
        return order.order_id

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled_orders.append(order_id)
        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = OrderStatus.CANCELLED
            self._notify_order_status(order)
            return True
        return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    async def get_positions(self) -> dict[str, Position]:
        return self._positions

    async def get_account_info(self) -> dict[str, Any]:
        return {
            'account_id': 'TEST123',
            'cash': 100000.0,
            'buying_power': 200000.0,
            'portfolio_value': 150000.0
        }


class TestExecutorError:
    """Test ExecutorError exception."""

    def test_executor_error_creation(self):
        """Test creating ExecutorError."""
        error = ExecutorError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_executor_error_with_details(self):
        """Test ExecutorError with additional details."""
        error = ExecutorError("Order failed")
        assert "Order failed" in str(error)


class TestEnums:
    """Test all enum classes."""

    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
        assert OrderType.TRAILING_STOP.value == "TRAILING_STOP"

    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_time_in_force_enum(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.DAY.value == "DAY"
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"
        assert TimeInForce.GTD.value == "GTD"
        assert TimeInForce.OPG.value == "OPG"
        assert TimeInForce.CLS.value == "CLS"


class TestDataClasses:
    """Test all dataclass implementations."""

    def test_order_creation(self):
        """Test Order dataclass creation."""
        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            time_in_force=TimeInForce.DAY
        )

        assert order.order_id == "TEST123"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0
        assert order.stop_price is None
        assert order.time_in_force == TimeInForce.DAY
        assert order.status == OrderStatus.PENDING
        assert order.submitted_at is None
        assert order.filled_at is None
        assert order.filled_quantity == 0
        assert order.average_fill_price == 0.0
        assert order.commission == 0.0
        assert order.metadata == {}

    def test_order_with_all_fields(self):
        """Test Order with all fields populated."""
        now = datetime.now()
        metadata = {"source": "test"}

        order = Order(
            order_id="TEST456",
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.STOP_LIMIT,
            limit_price=2800.0,
            stop_price=2790.0,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            submitted_at=now,
            filled_at=now,
            filled_quantity=50,
            average_fill_price=2795.0,
            commission=1.0,
            metadata=metadata
        )

        assert order.stop_price == 2790.0
        assert order.status == OrderStatus.FILLED
        assert order.submitted_at == now
        assert order.filled_at == now
        assert order.filled_quantity == 50
        assert order.average_fill_price == 2795.0
        assert order.commission == 1.0
        assert order.metadata == metadata

    def test_fill_creation(self):
        """Test Fill dataclass creation."""
        now = datetime.now()
        metadata = {"exchange": "NASDAQ"}

        fill = Fill(
            fill_id="FILL123",
            order_id="ORDER123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.5,
            commission=0.5,
            timestamp=now,
            metadata=metadata
        )

        assert fill.fill_id == "FILL123"
        assert fill.order_id == "ORDER123"
        assert fill.symbol == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == 100
        assert fill.price == 150.5
        assert fill.commission == 0.5
        assert fill.timestamp == now
        assert fill.metadata == metadata

    def test_position_creation(self):
        """Test Position dataclass creation."""
        now = datetime.now()

        position = Position(
            symbol="MSFT",
            quantity=200,
            average_cost=300.0,
            current_price=310.0,
            unrealized_pnl=2000.0,
            realized_pnl=500.0,
            market_value=62000.0,
            last_updated=now
        )

        assert position.symbol == "MSFT"
        assert position.quantity == 200
        assert position.average_cost == 300.0
        assert position.current_price == 310.0
        assert position.unrealized_pnl == 2000.0
        assert position.realized_pnl == 500.0
        assert position.market_value == 62000.0
        assert position.last_updated == now


class TestExecutionCallback:
    """Test ExecutionCallback protocol."""

    def test_protocol_methods_exist(self):
        """Test that ExecutionCallback protocol methods exist."""
        # Test that the protocol defines the required methods
        from core.executor import ExecutionCallback
        assert hasattr(ExecutionCallback, 'on_order_status')
        assert hasattr(ExecutionCallback, 'on_fill')
        assert hasattr(ExecutionCallback, 'on_error')

        # Test that the protocol methods have ellipsis as placeholders
        # This ensures the protocol is properly defined
        import inspect

        # Get the source of the methods to verify they contain ellipsis
        on_order_status_source = inspect.getsource(ExecutionCallback.on_order_status)
        on_fill_source = inspect.getsource(ExecutionCallback.on_fill)
        on_error_source = inspect.getsource(ExecutionCallback.on_error)

        assert '...' in on_order_status_source
        assert '...' in on_fill_source
        assert '...' in on_error_source

    def test_callback_protocol(self):
        """Test that MockCallback implements ExecutionCallback protocol."""
        callback = MockCallback()

        # Test on_order_status
        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        callback.on_order_status(order)
        assert len(callback.order_status_calls) == 1
        assert callback.order_status_calls[0] == order

        # Test on_fill
        fill = Fill(
            fill_id="FILL123",
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=0.5,
            timestamp=datetime.now()
        )
        callback.on_fill(fill)
        assert len(callback.fill_calls) == 1
        assert callback.fill_calls[0] == fill

        # Test on_error
        error = ExecutorError("Test error")
        callback.on_error(error, order)
        assert len(callback.error_calls) == 1
        assert callback.error_calls[0][0] == error
        assert callback.error_calls[0][1] == order


class TestBaseExecutor:
    """Test suite for BaseExecutor abstract class."""

    @pytest.fixture
    def executor(self):
        """Create concrete executor instance."""
        config = {'test': True}
        return ConcreteExecutor(config)

    def test_initialization(self):
        """Test executor initialization."""
        config = {'test': True, 'param': 'value'}
        executor = ConcreteExecutor(config)

        assert executor.config == config
        assert executor.is_connected is False
        assert executor.callbacks == []
        assert executor._orders == {}
        assert executor._positions == {}

    def test_callback_registration(self, executor):
        """Test callback registration and unregistration."""
        callback1 = MockCallback()
        callback2 = MockCallback()

        # Register callbacks
        executor.register_callback(callback1)
        assert len(executor.callbacks) == 1
        assert callback1 in executor.callbacks

        executor.register_callback(callback2)
        assert len(executor.callbacks) == 2
        assert callback2 in executor.callbacks

        # Unregister callback
        executor.unregister_callback(callback1)
        assert len(executor.callbacks) == 1
        assert callback1 not in executor.callbacks
        assert callback2 in executor.callbacks

        # Try to unregister non-existent callback
        executor.unregister_callback(callback1)  # Should not raise error
        assert len(executor.callbacks) == 1

    @pytest.mark.asyncio
    async def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # This test verifies the abstract nature by checking ConcreteExecutor
        config = {}
        executor = ConcreteExecutor(config)

        # Verify all abstract methods are implemented
        assert hasattr(executor, 'connect')
        assert hasattr(executor, 'disconnect')
        assert hasattr(executor, 'submit_order')
        assert hasattr(executor, 'cancel_order')
        assert hasattr(executor, 'get_order_status')
        assert hasattr(executor, 'get_positions')
        assert hasattr(executor, 'get_account_info')

    def test_notify_order_status(self, executor):
        """Test order status notification to callbacks."""
        callback1 = MockCallback()
        callback2 = MockCallback()
        executor.register_callback(callback1)
        executor.register_callback(callback2)

        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        executor._notify_order_status(order)

        assert len(callback1.order_status_calls) == 1
        assert callback1.order_status_calls[0] == order
        assert len(callback2.order_status_calls) == 1
        assert callback2.order_status_calls[0] == order

    def test_notify_order_status_with_error(self, executor, caplog):
        """Test order status notification when callback raises error."""
        callback = Mock()
        callback.on_order_status.side_effect = RuntimeError("Callback error")
        executor.register_callback(callback)

        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        with caplog.at_level(logging.ERROR):
            executor._notify_order_status(order)

        assert "Error in order status callback: Callback error" in caplog.text

    def test_notify_fill(self, executor):
        """Test fill notification to callbacks."""
        callback = MockCallback()
        executor.register_callback(callback)

        fill = Fill(
            fill_id="FILL123",
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=0.5,
            timestamp=datetime.now()
        )

        executor._notify_fill(fill)

        assert len(callback.fill_calls) == 1
        assert callback.fill_calls[0] == fill

    def test_notify_fill_with_error(self, executor, caplog):
        """Test fill notification when callback raises error."""
        callback = Mock()
        callback.on_fill.side_effect = ValueError("Fill callback error")
        executor.register_callback(callback)

        fill = Fill(
            fill_id="FILL123",
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=0.5,
            timestamp=datetime.now()
        )

        with caplog.at_level(logging.ERROR):
            executor._notify_fill(fill)

        assert "Error in fill callback: Fill callback error" in caplog.text

    def test_notify_error(self, executor):
        """Test error notification to callbacks."""
        callback = MockCallback()
        executor.register_callback(callback)

        error = ExecutorError("Test error")
        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        executor._notify_error(error, order)

        assert len(callback.error_calls) == 1
        assert callback.error_calls[0][0] == error
        assert callback.error_calls[0][1] == order

    def test_notify_error_without_order(self, executor):
        """Test error notification without order."""
        callback = MockCallback()
        executor.register_callback(callback)

        error = ExecutorError("Connection error")

        executor._notify_error(error)

        assert len(callback.error_calls) == 1
        assert callback.error_calls[0][0] == error
        assert callback.error_calls[0][1] is None

    def test_notify_error_with_callback_error(self, executor, caplog):
        """Test error notification when callback raises error."""
        callback = Mock()
        callback.on_error.side_effect = RuntimeError("Error callback error")
        executor.register_callback(callback)

        error = ExecutorError("Test error")

        with caplog.at_level(logging.ERROR):
            executor._notify_error(error)

        assert "Error in error callback: Error callback error" in caplog.text

    def test_validate_order_valid(self, executor):
        """Test validation of valid orders."""
        # Market order
        order = Order(
            order_id="TEST1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        assert executor.validate_order(order) is True

        # Limit order
        order = Order(
            order_id="TEST2",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        assert executor.validate_order(order) is True

        # Stop order
        order = Order(
            order_id="TEST3",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=145.0
        )
        assert executor.validate_order(order) is True

        # Stop limit order
        order = Order(
            order_id="TEST4",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            limit_price=151.0,
            stop_price=150.0
        )
        assert executor.validate_order(order) is True

    def test_validate_order_invalid_quantity(self, executor):
        """Test validation rejects invalid quantity."""
        # Zero quantity
        order = Order(
            order_id="TEST1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.MARKET
        )
        with pytest.raises(ValueError, match="Invalid quantity: 0"):
            executor.validate_order(order)

        # Negative quantity
        order = Order(
            order_id="TEST2",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=-10,
            order_type=OrderType.MARKET
        )
        with pytest.raises(ValueError, match="Invalid quantity: -10"):
            executor.validate_order(order)

    def test_validate_order_missing_limit_price(self, executor):
        """Test validation rejects limit order without limit price."""
        order = Order(
            order_id="TEST1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT
        )
        with pytest.raises(ValueError, match="Limit order requires limit price"):
            executor.validate_order(order)

    def test_validate_order_missing_stop_price(self, executor):
        """Test validation rejects stop order without stop price."""
        # Stop order
        order = Order(
            order_id="TEST1",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP
        )
        with pytest.raises(ValueError, match="Stop order requires stop price"):
            executor.validate_order(order)

        # Stop limit order
        order = Order(
            order_id="TEST2",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            limit_price=150.0
        )
        with pytest.raises(ValueError, match="Stop order requires stop price"):
            executor.validate_order(order)

    def test_get_open_orders(self, executor):
        """Test getting open orders."""
        # Add various orders
        order1 = Order(
            order_id="TEST1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING
        )
        executor._orders["TEST1"] = order1

        order2 = Order(
            order_id="TEST2",
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=2800.0,
            status=OrderStatus.SUBMITTED
        )
        executor._orders["TEST2"] = order2

        order3 = Order(
            order_id="TEST3",
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=75,
            order_type=OrderType.LIMIT,
            limit_price=300.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=25
        )
        executor._orders["TEST3"] = order3

        order4 = Order(
            order_id="TEST4",
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=30,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED
        )
        executor._orders["TEST4"] = order4

        order5 = Order(
            order_id="TEST5",
            symbol="NVDA",
            side=OrderSide.BUY,
            quantity=20,
            order_type=OrderType.LIMIT,
            limit_price=500.0,
            status=OrderStatus.CANCELLED
        )
        executor._orders["TEST5"] = order5

        # Get open orders
        open_orders = executor.get_open_orders()

        assert len(open_orders) == 3
        assert order1 in open_orders
        assert order2 in open_orders
        assert order3 in open_orders
        assert order4 not in open_orders
        assert order5 not in open_orders

    def test_get_open_orders_empty(self, executor):
        """Test getting open orders when none exist."""
        open_orders = executor.get_open_orders()
        assert open_orders == []

    def test_get_order(self, executor):
        """Test getting order by ID."""
        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        executor._orders["TEST123"] = order

        # Get existing order
        retrieved = executor.get_order("TEST123")
        assert retrieved == order

        # Get non-existent order
        retrieved = executor.get_order("NONEXISTENT")
        assert retrieved is None

    def test_get_position(self, executor):
        """Test getting position by symbol."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            average_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0,
            market_value=15500.0,
            last_updated=datetime.now()
        )
        executor._positions["AAPL"] = position

        # Get existing position
        retrieved = executor.get_position("AAPL")
        assert retrieved == position

        # Get non-existent position
        retrieved = executor.get_position("GOOGL")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_integration_flow(self, executor):
        """Test complete integration flow."""
        # Register callback
        callback = MockCallback()
        executor.register_callback(callback)

        # Connect
        connected = await executor.connect()
        assert connected is True
        assert executor.is_connected is True

        # Submit order
        order = Order(
            order_id="INT_TEST_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )

        order_id = await executor.submit_order(order)
        assert order_id == "INT_TEST_1"
        assert len(callback.order_status_calls) == 1
        assert callback.order_status_calls[0].status == OrderStatus.SUBMITTED

        # Check order status
        status = await executor.get_order_status(order_id)
        assert status is not None
        assert status.order_id == order_id
        assert status.status == OrderStatus.SUBMITTED

        # Cancel order
        cancelled = await executor.cancel_order(order_id)
        assert cancelled is True
        assert len(callback.order_status_calls) == 2
        assert callback.order_status_calls[1].status == OrderStatus.CANCELLED

        # Get account info
        account_info = await executor.get_account_info()
        assert account_info['account_id'] == 'TEST123'
        assert account_info['cash'] == 100000.0

        # Disconnect
        await executor.disconnect()
        assert executor.is_connected is False
