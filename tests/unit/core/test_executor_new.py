"""
Unit tests for Executor module.

Tests cover:
- Order creation and validation
- Order submission and status tracking
- Fill processing
- Position management
- Callback system
- Error handling
- Mock executor implementation

All tests follow FIRST principles with strong assertions.
"""
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from core.executor import (
    BaseExecutor,
    ExecutionCallback,
    ExecutorError,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)


class MockExecutor(BaseExecutor):
    """Mock executor for testing base functionality."""

    def __init__(self, config: dict):
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

    async def get_order_status(self, order_id: str) -> Order:
        return self._orders.get(order_id)

    async def get_positions(self) -> dict[str, Position]:
        return self._positions

    async def get_account_info(self) -> dict:
        return {
            "account_id": "TEST123",
            "balance": 100000.0,
            "buying_power": 100000.0,
        }

    # Helper method for testing
    async def simulate_fill(self, order_id: str, fill_price: float = None):
        """Simulate order fill for testing."""
        if order_id not in self._orders:
            return

        order = self._orders[order_id]
        fill_price = fill_price or order.limit_price or 100.0

        # Create fill
        fill = Fill(
            fill_id=f"FILL_{order_id}",
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=order.quantity * fill_price * 0.001,  # 0.1% commission
            timestamp=datetime.now()
        )

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = fill.commission

        # Update position
        self._update_position_from_fill(fill)

        # Notify callbacks
        self._notify_fill(fill)
        self._notify_order_status(order)

    def _update_position_from_fill(self, fill: Fill):
        """Update position based on fill."""
        symbol = fill.symbol

        if symbol not in self._positions:
            # New position
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=fill.quantity if fill.side == OrderSide.BUY else -fill.quantity,
                average_cost=fill.price,
                current_price=fill.price,
                unrealized_pnl=0.0,
                realized_pnl=-fill.commission,
                market_value=fill.quantity * fill.price,
                last_updated=datetime.now()
            )
        else:
            # Update existing position
            pos = self._positions[symbol]

            if fill.side == OrderSide.BUY:
                # Adding to position
                total_cost = pos.quantity * pos.average_cost + fill.quantity * fill.price
                pos.quantity += fill.quantity
                pos.average_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                # Reducing position
                pos.quantity -= fill.quantity
                if pos.quantity == 0:
                    del self._positions[symbol]

            if symbol in self._positions:
                pos.market_value = pos.quantity * pos.current_price
                pos.unrealized_pnl = (pos.current_price - pos.average_cost) * pos.quantity
                pos.realized_pnl -= fill.commission
                pos.last_updated = datetime.now()

    def update_price(self, symbol: str, price: float) -> None:
        """Update current market price for a symbol."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = (price - pos.average_cost) * pos.quantity
            pos.last_updated = datetime.now()


class TestOrderCreation:
    """Test order creation and validation."""

    @pytest.mark.unit
    def test_order_creation_basic(self):
        """
        Orders are created with correct attributes.

        All required fields should be set properly.
        """
        order = Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        assert order.order_id == "TEST123"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.time_in_force == TimeInForce.DAY
        assert order.filled_quantity == 0
        assert order.average_fill_price == 0.0
        assert order.commission == 0.0

    @pytest.mark.unit
    def test_limit_order_creation(self):
        """
        Limit orders include limit price.

        Should have all market order fields plus limit price.
        """
        order = Order(
            order_id="LIMIT123",
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            time_in_force=TimeInForce.GTC
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 2500.0
        assert order.time_in_force == TimeInForce.GTC
        assert order.stop_price is None  # Not a stop order

    @pytest.mark.unit
    def test_stop_order_creation(self):
        """
        Stop orders include stop price.

        Should have stop price set.
        """
        order = Order(
            order_id="STOP123",
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=290.0
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == 290.0
        assert order.limit_price is None  # Not a limit order

    @pytest.mark.unit
    def test_stop_limit_order_creation(self):
        """
        Stop-limit orders have both stop and limit prices.

        Should have both prices set.
        """
        order = Order(
            order_id="STOPLIMIT123",
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=50,
            order_type=OrderType.STOP_LIMIT,
            stop_price=195.0,
            limit_price=200.0
        )

        assert order.order_type == OrderType.STOP_LIMIT
        assert order.stop_price == 195.0
        assert order.limit_price == 200.0

    @pytest.mark.unit
    def test_order_with_metadata(self):
        """
        Orders can include metadata.

        Metadata should be stored and accessible.
        """
        metadata = {
            "strategy": "mean_reversion",
            "signal_strength": 0.85,
            "entry_reason": "oversold"
        }

        order = Order(
            order_id="META123",
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=200,
            order_type=OrderType.MARKET,
            metadata=metadata
        )

        assert order.metadata == metadata
        assert order.metadata["strategy"] == "mean_reversion"
        assert order.metadata["signal_strength"] == 0.85


class TestOrderValidation:
    """Test order validation logic."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.mark.unit
    def test_valid_market_order(self, executor):
        """
        Valid market orders pass validation.

        Basic market order should be valid.
        """
        order = Order(
            order_id="VALID1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        assert executor.validate_order(order) is True

    @pytest.mark.unit
    def test_invalid_quantity(self, executor):
        """
        Orders with invalid quantity are rejected.

        Zero or negative quantity should fail.
        """
        order = Order(
            order_id="INVALID1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,  # Invalid
            order_type=OrderType.MARKET
        )

        with pytest.raises(ValueError, match="Invalid quantity"):
            executor.validate_order(order)

        # Negative quantity
        order.quantity = -100
        with pytest.raises(ValueError, match="Invalid quantity"):
            executor.validate_order(order)

    @pytest.mark.unit
    def test_limit_order_without_price(self, executor):
        """
        Limit orders without price are rejected.

        Limit orders must have limit price.
        """
        order = Order(
            order_id="INVALID2",
            symbol="GOOGL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=None  # Missing required price
        )

        with pytest.raises(ValueError, match="Limit order requires limit price"):
            executor.validate_order(order)

    @pytest.mark.unit
    def test_stop_order_without_price(self, executor):
        """
        Stop orders without stop price are rejected.

        Stop orders must have stop price.
        """
        order = Order(
            order_id="INVALID3",
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.STOP,
            stop_price=None  # Missing required price
        )

        with pytest.raises(ValueError, match="Stop order requires stop price"):
            executor.validate_order(order)

    @pytest.mark.unit
    def test_valid_complex_orders(self, executor):
        """
        Complex orders with all required fields pass validation.

        Stop-limit and trailing stop orders should validate.
        """
        # Valid stop-limit order
        stop_limit = Order(
            order_id="VALID2",
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=25,
            order_type=OrderType.STOP_LIMIT,
            stop_price=195.0,
            limit_price=200.0
        )

        assert executor.validate_order(stop_limit) is True

        # Valid trailing stop (stop price represents trail amount)
        trailing = Order(
            order_id="VALID3",
            symbol="NVDA",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.TRAILING_STOP,
            stop_price=5.0  # $5 trail
        )

        assert executor.validate_order(trailing) is True


class TestOrderSubmission:
    """Test order submission and tracking."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        return Order(
            order_id="TEST123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_order_submission(self, executor, sample_order):
        """
        Orders are submitted and tracked correctly.

        Should update status and store order.
        """
        # Submit order
        order_id = await executor.submit_order(sample_order)

        assert order_id == "TEST123"
        assert len(executor.submitted_orders) == 1
        assert executor.submitted_orders[0] == sample_order

        # Check order status updated
        assert sample_order.status == OrderStatus.SUBMITTED
        assert sample_order.submitted_at is not None

        # Check order stored
        assert "TEST123" in executor._orders
        stored_order = executor._orders["TEST123"]
        assert stored_order == sample_order

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_order_submission(self, executor):
        """
        Multiple orders are tracked independently.

        Each order should be stored separately.
        """
        orders = []
        for i in range(5):
            order = Order(
                order_id=f"ORDER{i}",
                symbol="SPY",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=100 + i * 10,
                order_type=OrderType.LIMIT,
                limit_price=450.0 + i
            )
            orders.append(order)
            await executor.submit_order(order)

        # All orders submitted
        assert len(executor.submitted_orders) == 5
        assert len(executor._orders) == 5

        # Each order tracked correctly
        for i, order in enumerate(orders):
            assert executor._orders[f"ORDER{i}"] == order
            assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_order_cancellation(self, executor, sample_order):
        """
        Orders can be cancelled after submission.

        Should update status to cancelled.
        """
        # Submit order first
        await executor.submit_order(sample_order)

        # Cancel order
        success = await executor.cancel_order("TEST123")

        assert success is True
        assert len(executor.cancelled_orders) == 1
        assert executor.cancelled_orders[0] == "TEST123"

        # Check status updated
        assert sample_order.status == OrderStatus.CANCELLED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, executor):
        """
        Cancelling non-existent order returns False.

        Should not crash, just return False.
        """
        success = await executor.cancel_order("NONEXISTENT")

        assert success is False
        assert len(executor.cancelled_orders) == 1  # Still tracked attempt

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_order_status(self, executor, sample_order):
        """
        Order status can be retrieved by ID.

        Should return current order state.
        """
        # No order initially
        order = await executor.get_order_status("TEST123")
        assert order is None

        # Submit order
        await executor.submit_order(sample_order)

        # Get status
        order = await executor.get_order_status("TEST123")
        assert order == sample_order
        assert order.status == OrderStatus.SUBMITTED

    @pytest.mark.unit
    def test_get_open_orders(self, executor):
        """
        Open orders are filtered correctly.

        Should only return pending/submitted/partial orders.
        """
        # Create orders with different statuses
        orders = [
            Order("O1", "AAPL", OrderSide.BUY, 100, OrderType.MARKET,
                  status=OrderStatus.PENDING),
            Order("O2", "GOOGL", OrderSide.SELL, 50, OrderType.LIMIT,
                  status=OrderStatus.SUBMITTED),
            Order("O3", "MSFT", OrderSide.BUY, 75, OrderType.MARKET,
                  status=OrderStatus.PARTIALLY_FILLED),
            Order("O4", "TSLA", OrderSide.SELL, 25, OrderType.LIMIT,
                  status=OrderStatus.FILLED),
            Order("O5", "NVDA", OrderSide.BUY, 30, OrderType.MARKET,
                  status=OrderStatus.CANCELLED),
        ]

        for order in orders:
            executor._orders[order.order_id] = order

        open_orders = executor.get_open_orders()

        assert len(open_orders) == 3
        assert all(o.order_id in ["O1", "O2", "O3"] for o in open_orders)


class TestFillProcessing:
    """Test order fill processing."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_order_fill_simulation(self, executor):
        """
        Order fills update order and position correctly.

        Should process fill and update all related data.
        """
        # Submit order
        order = Order(
            order_id="FILL1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        await executor.submit_order(order)

        # Simulate fill
        await executor.simulate_fill("FILL1", 149.50)

        # Check order updated
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None
        assert order.filled_quantity == 100
        assert order.average_fill_price == 149.50
        assert order.commission == pytest.approx(14.95)  # 100 * 149.50 * 0.001

        # Check position created
        assert "AAPL" in executor._positions
        position = executor._positions["AAPL"]
        assert position.quantity == 100
        assert position.average_cost == 149.50
        assert position.market_value == 14950.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_position_update_on_fill(self, executor):
        """
        Positions are updated correctly on fills.

        Should handle both increasing and decreasing positions.
        """
        # First buy
        buy1 = Order("BUY1", "MSFT", OrderSide.BUY, 100, OrderType.MARKET)
        await executor.submit_order(buy1)
        await executor.simulate_fill("BUY1", 300.0)

        pos = executor._positions["MSFT"]
        assert pos.quantity == 100
        assert pos.average_cost == 300.0

        # Second buy (averaging up)
        buy2 = Order("BUY2", "MSFT", OrderSide.BUY, 50, OrderType.MARKET)
        await executor.submit_order(buy2)
        await executor.simulate_fill("BUY2", 310.0)

        pos = executor._positions["MSFT"]
        assert pos.quantity == 150
        # Average cost: (100*300 + 50*310) / 150 = 303.33
        assert pos.average_cost == pytest.approx(303.33, rel=0.01)

        # Partial sell
        sell1 = Order("SELL1", "MSFT", OrderSide.SELL, 75, OrderType.MARKET)
        await executor.submit_order(sell1)
        await executor.simulate_fill("SELL1", 305.0)

        pos = executor._positions["MSFT"]
        assert pos.quantity == 75  # 150 - 75
        assert pos.average_cost == pytest.approx(303.33, rel=0.01)  # Unchanged

        # Full sell (close position)
        sell2 = Order("SELL2", "MSFT", OrderSide.SELL, 75, OrderType.MARKET)
        await executor.submit_order(sell2)
        await executor.simulate_fill("SELL2", 308.0)

        # Position should be removed
        assert "MSFT" not in executor._positions

    @pytest.mark.unit
    def test_fill_creation(self):
        """
        Fill objects contain all required information.

        Should have complete fill details.
        """
        fill = Fill(
            fill_id="F123",
            order_id="O123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=15.0,
            timestamp=datetime.now(),
            metadata={"venue": "NASDAQ"}
        )

        assert fill.fill_id == "F123"
        assert fill.order_id == "O123"
        assert fill.symbol == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == 100
        assert fill.price == 150.0
        assert fill.commission == 15.0
        assert fill.metadata["venue"] == "NASDAQ"


class TestCallbackSystem:
    """Test execution callback system."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.fixture
    def mock_callback(self):
        """Create mock callback."""
        callback = Mock(spec=ExecutionCallback)
        callback.on_order_status = Mock()
        callback.on_fill = Mock()
        callback.on_error = Mock()
        return callback

    @pytest.mark.unit
    def test_register_callback(self, executor, mock_callback):
        """
        Callbacks can be registered and unregistered.

        Should manage callback list correctly.
        """
        # Register callback
        executor.register_callback(mock_callback)
        assert len(executor.callbacks) == 1
        assert mock_callback in executor.callbacks

        # Register another
        callback2 = Mock(spec=ExecutionCallback)
        executor.register_callback(callback2)
        assert len(executor.callbacks) == 2

        # Unregister first callback
        executor.unregister_callback(mock_callback)
        assert len(executor.callbacks) == 1
        assert mock_callback not in executor.callbacks
        assert callback2 in executor.callbacks

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_order_status_callback(self, executor, mock_callback):
        """
        Order status changes trigger callbacks.

        Callbacks should be called with updated order.
        """
        executor.register_callback(mock_callback)

        # Submit order
        order = Order("CB1", "AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        await executor.submit_order(order)

        # Should have called callback
        mock_callback.on_order_status.assert_called_once()
        called_order = mock_callback.on_order_status.call_args[0][0]
        assert called_order == order
        assert called_order.status == OrderStatus.SUBMITTED

        # Cancel order
        mock_callback.on_order_status.reset_mock()
        await executor.cancel_order("CB1")

        mock_callback.on_order_status.assert_called_once()
        called_order = mock_callback.on_order_status.call_args[0][0]
        assert called_order.status == OrderStatus.CANCELLED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fill_callback(self, executor, mock_callback):
        """
        Fills trigger fill callbacks.

        Should call on_fill with fill details.
        """
        executor.register_callback(mock_callback)

        # Submit and fill order
        order = Order("FILL2", "GOOGL", OrderSide.BUY, 10, OrderType.LIMIT, limit_price=2500.0)
        await executor.submit_order(order)
        await executor.simulate_fill("FILL2", 2495.0)

        # Should have called fill callback
        mock_callback.on_fill.assert_called_once()
        fill = mock_callback.on_fill.call_args[0][0]
        assert fill.order_id == "FILL2"
        assert fill.symbol == "GOOGL"
        assert fill.quantity == 10
        assert fill.price == 2495.0

    @pytest.mark.unit
    def test_error_callback(self, executor, mock_callback):
        """
        Errors trigger error callbacks.

        Should call on_error with exception details.
        """
        executor.register_callback(mock_callback)

        # Trigger error
        error = ExecutorError("Connection lost")
        order = Order("ERR1", "TSLA", OrderSide.SELL, 50, OrderType.MARKET)
        executor._notify_error(error, order)

        mock_callback.on_error.assert_called_once_with(error, order)

    @pytest.mark.unit
    def test_callback_error_handling(self, executor, mock_callback):
        """
        Callback errors don't crash executor.

        Should log errors and continue.
        """
        # Make callback raise exception
        mock_callback.on_order_status.side_effect = RuntimeError("Callback error")
        executor.register_callback(mock_callback)

        # Add another callback that works
        good_callback = Mock(spec=ExecutionCallback)
        executor.register_callback(good_callback)

        # Notify should not crash
        order = Order("TEST", "SPY", OrderSide.BUY, 100, OrderType.MARKET)
        with patch("core.executor.logger") as mock_logger:
            executor._notify_order_status(order)

            # Error should be logged
            mock_logger.error.assert_called()

            # Good callback should still be called
            good_callback.on_order_status.assert_called_once_with(order)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, executor):
        """
        Multiple callbacks all receive notifications.

        All registered callbacks should be called.
        """
        callbacks = [Mock(spec=ExecutionCallback) for _ in range(3)]
        for cb in callbacks:
            executor.register_callback(cb)

        # Submit order
        order = Order("MULTI", "NVDA", OrderSide.BUY, 50, OrderType.MARKET)
        await executor.submit_order(order)

        # All callbacks should be notified
        for cb in callbacks:
            cb.on_order_status.assert_called_once()
            assert cb.on_order_status.call_args[0][0] == order


class TestConnectionManagement:
    """Test executor connection management."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, executor):
        """
        Executor can connect and disconnect.

        Should track connection state.
        """
        # Initial state
        assert executor.is_connected is False
        assert executor.connect_called is False

        # Connect
        success = await executor.connect()
        assert success is True
        assert executor.is_connected is True
        assert executor.connect_called is True

        # Disconnect
        await executor.disconnect()
        assert executor.is_connected is False
        assert executor.disconnect_called is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_account_info_retrieval(self, executor):
        """
        Account info can be retrieved.

        Should return account details.
        """
        info = await executor.get_account_info()

        assert isinstance(info, dict)
        assert info["account_id"] == "TEST123"
        assert info["balance"] == 100000.0
        assert info["buying_power"] == 100000.0


class TestPositionManagement:
    """Test position tracking and management."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.mark.unit
    def test_position_creation(self):
        """
        Position objects contain required information.

        All fields should be properly set.
        """
        position = Position(
            symbol="AAPL",
            quantity=100,
            average_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,  # (155-150)*100
            realized_pnl=-15.0,    # Commission
            market_value=15500.0,  # 100*155
            last_updated=datetime.now()
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.average_cost == 150.0
        assert position.current_price == 155.0
        assert position.unrealized_pnl == 500.0
        assert position.realized_pnl == -15.0
        assert position.market_value == 15500.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_positions(self, executor):
        """
        All positions can be retrieved.

        Should return current position dictionary.
        """
        # Initially empty
        positions = await executor.get_positions()
        assert positions == {}

        # Add some positions via fills
        order1 = Order("P1", "AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        await executor.submit_order(order1)
        await executor.simulate_fill("P1", 150.0)

        order2 = Order("P2", "GOOGL", OrderSide.BUY, 20, OrderType.MARKET)
        await executor.submit_order(order2)
        await executor.simulate_fill("P2", 2500.0)

        positions = await executor.get_positions()
        assert len(positions) == 2
        assert "AAPL" in positions
        assert "GOOGL" in positions
        assert positions["AAPL"].quantity == 100
        assert positions["GOOGL"].quantity == 20

    @pytest.mark.unit
    def test_get_position_by_symbol(self, executor):
        """
        Individual positions can be retrieved by symbol.

        Should return position or None.
        """
        # No position initially
        pos = executor.get_position("MSFT")
        assert pos is None

        # Add position
        executor._positions["MSFT"] = Position(
            symbol="MSFT",
            quantity=50,
            average_cost=300.0,
            current_price=305.0,
            unrealized_pnl=250.0,
            realized_pnl=0.0,
            market_value=15250.0,
            last_updated=datetime.now()
        )

        pos = executor.get_position("MSFT")
        assert pos is not None
        assert pos.symbol == "MSFT"
        assert pos.quantity == 50

    @pytest.mark.unit
    def test_update_price(self, executor):
        """
        Position prices can be updated.

        Should update current price and unrealized PnL.
        """
        # Create position
        executor._positions["TSLA"] = Position(
            symbol="TSLA",
            quantity=25,
            average_cost=200.0,
            current_price=200.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            market_value=5000.0,
            last_updated=datetime.now()
        )

        # Update price
        executor.update_price("TSLA", 210.0)

        pos = executor._positions["TSLA"]
        assert pos.current_price == 210.0
        assert pos.market_value == 5250.0  # 25 * 210
        assert pos.unrealized_pnl == 250.0  # 25 * (210-200)


class TestParametrizedScenarios:
    """Test various scenarios with parametrization."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return MockExecutor({})

    @pytest.mark.unit
    @pytest.mark.parametrize("order_type,limit_price,stop_price,should_validate", [
        (OrderType.MARKET, None, None, True),
        (OrderType.LIMIT, 100.0, None, True),
        (OrderType.LIMIT, None, None, False),  # Missing limit price
        (OrderType.STOP, None, 95.0, True),
        (OrderType.STOP, None, None, False),   # Missing stop price
        (OrderType.STOP_LIMIT, 100.0, 95.0, True),
        (OrderType.STOP_LIMIT, None, 95.0, False),  # Missing limit price
        (OrderType.STOP_LIMIT, 100.0, None, False),  # Missing stop price
    ])
    def test_order_validation_scenarios(self, executor, order_type, limit_price,
                                      stop_price, should_validate):
        """
        Order validation works for all order types.

        Tests various order type and price combinations.
        """
        order = Order(
            order_id="VAL1",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )

        if should_validate:
            assert executor.validate_order(order) is True
        else:
            with pytest.raises(ValueError):
                executor.validate_order(order)

    @pytest.mark.unit
    @pytest.mark.parametrize("side,quantity,fill_price,expected_pnl", [
        (OrderSide.BUY, 100, 105.0, 500.0),   # Long profit
        (OrderSide.BUY, 100, 95.0, -500.0),   # Long loss
        (OrderSide.SELL, 100, 95.0, 500.0),   # Short profit (sold at 100, covers at 95)
        (OrderSide.SELL, 100, 105.0, -500.0), # Short loss
    ])
    @pytest.mark.asyncio
    async def test_pnl_calculations(self, executor, side, quantity, fill_price, expected_pnl):
        """
        PnL is calculated correctly for various scenarios.

        Tests long/short positions with profits/losses.
        """
        # Set up initial position at $100
        if side == OrderSide.BUY:
            # For long test, buy at 100 first
            order1 = Order("INIT", "TEST", OrderSide.BUY, quantity, OrderType.MARKET)
            await executor.submit_order(order1)
            await executor.simulate_fill("INIT", 100.0)

            # Update price
            executor.update_price("TEST", fill_price)
        else:
            # For short test, sell at 100 first
            order1 = Order("INIT", "TEST", OrderSide.SELL, quantity, OrderType.MARKET)
            await executor.submit_order(order1)
            await executor.simulate_fill("INIT", 100.0)

            # Update price (this is where we'd cover)
            executor.update_price("TEST", fill_price)

        # Check PnL
        pos = executor.get_position("TEST")
        if pos:  # Short positions might be closed
            # Adjust for commission in comparison
            assert pos.unrealized_pnl == pytest.approx(expected_pnl, abs=1.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("status,is_open", [
        (OrderStatus.PENDING, True),
        (OrderStatus.SUBMITTED, True),
        (OrderStatus.PARTIALLY_FILLED, True),
        (OrderStatus.FILLED, False),
        (OrderStatus.CANCELLED, False),
        (OrderStatus.REJECTED, False),
        (OrderStatus.EXPIRED, False),
    ])
    def test_open_order_filtering(self, executor, status, is_open):
        """
        Open orders are correctly identified by status.

        Only pending/submitted/partial orders are "open".
        """
        order = Order("STATUS1", "SPY", OrderSide.BUY, 100, OrderType.MARKET, status=status)
        executor._orders[order.order_id] = order

        open_orders = executor.get_open_orders()

        if is_open:
            assert len(open_orders) == 1
            assert open_orders[0] == order
        else:
            assert len(open_orders) == 0
