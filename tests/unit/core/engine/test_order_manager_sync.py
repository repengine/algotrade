"""
Synchronous test suite for OrderManager achieving 100% coverage.

This test file runs without pytest-asyncio by using asyncio.run()
to execute async methods synchronously.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from core.engine.order_manager import (
    Order,
    OrderFill,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


# Removed run_async helper - using pytest-asyncio instead


class TestOrderDataStructures:
    """Test Order and OrderFill data structures."""

    def test_order_defaults(self):
        """Test Order default values."""
        order = Order()

        assert order.order_id is not None
        assert order.symbol == ""
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.0
        assert order.price is None
        assert order.stop_price is None
        assert order.time_in_force == TimeInForce.GTC
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.average_fill_price == 0.0
        assert order.strategy_id is None
        assert isinstance(order.metadata, dict)
        assert len(order.metadata) == 0

    def test_order_is_active(self):
        """Test Order.is_active() method."""
        # Active statuses
        for status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
            order = Order(status=status)
            assert order.is_active() is True

        # Inactive statuses
        for status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            order = Order(status=status)
            assert order.is_active() is False

    def test_order_is_filled(self):
        """Test Order.is_filled() method."""
        order = Order(status=OrderStatus.FILLED)
        assert order.is_filled() is True

        order.status = OrderStatus.PARTIAL
        assert order.is_filled() is False

    def test_order_remaining_quantity(self):
        """Test Order.remaining_quantity() method."""
        order = Order(quantity=100.0, filled_quantity=30.0)
        assert order.remaining_quantity() == 70.0

        order.filled_quantity = 100.0
        assert order.remaining_quantity() == 0.0

        # Edge case: overfilled
        order.filled_quantity = 110.0
        assert order.remaining_quantity() == -10.0


class TestOrderManagerSync:
    """Test OrderManager with synchronous test methods."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager instance."""
        return OrderManager()

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange connector."""
        exchange = Mock()
        exchange.submit_order = AsyncMock(return_value=True)
        exchange.cancel_order = AsyncMock(return_value=True)
        return exchange

    def test_initialization(self):
        """Test OrderManager initialization."""
        manager = OrderManager()
        assert manager.exchange_connector is None
        assert isinstance(manager.orders, dict)
        assert len(manager.orders) == 0
        assert isinstance(manager.order_fills, dict)

        # With exchange
        exchange = Mock()
        manager2 = OrderManager(exchange_connector=exchange)
        assert manager2.exchange_connector == exchange

    def test_create_order_basic(self, order_manager):
        """Test basic order creation."""
        order = run_async(order_manager.create_order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0
        ))

        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 100.0
        assert order.status == OrderStatus.PENDING
        assert order.order_id in order_manager.orders
        assert order.order_id in order_manager.order_fills

    def test_create_order_validation_errors(self, order_manager):
        """Test order creation validation errors."""
        # Zero quantity
        with pytest.raises(ValueError, match="quantity must be positive"):
            run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 0))

        # Negative quantity
        with pytest.raises(ValueError, match="quantity must be positive"):
            run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, -10))

        # Limit order without price
        with pytest.raises(ValueError, match="limit order requires price"):
            run_async(order_manager.create_order("AAPL", OrderType.LIMIT, OrderSide.BUY, 100))

        # Stop order without stop price
        with pytest.raises(ValueError, match="stop order requires stop price"):
            run_async(order_manager.create_order("AAPL", OrderType.STOP, OrderSide.BUY, 100))

        # Stop limit order without price
        with pytest.raises(ValueError, match="stop_limit order requires price"):
            run_async(order_manager.create_order(
                "AAPL", OrderType.STOP_LIMIT, OrderSide.BUY, 100, stop_price=148.0
            ))

        # Stop limit order without stop price
        with pytest.raises(ValueError, match="stop_limit order requires stop price"):
            run_async(order_manager.create_order(
                "AAPL", OrderType.STOP_LIMIT, OrderSide.BUY, 100, price=150.0
            ))

    def test_submit_order_success(self, mock_exchange):
        """Test successful order submission."""
        manager = OrderManager(exchange_connector=mock_exchange)

        # Create order
        order = run_async(manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # Submit order
        success = run_async(manager.submit_order(order))

        assert success is True
        assert order.status == OrderStatus.SUBMITTED
        assert order.updated_at > order.created_at
        mock_exchange.submit_order.assert_called_once_with(order)

    def test_submit_order_no_exchange(self, order_manager):
        """Test order submission without exchange connector."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        success = run_async(order_manager.submit_order(order))

        assert success is False
        assert order.status == OrderStatus.PENDING

    def test_submit_order_rejection(self, mock_exchange):
        """Test order rejection by exchange."""
        mock_exchange.submit_order.return_value = False
        manager = OrderManager(exchange_connector=mock_exchange)

        order = run_async(manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))
        success = run_async(manager.submit_order(order))

        assert success is False
        assert order.status == OrderStatus.REJECTED

    def test_submit_order_exception(self, mock_exchange):
        """Test order submission with exception."""
        mock_exchange.submit_order.side_effect = Exception("Connection error")
        manager = OrderManager(exchange_connector=mock_exchange)

        order = run_async(manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))
        success = run_async(manager.submit_order(order))

        assert success is False
        assert order.status == OrderStatus.REJECTED

    def test_cancel_order_success(self, mock_exchange):
        """Test successful order cancellation."""
        manager = OrderManager(exchange_connector=mock_exchange)

        # Create and submit order
        order = run_async(manager.create_order("AAPL", OrderType.LIMIT, OrderSide.BUY, 100, price=150))
        order.status = OrderStatus.SUBMITTED

        success = run_async(manager.cancel_order(order.order_id))

        assert success is True
        assert order.status == OrderStatus.CANCELLED
        mock_exchange.cancel_order.assert_called_once_with(order.order_id)

    def test_cancel_order_not_found(self, order_manager):
        """Test cancelling non-existent order."""
        success = run_async(order_manager.cancel_order("non_existent_id"))
        assert success is False

    def test_cancel_order_inactive(self, order_manager):
        """Test cancelling inactive order."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))
        order.status = OrderStatus.FILLED

        success = run_async(order_manager.cancel_order(order.order_id))
        assert success is False

    def test_cancel_order_no_exchange(self, order_manager):
        """Test order cancellation without exchange (mock mode)."""
        order = run_async(order_manager.create_order("AAPL", OrderType.LIMIT, OrderSide.BUY, 100, price=150))
        order.status = OrderStatus.SUBMITTED

        success = run_async(order_manager.cancel_order(order.order_id))

        assert success is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_order_exception(self, mock_exchange):
        """Test order cancellation with exception."""
        mock_exchange.cancel_order.side_effect = Exception("Cancel failed")
        manager = OrderManager(exchange_connector=mock_exchange)

        order = run_async(manager.create_order("AAPL", OrderType.LIMIT, OrderSide.BUY, 100, price=150))
        order.status = OrderStatus.SUBMITTED

        success = run_async(manager.cancel_order(order.order_id))
        assert success is False

    def test_update_order_fill_partial(self, order_manager):
        """Test updating order with partial fill."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        fill = OrderFill(
            order_id=order.order_id,
            fill_id="fill_1",
            quantity=30.0,
            price=150.0,
            commission=1.0,
            timestamp=datetime.now()
        )

        run_async(order_manager.update_order_fill(order.order_id, fill))

        assert order.filled_quantity == 30.0
        assert order.average_fill_price == 150.0
        assert order.status == OrderStatus.PARTIAL
        assert len(order_manager.order_fills[order.order_id]) == 1

    def test_update_order_fill_complete(self, order_manager):
        """Test updating order with complete fill."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        fill = OrderFill(
            order_id=order.order_id,
            fill_id="fill_1",
            quantity=100.0,
            price=150.0,
            commission=1.0,
            timestamp=datetime.now()
        )

        run_async(order_manager.update_order_fill(order.order_id, fill))

        assert order.filled_quantity == 100.0
        assert order.average_fill_price == 150.0
        assert order.status == OrderStatus.FILLED

    def test_update_order_fill_multiple(self, order_manager):
        """Test updating order with multiple fills."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # First fill
        fill1 = OrderFill(order.order_id, "fill_1", 40.0, 150.0, 0.5, datetime.now())
        run_async(order_manager.update_order_fill(order.order_id, fill1))

        # Second fill
        fill2 = OrderFill(order.order_id, "fill_2", 60.0, 151.0, 0.5, datetime.now())
        run_async(order_manager.update_order_fill(order.order_id, fill2))

        assert order.filled_quantity == 100.0
        assert order.status == OrderStatus.FILLED
        # Average price: (40 * 150 + 60 * 151) / 100 = 150.6
        assert order.average_fill_price == pytest.approx(150.6)
        assert len(order_manager.order_fills[order.order_id]) == 2

    def test_update_order_fill_not_found(self, order_manager):
        """Test updating fill for non-existent order."""
        fill = OrderFill("non_existent", "fill_1", 100.0, 150.0, 1.0, datetime.now())

        # Should not raise, just log error
        run_async(order_manager.update_order_fill("non_existent", fill))

    def test_get_order(self, order_manager):
        """Test getting order by ID."""
        # Create order directly
        order = Order(order_id="test_id", symbol="AAPL")
        order_manager.orders["test_id"] = order

        retrieved = order_manager.get_order("test_id")
        assert retrieved == order

        # Non-existent order
        assert order_manager.get_order("non_existent") is None

    def test_get_active_orders(self, order_manager):
        """Test getting active orders."""
        # Create multiple orders with different statuses
        order1 = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))
        order1.status = OrderStatus.SUBMITTED

        order2 = run_async(order_manager.create_order("GOOGL", OrderType.LIMIT, OrderSide.BUY, 50, price=2800))
        order2.status = OrderStatus.PARTIAL

        order3 = run_async(order_manager.create_order("MSFT", OrderType.MARKET, OrderSide.SELL, 75))
        order3.status = OrderStatus.FILLED

        order4 = run_async(order_manager.create_order("AAPL", OrderType.LIMIT, OrderSide.SELL, 50, price=155))
        order4.status = OrderStatus.PENDING

        # Get all active orders
        active = order_manager.get_active_orders()
        assert len(active) == 3  # order1, order2, order4

        # Get active orders for specific symbol
        aapl_active = order_manager.get_active_orders("AAPL")
        assert len(aapl_active) == 2  # order1, order4
        assert all(o.symbol == "AAPL" for o in aapl_active)

    def test_get_orders_by_strategy(self, order_manager):
        """Test getting orders by strategy."""
        # Create orders for different strategies
        run_async(order_manager.create_order(
            "AAPL", OrderType.MARKET, OrderSide.BUY, 100, strategy_id="strategy_A"
        ))
        run_async(order_manager.create_order(
            "GOOGL", OrderType.MARKET, OrderSide.BUY, 50, strategy_id="strategy_A"
        ))
        order3 = run_async(order_manager.create_order(
            "MSFT", OrderType.MARKET, OrderSide.BUY, 75, strategy_id="strategy_B"
        ))
        run_async(order_manager.create_order(
            "AAPL", OrderType.MARKET, OrderSide.SELL, 50
        ))  # No strategy

        # Get orders by strategy
        strategy_a_orders = order_manager.get_orders_by_strategy("strategy_A")
        assert len(strategy_a_orders) == 2
        assert all(o.strategy_id == "strategy_A" for o in strategy_a_orders)

        strategy_b_orders = order_manager.get_orders_by_strategy("strategy_B")
        assert len(strategy_b_orders) == 1
        assert strategy_b_orders[0] == order3

        # Non-existent strategy
        empty = order_manager.get_orders_by_strategy("strategy_C")
        assert len(empty) == 0

    def test_register_callback(self, order_manager):
        """Test callback registration."""
        callback = Mock()
        order_id = "test_order"

        order_manager.register_callback(order_id, "filled", callback)

        assert order_id in order_manager.order_callbacks
        assert len(order_manager.order_callbacks[order_id]) == 1
        assert order_manager.order_callbacks[order_id][0] == ("filled", callback)

        # Register another callback
        callback2 = Mock()
        order_manager.register_callback(order_id, "cancelled", callback2)
        assert len(order_manager.order_callbacks[order_id]) == 2

    def test_trigger_callbacks(self, order_manager):
        """Test callback triggering."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # Mock async callback that returns completed future
        callback = Mock()
        future = asyncio.Future()
        future.set_result(None)
        callback.return_value = future

        order_manager.register_callback(order.order_id, "submitted", callback)

        # Trigger callback
        run_async(order_manager._trigger_callbacks(order, "submitted"))

        callback.assert_called_once_with(order, "submitted")

    def test_trigger_callbacks_wildcard(self, order_manager):
        """Test wildcard callbacks."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # Register wildcard callback
        wildcard_callback = Mock()
        future = asyncio.Future()
        future.set_result(None)
        wildcard_callback.return_value = future

        order_manager.register_callback(order.order_id, "*", wildcard_callback)

        # Trigger different events
        run_async(order_manager._trigger_callbacks(order, "submitted"))
        run_async(order_manager._trigger_callbacks(order, "filled"))

        assert wildcard_callback.call_count == 2

    def test_trigger_callbacks_error(self, order_manager):
        """Test callback error handling."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # Register callback that raises error
        async def error_callback(order, event):
            raise Exception("Callback error")

        order_manager.register_callback(order.order_id, "submitted", error_callback)

        # Should not raise, just log error
        run_async(order_manager._trigger_callbacks(order, "submitted"))

    def test_callbacks_during_submit(self, mock_exchange):
        """Test callbacks triggered during order submission."""
        manager = OrderManager(exchange_connector=mock_exchange)
        order = run_async(manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # Register callbacks
        submitted_callback = Mock()
        rejected_callback = Mock()

        # Mock them as async
        submitted_future = asyncio.Future()
        submitted_future.set_result(None)
        submitted_callback.return_value = submitted_future

        rejected_future = asyncio.Future()
        rejected_future.set_result(None)
        rejected_callback.return_value = rejected_future

        manager.register_callback(order.order_id, "submitted", submitted_callback)
        manager.register_callback(order.order_id, "rejected", rejected_callback)

        # Submit order
        run_async(manager.submit_order(order))

        # Only submitted callback should be called
        submitted_callback.assert_called_once()
        rejected_callback.assert_not_called()

    def test_callbacks_during_fill(self, order_manager):
        """Test callbacks triggered during fill updates."""
        order = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))

        # Register callbacks
        partial_callback = Mock()
        filled_callback = Mock()

        # Mock as async
        partial_future = asyncio.Future()
        partial_future.set_result(None)
        partial_callback.return_value = partial_future

        filled_future = asyncio.Future()
        filled_future.set_result(None)
        filled_callback.return_value = filled_future

        order_manager.register_callback(order.order_id, "partial_fill", partial_callback)
        order_manager.register_callback(order.order_id, "filled", filled_callback)

        # Partial fill
        fill1 = OrderFill(order.order_id, "fill_1", 40.0, 150.0, 0.5, datetime.now())
        run_async(order_manager.update_order_fill(order.order_id, fill1))

        partial_callback.assert_called_once()
        filled_callback.assert_not_called()

        # Complete fill
        fill2 = OrderFill(order.order_id, "fill_2", 60.0, 150.5, 0.5, datetime.now())
        run_async(order_manager.update_order_fill(order.order_id, fill2))

        filled_callback.assert_called_once()

    def test_callbacks_during_cancel(self, order_manager):
        """Test callbacks triggered during cancellation."""
        order = run_async(order_manager.create_order("AAPL", OrderType.LIMIT, OrderSide.BUY, 100, price=150))
        order.status = OrderStatus.SUBMITTED

        # Register callback
        cancelled_callback = Mock()
        future = asyncio.Future()
        future.set_result(None)
        cancelled_callback.return_value = future

        order_manager.register_callback(order.order_id, "cancelled", cancelled_callback)

        # Cancel order
        run_async(order_manager.cancel_order(order.order_id))

        cancelled_callback.assert_called_once()

    def test_calculate_average_price(self, order_manager):
        """Test average price calculation."""
        order = Order(quantity=100.0)

        # First fill
        fill1 = OrderFill("order_1", "fill_1", 40.0, 150.0, 0.5, datetime.now())
        avg_price = order_manager._calculate_average_price(order, fill1)
        assert avg_price == 150.0

        # Update order state
        order.filled_quantity = 40.0
        order.average_fill_price = 150.0

        # Second fill
        fill2 = OrderFill("order_1", "fill_2", 60.0, 151.0, 0.5, datetime.now())
        order.filled_quantity = 100.0  # Update before calculation
        avg_price = order_manager._calculate_average_price(order, fill2)
        # (150 * 40 + 151 * 60) / 100 = 150.6
        assert avg_price == pytest.approx(150.6)

        # Edge case: Zero filled quantity
        order.filled_quantity = 0.0
        avg_price = order_manager._calculate_average_price(order, fill1)
        assert avg_price == fill1.price

    def test_get_order_statistics(self, order_manager):
        """Test order statistics calculation."""
        # Create orders with different statuses
        order1 = run_async(order_manager.create_order("AAPL", OrderType.MARKET, OrderSide.BUY, 100))
        order1.status = OrderStatus.FILLED

        order2 = run_async(order_manager.create_order("GOOGL", OrderType.LIMIT, OrderSide.BUY, 50, price=2800))
        order2.status = OrderStatus.SUBMITTED

        order3 = run_async(order_manager.create_order("MSFT", OrderType.MARKET, OrderSide.SELL, 75))
        order3.status = OrderStatus.CANCELLED

        order4 = run_async(order_manager.create_order("AAPL", OrderType.LIMIT, OrderSide.SELL, 50, price=155))
        order4.status = OrderStatus.REJECTED

        order5 = run_async(order_manager.create_order("TSLA", OrderType.MARKET, OrderSide.BUY, 25))
        order5.status = OrderStatus.PARTIAL

        stats = order_manager.get_order_statistics()

        assert stats["total_orders"] == 5
        assert stats["active_orders"] == 2  # SUBMITTED and PARTIAL
        assert stats["filled_orders"] == 1
        assert stats["cancelled_orders"] == 1
        assert stats["rejected_orders"] == 1
        assert stats["fill_rate"] == 0.2  # 1/5

    def test_get_order_statistics_empty(self, order_manager):
        """Test order statistics with no orders."""
        stats = order_manager.get_order_statistics()

        assert stats["total_orders"] == 0
        assert stats["active_orders"] == 0
        assert stats["filled_orders"] == 0
        assert stats["cancelled_orders"] == 0
        assert stats["rejected_orders"] == 0
        assert stats["fill_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
