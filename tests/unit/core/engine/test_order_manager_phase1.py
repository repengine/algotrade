"""Test suite for Phase 1 OrderManager implementations."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from core.engine.order_manager import (
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


class TestPhase1OrderManager:
    """Test suite for Phase 1 OrderManager critical methods."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager instance."""
        return OrderManager()

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        return Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test_strategy"
        )

    def test_add_order_valid(self, order_manager, sample_order):
        """Test adding a valid order."""
        order_id = "ORDER123"
        
        # Add order
        order_manager.add_order(order_id, sample_order)
        
        # Verify order was added
        assert order_id in order_manager.orders
        assert order_manager.orders[order_id] == sample_order
        assert sample_order.order_id == order_id
        assert order_id in order_manager.order_fills
        assert len(order_manager.order_fills[order_id]) == 0

    def test_add_order_duplicate_detection(self, order_manager, sample_order):
        """Test duplicate order detection."""
        # Add first order
        order_manager.add_order("ORDER1", sample_order)
        
        # Create duplicate order
        duplicate_order = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test_strategy"
        )
        
        # Should raise ValueError for duplicate
        with pytest.raises(ValueError, match="Duplicate order detected"):
            order_manager.add_order("ORDER2", duplicate_order)

    def test_add_order_duplicate_different_attributes(self, order_manager):
        """Test that orders with different attributes are not considered duplicates."""
        # Add first order
        order1 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="strategy1"
        )
        order_manager.add_order("ORDER1", order1)
        
        # Different symbol - not duplicate
        order2 = Order(
            symbol="GOOGL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="strategy1"
        )
        order_manager.add_order("ORDER2", order2)
        
        # Different side - not duplicate
        order3 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
            strategy_id="strategy1"
        )
        order_manager.add_order("ORDER3", order3)
        
        # Different quantity - not duplicate
        order4 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=200,
            strategy_id="strategy1"
        )
        order_manager.add_order("ORDER4", order4)
        
        # All orders should be added
        assert len(order_manager.orders) == 4

    def test_add_order_limit_price_duplicate(self, order_manager):
        """Test duplicate detection for limit orders considers price."""
        # Add first limit order
        order1 = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            strategy_id="strategy1"
        )
        order_manager.add_order("ORDER1", order1)
        
        # Same limit order - duplicate
        order2 = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            strategy_id="strategy1"
        )
        with pytest.raises(ValueError, match="Duplicate order detected"):
            order_manager.add_order("ORDER2", order2)
        
        # Different price - not duplicate
        order3 = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            price=151.0,
            strategy_id="strategy1"
        )
        order_manager.add_order("ORDER3", order3)
        
        assert len(order_manager.orders) == 2

    def test_add_order_invalid_quantity(self, order_manager):
        """Test adding order with invalid quantity."""
        order = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0,  # Invalid
            strategy_id="test"
        )
        
        with pytest.raises(ValueError, match="Order quantity must be positive"):
            order_manager.add_order("ORDER1", order)

    def test_update_order_status_basic(self, order_manager, sample_order):
        """Test basic order status update."""
        order_id = "ORDER123"
        order_manager.add_order(order_id, sample_order)
        
        # Update status to submitted
        order_manager.update_order_status(order_id, OrderStatus.SUBMITTED)
        
        assert sample_order.status == OrderStatus.SUBMITTED
        assert sample_order.updated_at > sample_order.created_at

    def test_update_order_status_with_fill(self, order_manager, sample_order):
        """Test order status update with fill information."""
        order_id = "ORDER123"
        order_manager.add_order(order_id, sample_order)
        
        # Update with fill data
        fill_data = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.5,
            'commission': 1.0,
            'timestamp': datetime.now()
        }
        
        order_manager.update_order_status(order_id, OrderStatus.FILLED, fill_data)
        
        # Check order updates
        assert sample_order.status == OrderStatus.FILLED
        assert sample_order.filled_quantity == 100
        assert sample_order.average_fill_price == 150.5
        
        # Check fill was recorded
        assert len(order_manager.order_fills[order_id]) == 1
        fill = order_manager.order_fills[order_id][0]
        assert fill.quantity == 100
        assert fill.price == 150.5
        assert fill.commission == 1.0

    def test_update_order_status_partial_fills(self, order_manager):
        """Test handling of partial fills."""
        order = Order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=1000,
            price=150.0,
            strategy_id="test"
        )
        order_id = "LARGE_ORDER"
        order_manager.add_order(order_id, order)
        
        # First partial fill
        fill1 = {
            'symbol': 'AAPL',
            'quantity': 300,
            'price': 149.95,
            'timestamp': datetime.now()
        }
        order_manager.update_order_status(order_id, OrderStatus.PARTIAL, fill1)
        
        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 300
        assert order.average_fill_price == 149.95
        
        # Second partial fill
        fill2 = {
            'symbol': 'AAPL',
            'quantity': 700,
            'price': 150.00,
            'timestamp': datetime.now()
        }
        order_manager.update_order_status(order_id, OrderStatus.FILLED, fill2)
        
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1000
        # Average price: (300 * 149.95 + 700 * 150.00) / 1000
        expected_avg = (300 * 149.95 + 700 * 150.00) / 1000
        assert abs(order.average_fill_price - expected_avg) < 0.01

    def test_update_order_status_nonexistent(self, order_manager):
        """Test updating status of non-existent order."""
        # Should not raise, just log error
        order_manager.update_order_status("FAKE_ORDER", OrderStatus.FILLED)
        # No assertion needed - just ensure no exception

    @pytest.mark.asyncio
    async def test_update_order_status_callbacks(self, order_manager, sample_order):
        """Test that callbacks are triggered on status updates."""
        order_id = "ORDER123"
        order_manager.add_order(order_id, sample_order)
        
        # Register callback
        callback_called = False
        async def test_callback(order, event):
            nonlocal callback_called
            callback_called = True
            assert order == sample_order
            assert event == "filled"
        
        order_manager.register_callback(order_id, "filled", test_callback)
        
        # Update status
        order_manager.update_order_status(order_id, OrderStatus.FILLED)
        
        # Allow async callback to execute
        await asyncio.sleep(0.1)
        
        assert callback_called

    def test_duplicate_detection_time_window(self, order_manager):
        """Test that duplicate detection respects time window."""
        import time
        
        # Mock the order creation time to be outside window
        order1 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test"
        )
        order1.created_at = datetime(2020, 1, 1)  # Old timestamp
        order_manager.add_order("ORDER1", order1)
        
        # Same order attributes but outside time window - not duplicate
        order2 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test"
        )
        order_manager.add_order("ORDER2", order2)  # Should not raise
        
        assert len(order_manager.orders) == 2

    def test_duplicate_detection_inactive_orders(self, order_manager):
        """Test that inactive orders are not considered for duplicate detection."""
        # Add and fill first order
        order1 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test"
        )
        order_manager.add_order("ORDER1", order1)
        order_manager.update_order_status("ORDER1", OrderStatus.FILLED)
        
        # Same attributes but first order is filled - not duplicate
        order2 = Order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test"
        )
        order_manager.add_order("ORDER2", order2)  # Should not raise
        
        assert len(order_manager.orders) == 2

    def test_order_id_update(self, order_manager):
        """Test that order ID is updated when different from provided ID."""
        order = Order(
            order_id="ORIGINAL_ID",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="test"
        )
        
        new_id = "NEW_ID"
        order_manager.add_order(new_id, order)
        
        # Order ID should be updated
        assert order.order_id == new_id
        assert new_id in order_manager.orders
        assert "ORIGINAL_ID" not in order_manager.orders