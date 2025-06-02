"""
Tests for enhanced order manager.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from algostack.adapters.paper_executor import PaperExecutor
from algostack.core.engine.enhanced_order_manager import (
    EnhancedOrderManager,
    OrderEventType,
)
from algostack.core.executor import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)


@pytest.fixture
def order_manager():
    """Create order manager fixture."""
    return EnhancedOrderManager()


@pytest.fixture
def paper_executor():
    """Create paper executor fixture."""
    config = {
        "initial_capital": 100000,
        "commission": 1.0,
        "slippage": 0.0,
        "fill_delay": 0.01,
    }
    return PaperExecutor(config)


class TestEnhancedOrderManager:
    """Test enhanced order manager."""
    
    @pytest.mark.asyncio
    async def test_add_executor(self, order_manager, paper_executor):
        """Test adding executor."""
        order_manager.add_executor("paper", paper_executor)
        
        assert "paper" in order_manager.executors
        assert order_manager.active_executor == "paper"
    
    @pytest.mark.asyncio
    async def test_create_order(self, order_manager):
        """Test order creation."""
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            strategy_id="test_strategy",
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.metadata["strategy_id"] == "test_strategy"
        assert order.order_id in order_manager._orders
    
    @pytest.mark.asyncio
    async def test_submit_order(self, order_manager, paper_executor):
        """Test order submission."""
        # Setup
        await paper_executor.connect()
        paper_executor.update_price("AAPL", 150.0)
        order_manager.add_executor("paper", paper_executor)
        
        # Create and submit order
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        
        success = await order_manager.submit_order(order)
        assert success is True
        assert order.metadata.get("executor") == "paper"
        
        await paper_executor.disconnect()
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager, paper_executor):
        """Test order cancellation."""
        # Setup
        await paper_executor.connect()
        paper_executor.update_price("AAPL", 150.0)
        order_manager.add_executor("paper", paper_executor)
        
        # Create and submit limit order
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=140.0,  # Won't fill immediately
        )
        
        await order_manager.submit_order(order)
        
        # Cancel order
        success = await order_manager.cancel_order(order.order_id)
        assert success is True
        
        await paper_executor.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_active_orders(self, order_manager, paper_executor):
        """Test getting active orders."""
        # Setup
        await paper_executor.connect()
        paper_executor.update_price("AAPL", 150.0)
        paper_executor.update_price("MSFT", 300.0)
        order_manager.add_executor("paper", paper_executor)
        
        # Create multiple orders
        order1 = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=140.0,
        )
        
        order2 = await order_manager.create_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=290.0,
        )
        
        await order_manager.submit_order(order1)
        await order_manager.submit_order(order2)
        
        # Get all active orders
        active_orders = order_manager.get_active_orders()
        assert len(active_orders) == 2
        
        # Get active orders for specific symbol
        aapl_orders = order_manager.get_active_orders("AAPL")
        assert len(aapl_orders) == 1
        assert aapl_orders[0].symbol == "AAPL"
        
        await paper_executor.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_strategy_orders(self, order_manager):
        """Test getting orders by strategy."""
        # Create orders for different strategies
        order1 = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            strategy_id="strategy1",
        )
        
        order2 = await order_manager.create_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=50,
            strategy_id="strategy1",
        )
        
        order3 = await order_manager.create_order(
            symbol="GOOGL",
            side=OrderSide.BUY,
            quantity=25,
            strategy_id="strategy2",
        )
        
        # Check strategy orders
        strategy1_orders = order_manager.get_strategy_orders("strategy1")
        assert len(strategy1_orders) == 2
        
        strategy2_orders = order_manager.get_strategy_orders("strategy2")
        assert len(strategy2_orders) == 1
    
    @pytest.mark.asyncio
    async def test_order_event_callbacks(self, order_manager, paper_executor):
        """Test order event callbacks."""
        # Setup callback tracking
        events = []
        
        def callback(order, event_type, data):
            events.append((order.order_id, event_type, data))
        
        order_manager.register_event_callback(OrderEventType.CREATED, callback)
        order_manager.register_event_callback(OrderEventType.SUBMITTED, callback)
        order_manager.register_event_callback(OrderEventType.FILLED, callback)
        
        # Setup executor
        await paper_executor.connect()
        paper_executor.update_price("AAPL", 150.0)
        order_manager.add_executor("paper", paper_executor)
        
        # Create and submit order
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        
        await order_manager.submit_order(order)
        
        # Wait for fill
        await asyncio.sleep(0.1)
        
        # Check events
        event_types = [e[1] for e in events]
        assert OrderEventType.CREATED in event_types
        assert OrderEventType.SUBMITTED in event_types
        assert OrderEventType.FILLED in event_types
        
        await paper_executor.disconnect()
    
    @pytest.mark.asyncio
    async def test_position_retrieval(self, order_manager, paper_executor):
        """Test position retrieval from executor."""
        # Setup
        await paper_executor.connect()
        paper_executor.update_price("AAPL", 150.0)
        order_manager.add_executor("paper", paper_executor)
        
        # Submit and fill an order
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        
        await order_manager.submit_order(order)
        await asyncio.sleep(0.1)  # Wait for fill
        
        # Get positions
        positions = await order_manager.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 100
        
        await paper_executor.disconnect()
    
    @pytest.mark.asyncio
    async def test_order_statistics(self, order_manager, paper_executor):
        """Test order statistics tracking."""
        # Setup
        await paper_executor.connect()
        paper_executor.update_price("AAPL", 150.0)
        order_manager.add_executor("paper", paper_executor)
        
        # Submit multiple orders
        for i in range(5):
            order = await order_manager.create_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
            )
            await order_manager.submit_order(order)
        
        # Wait for fills
        await asyncio.sleep(0.1)
        
        # Check statistics
        stats = order_manager.get_order_statistics()
        assert stats["total_orders"] == 5
        assert stats["filled_orders"] == 5
        assert stats["fill_rate"] == 1.0
        assert stats["total_commission"] > 0
        
        await paper_executor.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_executors(self, order_manager):
        """Test managing multiple executors."""
        # Create multiple executors
        paper1 = PaperExecutor({"initial_capital": 100000})
        paper2 = PaperExecutor({"initial_capital": 200000})
        
        await paper1.connect()
        await paper2.connect()
        
        # Add executors
        order_manager.add_executor("paper1", paper1)
        order_manager.add_executor("paper2", paper2)
        
        # Check active executor
        assert order_manager.active_executor == "paper1"  # First added
        
        # Switch active executor
        order_manager.set_active_executor("paper2")
        assert order_manager.active_executor == "paper2"
        
        # Submit order to specific executor
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        
        success = await order_manager.submit_order(order, executor_name="paper1")
        assert success is True
        assert order.metadata["executor"] == "paper1"
        
        await paper1.disconnect()
        await paper2.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, order_manager, paper_executor):
        """Test error handling in order management."""
        # Don't connect executor
        order_manager.add_executor("paper", paper_executor)
        
        # Try to submit order - should fail
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        
        with pytest.raises(RuntimeError, match="not connected"):
            await order_manager.submit_order(order)
    
    @pytest.mark.asyncio
    async def test_wildcard_callbacks(self, order_manager):
        """Test wildcard event callbacks."""
        # Track all events
        all_events = []
        
        def wildcard_callback(order, event_type, data):
            all_events.append(event_type)
        
        order_manager.register_event_callback("*", wildcard_callback)
        
        # Create order
        order = await order_manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        
        # Check wildcard received all events
        assert OrderEventType.CREATED in all_events