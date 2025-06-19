"""Test suite for Phase 1 Order State Synchronization."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.executor import Fill, Order, OrderSide, OrderStatus, OrderType
from src.core.order_state_sync import (
    OrderStateSynchronizer,
    OrderSyncState,
    SyncStatus,
)


class TestPhase1OrderStateSynchronization:
    """Test suite for Phase 1 order state synchronization."""

    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        manager = Mock()
        manager.orders = {}
        manager.get_active_orders = Mock(return_value=[])
        return manager

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor."""
        executor = Mock()
        executor.get_orders = AsyncMock(return_value=[])
        return executor

    @pytest.fixture
    def synchronizer(self, mock_order_manager, mock_executor):
        """Create OrderStateSynchronizer instance."""
        config = {
            "sync_interval": 1.0,
            "stale_order_hours": 24,
            "duplicate_window": 2.0
        }
        return OrderStateSynchronizer(mock_order_manager, mock_executor, config)

    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        order = Order(
            order_id="ORDER123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            status=OrderStatus.SUBMITTED
        )
        order.filled_quantity = 0
        order.submitted_at = datetime.now()
        return order

    @pytest.mark.asyncio
    async def test_start_stop_synchronization(self, synchronizer):
        """Test starting and stopping synchronization."""
        # Start synchronization
        await synchronizer.start()
        assert synchronizer._running is True
        assert synchronizer._sync_task is not None
        
        # Allow sync loop to run briefly
        await asyncio.sleep(0.1)
        
        # Stop synchronization
        await synchronizer.stop()
        assert synchronizer._running is False

    @pytest.mark.asyncio
    async def test_sync_matched_orders(self, synchronizer, mock_order_manager, mock_executor, sample_order):
        """Test synchronization of orders that match between systems."""
        # Setup local order
        mock_order_manager.orders = {"ORDER123": sample_order}
        mock_order_manager.get_active_orders.return_value = [sample_order]
        
        # Setup matching broker order
        broker_order = {
            "orderId": "ORDER123",
            "status": "Submitted",
            "filledQuantity": 0,
            "avgFillPrice": 0
        }
        mock_executor.get_orders.return_value = [broker_order]
        
        # Perform sync
        results = await synchronizer.sync_all_orders()
        
        assert results["synced"] == 1
        assert results["mismatched"] == 0
        assert results["local_only"] == 0
        assert results["broker_only"] == 0
        
        # Check sync state
        sync_state = synchronizer.get_sync_state("ORDER123")
        assert sync_state is not None
        assert sync_state.sync_status == SyncStatus.SYNCED
        assert len(sync_state.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_sync_status_mismatch(self, synchronizer, mock_order_manager, mock_executor, sample_order):
        """Test synchronization when order status differs."""
        # Local order is SUBMITTED
        mock_order_manager.orders = {"ORDER123": sample_order}
        mock_order_manager.get_active_orders.return_value = [sample_order]
        
        # Broker shows order as FILLED
        broker_order = {
            "orderId": "ORDER123",
            "status": "Filled",
            "filledQuantity": 100,
            "avgFillPrice": 150.0
        }
        mock_executor.get_orders.return_value = [broker_order]
        
        # Perform sync
        results = await synchronizer.sync_all_orders()
        
        assert results["mismatched"] == 1
        
        # Order status should be updated to FILLED
        assert sample_order.status == OrderStatus.FILLED
        
        # Check sync state
        sync_state = synchronizer.get_sync_state("ORDER123")
        assert sync_state.sync_status == SyncStatus.MISMATCHED
        assert len(sync_state.discrepancies) > 0
        assert "Status mismatch" in sync_state.discrepancies[0]

    @pytest.mark.asyncio
    async def test_missed_fill_detection(self, synchronizer, mock_order_manager, mock_executor, sample_order):
        """Test detection and processing of missed fills."""
        # Setup callbacks to track missed fills
        missed_fills = []
        async def missed_fill_callback(order, fill):
            missed_fills.append((order, fill))
        
        synchronizer.register_missed_fill_callback(missed_fill_callback)
        
        # Local order shows no fills
        sample_order.filled_quantity = 0
        mock_order_manager.orders = {"ORDER123": sample_order}
        mock_order_manager.get_active_orders.return_value = [sample_order]
        
        # Broker shows partial fill
        broker_order = {
            "orderId": "ORDER123",
            "status": "Submitted",
            "filledQuantity": 50,
            "avgFillPrice": 149.95,
            "commission": 1.0
        }
        mock_executor.get_orders.return_value = [broker_order]
        
        # Perform sync
        await synchronizer.sync_all_orders()
        
        # Check fill was processed
        assert sample_order.filled_quantity == 50
        assert sample_order.average_fill_price == 149.95
        
        # Check callback was triggered
        assert len(missed_fills) == 1
        order, fill = missed_fills[0]
        assert order == sample_order
        assert fill.quantity == 50
        assert fill.price == 149.95
        
        # Check metrics
        assert synchronizer.metrics.missed_fills_detected == 1

    @pytest.mark.asyncio
    async def test_local_only_order_handling(self, synchronizer, mock_order_manager, mock_executor, sample_order):
        """Test handling of orders that exist only locally."""
        # Local order exists
        mock_order_manager.orders = {"ORDER123": sample_order}
        mock_order_manager.get_active_orders.return_value = [sample_order]
        
        # No broker order
        mock_executor.get_orders.return_value = []
        
        # Perform sync
        results = await synchronizer.sync_all_orders()
        
        assert results["local_only"] == 1
        
        # Check sync state
        sync_state = synchronizer.get_sync_state("ORDER123")
        assert sync_state.sync_status == SyncStatus.LOCAL_ONLY
        assert sync_state.retry_count == 1

    @pytest.mark.asyncio
    async def test_broker_only_order_handling(self, synchronizer, mock_order_manager, mock_executor):
        """Test handling of orders that exist only at broker."""
        # No local orders
        mock_order_manager.get_active_orders.return_value = []
        
        # Broker has an order
        broker_order = {
            "orderId": "BROKER123",
            "status": "Submitted",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100
        }
        mock_executor.get_orders.return_value = [broker_order]
        
        # Setup callback to track orphaned orders
        orphaned_orders = []
        async def mismatch_callback(sync_state):
            if sync_state.sync_status == SyncStatus.BROKER_ONLY:
                orphaned_orders.append(sync_state)
        
        synchronizer.register_mismatch_callback(mismatch_callback)
        
        # Perform sync
        results = await synchronizer.sync_all_orders()
        
        assert results["broker_only"] == 1
        
        # Check callback was triggered
        assert len(orphaned_orders) == 1
        assert orphaned_orders[0].order_id == "BROKER123"
        
        # Check metrics
        assert synchronizer.metrics.orphaned_orders_found == 1

    @pytest.mark.asyncio
    async def test_duplicate_order_detection(self, synchronizer):
        """Test duplicate order detection."""
        # Check first order - not duplicate
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert is_duplicate is False
        
        # Check same order immediately - should be duplicate
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert is_duplicate is True
        
        # Check metrics
        assert synchronizer.metrics.duplicate_orders_prevented == 1
        
        # Different attributes - not duplicate
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "SELL", 100)
        assert is_duplicate is False
        
        is_duplicate = await synchronizer.check_duplicate_order("GOOGL", "BUY", 100)
        assert is_duplicate is False
        
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 200)
        assert is_duplicate is False

    @pytest.mark.asyncio
    async def test_duplicate_window_expiration(self, synchronizer):
        """Test that duplicate detection respects time window."""
        # First order
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert is_duplicate is False
        
        # Wait for window to expire
        await asyncio.sleep(2.1)  # Window is 2 seconds
        
        # Same order after window - not duplicate
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert is_duplicate is False

    @pytest.mark.asyncio
    async def test_stale_order_detection(self, synchronizer, mock_order_manager, mock_executor):
        """Test detection of stale orders."""
        # Create old order
        old_order = Order(
            order_id="OLD123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            status=OrderStatus.SUBMITTED
        )
        old_order.submitted_at = datetime.now() - timedelta(hours=25)  # Over 24 hours old
        
        mock_order_manager.orders = {"OLD123": old_order}
        mock_order_manager.get_active_orders.return_value = [old_order]
        mock_executor.get_orders.return_value = []
        
        # Perform sync
        await synchronizer.sync_all_orders()
        
        # Order should be cancelled
        assert old_order.status == OrderStatus.CANCELLED
        
        # Check sync state
        sync_state = synchronizer.get_sync_state("OLD123")
        assert sync_state.sync_status == SyncStatus.STALE

    @pytest.mark.asyncio
    async def test_sync_metrics(self, synchronizer, mock_order_manager, mock_executor, sample_order):
        """Test synchronization metrics tracking."""
        # Setup some orders
        mock_order_manager.orders = {"ORDER123": sample_order}
        mock_order_manager.get_active_orders.return_value = [sample_order]
        
        broker_order = {
            "orderId": "ORDER123",
            "status": "Submitted",
            "filledQuantity": 0
        }
        mock_executor.get_orders.return_value = [broker_order]
        
        # Perform multiple syncs
        for _ in range(3):
            await synchronizer.sync_all_orders()
        
        # Check metrics
        metrics = synchronizer.get_metrics()
        assert metrics["total_syncs"] == 3
        assert metrics["successful_syncs"] == 3
        assert metrics["failed_syncs"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["average_sync_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_sync_error_handling(self, synchronizer, mock_order_manager, mock_executor):
        """Test error handling during synchronization."""
        # Make executor raise an error
        mock_executor.get_orders.side_effect = Exception("Connection error")
        
        # Perform sync - should not raise
        results = await synchronizer.sync_all_orders()
        
        assert results["errors"] == 1
        
        # Check metrics
        assert synchronizer.metrics.failed_syncs == 1

    @pytest.mark.asyncio
    async def test_callback_registration(self, synchronizer):
        """Test callback registration and execution."""
        # Track callback executions
        callback_data = {
            "mismatch": [],
            "duplicate": [],
            "missed_fill": []
        }
        
        async def mismatch_cb(sync_state):
            callback_data["mismatch"].append(sync_state)
        
        async def duplicate_cb(dup_info):
            callback_data["duplicate"].append(dup_info)
        
        async def missed_fill_cb(order, fill):
            callback_data["missed_fill"].append((order, fill))
        
        # Register callbacks
        synchronizer.register_mismatch_callback(mismatch_cb)
        synchronizer.register_duplicate_callback(duplicate_cb)
        synchronizer.register_missed_fill_callback(missed_fill_cb)
        
        # Trigger duplicate detection
        await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        await synchronizer.check_duplicate_order("AAPL", "BUY", 100)  # Duplicate
        
        assert len(callback_data["duplicate"]) == 1
        assert callback_data["duplicate"][0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_status_mapping(self, synchronizer):
        """Test broker status mapping."""
        # Test various broker status mappings
        assert synchronizer._map_broker_status("PendingSubmit") == OrderStatus.PENDING
        assert synchronizer._map_broker_status("Submitted") == OrderStatus.SUBMITTED
        assert synchronizer._map_broker_status("Filled") == OrderStatus.FILLED
        assert synchronizer._map_broker_status("Cancelled") == OrderStatus.CANCELLED
        assert synchronizer._map_broker_status("Rejected") == OrderStatus.REJECTED
        assert synchronizer._map_broker_status("Unknown") == OrderStatus.PENDING  # Default

    def test_broker_authoritative_status(self, synchronizer):
        """Test which statuses are considered broker-authoritative."""
        assert synchronizer._is_broker_authoritative(OrderStatus.FILLED) is True
        assert synchronizer._is_broker_authoritative(OrderStatus.CANCELLED) is True
        assert synchronizer._is_broker_authoritative(OrderStatus.REJECTED) is True
        assert synchronizer._is_broker_authoritative(OrderStatus.EXPIRED) is True
        assert synchronizer._is_broker_authoritative(OrderStatus.SUBMITTED) is False
        assert synchronizer._is_broker_authoritative(OrderStatus.PENDING) is False