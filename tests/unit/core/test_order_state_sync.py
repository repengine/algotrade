"""
Unit tests for Order State Synchronization.

Tests the order state synchronization mechanism that prevents duplicate orders
and detects missed fills.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from core.executor import Order, OrderSide, OrderStatus, OrderType
from core.order_state_sync import (
    OrderStateSynchronizer,
    SyncStatus,
)


@pytest.fixture
def mock_order_manager():
    """Create mock order manager."""
    manager = Mock()
    manager.orders = {}
    manager.get_active_orders = Mock(return_value=[])
    return manager


@pytest.fixture
def mock_executor():
    """Create mock executor."""
    executor = Mock()
    executor.get_orders = AsyncMock(return_value=[])
    return executor


@pytest.fixture
def synchronizer(mock_order_manager, mock_executor):
    """Create order state synchronizer."""
    config = {
        "sync_interval": 0.1,  # Fast sync for tests
        "stale_order_hours": 1,
        "duplicate_window": 2.0,
    }
    return OrderStateSynchronizer(mock_order_manager, mock_executor, config)


@pytest.fixture
def sample_order():
    """Create sample order."""
    return Order(
        order_id="TEST001",
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        limit_price=150.0,
        status=OrderStatus.SUBMITTED,
        filled_quantity=0,
        submitted_at=datetime.now()
    )


@pytest.fixture
def sample_broker_order():
    """Create sample broker order."""
    return {
        "orderId": "TEST001",
        "status": "Submitted",
        "filledQuantity": 0,
        "avgFillPrice": 0,
        "commission": 0
    }


class TestOrderStateSynchronizer:
    """Test order state synchronizer."""

    @pytest.mark.asyncio
    async def test_initialization(self, synchronizer):
        """Test synchronizer initialization."""
        assert synchronizer.sync_interval == 0.1
        assert synchronizer.stale_order_threshold == 1
        assert synchronizer.duplicate_window_seconds == 2.0
        assert not synchronizer._running
        assert len(synchronizer.sync_states) == 0
        assert synchronizer.metrics.total_syncs == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, synchronizer):
        """Test starting and stopping synchronizer."""
        # Start
        await synchronizer.start()
        assert synchronizer._running
        assert synchronizer._sync_task is not None

        # Give it time to run
        await asyncio.sleep(0.2)

        # Stop
        await synchronizer.stop()
        assert not synchronizer._running
        assert synchronizer.metrics.total_syncs > 0

    @pytest.mark.asyncio
    async def test_sync_matching_orders(
        self,
        synchronizer,
        mock_order_manager,
        mock_executor,
        sample_order,
        sample_broker_order
    ):
        """Test syncing orders that match."""
        # Setup
        mock_order_manager.get_active_orders.return_value = [sample_order]
        mock_executor.get_orders.return_value = [sample_broker_order]

        # Sync
        results = await synchronizer.sync_all_orders()

        # Verify
        assert results["synced"] == 1
        assert results["mismatched"] == 0
        assert results["local_only"] == 0
        assert results["broker_only"] == 0

        # Check sync state
        sync_state = synchronizer.get_sync_state("TEST001")
        assert sync_state is not None
        assert sync_state.sync_status == SyncStatus.SYNCED
        assert len(sync_state.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_sync_status_mismatch(
        self,
        synchronizer,
        mock_order_manager,
        mock_executor,
        sample_order,
        sample_broker_order
    ):
        """Test syncing orders with status mismatch."""
        # Setup - broker shows filled
        sample_broker_order["status"] = "Filled"
        sample_broker_order["filledQuantity"] = 100
        sample_broker_order["avgFillPrice"] = 149.50

        mock_order_manager.get_active_orders.return_value = [sample_order]
        mock_executor.get_orders.return_value = [sample_broker_order]

        # Sync
        results = await synchronizer.sync_all_orders()

        # Verify
        assert results["mismatched"] == 1
        assert sample_order.status == OrderStatus.FILLED  # Updated
        assert sample_order.filled_quantity == 100
        assert synchronizer.metrics.missed_fills_detected == 1

    @pytest.mark.asyncio
    async def test_local_only_order(
        self,
        synchronizer,
        mock_order_manager,
        mock_executor,
        sample_order
    ):
        """Test order that exists only locally."""
        # Setup
        mock_order_manager.get_active_orders.return_value = [sample_order]
        mock_executor.get_orders.return_value = []

        # Sync
        results = await synchronizer.sync_all_orders()

        # Verify
        assert results["local_only"] == 1

        # Check sync state
        sync_state = synchronizer.get_sync_state("TEST001")
        assert sync_state is not None
        assert sync_state.retry_count == 1

    @pytest.mark.asyncio
    async def test_broker_only_order(
        self,
        synchronizer,
        mock_order_manager,
        mock_executor,
        sample_broker_order
    ):
        """Test order that exists only at broker."""
        # Setup
        mock_order_manager.get_active_orders.return_value = []
        mock_executor.get_orders.return_value = [sample_broker_order]

        # Track callbacks
        callback_called = False
        async def mismatch_callback(sync_state):
            nonlocal callback_called
            callback_called = True
            assert sync_state.sync_status == SyncStatus.BROKER_ONLY

        synchronizer.register_mismatch_callback(mismatch_callback)

        # Sync
        results = await synchronizer.sync_all_orders()

        # Verify
        assert results["broker_only"] == 1
        assert synchronizer.metrics.orphaned_orders_found == 1
        assert callback_called

    @pytest.mark.asyncio
    async def test_stale_order_handling(
        self,
        synchronizer,
        mock_order_manager,
        mock_executor
    ):
        """Test handling of stale orders."""
        # Create old order
        old_order = Order(
            order_id="OLD001",
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.0,
            status=OrderStatus.SUBMITTED,
            filled_quantity=0,
            submitted_at=datetime.now() - timedelta(hours=2)
        )

        mock_order_manager.get_active_orders.return_value = [old_order]
        mock_executor.get_orders.return_value = []

        # Sync
        await synchronizer.sync_all_orders()

        # Verify order was cancelled
        assert old_order.status == OrderStatus.CANCELLED

        sync_state = synchronizer.get_sync_state("OLD001")
        assert sync_state.sync_status == SyncStatus.STALE

    @pytest.mark.asyncio
    async def test_duplicate_order_detection(self, synchronizer):
        """Test duplicate order detection."""
        # First order
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert not is_duplicate

        # Immediate duplicate
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert is_duplicate
        assert synchronizer.metrics.duplicate_orders_prevented == 1

        # Wait for window to pass
        await asyncio.sleep(2.1)

        # Should not be duplicate anymore
        is_duplicate = await synchronizer.check_duplicate_order("AAPL", "BUY", 100)
        assert not is_duplicate

    @pytest.mark.asyncio
    async def test_missed_fill_processing(
        self,
        synchronizer,
        mock_order_manager,
        mock_executor,
        sample_order,
        sample_broker_order
    ):
        """Test processing of missed fills."""
        # Setup partial fill at broker
        sample_broker_order["filledQuantity"] = 50
        sample_broker_order["avgFillPrice"] = 149.75

        mock_order_manager.get_active_orders.return_value = [sample_order]
        mock_executor.get_orders.return_value = [sample_broker_order]

        # Track callbacks
        fill_detected = False
        async def fill_callback(order, fill):
            nonlocal fill_detected
            fill_detected = True
            assert order.order_id == "TEST001"
            assert fill.quantity == 50
            assert fill.price == 149.75

        synchronizer.register_missed_fill_callback(fill_callback)

        # Sync
        await synchronizer.sync_all_orders()

        # Verify
        assert sample_order.filled_quantity == 50
        assert sample_order.average_fill_price == 149.75
        assert fill_detected
        assert synchronizer.metrics.missed_fills_detected == 1

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, synchronizer, mock_order_manager, mock_executor):
        """Test metrics tracking."""
        # Setup some orders
        orders = [
            Order(
                order_id=f"TEST{i}",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now()
            )
            for i in range(3)
        ]
        broker_orders = [
            {"orderId": f"TEST{i}", "status": "Submitted", "filledQuantity": 0}
            for i in range(2)
        ]

        mock_order_manager.get_active_orders.return_value = orders
        mock_executor.get_orders.return_value = broker_orders

        # Run multiple syncs
        for _ in range(5):
            await synchronizer.sync_all_orders()

        # Check metrics
        metrics = synchronizer.get_metrics()
        assert metrics["total_syncs"] == 5
        assert metrics["successful_syncs"] == 5
        assert metrics["failed_syncs"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["average_sync_duration_ms"] > 0
        assert metrics["active_sync_states"] == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, synchronizer, mock_order_manager, mock_executor):
        """Test error handling during sync."""
        # Make executor throw error
        mock_executor.get_orders.side_effect = Exception("Connection error")

        # Sync should handle error gracefully
        results = await synchronizer.sync_all_orders()

        assert results["errors"] == 1
        assert synchronizer.metrics.failed_syncs == 1
        assert synchronizer.metrics.successful_syncs == 0

    @pytest.mark.asyncio
    async def test_concurrent_sync_protection(self, synchronizer):
        """Test that concurrent syncs are prevented."""
        # Start multiple syncs
        sync_tasks = [
            asyncio.create_task(synchronizer.sync_all_orders())
            for _ in range(5)
        ]

        # Wait for all to complete
        results = await asyncio.gather(*sync_tasks)

        # Should have completed without errors
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_status_mapping(self, synchronizer):
        """Test broker status mapping."""
        mappings = {
            "PendingSubmit": OrderStatus.PENDING,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
            "Unknown": OrderStatus.PENDING,  # Default
        }

        for broker_status, expected in mappings.items():
            assert synchronizer._map_broker_status(broker_status) == expected

    def test_broker_authoritative_states(self, synchronizer):
        """Test which states are considered broker-authoritative."""
        authoritative = [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]

        non_authoritative = [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED
        ]

        for status in authoritative:
            assert synchronizer._is_broker_authoritative(status)

        for status in non_authoritative:
            assert not synchronizer._is_broker_authoritative(status)
