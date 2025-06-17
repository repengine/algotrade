"""
Order State Synchronization Module.

This module provides comprehensive order state synchronization between
local order management and broker systems. It prevents duplicate orders,
detects missed fills, and ensures consistent state across systems.

CRITICAL FOR CAPITAL PRESERVATION: Prevents unintended positions from
order state mismatches.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Set, Dict, List

from algostack.core.executor import Order, OrderStatus, Fill
from algostack.utils.logging import setup_logger

logger = setup_logger(__name__)


class SyncStatus(Enum):
    """Synchronization status for orders."""
    
    SYNCED = "synced"
    LOCAL_ONLY = "local_only"
    BROKER_ONLY = "broker_only"
    MISMATCHED = "mismatched"
    STALE = "stale"


@dataclass
class OrderSyncState:
    """Represents the synchronization state of an order."""
    
    order_id: str
    local_order: Optional[Order] = None
    broker_order: Optional[dict] = None
    sync_status: SyncStatus = SyncStatus.LOCAL_ONLY
    last_sync_time: datetime = field(default_factory=datetime.now)
    discrepancies: List[str] = field(default_factory=list)
    fill_discrepancies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class SyncMetrics:
    """Metrics for order synchronization."""
    
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    discrepancies_found: int = 0
    discrepancies_resolved: int = 0
    duplicate_orders_prevented: int = 0
    missed_fills_detected: int = 0
    orphaned_orders_found: int = 0
    last_sync_duration_ms: float = 0.0
    average_sync_duration_ms: float = 0.0


class OrderStateSynchronizer:
    """
    Manages order state synchronization between local and broker systems.
    
    Key responsibilities:
    - Periodic state reconciliation
    - Duplicate order prevention
    - Missed fill detection
    - Orphaned order cleanup
    - State mismatch resolution
    """
    
    def __init__(
        self,
        order_manager: Any,
        executor: Any,
        config: Optional[dict] = None
    ):
        """
        Initialize order state synchronizer.
        
        Args:
            order_manager: Local order manager instance
            executor: Broker executor instance
            config: Configuration options
        """
        self.order_manager = order_manager
        self.executor = executor
        self.config = config or {}
        
        # Configuration
        self.sync_interval = self.config.get("sync_interval", 5.0)  # seconds
        self.stale_order_threshold = self.config.get("stale_order_hours", 24)
        self.enable_auto_resolve = self.config.get("enable_auto_resolve", False)
        self.duplicate_window_seconds = self.config.get("duplicate_window", 2.0)
        
        # State tracking
        self.sync_states: Dict[str, OrderSyncState] = {}
        self.recent_orders: Dict[str, datetime] = {}  # For duplicate detection
        self.metrics = SyncMetrics()
        
        # Callbacks
        self.mismatch_callbacks: List[Callable] = []
        self.duplicate_callbacks: List[Callable] = []
        self.missed_fill_callbacks: List[Callable] = []
        
        # Sync control
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._sync_lock = asyncio.Lock()
        
        logger.info("Order state synchronizer initialized")
    
    async def start(self) -> None:
        """Start synchronization process."""
        if self._running:
            logger.warning("Synchronizer already running")
            return
        
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Order state synchronization started")
    
    async def stop(self) -> None:
        """Stop synchronization process."""
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Order state synchronization stopped")
    
    async def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self._running:
            try:
                await self.sync_all_orders()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def sync_all_orders(self) -> Dict[str, Any]:
        """
        Perform full synchronization of all active orders.
        
        Returns:
            Synchronization results summary
        """
        async with self._sync_lock:
            start_time = datetime.now()
            results = {
                "synced": 0,
                "mismatched": 0,
                "local_only": 0,
                "broker_only": 0,
                "errors": 0
            }
            
            try:
                # Get local orders
                local_orders = self._get_active_local_orders()
                local_order_ids = {o.order_id for o in local_orders}
                
                # Get broker orders
                broker_orders = await self._get_broker_orders()
                broker_order_ids = {o.get("orderId", o.get("id")) for o in broker_orders}
                
                # Find orders in both systems
                common_ids = local_order_ids & broker_order_ids
                local_only_ids = local_order_ids - broker_order_ids
                broker_only_ids = broker_order_ids - local_order_ids
                
                # Sync common orders
                for order_id in common_ids:
                    local_order = next(o for o in local_orders if o.order_id == order_id)
                    broker_order = next(o for o in broker_orders if o.get("orderId", o.get("id")) == order_id)
                    
                    sync_state = await self._sync_order(local_order, broker_order)
                    
                    if sync_state.sync_status == SyncStatus.SYNCED:
                        results["synced"] += 1
                    else:
                        results["mismatched"] += 1
                
                # Handle local-only orders
                for order_id in local_only_ids:
                    local_order = next(o for o in local_orders if o.order_id == order_id)
                    await self._handle_local_only_order(local_order)
                    results["local_only"] += 1
                
                # Handle broker-only orders
                for order_id in broker_only_ids:
                    broker_order = next(o for o in broker_orders if o.get("orderId", o.get("id")) == order_id)
                    await self._handle_broker_only_order(broker_order)
                    results["broker_only"] += 1
                
                # Update metrics
                self.metrics.total_syncs += 1
                self.metrics.successful_syncs += 1
                
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics.last_sync_duration_ms = duration_ms
                self._update_average_duration(duration_ms)
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                results["errors"] += 1
                self.metrics.failed_syncs += 1
            
            return results
    
    async def _sync_order(
        self,
        local_order: Order,
        broker_order: dict
    ) -> OrderSyncState:
        """Synchronize a single order."""
        sync_state = self.sync_states.get(
            local_order.order_id,
            OrderSyncState(order_id=local_order.order_id)
        )
        
        sync_state.local_order = local_order
        sync_state.broker_order = broker_order
        sync_state.last_sync_time = datetime.now()
        sync_state.discrepancies.clear()
        
        # Check status mismatch
        broker_status = self._map_broker_status(broker_order.get("status"))
        if local_order.status != broker_status:
            sync_state.discrepancies.append(
                f"Status mismatch: local={local_order.status}, broker={broker_status}"
            )
            
            # Update local status if broker is authoritative
            if self._is_broker_authoritative(broker_status):
                local_order.status = broker_status
                logger.info(f"Updated order {local_order.order_id} status to {broker_status}")
        
        # Check fill discrepancies
        broker_filled = broker_order.get("filledQuantity", 0)
        if abs(local_order.filled_quantity - broker_filled) > 0.0001:
            sync_state.fill_discrepancies.append(
                f"Fill mismatch: local={local_order.filled_quantity}, broker={broker_filled}"
            )
            
            # Update local fills
            if broker_filled > local_order.filled_quantity:
                await self._process_missed_fill(local_order, broker_order)
                self.metrics.missed_fills_detected += 1
        
        # Determine sync status
        if sync_state.discrepancies or sync_state.fill_discrepancies:
            sync_state.sync_status = SyncStatus.MISMATCHED
            self.metrics.discrepancies_found += 1
            
            # Notify callbacks
            for callback in self.mismatch_callbacks:
                await callback(sync_state)
        else:
            sync_state.sync_status = SyncStatus.SYNCED
        
        self.sync_states[local_order.order_id] = sync_state
        return sync_state
    
    async def _handle_local_only_order(self, order: Order) -> None:
        """Handle order that exists only locally."""
        # Check if order is stale
        age_hours = (datetime.now() - order.submitted_at).total_seconds() / 3600
        
        if age_hours > self.stale_order_threshold:
            # Mark as stale
            sync_state = OrderSyncState(
                order_id=order.order_id,
                local_order=order,
                sync_status=SyncStatus.STALE
            )
            self.sync_states[order.order_id] = sync_state
            
            # Cancel if still active
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                logger.warning(f"Cancelling stale local order: {order.order_id}")
                order.status = OrderStatus.CANCELLED
        else:
            # Might be a new order not yet at broker
            sync_state = self.sync_states.get(
                order.order_id,
                OrderSyncState(order_id=order.order_id, local_order=order)
            )
            
            sync_state.local_order = order
            sync_state.sync_status = SyncStatus.LOCAL_ONLY
            sync_state.retry_count += 1
            
            if sync_state.retry_count > sync_state.max_retries:
                logger.error(f"Order {order.order_id} not found at broker after {sync_state.max_retries} retries")
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reason"] = "Not found at broker"
            
            # Store the sync state
            self.sync_states[order.order_id] = sync_state
    
    async def _handle_broker_only_order(self, broker_order: dict) -> None:
        """Handle order that exists only at broker."""
        order_id = broker_order.get("orderId", broker_order.get("id"))
        
        sync_state = OrderSyncState(
            order_id=order_id,
            broker_order=broker_order,
            sync_status=SyncStatus.BROKER_ONLY
        )
        
        self.sync_states[order_id] = sync_state
        self.metrics.orphaned_orders_found += 1
        
        logger.warning(f"Found orphaned broker order: {order_id}")
        
        # Notify callbacks
        for callback in self.mismatch_callbacks:
            await callback(sync_state)
    
    async def check_duplicate_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> bool:
        """
        Check if an order would be a duplicate.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            
        Returns:
            True if order appears to be duplicate
        """
        key = f"{symbol}:{side}:{quantity}"
        now = datetime.now()
        
        # Check recent orders
        if key in self.recent_orders:
            last_order_time = self.recent_orders[key]
            if (now - last_order_time).total_seconds() < self.duplicate_window_seconds:
                logger.warning(f"Potential duplicate order detected: {key}")
                self.metrics.duplicate_orders_prevented += 1
                
                # Notify callbacks
                for callback in self.duplicate_callbacks:
                    await callback({
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "last_order_time": last_order_time
                    })
                
                return True
        
        # Record this order
        self.recent_orders[key] = now
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self.duplicate_window_seconds * 2)
        self.recent_orders = {
            k: v for k, v in self.recent_orders.items()
            if v > cutoff
        }
        
        return False
    
    async def _process_missed_fill(
        self,
        local_order: Order,
        broker_order: dict
    ) -> None:
        """Process a fill that was missed locally."""
        broker_filled = broker_order.get("filledQuantity", 0)
        missed_quantity = broker_filled - local_order.filled_quantity
        
        if missed_quantity > 0:
            # Create fill record
            fill = Fill(
                fill_id=f"SYNC_{datetime.now().timestamp()}",
                order_id=local_order.order_id,
                symbol=local_order.symbol,
                side=local_order.side,
                quantity=missed_quantity,
                price=broker_order.get("avgFillPrice", broker_order.get("price", 0)),
                commission=broker_order.get("commission", 0),
                timestamp=datetime.now()
            )
            
            # Update order
            local_order.filled_quantity = broker_filled
            local_order.average_fill_price = broker_order.get("avgFillPrice", 0)
            
            if broker_filled >= local_order.quantity:
                local_order.status = OrderStatus.FILLED
            
            # Notify callbacks
            for callback in self.missed_fill_callbacks:
                await callback(local_order, fill)
            
            logger.warning(
                f"Processed missed fill for order {local_order.order_id}: "
                f"{missed_quantity} @ {fill.price}"
            )
    
    def _get_active_local_orders(self) -> List[Order]:
        """Get active orders from local order manager."""
        if hasattr(self.order_manager, 'get_active_orders'):
            return self.order_manager.get_active_orders()
        elif hasattr(self.order_manager, 'orders'):
            return [
                o for o in self.order_manager.orders.values()
                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
            ]
        return []
    
    async def _get_broker_orders(self) -> List[dict]:
        """Get orders from broker."""
        if hasattr(self.executor, 'get_orders'):
            return await self.executor.get_orders()
        elif hasattr(self.executor, 'adapter') and hasattr(self.executor.adapter, 'get_orders'):
            return await self.executor.adapter.get_orders()
        return []
    
    def _map_broker_status(self, broker_status: str) -> OrderStatus:
        """Map broker status to local OrderStatus."""
        status_map = {
            "PendingSubmit": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
        }
        
        return status_map.get(broker_status, OrderStatus.PENDING)
    
    def _is_broker_authoritative(self, status: OrderStatus) -> bool:
        """Check if broker status should override local."""
        # Broker is authoritative for terminal states
        return status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    def _update_average_duration(self, duration_ms: float) -> None:
        """Update average sync duration."""
        if self.metrics.average_sync_duration_ms == 0:
            self.metrics.average_sync_duration_ms = duration_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_sync_duration_ms = (
                alpha * duration_ms + 
                (1 - alpha) * self.metrics.average_sync_duration_ms
            )
    
    def register_mismatch_callback(self, callback: Callable) -> None:
        """Register callback for order mismatches."""
        self.mismatch_callbacks.append(callback)
    
    def register_duplicate_callback(self, callback: Callable) -> None:
        """Register callback for duplicate order detection."""
        self.duplicate_callbacks.append(callback)
    
    def register_missed_fill_callback(self, callback: Callable) -> None:
        """Register callback for missed fills."""
        self.missed_fill_callbacks.append(callback)
    
    def get_sync_state(self, order_id: str) -> Optional[OrderSyncState]:
        """Get synchronization state for an order."""
        return self.sync_states.get(order_id)
    
    def get_metrics(self) -> dict:
        """Get synchronization metrics."""
        return {
            "total_syncs": self.metrics.total_syncs,
            "successful_syncs": self.metrics.successful_syncs,
            "failed_syncs": self.metrics.failed_syncs,
            "success_rate": (
                self.metrics.successful_syncs / self.metrics.total_syncs
                if self.metrics.total_syncs > 0 else 0
            ),
            "discrepancies_found": self.metrics.discrepancies_found,
            "discrepancies_resolved": self.metrics.discrepancies_resolved,
            "duplicate_orders_prevented": self.metrics.duplicate_orders_prevented,
            "missed_fills_detected": self.metrics.missed_fills_detected,
            "orphaned_orders_found": self.metrics.orphaned_orders_found,
            "average_sync_duration_ms": self.metrics.average_sync_duration_ms,
            "active_sync_states": len(self.sync_states)
        }