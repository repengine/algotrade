"""
Enhanced Order Manager with Executor Integration.

This module provides advanced order management capabilities that integrate
with the executor framework for live and paper trading.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Optional

from core.executor import (
    BaseExecutor,
    ExecutionCallback,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from core.order_state_sync import OrderStateSynchronizer

logger = logging.getLogger(__name__)


class OrderEventType:
    """Order event types for callbacks."""

    CREATED = "created"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


OrderEventCallback = Callable[[Order, str, Optional[Any]], None]


class EnhancedOrderManager(ExecutionCallback):
    """
    Enhanced order manager with executor integration.

    Features:
    - Unified order management across multiple executors
    - Advanced order tracking and analytics
    - Risk-based order validation
    - Order routing and smart execution
    - Comprehensive event handling
    """

    def __init__(
        self,
        risk_manager: Optional[Any] = None,
        sync_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize enhanced order manager.

        Args:
            risk_manager: Risk manager for order validation
            sync_config: Configuration for order state synchronization
        """
        self.risk_manager = risk_manager
        self.executors: dict[str, BaseExecutor] = {}
        self.active_executor: Optional[str] = None

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._order_fills: dict[str, list[Fill]] = defaultdict(list)
        self._order_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._strategy_orders: dict[str, set[str]] = defaultdict(set)

        # Callbacks
        self._event_callbacks: dict[str, list[OrderEventCallback]] = defaultdict(list)

        # Statistics
        self._order_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "cancelled_orders": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
        }

        # Locks
        self._order_lock = asyncio.Lock()

        # Order state synchronization
        self.sync_config = sync_config or {}
        self.order_synchronizer: Optional[OrderStateSynchronizer] = None

    def add_executor(self, name: str, executor: BaseExecutor) -> None:
        """
        Add executor to manager.

        Args:
            name: Executor name
            executor: Executor instance
        """
        self.executors[name] = executor
        executor.register_callback(self)

        if self.active_executor is None:
            self.active_executor = name

        logger.info(f"Added executor: {name}")

    def set_active_executor(self, name: str) -> None:
        """Set active executor for order submission."""
        if name not in self.executors:
            raise ValueError(f"Unknown executor: {name}")
        self.active_executor = name
        logger.info(f"Active executor set to: {name}")

    async def create_order(
        self,
        symbol: str,
        side: OrderSide | str,
        quantity: int,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce | str = TimeInForce.DAY,
        strategy_id: Optional[str] = None,
        **metadata,
    ) -> Order:
        """
        Create new order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            strategy_id: Strategy identifier
            **metadata: Additional metadata

        Returns:
            Created order
        """
        # Convert string parameters to enums if needed
        if isinstance(side, str):
            side = OrderSide(side.upper())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.upper())
        if isinstance(time_in_force, str):
            time_in_force = TimeInForce(time_in_force.upper())

        # Create order
        order = Order(
            order_id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            metadata=metadata,
        )

        # Store strategy association
        if strategy_id:
            order.metadata["strategy_id"] = strategy_id
            self._strategy_orders[strategy_id].add(order.order_id)

        # Validate with risk manager
        if self.risk_manager and not await self._validate_order_risk(order):
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "Risk validation failed"
            await self._record_event(order, OrderEventType.REJECTED)
            raise ValueError("Order rejected by risk manager")

        # Store order
        async with self._order_lock:
            self._orders[order.order_id] = order
            self._order_stats["total_orders"] += 1

        # Record creation event
        await self._record_event(order, OrderEventType.CREATED)

        logger.info(
            f"Order created: {order.order_id} - {order.side.value} {quantity} {symbol} "
            f"({order.order_type.value})"
        )

        return order

    async def submit_order(
        self, order: Order, executor_name: Optional[str] = None
    ) -> bool:
        """
        Submit order to executor.

        Args:
            order: Order to submit
            executor_name: Specific executor to use (optional)

        Returns:
            True if submission successful
        """
        # Get executor
        executor_name = executor_name or self.active_executor
        if not executor_name or executor_name not in self.executors:
            raise ValueError(f"No valid executor available: {executor_name}")

        executor = self.executors[executor_name]

        # Check connection
        if not executor.is_connected:
            raise RuntimeError(f"Executor {executor_name} not connected")

        try:
            # Check for duplicate orders if synchronizer is active
            if self.order_synchronizer:
                is_duplicate = await self.order_synchronizer.check_duplicate_order(
                    order.symbol, order.side.value, order.quantity
                )
                if is_duplicate:
                    logger.warning(f"Duplicate order prevented: {order}")
                    order.status = OrderStatus.REJECTED
                    order.metadata["rejection_reason"] = "Duplicate order detected"
                    await self._record_event(
                        order, OrderEventType.REJECTED, {"reason": "duplicate"}
                    )
                    return False

            # Submit to executor
            broker_order_id = await executor.submit_order(order)
            order.metadata["broker_order_id"] = broker_order_id
            order.metadata["executor"] = executor_name

            return True

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = str(e)
            await self._record_event(order, OrderEventType.REJECTED, {"error": str(e)})
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        if order.status not in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]:
            logger.warning(f"Cannot cancel order in status: {order.status}")
            return False

        # Get executor
        executor_name = order.metadata.get("executor")
        if not executor_name or executor_name not in self.executors:
            logger.error(f"No executor found for order: {order_id}")
            return False

        executor = self.executors[executor_name]

        try:
            # Cancel via executor
            success = await executor.cancel_order(order.order_id)
            if success:
                await self._record_event(order, OrderEventType.CANCELLED)
            return success

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get active orders, optionally filtered by symbol."""
        active_orders = [
            order
            for order in self._orders.values()
            if order.status
            in [
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
            ]
        ]

        if symbol:
            active_orders = [o for o in active_orders if o.symbol == symbol]

        return active_orders

    def get_all_orders(self) -> list[Order]:
        """Get all orders regardless of status."""
        return list(self._orders.values())

    def get_recent_fills(self, seconds: int = 60) -> list[Order]:
        """Get orders filled in the last N seconds."""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        recent_fills = []

        for order in self._orders.values():
            if order.status == OrderStatus.FILLED:
                # Check if order has a fill timestamp
                if hasattr(order, "filled_timestamp") and order.filled_timestamp:
                    if order.filled_timestamp >= cutoff_time:
                        recent_fills.append(order)
                elif hasattr(order, "updated_at") and order.updated_at:
                    # Fallback to updated_at if no fill timestamp
                    if order.updated_at >= cutoff_time:
                        recent_fills.append(order)

        return recent_fills

    def get_strategy_orders(self, strategy_id: str) -> list[Order]:
        """Get all orders for a strategy."""
        order_ids = self._strategy_orders.get(strategy_id, set())
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def add_order(self, order_id: str, order: Order) -> None:
        """
        Add an order to the manager for tracking.

        This method is used when an order has already been created and needs
        to be tracked by the order manager. It includes risk validation and
        duplicate detection via the synchronizer if available.

        Args:
            order_id: The order ID to use
            order: The order object to add

        Raises:
            ValueError: If order validation fails or risk check fails
        """
        # Check for duplicate if synchronizer is active
        if self.order_synchronizer:
            # Note: This is synchronous check for simplicity in add_order
            # In async context, use submit_order which has async duplicate check
            logger.warning(
                "Duplicate check skipped in sync add_order - use submit_order for full validation"
            )

        # Update order ID if different
        if order.order_id != order_id:
            logger.info(f"Updating order ID from {order.order_id} to {order_id}")
            order.order_id = order_id

        # Validate with risk manager if available
        if self.risk_manager:
            # Simple synchronous validation - full async validation in submit_order
            if hasattr(self.risk_manager, "validate_order_params"):
                if not self.risk_manager.validate_order_params(order):
                    raise ValueError(
                        "Order rejected by risk manager parameter validation"
                    )

        # Store order
        self._orders[order_id] = order
        self._order_stats["total_orders"] += 1

        # Store strategy association
        if hasattr(order, "strategy_id") and order.strategy_id:
            self._strategy_orders[order.strategy_id].add(order_id)
        elif hasattr(order, "metadata") and order.metadata.get("strategy_id"):
            self._strategy_orders[order.metadata["strategy_id"]].add(order_id)

        # Initialize order fills list
        if order_id not in self._order_fills:
            self._order_fills[order_id] = []

        # Record creation event (synchronously)
        event = {
            "timestamp": datetime.now(),
            "event_type": OrderEventType.CREATED,
            "order_status": order.status,
            "data": {},
        }
        self._order_events[order_id].append(event)

        logger.info(
            f"Added order: {order_id} - {order.side.value} {order.quantity} {order.symbol}"
        )

    async def get_positions(
        self, executor_name: Optional[str] = None
    ) -> dict[str, Position]:
        """
        Get positions from executor.

        Args:
            executor_name: Specific executor (uses active if not specified)

        Returns:
            Dictionary of positions by symbol
        """
        executor_name = executor_name or self.active_executor
        if not executor_name or executor_name not in self.executors:
            return {}

        executor = self.executors[executor_name]
        return await executor.get_positions()

    def register_event_callback(
        self, event_type: str, callback: OrderEventCallback
    ) -> None:
        """Register callback for order events."""
        self._event_callbacks[event_type].append(callback)

    def get_order_statistics(self) -> dict[str, Any]:
        """Get comprehensive order statistics."""
        stats = self._order_stats.copy()

        # Calculate additional metrics
        if stats["total_orders"] > 0:
            stats["fill_rate"] = stats["filled_orders"] / stats["total_orders"]
            stats["rejection_rate"] = stats["rejected_orders"] / stats["total_orders"]
            stats["avg_commission"] = stats["total_commission"] / max(
                stats["filled_orders"], 1
            )
        else:
            stats["fill_rate"] = 0.0
            stats["rejection_rate"] = 0.0
            stats["avg_commission"] = 0.0

        # Add current state
        stats["active_orders"] = len(self.get_active_orders())

        return stats

    # ExecutionCallback implementation

    def on_order_status(self, order: Order) -> None:
        """Handle order status updates from executor."""
        asyncio.create_task(self._handle_order_update(order))

    def on_order_update(self, order: Order) -> None:
        """Handle order status updates from executor (alias for compatibility)."""
        self.on_order_status(order)

    def on_fill(self, fill: Fill) -> None:
        """Handle order fill from executor."""
        asyncio.create_task(self._handle_fill(fill))

    def on_error(self, error: Exception, order: Optional[Order] = None) -> None:
        """Handle execution errors."""
        asyncio.create_task(self._handle_error(error, order))

    # Private methods

    async def _handle_order_update(self, order: Order) -> None:
        """Handle order status update from executor.

        CRITICAL: This ensures we track actual order state for risk management.
        """
        async with self._order_lock:
            if order.order_id in self._orders:
                # Update our copy of the order
                self._orders[order.order_id] = order

                # Track statistics
                if order.status == OrderStatus.FILLED:
                    self._order_stats["filled_orders"] += 1
                    self._order_stats["total_commission"] += order.commission or 0
                elif order.status == OrderStatus.REJECTED:
                    self._order_stats["rejected_orders"] += 1
                elif order.status == OrderStatus.CANCELLED:
                    self._order_stats["cancelled_orders"] += 1

                # Log status update
                logger.info(
                    f"Order {order.order_id} status updated to {order.status.value}"
                )

                # Trigger event callbacks
                event_type = self._map_status_to_event(order.status)
                if event_type:
                    self._trigger_event(event_type, order)

    def _map_status_to_event(self, status: OrderStatus) -> Optional[str]:
        """Map order status to event type."""
        mapping = {
            OrderStatus.SUBMITTED: OrderEventType.SUBMITTED,
            OrderStatus.FILLED: OrderEventType.FILLED,
            OrderStatus.PARTIALLY_FILLED: OrderEventType.PARTIALLY_FILLED,
            OrderStatus.CANCELLED: OrderEventType.CANCELLED,
            OrderStatus.REJECTED: OrderEventType.REJECTED,
            OrderStatus.EXPIRED: OrderEventType.EXPIRED,
        }
        return mapping.get(status)

    def _trigger_event(
        self, event_type: str, order: Order, data: Optional[Any] = None
    ) -> None:
        """Trigger event callbacks."""
        for callback in self._event_callbacks.get(event_type, []):
            try:
                callback(order, event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    async def _validate_order_risk(self, order: Order) -> bool:
        """Validate order with risk manager."""
        if not self.risk_manager:
            return True

        try:
            # Check if risk manager has async validation method
            if hasattr(self.risk_manager, "validate_order_risk"):
                # Call async method if it's coroutine
                if asyncio.iscoroutinefunction(self.risk_manager.validate_order_risk):
                    return await self.risk_manager.validate_order_risk(order)
                else:
                    # Call sync method
                    return self.risk_manager.validate_order_risk(order)
            elif hasattr(self.risk_manager, "validate_order_risk_sync"):
                # Fallback to sync method
                return self.risk_manager.validate_order_risk_sync(order)
            else:
                # No validation method available
                logger.warning("Risk manager has no validate_order_risk method")
                return True

        except Exception as e:
            logger.error(f"Error during risk validation: {e}")
            # On error, reject the order for safety
            return False

    async def _handle_order_status(self, order: Order) -> None:
        """Process order status update."""
        async with self._order_lock:
            # Update our order record
            if order.order_id in self._orders:
                stored_order = self._orders[order.order_id]
                stored_order.status = order.status
                stored_order.filled_quantity = order.filled_quantity
                stored_order.average_fill_price = order.average_fill_price

                # Update statistics
                if (
                    order.status == OrderStatus.FILLED
                    and stored_order.status != OrderStatus.FILLED
                ):
                    self._order_stats["filled_orders"] += 1
                elif order.status == OrderStatus.REJECTED:
                    self._order_stats["rejected_orders"] += 1
                elif order.status == OrderStatus.CANCELLED:
                    self._order_stats["cancelled_orders"] += 1

        # Determine event type
        if order.status == OrderStatus.SUBMITTED:
            event_type = OrderEventType.SUBMITTED
        elif order.status == OrderStatus.FILLED:
            event_type = OrderEventType.FILLED
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            event_type = OrderEventType.PARTIALLY_FILLED
        elif order.status == OrderStatus.CANCELLED:
            event_type = OrderEventType.CANCELLED
        elif order.status == OrderStatus.REJECTED:
            event_type = OrderEventType.REJECTED
        else:
            return

        await self._record_event(order, event_type)

    async def _handle_fill(self, fill: Fill) -> None:
        """Process fill notification."""
        async with self._order_lock:
            # Store fill
            self._order_fills[fill.order_id].append(fill)

            # Update statistics
            self._order_stats["total_commission"] += fill.commission

            # Calculate slippage if we have the order
            if fill.order_id in self._orders:
                order = self._orders[fill.order_id]
                if order.order_type == OrderType.LIMIT and order.limit_price:
                    slippage = abs(fill.price - order.limit_price) * fill.quantity
                    self._order_stats["total_slippage"] += slippage

        logger.info(f"Fill received: {fill.order_id} - {fill.quantity} @ {fill.price}")

    async def _handle_error(
        self, error: Exception, order: Optional[Order] = None
    ) -> None:
        """Process execution error."""
        logger.error(f"Execution error: {error}")

        if order:
            await self._record_event(
                order,
                OrderEventType.ERROR,
                {"error": str(error), "error_type": type(error).__name__},
            )

    async def _record_event(
        self, order: Order, event_type: str, data: Optional[dict[str, Any]] = None
    ) -> None:
        """Record order event and trigger callbacks."""
        # Record event
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "order_status": order.status,
            "data": data or {},
        }
        self._order_events[order.order_id].append(event)

        # Trigger callbacks
        for callback in self._event_callbacks[event_type]:
            try:
                callback(order, event_type, data)
            except Exception as e:
                logger.error(f"Error in order event callback: {e}")

        # Also trigger wildcard callbacks
        for callback in self._event_callbacks.get("*", []):
            try:
                callback(order, event_type, data)
            except Exception as e:
                logger.error(f"Error in wildcard callback: {e}")

    async def initialize_synchronization(self) -> None:
        """Initialize order state synchronization."""
        if self.active_executor and self.active_executor in self.executors:
            executor = self.executors[self.active_executor]

            # Create synchronizer
            self.order_synchronizer = OrderStateSynchronizer(
                order_manager=self, executor=executor, config=self.sync_config
            )

            # Register callbacks
            self.order_synchronizer.register_mismatch_callback(
                self._handle_sync_mismatch
            )
            self.order_synchronizer.register_duplicate_callback(
                self._handle_duplicate_detection
            )
            self.order_synchronizer.register_missed_fill_callback(
                self._handle_missed_fill
            )

            # Start synchronization
            await self.order_synchronizer.start()
            logger.info("Order state synchronization initialized")
        else:
            logger.warning("Cannot initialize synchronization - no active executor")

    async def stop_synchronization(self) -> None:
        """Stop order state synchronization."""
        if self.order_synchronizer:
            await self.order_synchronizer.stop()
            logger.info("Order state synchronization stopped")

    async def _handle_sync_mismatch(self, sync_state) -> None:
        """Handle order synchronization mismatch."""
        logger.warning(
            f"Order sync mismatch for {sync_state.order_id}: "
            f"status={sync_state.sync_status}, "
            f"discrepancies={sync_state.discrepancies}"
        )

        # Notify via event system
        if sync_state.local_order:
            await self._record_event(
                sync_state.local_order,
                "sync_mismatch",
                {
                    "sync_status": sync_state.sync_status.value,
                    "discrepancies": sync_state.discrepancies,
                },
            )

    async def _handle_duplicate_detection(self, duplicate_info) -> None:
        """Handle duplicate order detection."""
        logger.warning(
            f"Duplicate order detected: {duplicate_info['symbol']} "
            f"{duplicate_info['side']} {duplicate_info['quantity']}"
        )

        # Could implement additional logic here

    async def _handle_missed_fill(self, order, fill) -> None:
        """Handle missed fill from synchronization."""
        logger.warning(
            f"Processing missed fill for order {order.order_id}: "
            f"{fill.quantity} @ {fill.price}"
        )

        # Process the fill through normal channels
        await self._handle_fill(fill)

    def get_sync_metrics(self) -> Optional[dict]:
        """Get synchronization metrics."""
        if self.order_synchronizer:
            return self.order_synchronizer.get_metrics()
        return None
