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

from algostack.core.executor import (
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

    def __init__(self, risk_manager=None):
        """
        Initialize enhanced order manager.

        Args:
            risk_manager: Risk manager for order validation
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
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
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
            f"Order created: {order.order_id} - {side.value} {quantity} {symbol} "
            f"({order_type.value})"
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

    def get_strategy_orders(self, strategy_id: str) -> list[Order]:
        """Get all orders for a strategy."""
        order_ids = self._strategy_orders.get(strategy_id, set())
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

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
        asyncio.create_task(self._handle_order_status(order))

    def on_fill(self, fill: Fill) -> None:
        """Handle fill notifications from executor."""
        asyncio.create_task(self._handle_fill(fill))

    def on_error(self, error: Exception, order: Optional[Order] = None) -> None:
        """Handle execution errors."""
        asyncio.create_task(self._handle_error(error, order))

    # Private methods

    async def _validate_order_risk(self, order: Order) -> bool:
        """Validate order with risk manager."""
        if not self.risk_manager:
            return True

        # TODO: Implement risk validation
        # This would check position limits, buying power, etc.
        return True

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
