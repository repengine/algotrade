"""
Order Manager Module

This module handles order lifecycle management, including order creation,
submission, tracking, and execution.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from utils.logging import setup_logger


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options"""

    GTC = "good_till_cancelled"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    GTD = "good_till_date"
    DAY = "day"


@dataclass
class Order:
    """Order data structure"""

    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        ]

    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED

    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill"""
        return self.quantity - self.filled_quantity


@dataclass
class OrderFill:
    """Order fill information"""

    order_id: str
    fill_id: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    exchange: str = ""


class OrderManager:
    """
    Manages order lifecycle and execution.

    Responsibilities:
    - Order validation
    - Order submission
    - Order tracking
    - Fill management
    - Order event handling
    """

    def __init__(self, exchange_connector=None):
        """
        Initialize order manager.

        Args:
            exchange_connector: Exchange connection interface
        """
        self.logger = setup_logger(__name__)
        self.exchange_connector = exchange_connector
        self.orders: dict[str, Order] = {}
        self.order_fills: dict[str, list[OrderFill]] = {}
        self.order_callbacks: dict[str, list[Callable]] = {}
        self._order_lock = asyncio.Lock()

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        strategy_id: Optional[str] = None,
        **kwargs,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading symbol
            order_type: Type of order
            side: Buy or sell
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force option
            strategy_id: ID of the strategy creating this order
            **kwargs: Additional order parameters

        Returns:
            Created order object
        """
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_id=strategy_id,
            metadata=kwargs,
        )

        # Validate order
        self._validate_order(order)

        async with self._order_lock:
            self.orders[order.order_id] = order
            self.order_fills[order.order_id] = []

        self.logger.info(
            f"Created order: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}"
        )
        return order

    async def submit_order(self, order: Order) -> bool:
        """
        Submit order to exchange.

        Args:
            order: Order to submit

        Returns:
            True if submission successful
        """
        if not self.exchange_connector:
            self.logger.error("No exchange connector available")
            return False

        try:
            # Submit to exchange
            success = await self.exchange_connector.submit_order(order)

            if success:
                order.status = OrderStatus.SUBMITTED
                order.updated_at = datetime.now()
                await self._trigger_callbacks(order, "submitted")
                self.logger.info(f"Order submitted: {order.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                await self._trigger_callbacks(order, "rejected")
                self.logger.error(f"Order rejected: {order.order_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancellation successful
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False

        if not order.is_active():
            self.logger.warning(f"Cannot cancel inactive order: {order_id}")
            return False

        try:
            if self.exchange_connector:
                success = await self.exchange_connector.cancel_order(order_id)
            else:
                success = True  # Mock cancellation for testing

            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                await self._trigger_callbacks(order, "cancelled")
                self.logger.info(f"Order cancelled: {order_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def update_order_fill(self, order_id: str, fill: OrderFill) -> None:
        """
        Update order with fill information.

        Args:
            order_id: Order ID
            fill: Fill information
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order not found for fill: {order_id}")
            return

        async with self._order_lock:
            # Update order
            order.filled_quantity += fill.quantity
            order.average_fill_price = self._calculate_average_price(order, fill)
            order.updated_at = datetime.now()

            # Store fill
            self.order_fills[order_id].append(fill)

            # Update status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                await self._trigger_callbacks(order, "filled")
            else:
                order.status = OrderStatus.PARTIAL
                await self._trigger_callbacks(order, "partial_fill")

        self.logger.info(
            f"Order fill processed: {order_id} - {fill.quantity} @ {fill.price}"
        )

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all active orders, optionally filtered by symbol"""
        active_orders = [o for o in self.orders.values() if o.is_active()]

        if symbol:
            active_orders = [o for o in active_orders if o.symbol == symbol]

        return active_orders

    def get_orders_by_strategy(self, strategy_id: str) -> list[Order]:
        """Get all orders for a specific strategy"""
        return [o for o in self.orders.values() if o.strategy_id == strategy_id]

    def register_callback(self, order_id: str, event: str, callback: Callable) -> None:
        """Register callback for order events"""
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        self.order_callbacks[order_id].append((event, callback))

    def _validate_order(self, order: Order) -> None:
        """Validate order parameters"""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if (
            order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
            and order.price is None
        ):
            raise ValueError(f"{order.order_type.value} order requires price")

        if (
            order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and order.stop_price is None
        ):
            raise ValueError(f"{order.order_type.value} order requires stop price")

    def _calculate_average_price(self, order: Order, new_fill: OrderFill) -> float:
        """Calculate average fill price"""
        if order.average_fill_price == 0:
            return new_fill.price

        total_value = (
            order.average_fill_price * (order.filled_quantity - new_fill.quantity)
            + new_fill.price * new_fill.quantity
        )
        return (
            total_value / order.filled_quantity
            if order.filled_quantity > 0
            else new_fill.price
        )

    async def _trigger_callbacks(self, order: Order, event: str) -> None:
        """Trigger callbacks for order events"""
        callbacks = self.order_callbacks.get(order.order_id, [])
        for callback_event, callback in callbacks:
            if callback_event == event or callback_event == "*":
                try:
                    await callback(order, event)
                except Exception as e:
                    self.logger.error(f"Error in order callback: {e}")

    def get_order_statistics(self) -> dict[str, Any]:
        """Get order statistics"""
        total_orders = len(self.orders)
        active_orders = len([o for o in self.orders.values() if o.is_active()])
        filled_orders = len([o for o in self.orders.values() if o.is_filled()])

        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": len(
                [o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]
            ),
            "rejected_orders": len(
                [o for o in self.orders.values() if o.status == OrderStatus.REJECTED]
            ),
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0.0,
        }
