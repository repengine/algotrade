"""
Base executor interface for broker adapters.

This module provides the abstract base class for all execution adapters
(IBKR, paper trading, etc.) and common execution utilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class ExecutorError(Exception):
    """Base exception for executor errors."""
    pass


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force enumeration."""

    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    OPG = "OPG"  # At the Opening
    CLS = "CLS"  # At the Close


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    commission: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    """Trade fill representation."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position representation."""

    symbol: str
    quantity: int
    average_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    last_updated: datetime


class ExecutionCallback(Protocol):
    """Protocol for execution callbacks."""

    def on_order_status(self, order: Order) -> None:
        """Called when order status changes."""
        ...  # pragma: no cover

    def on_fill(self, fill: Fill) -> None:
        """Called when order is filled."""
        ...  # pragma: no cover

    def on_error(self, error: Exception, order: Optional[Order] = None) -> None:
        """Called when execution error occurs."""
        ...  # pragma: no cover


class BaseExecutor(ABC):
    """Abstract base class for all execution adapters."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize executor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_connected = False
        self.callbacks: list[ExecutionCallback] = []
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}

    def register_callback(self, callback: ExecutionCallback) -> None:
        """Register execution callback."""
        self.callbacks.append(callback)

    def unregister_callback(self, callback: ExecutionCallback) -> None:
        """Unregister execution callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit order to broker.

        Args:
            order: Order to submit

        Returns:
            Order ID from broker
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get order status.

        Args:
            order_id: Order ID to check

        Returns:
            Order object with current status
        """
        pass

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        """
        Get all positions.

        Returns:
            Dictionary of symbol to Position
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary with account details
        """
        pass

    def _notify_order_status(self, order: Order) -> None:
        """Notify callbacks of order status change."""
        for callback in self.callbacks:
            try:
                callback.on_order_status(order)
            except Exception as e:
                logger.error(f"Error in order status callback: {e}")

    def _notify_fill(self, fill: Fill) -> None:
        """Notify callbacks of fill."""
        for callback in self.callbacks:
            try:
                callback.on_fill(fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")

    def _notify_error(self, error: Exception, order: Optional[Order] = None) -> None:
        """Notify callbacks of error."""
        for callback in self.callbacks:
            try:
                callback.on_error(error, order)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def validate_order(self, order: Order) -> bool:
        """
        Validate order before submission.

        Args:
            order: Order to validate

        Returns:
            True if order is valid
        """
        # Basic validation
        if order.quantity <= 0:
            raise ValueError(f"Invalid quantity: {order.quantity}")

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("Limit order requires limit price")

        if (
            order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and order.stop_price is None
        ):
            raise ValueError("Stop order requires stop price")

        if order.order_type == OrderType.STOP_LIMIT and order.limit_price is None:
            raise ValueError("Stop limit order requires limit price")

        return True

    def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        return [
            order
            for order in self._orders.values()
            if order.status
            in [
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
            ]
        ]

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update current market price for a symbol.

        This is a default implementation that can be overridden by subclasses.

        Args:
            symbol: Symbol to update
            price: Current market price
        """
        # Default implementation does nothing
        # Subclasses (like PaperExecutor) should override to track prices
        return None
