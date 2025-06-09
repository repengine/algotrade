"""
Paper trading executor for testing strategies without real money.

This module provides a simulated execution environment that mimics
real broker behavior for testing and development.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from algostack.core.executor import (
    BaseExecutor,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class PaperExecutor(BaseExecutor):
    """Paper trading executor for simulation."""
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize paper executor.
        
        Args:
            config: Configuration with:
                - initial_capital: Starting capital
                - commission: Commission per trade
                - slippage: Slippage percentage
                - fill_delay: Delay in seconds before fill
                - price_data_source: Data source for current prices
        """
        super().__init__(config)
        self.initial_capital = config.get("initial_capital", 100000.0)
        self.commission = config.get("commission", 1.0)
        self.slippage = config.get("slippage", 0.0001)  # 0.01%
        self.fill_delay = config.get("fill_delay", 0.1)
        
        # Account state
        self.cash = self.initial_capital
        self.buying_power = self.initial_capital
        self._price_data: dict[str, float] = {}
        self._fill_tasks: dict[str, asyncio.Task] = {}
        
    async def connect(self) -> bool:
        """Connect to paper trading (always succeeds)."""
        logger.info("Paper trading executor connected")
        self.is_connected = True
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from paper trading."""
        # Cancel any pending fill tasks
        for task in self._fill_tasks.values():
            task.cancel()
        self._fill_tasks.clear()
        
        self.is_connected = False
        logger.info("Paper trading executor disconnected")
    
    async def submit_order(self, order: Order) -> str:
        """
        Submit order for paper execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
        """
        if not self.is_connected:
            raise RuntimeError("Executor not connected")
            
        # Validate order
        self.validate_order(order)
        
        # Generate order ID if not provided
        if not order.order_id:
            order.order_id = f"PAPER-{uuid.uuid4().hex[:8]}"
            
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            estimated_cost = self._estimate_order_cost(order)
            if estimated_cost > self.buying_power:
                order.status = OrderStatus.REJECTED
                self._notify_order_status(order)
                raise ValueError(f"Insufficient buying power: {self.buying_power:.2f} < {estimated_cost:.2f}")
        
        # Update order status
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        self._orders[order.order_id] = order
        
        # Notify submission
        self._notify_order_status(order)
        
        # Schedule fill simulation
        task = asyncio.create_task(self._simulate_fill(order))
        self._fill_tasks[order.order_id] = task
        
        logger.info(f"Order submitted: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
        
        return order.order_id
    
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
            return False
            
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
            
        # Cancel fill task if exists
        if order_id in self._fill_tasks:
            self._fill_tasks[order_id].cancel()
            del self._fill_tasks[order_id]
            
        # Update status
        order.status = OrderStatus.CANCELLED
        self._notify_order_status(order)
        
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self._orders.get(order_id)
    
    async def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()
    
    async def get_account_info(self) -> dict[str, Any]:
        """Get account information."""
        # Calculate account values
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        total_value = self.cash + positions_value
        
        return {
            "account_id": "PAPER-TRADING",
            "cash": self.cash,
            "buying_power": self.buying_power,
            "positions_value": positions_value,
            "total_value": total_value,
            "initial_capital": self.initial_capital,
            "total_pnl": total_value - self.initial_capital,
            "is_paper": True,
        }
    
    def update_price(self, symbol: str, price: float) -> None:
        """
        Update current price for symbol.
        
        Args:
            symbol: Symbol to update
            price: Current price
        """
        self._price_data[symbol] = price
        
        # Update position market values
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = (price - pos.average_cost) * pos.quantity
            pos.last_updated = datetime.now()
    
    def update_prices(self, prices: dict[str, float]) -> None:
        """Update multiple prices at once."""
        for symbol, price in prices.items():
            self.update_price(symbol, price)
    
    async def _simulate_fill(self, order: Order) -> None:
        """Simulate order fill with delay and slippage."""
        try:
            # Wait for fill delay
            await asyncio.sleep(self.fill_delay)
            
            # Check if order was cancelled
            if order.status == OrderStatus.CANCELLED:
                return
                
            # Get current price
            current_price = self._price_data.get(order.symbol)
            if current_price is None:
                order.status = OrderStatus.REJECTED
                self._notify_order_status(order)
                logger.error(f"No price data for {order.symbol}")
                return
                
            # Determine fill price based on order type
            fill_price = self._calculate_fill_price(order, current_price)
            
            if fill_price is None:
                # Order cannot be filled at current price
                return
                
            # Execute fill
            await self._execute_fill(order, fill_price)
            
        except asyncio.CancelledError:
            logger.debug(f"Fill simulation cancelled for {order.order_id}")
        except Exception as e:
            logger.error(f"Error simulating fill: {e}")
            self._notify_error(e, order)
    
    def _calculate_fill_price(self, order: Order, current_price: float) -> Optional[float]:
        """Calculate fill price with slippage."""
        slippage_mult = 1 + self.slippage if order.side == OrderSide.BUY else 1 - self.slippage
        
        if order.order_type == OrderType.MARKET:
            return current_price * slippage_mult
            
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                return min(current_price * slippage_mult, order.limit_price)
            elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                return max(current_price * slippage_mult, order.limit_price)
                
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                return current_price * slippage_mult
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                return current_price * slippage_mult
                
        return None
    
    async def _execute_fill(self, order: Order, fill_price: float) -> None:
        """Execute order fill."""
        # Create fill
        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=self.commission,
            timestamp=datetime.now(),
        )
        
        # Update cash and positions
        if order.side == OrderSide.BUY:
            total_cost = fill_price * order.quantity + self.commission
            self.cash -= total_cost
            self.buying_power = self.cash  # Simple buying power calculation
            
            # Update or create position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                pos.average_cost = (
                    (pos.average_cost * pos.quantity + fill_price * order.quantity) 
                    / total_quantity
                )
                pos.quantity = total_quantity
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_cost=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    market_value=fill_price * order.quantity,
                    last_updated=datetime.now(),
                )
                
        else:  # SELL
            if order.symbol not in self._positions:
                # Short selling not implemented in basic paper trading
                order.status = OrderStatus.REJECTED
                self._notify_order_status(order)
                return
                
            pos = self._positions[order.symbol]
            if pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                self._notify_order_status(order)
                return
                
            # Calculate realized P&L
            realized_pnl = (fill_price - pos.average_cost) * order.quantity - self.commission
            pos.realized_pnl += realized_pnl
            
            # Update position
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                del self._positions[order.symbol]
            else:
                pos.market_value = pos.quantity * fill_price
                pos.unrealized_pnl = (fill_price - pos.average_cost) * pos.quantity
                
            # Update cash
            self.cash += fill_price * order.quantity - self.commission
            self.buying_power = self.cash
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = self.commission
        
        # Notify callbacks
        self._notify_fill(fill)
        self._notify_order_status(order)
        
        logger.info(
            f"Order filled: {order.order_id} - {order.side.value} {order.quantity} "
            f"{order.symbol} @ {fill_price:.2f}"
        )
    
    def _estimate_order_cost(self, order: Order) -> float:
        """Estimate order cost for buying power check."""
        current_price = self._price_data.get(order.symbol, 0)
        
        if order.order_type == OrderType.MARKET:
            price = current_price
        elif order.order_type == OrderType.LIMIT:
            price = order.limit_price
        elif order.order_type == OrderType.STOP:
            price = max(current_price, order.stop_price) if order.side == OrderSide.BUY else current_price
        else:
            price = current_price
            
        return price * order.quantity + self.commission