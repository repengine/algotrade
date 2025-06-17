"""
Execution Handler Module

This module manages order execution logic, including smart order routing,
execution algorithms, and order slicing.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Union

from core.engine.order_manager import Order, OrderType
from utils.logging import setup_logger


class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""

    MARKET = "market"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    SMART = "smart"
    POV = "pov"  # Percentage of Volume


@dataclass
class ExecutionParams:
    """Execution parameters"""

    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation_rate: float = 0.1  # 10% of volume
    slice_size: Optional[float] = None
    price_limit: Optional[float] = None
    urgency: float = 0.5  # 0-1, where 1 is most urgent
    hidden_quantity: Optional[float] = None  # For iceberg orders


@dataclass
class ExecutionPlan:
    """Execution plan for an order"""

    parent_order: Order
    child_orders: list[Order] = field(default_factory=list)
    execution_params: ExecutionParams = field(default_factory=ExecutionParams)
    total_executed: float = 0.0
    average_price: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    status: str = "pending"


class ExecutionHandler:
    """
    Handles smart order execution using various algorithms.

    Features:
    - Multiple execution algorithms (TWAP, VWAP, Iceberg, etc.)
    - Smart order routing
    - Order slicing and scheduling
    - Execution analytics
    """

    def __init__(
        self,
        order_manager: Optional[Any] = None,
        market_data_provider: Optional[Any] = None,
        # Backward compatibility
        executor: Optional[Any] = None
    ) -> None:
        """
        Initialize execution handler.

        Args:
            order_manager: Order manager instance (new API)
            market_data_provider: Market data provider for execution decisions
            executor: Executor instance (backward compatibility)
        """
        self.logger = setup_logger(__name__)

        # Handle backward compatibility
        if executor is not None:
            # Old API - executor-based
            self.executor = executor
            self.order_manager = None
            self.market_data_provider = None
            # Add backward compatibility attributes
            self.max_retries = 3
            self.retry_delay = 1.0
        else:
            # New API - order manager based
            self.order_manager = order_manager
            self.market_data_provider = market_data_provider
            self.executor = None

        self.execution_plans: dict[str, ExecutionPlan] = {}
        self.active_executions: dict[str, asyncio.Task] = {}
        self._execution_lock = asyncio.Lock()

    async def execute_order(
        self, order: Union[Order, Dict[str, Any]], params: Optional[ExecutionParams] = None
    ) -> Union[ExecutionPlan, Dict[str, Any]]:
        """
        Execute an order using the specified algorithm.

        Args:
            order: Order to execute (Order object or dict for backward compatibility)
            params: Execution parameters

        Returns:
            Execution plan object or dict for backward compatibility
        """
        # Handle backward compatibility
        if self.executor is not None:
            # Old API - simple executor-based execution
            return await self._execute_order_legacy(order)

        # New API - advanced execution algorithms
        params = params or ExecutionParams()

        # Create execution plan
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        async with self._execution_lock:
            self.execution_plans[order.order_id] = plan

        # Start execution based on algorithm
        if params.algorithm == ExecutionAlgorithm.MARKET:
            task = asyncio.create_task(self._execute_market_order(plan))
        elif params.algorithm == ExecutionAlgorithm.TWAP:
            task = asyncio.create_task(self._execute_twap(plan))
        elif params.algorithm == ExecutionAlgorithm.VWAP:
            task = asyncio.create_task(self._execute_vwap(plan))
        elif params.algorithm == ExecutionAlgorithm.ICEBERG:
            task = asyncio.create_task(self._execute_iceberg(plan))
        elif params.algorithm == ExecutionAlgorithm.POV:
            task = asyncio.create_task(self._execute_pov(plan))
        else:
            task = asyncio.create_task(self._execute_smart_order(plan))

        self.active_executions[order.order_id] = task

        self.logger.info(
            f"Started {params.algorithm.value} execution for order {order.order_id}"
        )
        return plan

    async def cancel_execution(self, order_id: str) -> bool:
        """
        Cancel an active execution.

        Args:
            order_id: Parent order ID

        Returns:
            True if cancellation successful
        """
        if order_id not in self.active_executions:
            self.logger.warning(f"No active execution found for order {order_id}")
            return False

        task = self.active_executions[order_id]
        task.cancel()

        # Cancel all child orders
        plan = self.execution_plans.get(order_id)
        if plan:
            for child_order in plan.child_orders:
                if child_order.is_active():
                    await self.order_manager.cancel_order(child_order.order_id)

            plan.status = "cancelled"

        del self.active_executions[order_id]
        self.logger.info(f"Cancelled execution for order {order_id}")
        return True

    async def _execute_market_order(self, plan: ExecutionPlan) -> None:
        """Execute as a simple market order"""
        try:
            plan.status = "executing"
            order = plan.parent_order

            # Submit the order directly
            success = await self.order_manager.submit_order(order)

            if success:
                plan.child_orders.append(order)
                plan.status = "completed"
            else:
                plan.status = "failed"

        except Exception as e:
            self.logger.error(f"Error in market order execution: {e}")
            plan.status = "error"

    async def _execute_twap(self, plan: ExecutionPlan) -> None:
        """Execute using Time-Weighted Average Price algorithm"""
        try:
            plan.status = "executing"
            order = plan.parent_order
            params = plan.execution_params

            # Calculate time slices
            start_time = params.start_time or datetime.now()
            end_time = params.end_time or (start_time + timedelta(hours=1))
            duration = (end_time - start_time).total_seconds()

            # Determine slice count and size
            num_slices = max(
                10, int(duration / 60)
            )  # At least 10 slices or 1 per minute
            slice_quantity = order.quantity / num_slices
            slice_interval = duration / num_slices

            self.logger.info(
                f"TWAP execution: {num_slices} slices of {slice_quantity:.2f} every {slice_interval:.0f}s"
            )

            for i in range(num_slices):
                if plan.status != "executing":
                    break

                # Create child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    order_type=(
                        OrderType.LIMIT if params.price_limit else OrderType.MARKET
                    ),
                    side=order.side,
                    quantity=slice_quantity,
                    price=params.price_limit,
                    strategy_id=order.strategy_id,
                )

                plan.child_orders.append(child_order)

                # Submit order
                await self.order_manager.submit_order(child_order)

                # Update execution stats
                plan.total_executed += slice_quantity

                # Wait for next slice
                if i < num_slices - 1:
                    await asyncio.sleep(slice_interval)

            plan.status = "completed"
            plan.completion_time = datetime.now()

        except asyncio.CancelledError:
            plan.status = "cancelled"
            raise
        except Exception as e:
            self.logger.error(f"Error in TWAP execution: {e}")
            plan.status = "error"

    async def _execute_vwap(self, plan: ExecutionPlan) -> None:
        """Execute using Volume-Weighted Average Price algorithm"""
        try:
            plan.status = "executing"
            order = plan.parent_order
            params = plan.execution_params

            if not self.market_data_provider:
                self.logger.warning("No market data provider, falling back to TWAP")
                await self._execute_twap(plan)
                return

            # Get volume profile
            volume_profile = await self._get_volume_profile(order.symbol)

            # Calculate execution schedule based on historical volume
            schedule = self._calculate_vwap_schedule(order.quantity, volume_profile)

            for time_slot, quantity in schedule:
                if plan.status != "executing":
                    break

                # Wait until time slot
                wait_time = (time_slot - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Create and submit child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    order_type=(
                        OrderType.LIMIT if params.price_limit else OrderType.MARKET
                    ),
                    side=order.side,
                    quantity=quantity,
                    price=params.price_limit,
                    strategy_id=order.strategy_id,
                )

                plan.child_orders.append(child_order)
                await self.order_manager.submit_order(child_order)

                plan.total_executed += quantity

            plan.status = "completed"
            plan.completion_time = datetime.now()

        except asyncio.CancelledError:
            plan.status = "cancelled"
            raise
        except Exception as e:
            self.logger.error(f"Error in VWAP execution: {e}")
            plan.status = "error"

    async def _execute_iceberg(self, plan: ExecutionPlan) -> None:
        """Execute iceberg order (hidden quantity)"""
        try:
            plan.status = "executing"
            order = plan.parent_order
            params = plan.execution_params

            # Determine visible and hidden quantities
            visible_quantity = params.slice_size or (
                order.quantity * 0.1
            )  # 10% visible by default
            params.hidden_quantity or (
                order.quantity - visible_quantity
            )

            remaining = order.quantity

            while remaining > 0 and plan.status == "executing":
                # Calculate next slice
                slice_size = min(visible_quantity, remaining)

                # Create child order
                child_order = await self.order_manager.create_order(
                    symbol=order.symbol,
                    order_type=order.order_type,
                    side=order.side,
                    quantity=slice_size,
                    price=order.price,
                    strategy_id=order.strategy_id,
                )

                plan.child_orders.append(child_order)
                await self.order_manager.submit_order(child_order)

                # Wait for fill or timeout
                await self._wait_for_order_completion(child_order, timeout=60)

                if child_order.is_filled():
                    plan.total_executed += child_order.filled_quantity
                    remaining -= child_order.filled_quantity
                else:
                    # Handle partial fill or rejection
                    if child_order.filled_quantity > 0:
                        plan.total_executed += child_order.filled_quantity
                        remaining -= child_order.filled_quantity
                    else:
                        break

            plan.status = "completed" if remaining == 0 else "partial"
            plan.completion_time = datetime.now()

        except asyncio.CancelledError:
            plan.status = "cancelled"
            raise
        except Exception as e:
            self.logger.error(f"Error in iceberg execution: {e}")
            plan.status = "error"

    async def _execute_pov(self, plan: ExecutionPlan) -> None:
        """Execute as Percentage of Volume"""
        try:
            plan.status = "executing"
            order = plan.parent_order
            params = plan.execution_params

            if not self.market_data_provider:
                self.logger.warning("No market data provider, falling back to TWAP")
                await self._execute_twap(plan)
                return

            remaining = order.quantity

            while remaining > 0 and plan.status == "executing":
                # Get current market volume
                market_volume = await self._get_market_volume(
                    order.symbol, lookback_seconds=60
                )

                # Calculate our slice based on participation rate
                slice_size = min(
                    market_volume * params.max_participation_rate, remaining
                )

                if slice_size > 0:
                    # Create and submit child order
                    child_order = await self.order_manager.create_order(
                        symbol=order.symbol,
                        order_type=OrderType.MARKET,
                        side=order.side,
                        quantity=slice_size,
                        strategy_id=order.strategy_id,
                    )

                    plan.child_orders.append(child_order)
                    await self.order_manager.submit_order(child_order)

                    # Wait for completion
                    await self._wait_for_order_completion(child_order, timeout=30)

                    if child_order.filled_quantity > 0:
                        plan.total_executed += child_order.filled_quantity
                        remaining -= child_order.filled_quantity

                # Wait before next check
                await asyncio.sleep(5)

            plan.status = "completed" if remaining == 0 else "partial"
            plan.completion_time = datetime.now()

        except asyncio.CancelledError:
            plan.status = "cancelled"
            raise
        except Exception as e:
            self.logger.error(f"Error in POV execution: {e}")
            plan.status = "error"

    async def _execute_smart_order(self, plan: ExecutionPlan) -> None:
        """Execute using smart order routing logic"""
        # Implement smart order routing based on:
        # - Market conditions
        # - Liquidity
        # - Spread
        # - Urgency
        # For now, fallback to TWAP
        await self._execute_twap(plan)

    async def _get_volume_profile(self, symbol: str) -> list[tuple[datetime, float]]:
        """Get historical volume profile for VWAP calculation"""
        # TODO: Implement actual volume profile retrieval
        # For now, return mock data
        now = datetime.now()
        return [(now + timedelta(minutes=i), 1000.0) for i in range(60)]

    def _calculate_vwap_schedule(
        self, total_quantity: float, volume_profile: list[tuple[datetime, float]]
    ) -> list[tuple[datetime, float]]:
        """Calculate VWAP execution schedule"""
        total_volume = sum(v for _, v in volume_profile)
        schedule = []

        for time_slot, volume in volume_profile:
            quantity = total_quantity * (volume / total_volume)
            schedule.append((time_slot, quantity))

        return schedule

    async def _get_market_volume(self, symbol: str, lookback_seconds: int) -> float:
        """Get recent market volume"""
        # TODO: Implement actual market volume retrieval
        return 10000.0  # Mock value

    async def _wait_for_order_completion(self, order: Order, timeout: float) -> None:
        """Wait for order to complete or timeout"""
        start_time = datetime.now()

        while (
            order.is_active()
            and (datetime.now() - start_time).total_seconds() < timeout
        ):
            await asyncio.sleep(1)

    async def _execute_order_legacy(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order using legacy executor API with retry logic."""
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Call executor's place_order method
                result = await self.executor.place_order(order)
                return result
            except Exception as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    self.logger.warning(f"Execution attempt {retries} failed: {e}, retrying...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All execution attempts failed: {e}")
                    raise

        # Should not reach here
        raise last_error if last_error else Exception("Execution failed")

    async def execute_with_retry(self, order: Dict[str, Any], max_retries: Optional[int] = None) -> Dict[str, Any]:
        """Execute order with retry logic (backward compatibility method)."""
        if max_retries is not None:
            old_max_retries = self.max_retries
            self.max_retries = max_retries
            try:
                return await self._execute_order_legacy(order)
            finally:
                self.max_retries = old_max_retries
        else:
            return await self._execute_order_legacy(order)

    def calculate_slippage(self, order: Union[Dict[str, Any], float], fill: Optional[Dict[str, Any]] = None, side: Optional[str] = None) -> Union[Dict[str, Any], float]:
        """Calculate slippage for an execution (backward compatibility method)."""
        # Handle new API (dict inputs)
        if isinstance(order, dict) and fill is not None:
            expected_price = order.get('expected_price', order.get('price', 0))
            actual_price = fill.get('avg_fill_price', 0)
            side = order.get('side', 'BUY')
            quantity = fill.get('filled_quantity', 0)

            # Calculate slippage amount per share
            if side.upper() == 'BUY':
                # For buys, negative slippage means we paid more than expected
                slippage_amount = expected_price - actual_price
            else:
                # For sells, negative slippage means we received less than expected
                slippage_amount = actual_price - expected_price

            # Calculate percentage
            slippage_pct = (slippage_amount / expected_price * 100) if expected_price > 0 else 0

            # Calculate total cost impact
            slippage_cost = slippage_amount * quantity

            return {
                'amount': slippage_amount,
                'percentage': slippage_pct,
                'cost': slippage_cost
            }

        # Handle old API (individual parameters)
        else:
            expected_price = order
            actual_price = fill if fill is not None else 0
            side = side or 'BUY'

            if side.upper() == 'BUY':
                # For buys, positive slippage means we paid more than expected
                return (actual_price - expected_price) / expected_price
            else:
                # For sells, positive slippage means we received less than expected
                return (expected_price - actual_price) / expected_price

    async def smart_route_order(self, order: Dict[str, Any], venues: list[str]) -> Dict[str, Any]:
        """Smart route order to best venue (backward compatibility method)."""
        # For now, just use the first venue or execute normally
        if venues and hasattr(self.executor, 'set_venue'):
            self.executor.set_venue(venues[0])
        return await self._execute_order_legacy(order)

    def get_execution_stats(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for an order"""
        plan = self.execution_plans.get(order_id)
        if not plan:
            return None

        filled_orders = [o for o in plan.child_orders if o.is_filled()]
        total_filled = sum(o.filled_quantity for o in filled_orders)

        if total_filled > 0:
            avg_price = (
                sum(o.average_fill_price * o.filled_quantity for o in filled_orders)
                / total_filled
            )
        else:
            avg_price = 0.0

        return {
            "parent_order_id": order_id,
            "algorithm": plan.execution_params.algorithm.value,
            "status": plan.status,
            "total_quantity": plan.parent_order.quantity,
            "executed_quantity": plan.total_executed,
            "fill_rate": plan.total_executed / plan.parent_order.quantity,
            "average_price": avg_price,
            "child_orders": len(plan.child_orders),
            "start_time": plan.start_time.isoformat(),
            "completion_time": (
                plan.completion_time.isoformat() if plan.completion_time else None
            ),
            "duration_seconds": (
                (plan.completion_time - plan.start_time).total_seconds()
                if plan.completion_time
                else None
            ),
        }
