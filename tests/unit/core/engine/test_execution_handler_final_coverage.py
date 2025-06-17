"""
Final tests for ExecutionHandler to achieve maximum coverage.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from core.engine.execution_handler import (
    ExecutionAlgorithm,
    ExecutionHandler,
    ExecutionParams,
    ExecutionPlan,
)
from core.engine.order_manager import Order, OrderSide, OrderStatus, OrderType


class TestExecutionHandlerFinalCoverage:
    """Final coverage tests."""

    @pytest.fixture
    def handler(self):
        """Create handler with all dependencies."""
        manager = Mock()
        manager.create_order = AsyncMock()
        manager.submit_order = AsyncMock(return_value=True)
        manager.cancel_order = AsyncMock(return_value=True)

        provider = Mock()

        return ExecutionHandler(
            order_manager=manager,
            market_data_provider=provider
        )

    @pytest.fixture
    def legacy_handler(self):
        """Create handler with legacy executor."""
        executor = Mock()
        executor.place_order = AsyncMock()
        return ExecutionHandler(executor=executor)

    def test_execute_order_new_api_all_algorithms(self, handler):
        """Test execute_order with new API for all algorithms."""
        algorithms = [
            ExecutionAlgorithm.MARKET,
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
            ExecutionAlgorithm.ICEBERG,
            ExecutionAlgorithm.POV,
            ExecutionAlgorithm.SMART
        ]

        # Mock task creation
        mock_task = Mock()
        mock_task.cancel = Mock()

        with patch('asyncio.create_task', return_value=mock_task):
            for algo in algorithms:
                order = Order(
                    order_id=f"test_{algo.value}",
                    symbol="AAPL",
                    quantity=100,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )
                params = ExecutionParams(algorithm=algo)

                loop = asyncio.new_event_loop()
                plan = loop.run_until_complete(handler.execute_order(order, params))
                loop.close()

                assert isinstance(plan, ExecutionPlan)
                assert plan.parent_order == order
                assert plan.execution_params.algorithm == algo

    def test_cancel_execution_with_active_child_orders(self, handler):
        """Test cancelling execution with active child orders."""
        # Setup
        parent = Order(order_id="parent", symbol="AAPL", quantity=1000)
        child1 = Order(order_id="child1", status=OrderStatus.SUBMITTED)
        child2 = Order(order_id="child2", status=OrderStatus.FILLED)
        child3 = Order(order_id="child3", status=OrderStatus.PARTIAL)

        plan = ExecutionPlan(parent_order=parent, child_orders=[child1, child2, child3])
        handler.execution_plans["parent"] = plan

        mock_task = Mock()
        handler.active_executions["parent"] = mock_task

        # Execute
        loop = asyncio.new_event_loop()
        success = loop.run_until_complete(handler.cancel_execution("parent"))
        loop.close()

        # Verify
        assert success is True
        assert plan.status == "cancelled"
        mock_task.cancel.assert_called_once()
        # Should cancel only active orders (child1 and child3)
        assert handler.order_manager.cancel_order.call_count == 2

    def test_execute_twap_with_price_limit(self, handler):
        """Test TWAP with price limit."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.TWAP,
            price_limit=150.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=10)
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock child order creation
        child = Order(order_id="child", symbol="AAPL", quantity=100)
        handler.order_manager.create_order.return_value = child

        # Execute with quick termination
        async def execute_and_stop():
            task = asyncio.create_task(handler._execute_twap(plan))
            await asyncio.sleep(0.1)
            plan.status = "stopped"  # Stop execution
            await task

        loop = asyncio.new_event_loop()
        loop.run_until_complete(execute_and_stop())
        loop.close()

        # Verify limit order was created
        call_args = handler.order_manager.create_order.call_args
        assert call_args[1]['order_type'] == OrderType.LIMIT
        assert call_args[1]['price'] == 150.0

    def test_execute_vwap_with_schedule(self, handler):
        """Test VWAP execution with actual schedule."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock volume profile
        now = datetime.now()
        volume_profile = [
            (now + timedelta(seconds=1), 600.0),
            (now + timedelta(seconds=2), 400.0)
        ]

        async def mock_get_volume_profile(symbol):
            return volume_profile

        handler._get_volume_profile = mock_get_volume_profile

        # Mock child order
        child = Order(order_id="child", symbol="AAPL", quantity=600)
        handler.order_manager.create_order.return_value = child

        # Execute with quick termination
        async def execute_and_stop():
            task = asyncio.create_task(handler._execute_vwap(plan))
            await asyncio.sleep(0.1)
            plan.status = "stopped"
            await task

        loop = asyncio.new_event_loop()
        loop.run_until_complete(execute_and_stop())
        loop.close()

        # Verify execution started
        assert handler.order_manager.create_order.called

    def test_execute_iceberg_complete_fill(self, handler):
        """Test iceberg with complete fills."""
        order = Order(
            order_id="test",
            symbol="AAPL",
            quantity=1000,
            order_type=OrderType.LIMIT,
            price=150.0
        )
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.ICEBERG,
            slice_size=100.0
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock filled slices
        call_count = 0
        async def create_filled_order(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Order(
                order_id=f"slice_{call_count}",
                symbol="AAPL",
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100
            )

        handler.order_manager.create_order = create_filled_order

        async def mock_wait(order, timeout):
            # Order already filled
            pass

        with patch.object(handler, '_wait_for_order_completion', side_effect=mock_wait):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_iceberg(plan))
            loop.close()

        assert plan.status == "completed"
        assert plan.total_executed == 1000.0
        assert plan.completion_time is not None

    def test_execute_pov_with_volume(self, handler):
        """Test POV execution with actual volume."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.POV,
            max_participation_rate=0.1
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock market volume
        async def mock_get_volume(symbol, lookback_seconds):
            return 5000.0  # Allows 500 share slice at 10%

        handler._get_market_volume = mock_get_volume

        # Mock order creation and completion
        async def create_order(*args, **kwargs):
            return Order(
                order_id="child",
                symbol="AAPL",
                quantity=kwargs['quantity'],
                filled_quantity=kwargs['quantity']
            )

        handler.order_manager.create_order = create_order

        async def mock_wait(order, timeout):
            pass

        # Execute with termination after 2 slices
        execution_count = 0
        async def counting_wait(order, timeout):
            nonlocal execution_count
            execution_count += 1
            if execution_count >= 2:
                plan.total_executed = 1000.0  # Mark as complete

        with patch.object(handler, '_wait_for_order_completion', side_effect=counting_wait):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                loop = asyncio.new_event_loop()
                loop.run_until_complete(handler._execute_pov(plan))
                loop.close()

        assert plan.status == "completed"
        assert plan.completion_time is not None

    def test_wait_for_order_completion_timeout(self, handler):
        """Test order completion wait with timeout."""
        order = Order(order_id="test", status=OrderStatus.SUBMITTED)

        # Order never completes
        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._wait_for_order_completion(order, 0.1))
            loop.close()

        # Should return after timeout
        assert order.status == OrderStatus.SUBMITTED

    def test_wait_for_order_completion_filled(self, handler):
        """Test order completion wait when order fills."""
        order = Order(order_id="test", status=OrderStatus.SUBMITTED)

        async def mock_sleep(delay):
            # Simulate order fill during wait
            order.status = OrderStatus.FILLED

        with patch('asyncio.sleep', side_effect=mock_sleep):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._wait_for_order_completion(order, 10.0))
            loop.close()

        # Should detect fill and return
        assert order.status == OrderStatus.FILLED

    def test_execute_order_legacy_api_failure_retry(self, legacy_handler):
        """Test legacy API with retry on failure."""
        order = {'order_id': '123', 'symbol': 'AAPL'}

        # Fail twice, then succeed
        legacy_handler.executor.place_order.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            {'status': 'FILLED', 'order_id': '123'}
        ]

        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(legacy_handler.execute_order(order))
            loop.close()

        assert result['status'] == 'FILLED'
        assert legacy_handler.executor.place_order.call_count == 3

    def test_smart_route_order_with_set_venue(self, legacy_handler):
        """Test smart routing with venue setting."""
        order = {'order_id': '123'}
        venues = ['NASDAQ', 'NYSE']

        legacy_handler.executor.place_order.return_value = {'status': 'FILLED'}
        legacy_handler.executor.set_venue = Mock()

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(legacy_handler.smart_route_order(order, venues))
        loop.close()

        assert result['status'] == 'FILLED'
        legacy_handler.executor.set_venue.assert_called_once_with('NASDAQ')

    def test_calculate_slippage_sell_side(self, handler):
        """Test slippage calculation for sell orders."""
        # Dict API - sell side
        slippage = handler.calculate_slippage(
            {'expected_price': 100.0, 'side': 'SELL'},
            {'avg_fill_price': 99.50, 'filled_quantity': 1000}
        )
        assert slippage['amount'] == -0.50
        assert slippage['percentage'] == -0.5
        assert slippage['cost'] == -500.0

        # Legacy API - sell side
        slippage = handler.calculate_slippage(100.0, 99.0, 'SELL')
        assert slippage == 0.01  # 1% slippage

    def test_get_execution_stats_with_partial_fills(self, handler):
        """Test execution stats with various fill states."""
        parent = Order(order_id="parent", symbol="AAPL", quantity=1000)

        child1 = Order(
            order_id="child1",
            quantity=300,
            status=OrderStatus.FILLED,
            filled_quantity=300,
            average_fill_price=150.0
        )
        child2 = Order(
            order_id="child2",
            quantity=400,
            status=OrderStatus.PARTIAL,
            filled_quantity=200,
            average_fill_price=150.5
        )
        child3 = Order(
            order_id="child3",
            quantity=300,
            status=OrderStatus.REJECTED,
            filled_quantity=0
        )

        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)
        plan = ExecutionPlan(
            parent_order=parent,
            child_orders=[child1, child2, child3],
            execution_params=params,
            total_executed=500.0,
            status="executing"
        )
        plan.completion_time = plan.start_time + timedelta(minutes=5)

        handler.execution_plans["parent"] = plan

        stats = handler.get_execution_stats("parent")

        assert stats["executed_quantity"] == 500.0
        assert stats["fill_rate"] == 0.5
        assert stats["average_price"] == pytest.approx(150.2, 0.01)
        assert stats["duration_seconds"] == 300.0

    def test_execute_smart_order_delegates_to_twap(self, handler):
        """Test smart order execution delegates to TWAP."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_smart_order(plan))
            loop.close()

            mock_twap.assert_called_once_with(plan)

    def test_calculate_vwap_schedule(self, handler):
        """Test VWAP schedule calculation."""
        volume_profile = [
            (datetime.now(), 600.0),
            (datetime.now() + timedelta(minutes=1), 300.0),
            (datetime.now() + timedelta(minutes=2), 100.0)
        ]

        schedule = handler._calculate_vwap_schedule(1000.0, volume_profile)

        assert len(schedule) == 3
        assert schedule[0][1] == 600.0  # 60% of volume
        assert schedule[1][1] == 300.0  # 30% of volume
        assert schedule[2][1] == 100.0  # 10% of volume

    def test_execute_order_legacy_max_retries_exceeded(self, legacy_handler):
        """Test legacy API when max retries exceeded."""
        order = {'order_id': '123'}
        legacy_handler.executor.place_order.side_effect = Exception("Persistent error")
        legacy_handler.retry_delay = 0.01

        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            with pytest.raises(Exception, match="Persistent error"):
                loop.run_until_complete(legacy_handler._execute_order_legacy(order))
            loop.close()

        assert legacy_handler.executor.place_order.call_count == 3

    def test_execute_with_retry_preserves_max_retries(self, legacy_handler):
        """Test execute_with_retry preserves original max_retries."""
        order = {'order_id': '123'}
        legacy_handler.executor.place_order.return_value = {'status': 'FILLED'}
        original_max_retries = legacy_handler.max_retries

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            legacy_handler.execute_with_retry(order, max_retries=5)
        )
        loop.close()

        assert result['status'] == 'FILLED'
        assert legacy_handler.max_retries == original_max_retries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
