"""
Minimal test suite for ExecutionHandler achieving 100% coverage.

This test file uses careful mocking to avoid hanging async operations.
"""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from core.engine.execution_handler import (
    ExecutionAlgorithm,
    ExecutionHandler,
    ExecutionParams,
    ExecutionPlan,
)
from core.engine.order_manager import Order, OrderStatus, OrderType


@contextmanager
def mock_async_task():
    """Context manager to mock asyncio.create_task."""
    mock_task = Mock()
    mock_task.cancel = Mock()
    with patch('asyncio.create_task', return_value=mock_task):
        yield mock_task


class TestExecutionHandlerMinimal:
    """Minimal tests for 100% coverage."""

    def test_initialization_both_apis(self):
        """Test both initialization paths."""
        # Legacy API
        executor = Mock()
        h1 = ExecutionHandler(executor=executor)
        assert h1.executor == executor
        assert h1.order_manager is None
        assert h1.max_retries == 3
        assert h1.retry_delay == 1.0

        # New API
        manager = Mock()
        provider = Mock()
        h2 = ExecutionHandler(order_manager=manager, market_data_provider=provider)
        assert h2.order_manager == manager
        assert h2.market_data_provider == provider
        assert h2.executor is None

    @pytest.mark.asyncio
    async def test_execute_order_all_algorithms(self):
        """Test execute_order with all algorithms."""
        manager = Mock()
        manager.create_order = AsyncMock()
        manager.submit_order = AsyncMock(return_value=True)

        handler = ExecutionHandler(order_manager=manager, market_data_provider=Mock())

        # Test each algorithm
        algorithms = [
            ExecutionAlgorithm.MARKET,
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
            ExecutionAlgorithm.ICEBERG,
            ExecutionAlgorithm.POV,
            ExecutionAlgorithm.SMART
        ]

        for algo in algorithms:
            order = Order(order_id=f"test_{algo.value}", symbol="AAPL", quantity=100)
            params = ExecutionParams(algorithm=algo)

            with mock_async_task():
                plan = await handler.execute_order(order, params)
                assert isinstance(plan, ExecutionPlan)
                assert plan.execution_params.algorithm == algo

    @pytest.mark.asyncio
    async def test_execute_order_legacy(self):
        """Test legacy executor path."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}

        result = await handler.execute_order(order)
        assert result['status'] == 'FILLED'

    @pytest.mark.asyncio
    async def test_cancel_execution_all_paths(self):
        """Test all cancel execution paths."""
        manager = Mock()
        manager.cancel_order = AsyncMock(return_value=True)
        handler = ExecutionHandler(order_manager=manager)

        # Not found
        success = await handler.cancel_execution("not_found")
        assert success is False

        # Success path
        order = Order(order_id="test")
        child = Order(order_id="child", status=OrderStatus.SUBMITTED)
        plan = ExecutionPlan(parent_order=order, child_orders=[child])

        handler.execution_plans["test"] = plan
        handler.active_executions["test"] = Mock(cancel=Mock())

        success = await handler.cancel_execution("test")
        assert success is True
        assert plan.status == "cancelled"

    @pytest.mark.asyncio
    async def test_execute_market_order_all_paths(self):
        """Test all market order execution paths."""
        manager = Mock()
        handler = ExecutionHandler(order_manager=manager)

        # Success
        manager.submit_order = AsyncMock(return_value=True)
        plan1 = ExecutionPlan(parent_order=Order())
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_market_order(plan1)
        assert plan1.status == "completed"

        # Failure
        manager.submit_order = AsyncMock(return_value=False)
        plan2 = ExecutionPlan(parent_order=Order())
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_market_order(plan2)
        assert plan2.status == "failed"

        # Exception
        manager.submit_order = AsyncMock(side_effect=Exception("Error"))
        plan3 = ExecutionPlan(parent_order=Order())
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_market_order(plan3)
        assert plan3.status == "error"

    @pytest.mark.asyncio
    async def test_execute_twap_all_paths(self):
        """Test TWAP execution paths."""
        manager = Mock()
        manager.create_order = AsyncMock(return_value=Order())
        manager.submit_order = AsyncMock()
        handler = ExecutionHandler(order_manager=manager)

        # Normal execution
        order = Order(quantity=100)
        params = ExecutionParams(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            price_limit=150.0
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_twap(plan)

        assert plan.status == "completed"
        assert plan.completion_time is not None

        # Test cancellation during execution
        plan2 = ExecutionPlan(parent_order=Order(quantity=100))
        
        # Mock create_order to set status to cancelled after first order
        call_count = 0
        async def mock_create_order(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                plan2.status = "cancelled"
            return Order()
        
        manager.create_order = mock_create_order
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_twap(plan2)
        
        # Should have created at least one order but stopped early
        assert len(plan2.child_orders) > 0
        assert len(plan2.child_orders) < 10  # Less than the minimum 10 slices

        # Error
        manager.create_order = AsyncMock(side_effect=Exception("Error"))
        plan3 = ExecutionPlan(parent_order=Order())
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_twap(plan3)
        assert plan3.status == "error"

    @pytest.mark.asyncio
    async def test_execute_vwap_all_paths(self):
        """Test VWAP execution paths."""
        manager = Mock()
        manager.create_order = AsyncMock(return_value=Order())
        manager.submit_order = AsyncMock()
        provider = Mock()
        handler = ExecutionHandler(order_manager=manager, market_data_provider=provider)

        # Normal execution
        with patch.object(handler, '_get_volume_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = [(datetime.now(), 1000.0)]

            order = Order(quantity=1000)
            params = ExecutionParams(price_limit=150.0)
            plan = ExecutionPlan(parent_order=order, execution_params=params)

            with patch('asyncio.sleep', new_callable=AsyncMock):
                await handler._execute_vwap(plan)

            assert plan.status == "completed"

        # No market data provider
        handler2 = ExecutionHandler(order_manager=manager)
        plan2 = ExecutionPlan(parent_order=Order())

        with patch.object(handler2, '_execute_twap', new_callable=AsyncMock):
            await handler2._execute_vwap(plan2)

        # Test actual cancellation via CancelledError
        plan3 = ExecutionPlan(parent_order=Order())
        
        # Mock to raise CancelledError
        with patch.object(handler, '_get_volume_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.side_effect = asyncio.CancelledError()
            
            with pytest.raises(asyncio.CancelledError):
                await handler._execute_vwap(plan3)
            
            assert plan3.status == "cancelled"

        # Error
        with patch.object(handler, '_get_volume_profile', side_effect=Exception("Error")):
            plan4 = ExecutionPlan(parent_order=Order())
            with patch('asyncio.sleep', new_callable=AsyncMock):
                await handler._execute_vwap(plan4)
            assert plan4.status == "error"

    @pytest.mark.asyncio
    async def test_execute_iceberg_all_paths(self):
        """Test Iceberg execution paths."""
        manager = Mock()
        handler = ExecutionHandler(order_manager=manager)

        # Normal execution - complete fill
        order = Order(quantity=100, order_type=OrderType.LIMIT, price=150.0)
        params = ExecutionParams(slice_size=50.0, hidden_quantity=50.0)
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        filled_child = Order(status=OrderStatus.FILLED, filled_quantity=50.0)
        manager.create_order = AsyncMock(side_effect=[filled_child, filled_child])
        manager.submit_order = AsyncMock()

        with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
            await handler._execute_iceberg(plan)

        assert plan.status == "completed"
        assert plan.total_executed == 100.0

        # Partial fill then break
        order2 = Order(quantity=100)
        plan2 = ExecutionPlan(parent_order=order2, execution_params=ExecutionParams())

        partial_child = Order(status=OrderStatus.PARTIAL, filled_quantity=30.0)
        unfilled_child = Order(status=OrderStatus.CANCELLED, filled_quantity=0.0)
        manager.create_order = AsyncMock(side_effect=[partial_child, unfilled_child])

        with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
            await handler._execute_iceberg(plan2)

        assert plan2.status == "partial"
        assert plan2.total_executed == 30.0

        # Test actual cancellation via CancelledError
        order3 = Order(quantity=100, order_type=OrderType.LIMIT, price=150.0)
        plan3 = ExecutionPlan(parent_order=order3, execution_params=ExecutionParams(slice_size=50.0))
        
        # Mock to raise CancelledError
        manager.create_order = AsyncMock(side_effect=asyncio.CancelledError())
        
        with pytest.raises(asyncio.CancelledError):
            await handler._execute_iceberg(plan3)
        
        assert plan3.status == "cancelled"

        # Error
        manager.create_order = AsyncMock(side_effect=Exception("Error"))
        order4 = Order(quantity=100, order_type=OrderType.LIMIT, price=150.0)
        plan4 = ExecutionPlan(parent_order=order4, execution_params=ExecutionParams(slice_size=50.0))
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._execute_iceberg(plan4)
        assert plan4.status == "error"

    @pytest.mark.asyncio
    async def test_execute_pov_all_paths(self):
        """Test POV execution paths."""
        manager = Mock()
        provider = Mock()
        handler = ExecutionHandler(order_manager=manager, market_data_provider=provider)

        # Normal execution
        order = Order(quantity=1000)
        params = ExecutionParams(max_participation_rate=0.1)
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_vol:
            mock_vol.return_value = 10000.0

            filled_child = Order(filled_quantity=1000.0)
            manager.create_order = AsyncMock(return_value=filled_child)
            manager.submit_order = AsyncMock()

            with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    await handler._execute_pov(plan)

        assert plan.status == "completed"

        # No market data provider
        handler2 = ExecutionHandler(order_manager=manager)
        plan2 = ExecutionPlan(parent_order=Order())

        with patch.object(handler2, '_execute_twap', new_callable=AsyncMock):
            await handler2._execute_pov(plan2)

        # No volume - skip slice
        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_vol:
            mock_vol.return_value = 0.0

            order3 = Order(quantity=100)
            plan3 = ExecutionPlan(parent_order=order3)
            plan3.status = "executing"  # Will be changed by loop

            # Use counter to break loop
            call_count = 0
            async def mock_sleep(delay):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    plan3.status = "stopped"

            with patch('asyncio.sleep', mock_sleep):
                await handler._execute_pov(plan3)

        # Test actual cancellation via CancelledError
        order4 = Order(quantity=1000)
        plan4 = ExecutionPlan(parent_order=order4, execution_params=ExecutionParams(max_participation_rate=0.1))
        
        # Mock to raise CancelledError
        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_vol:
            mock_vol.side_effect = asyncio.CancelledError()
            
            with pytest.raises(asyncio.CancelledError):
                await handler._execute_pov(plan4)
            
            assert plan4.status == "cancelled"

        # Error
        with patch.object(handler, '_get_market_volume', side_effect=Exception("Error")):
            order5 = Order(quantity=1000)
            plan5 = ExecutionPlan(parent_order=order5, execution_params=ExecutionParams(max_participation_rate=0.1))
            with patch('asyncio.sleep', new_callable=AsyncMock):
                await handler._execute_pov(plan5)
            assert plan5.status == "error"

    @pytest.mark.asyncio
    async def test_execute_smart_order(self):
        """Test smart order execution."""
        handler = ExecutionHandler(order_manager=Mock())
        plan = ExecutionPlan(parent_order=Order())

        with patch.object(handler, '_execute_twap', new_callable=AsyncMock):
            await handler._execute_smart_order(plan)

    @pytest.mark.asyncio
    async def test_helper_methods(self):
        """Test all helper methods."""
        handler = ExecutionHandler(order_manager=Mock())

        # _get_volume_profile
        profile = await handler._get_volume_profile("AAPL")
        assert len(profile) == 60

        # _calculate_vwap_schedule
        schedule = handler._calculate_vwap_schedule(1000.0, [(datetime.now(), 500.0), (datetime.now(), 500.0)])
        assert len(schedule) == 2
        assert schedule[0][1] == 500.0

        # _get_market_volume
        volume = await handler._get_market_volume("AAPL", 60)
        assert volume == 10000.0

        # _wait_for_order_completion
        order = Order(status=OrderStatus.FILLED)
        await handler._wait_for_order_completion(order, 1.0)

        order2 = Order(status=OrderStatus.SUBMITTED)
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await handler._wait_for_order_completion(order2, 0.1)

    @pytest.mark.asyncio
    async def test_execute_order_legacy_all_paths(self):
        """Test legacy execution with all paths."""
        executor = Mock()
        handler = ExecutionHandler(executor=executor)

        # Success
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await handler._execute_order_legacy({})
        assert result['status'] == 'FILLED'

        # Retry then success
        executor.place_order = AsyncMock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"),
            {'status': 'FILLED'}
        ])
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await handler._execute_order_legacy({})
        assert result['status'] == 'FILLED'

        # Max retries exceeded
        executor.place_order = AsyncMock(side_effect=Exception("Persistent"))
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Persistent"):
                await handler._execute_order_legacy({})

    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        """Test execute_with_retry method."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})
        handler = ExecutionHandler(executor=executor)

        # With custom retries
        result = await handler.execute_with_retry({}, max_retries=5)
        assert result['status'] == 'FILLED'
        assert handler.max_retries == 3  # Restored

        # Without custom retries
        result = await handler.execute_with_retry({})
        assert result['status'] == 'FILLED'

    def test_calculate_slippage_all_paths(self):
        """Test all slippage calculation paths."""
        handler = ExecutionHandler()

        # Dict API - Buy
        slippage = handler.calculate_slippage(
            {'expected_price': 100, 'side': 'BUY'},
            {'avg_fill_price': 101, 'filled_quantity': 10}
        )
        assert slippage['amount'] == -1.0
        assert slippage['percentage'] == -1.0
        assert slippage['cost'] == -10.0

        # Dict API - Sell
        slippage = handler.calculate_slippage(
            {'expected_price': 100, 'side': 'SELL'},
            {'avg_fill_price': 99, 'filled_quantity': 10}
        )
        assert slippage['amount'] == -1.0

        # Dict API - Zero price
        slippage = handler.calculate_slippage(
            {'expected_price': 0, 'side': 'BUY'},
            {'avg_fill_price': 100, 'filled_quantity': 10}
        )
        assert slippage['percentage'] == 0

        # Dict API - price fallback
        slippage = handler.calculate_slippage(
            {'price': 100, 'side': 'BUY'},
            {'avg_fill_price': 101, 'filled_quantity': 10}
        )
        assert slippage['amount'] == -1.0

        # Legacy API - Buy
        slippage = handler.calculate_slippage(100.0, 101.0, 'BUY')
        assert slippage == 0.01

        # Legacy API - Sell
        slippage = handler.calculate_slippage(100.0, 99.0, 'SELL')
        assert slippage == 0.01

        # Legacy API - No fill
        slippage = handler.calculate_slippage(100.0, side='BUY')
        assert slippage == -1.0

    @pytest.mark.asyncio
    async def test_smart_route_order(self):
        """Test smart order routing."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})
        handler = ExecutionHandler(executor=executor)

        # With venues and set_venue
        executor.set_venue = Mock()
        result = await handler.smart_route_order({}, ['NASDAQ', 'NYSE'])
        assert result['status'] == 'FILLED'
        executor.set_venue.assert_called_once_with('NASDAQ')

        # Without venues
        result = await handler.smart_route_order({}, [])
        assert result['status'] == 'FILLED'

        # No set_venue method
        del executor.set_venue
        result = await handler.smart_route_order({}, ['NASDAQ'])
        assert result['status'] == 'FILLED'

    def test_get_execution_stats_all_paths(self):
        """Test execution statistics."""
        handler = ExecutionHandler()

        # Not found
        assert handler.get_execution_stats("not_found") is None

        # With filled orders
        parent = Order(order_id="parent", quantity=1000)
        child1 = Order(
            status=OrderStatus.FILLED,
            filled_quantity=500,
            average_fill_price=100.0
        )
        child2 = Order(
            status=OrderStatus.FILLED,
            filled_quantity=300,
            average_fill_price=101.0
        )

        plan = ExecutionPlan(
            parent_order=parent,
            child_orders=[child1, child2],
            execution_params=ExecutionParams(algorithm=ExecutionAlgorithm.TWAP),
            total_executed=800.0,
            status="completed"
        )
        plan.completion_time = plan.start_time + timedelta(minutes=5)

        handler.execution_plans["parent"] = plan

        stats = handler.get_execution_stats("parent")
        assert stats["parent_order_id"] == "parent"
        assert stats["algorithm"] == "twap"
        assert stats["executed_quantity"] == 800.0
        assert stats["average_price"] == pytest.approx(100.375)
        assert stats["duration_seconds"] == 300.0

        # No filled orders
        plan2 = ExecutionPlan(
            parent_order=Order(quantity=100),
            child_orders=[Order(status=OrderStatus.PENDING)],
            total_executed=0.0
        )
        handler.execution_plans["empty"] = plan2

        stats2 = handler.get_execution_stats("empty")
        assert stats2["average_price"] == 0.0
        assert stats2["completion_time"] is None
        assert stats2["duration_seconds"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
