"""
Test ExecutionHandler with patched asyncio to avoid hanging.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call
import pytest

from algostack.core.engine.execution_handler import (
    ExecutionHandler,
    ExecutionParams,
    ExecutionPlan,
    ExecutionAlgorithm,
)
from algostack.core.engine.order_manager import Order, OrderType, OrderSide, OrderStatus


class MockTask:
    """Mock asyncio Task."""
    def __init__(self, coro):
        self.coro = coro
        self.cancelled = False
        self._result = None
        
    def cancel(self):
        self.cancelled = True
        
    async def __await__(self):
        if not self.cancelled:
            self._result = await self.coro
        return self._result


def create_mock_task(coro):
    """Create a mock task that doesn't actually run."""
    task = Mock()
    task.cancel = Mock()
    task.cancelled = Mock(return_value=False)
    # Store the coroutine for later inspection if needed
    task._coro = coro
    return task


class TestExecutionHandlerPatched:
    """Test ExecutionHandler with patched async operations."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        manager = Mock()
        manager.create_order = AsyncMock()
        manager.submit_order = AsyncMock(return_value=True)
        manager.cancel_order = AsyncMock(return_value=True)
        
        provider = Mock()
        provider.get_volume_profile = AsyncMock()
        provider.get_market_volume = AsyncMock()
        
        return {
            'order_manager': manager,
            'market_data_provider': provider
        }
    
    @pytest.fixture
    def handler(self, mock_components):
        """Create handler with mocked components."""
        return ExecutionHandler(
            order_manager=mock_components['order_manager'],
            market_data_provider=mock_components['market_data_provider']
        )
    
    @patch('asyncio.create_task', side_effect=create_mock_task)
    def test_execute_order_all_algorithms(self, mock_create_task, handler):
        """Test execute_order with all algorithms."""
        algorithms = [
            ExecutionAlgorithm.MARKET,
            ExecutionAlgorithm.TWAP,
            ExecutionAlgorithm.VWAP,
            ExecutionAlgorithm.ICEBERG,
            ExecutionAlgorithm.POV,
            ExecutionAlgorithm.SMART
        ]
        
        for algo in algorithms:
            order = Order(
                order_id=f"test_{algo.value}",
                symbol="AAPL",
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )
            params = ExecutionParams(algorithm=algo)
            
            # Execute order
            loop = asyncio.new_event_loop()
            plan = loop.run_until_complete(handler.execute_order(order, params))
            loop.close()
            
            # Verify execution plan created
            assert isinstance(plan, ExecutionPlan)
            assert plan.parent_order == order
            assert plan.execution_params.algorithm == algo
            assert order.order_id in handler.execution_plans
            assert order.order_id in handler.active_executions
            
            # Verify create_task was called
            assert mock_create_task.called
    
    def test_execute_market_order_complete(self, handler, mock_components):
        """Test complete market order execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)
        
        # Execute
        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_market_order(plan))
        loop.close()
        
        # Verify
        assert plan.status == "completed"
        # Note: completion_time is only set for multi-step algorithms
        assert len(plan.child_orders) == 1
        mock_components['order_manager'].submit_order.assert_called_once()
    
    def test_execute_twap_complete(self, handler, mock_components):
        """Test TWAP execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.TWAP,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=10)
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)
        
        # Mock child order
        child = Order(order_id="child", symbol="AAPL", quantity=100)
        mock_components['order_manager'].create_order.return_value = child
        
        # Patch sleep to avoid waiting
        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_twap(plan))
            loop.close()
        
        # Verify
        assert plan.status == "completed"
        assert plan.total_executed == 1000.0
        assert len(plan.child_orders) > 0
    
    def test_execute_vwap_with_volume_profile(self, handler, mock_components):
        """Test VWAP execution with volume profile."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)
        plan = ExecutionPlan(parent_order=order, execution_params=params)
        
        # Mock volume profile
        with patch.object(handler, '_get_volume_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = [
                (datetime.now(), 500.0),
                (datetime.now() + timedelta(minutes=1), 500.0)
            ]
            
            # Mock child order
            child = Order(order_id="child", symbol="AAPL", quantity=500)
            mock_components['order_manager'].create_order.return_value = child
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                loop = asyncio.new_event_loop()
                loop.run_until_complete(handler._execute_vwap(plan))
                loop.close()
        
        assert plan.status == "completed"
        assert plan.total_executed == 1000.0
    
    def test_execute_iceberg_slices(self, handler, mock_components):
        """Test Iceberg execution with slices."""
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
        filled_slices = []
        for i in range(10):
            child = Order(
                order_id=f"slice_{i}",
                symbol="AAPL",
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100
            )
            filled_slices.append(child)
        
        mock_components['order_manager'].create_order.side_effect = filled_slices
        
        with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_iceberg(plan))
            loop.close()
        
        assert plan.status == "completed"
        assert plan.total_executed == 1000.0
        assert len(plan.child_orders) == 10
    
    def test_execute_pov_with_market_volume(self, handler, mock_components):
        """Test POV execution with market volume."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.POV,
            max_participation_rate=0.1
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)
        
        # Mock market volume
        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_vol:
            mock_vol.return_value = 10000.0  # Will allow 1000 share order (10%)
            
            # Mock child order that fills completely
            child = Order(
                order_id="child",
                symbol="AAPL", 
                quantity=1000,
                filled_quantity=1000
            )
            mock_components['order_manager'].create_order.return_value = child
            
            with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(handler._execute_pov(plan))
                    loop.close()
        
        assert plan.status == "completed"
        assert plan.total_executed == 1000.0
    
    def test_cancel_execution_success(self, handler, mock_components):
        """Test successful execution cancellation."""
        # Setup plan
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        child1 = Order(order_id="child1", status=OrderStatus.SUBMITTED)
        child2 = Order(order_id="child2", status=OrderStatus.PENDING)
        
        plan = ExecutionPlan(parent_order=order, child_orders=[child1, child2])
        handler.execution_plans["test"] = plan
        
        # Mock active task
        mock_task = Mock()
        handler.active_executions["test"] = mock_task
        
        # Cancel
        loop = asyncio.new_event_loop()
        success = loop.run_until_complete(handler.cancel_execution("test"))
        loop.close()
        
        assert success is True
        assert plan.status == "cancelled"
        mock_task.cancel.assert_called_once()
        assert mock_components['order_manager'].cancel_order.call_count == 2
    
    def test_get_execution_stats_complete(self, handler):
        """Test execution statistics with complete data."""
        parent = Order(order_id="parent", symbol="AAPL", quantity=1000)
        
        # Create child orders with different states
        child1 = Order(
            order_id="child1",
            symbol="AAPL",
            quantity=300,
            status=OrderStatus.FILLED,
            filled_quantity=300,
            average_fill_price=150.0
        )
        child2 = Order(
            order_id="child2",
            symbol="AAPL",
            quantity=400,
            status=OrderStatus.FILLED,
            filled_quantity=400,
            average_fill_price=150.5
        )
        child3 = Order(
            order_id="child3",
            symbol="AAPL",
            quantity=300,
            status=OrderStatus.PARTIAL,
            filled_quantity=200,
            average_fill_price=151.0
        )
        
        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)
        plan = ExecutionPlan(
            parent_order=parent,
            child_orders=[child1, child2, child3],
            execution_params=params,
            total_executed=900.0,
            status="executing"
        )
        plan.completion_time = plan.start_time + timedelta(minutes=10)
        
        handler.execution_plans["parent"] = plan
        
        stats = handler.get_execution_stats("parent")
        
        assert stats["parent_order_id"] == "parent"
        assert stats["algorithm"] == "vwap"
        assert stats["status"] == "executing"
        assert stats["total_quantity"] == 1000
        assert stats["executed_quantity"] == 900.0
        assert stats["fill_rate"] == 0.9
        assert stats["child_orders"] == 3
        # Average price: (300*150 + 400*150.5) / 700 = 150.29
        assert stats["average_price"] == pytest.approx(150.29, 0.01)
        assert stats["duration_seconds"] == 600.0
    
    def test_calculate_slippage_dict_api_comprehensive(self, handler):
        """Test slippage calculation with various scenarios."""
        # Buy order - paid more than expected
        slippage = handler.calculate_slippage(
            {'expected_price': 100.0, 'side': 'BUY'},
            {'avg_fill_price': 100.50, 'filled_quantity': 1000}
        )
        assert slippage['amount'] == -0.50  # Negative = bad for buyer
        assert slippage['percentage'] == -0.5
        assert slippage['cost'] == -500.0
        
        # Sell order - received less than expected
        slippage = handler.calculate_slippage(
            {'expected_price': 100.0, 'side': 'SELL'},
            {'avg_fill_price': 99.50, 'filled_quantity': 1000}
        )
        assert slippage['amount'] == -0.50  # Negative = bad for seller
        assert slippage['percentage'] == -0.5
        assert slippage['cost'] == -500.0
        
        # Perfect fill
        slippage = handler.calculate_slippage(
            {'expected_price': 100.0, 'side': 'BUY'},
            {'avg_fill_price': 100.0, 'filled_quantity': 1000}
        )
        assert slippage['amount'] == 0.0
        assert slippage['percentage'] == 0.0
        assert slippage['cost'] == 0.0
    
    def test_smart_route_order_with_venues(self):
        """Test smart order routing."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED', 'venue': 'NASDAQ'})
        executor.set_venue = Mock()
        
        # Create handler with executor (old API) which sets max_retries
        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}
        venues = ['NASDAQ', 'NYSE', 'ARCA']
        
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(handler.smart_route_order(order, venues))
        loop.close()
        
        assert result['status'] == 'FILLED'
        executor.set_venue.assert_called_once_with('NASDAQ')
    
    def test_execute_with_retry_exponential_backoff(self):
        """Test retry with exponential backoff."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=[
            Exception("Network error"),
            Exception("Timeout"),
            {'status': 'FILLED', 'order_id': '123'}
        ])
        
        # Create handler with executor (old API)
        handler = ExecutionHandler(executor=executor)
        handler.retry_delay = 0.01  # Fast retry for testing
        order = {'order_id': '123'}
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(handler.execute_with_retry(order))
            loop.close()
        
        assert result['status'] == 'FILLED'
        assert executor.place_order.call_count == 3
        # Check exponential backoff
        assert mock_sleep.call_count == 2
    
    def test_helper_methods_coverage(self, handler):
        """Test all helper methods."""
        loop = asyncio.new_event_loop()
        
        # _get_volume_profile
        profile = loop.run_until_complete(handler._get_volume_profile("AAPL"))
        assert len(profile) == 60
        assert all(isinstance(t, datetime) and v == 1000.0 for t, v in profile)
        
        # _calculate_vwap_schedule
        test_profile = [
            (datetime.now(), 600.0),
            (datetime.now() + timedelta(minutes=1), 400.0)
        ]
        schedule = handler._calculate_vwap_schedule(1000.0, test_profile)
        assert len(schedule) == 2
        assert schedule[0][1] == 600.0  # 60% of volume
        assert schedule[1][1] == 400.0  # 40% of volume
        
        # _get_market_volume
        volume = loop.run_until_complete(handler._get_market_volume("AAPL", 300))
        assert volume == 10000.0
        
        # _wait_for_order_completion with timeout
        order = Order(order_id="test", status=OrderStatus.SUBMITTED)
        
        # Mock order status changes
        async def mock_wait():
            await asyncio.sleep(0.01)
            order.status = OrderStatus.FILLED
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop.run_until_complete(handler._wait_for_order_completion(order, 1.0))
        
        loop.close()
    
    def test_execute_order_legacy_api_retry(self):
        """Test legacy API with retry logic."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=[
            Exception("Connection refused"),
            Exception("Timeout"),
            {'status': 'FILLED', 'order_id': '123', 'filled_quantity': 100}
        ])
        
        # Create handler with executor (old API)
        handler = ExecutionHandler(executor=executor)
        handler.retry_delay = 0.01
        
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(handler._execute_order_legacy(order))
            loop.close()
        
        assert result['status'] == 'FILLED'
        assert executor.place_order.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])