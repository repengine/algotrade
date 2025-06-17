"""
Edge case tests for ExecutionHandler to improve coverage.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from core.engine.execution_handler import (
    ExecutionHandler,
    ExecutionPlan,
)
from core.engine.order_manager import Order, OrderStatus


class TestExecutionHandlerEdgeCases:
    """Test edge cases and error paths."""

    def test_execute_order_legacy_api_direct(self):
        """Test legacy API execution path."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={
            'status': 'FILLED',
            'order_id': '123',
            'filled_quantity': 100,
            'avg_fill_price': 150.0
        })

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(handler.execute_order(order))
        loop.close()

        assert result['status'] == 'FILLED'
        executor.place_order.assert_called_once_with(order)

    def test_execute_market_order_failure(self):
        """Test market order execution failure."""
        manager = Mock()
        manager.submit_order = AsyncMock(return_value=False)

        handler = ExecutionHandler(order_manager=manager)
        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_market_order(plan))
        loop.close()

        assert plan.status == "failed"

    def test_execute_market_order_exception(self):
        """Test market order execution with exception."""
        manager = Mock()
        manager.submit_order = AsyncMock(side_effect=Exception("Network error"))

        handler = ExecutionHandler(order_manager=manager)
        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_market_order(plan))
        loop.close()

        assert plan.status == "error"

    def test_execute_twap_cancelled(self):
        """Test TWAP execution cancellation."""
        manager = Mock()
        manager.create_order = AsyncMock()
        handler = ExecutionHandler(order_manager=manager)
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_twap(plan))
        loop.close()

        # Should exit early without creating orders
        manager.create_order.assert_not_called()

    def test_execute_twap_error(self):
        """Test TWAP execution with error."""
        manager = Mock()
        manager.create_order = AsyncMock(side_effect=Exception("Order creation failed"))

        handler = ExecutionHandler(order_manager=manager)
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_twap(plan))
        loop.close()

        assert plan.status == "error"

    def test_execute_vwap_no_market_data(self):
        """Test VWAP execution without market data provider."""
        handler = ExecutionHandler(order_manager=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_vwap(plan))
            loop.close()

            mock_twap.assert_called_once_with(plan)

    def test_execute_vwap_cancelled(self):
        """Test VWAP execution cancellation."""
        manager = Mock()
        manager.create_order = AsyncMock()
        handler = ExecutionHandler(order_manager=manager, market_data_provider=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_vwap(plan))
        loop.close()

        # Should exit early without creating orders
        manager.create_order.assert_not_called()

    def test_execute_vwap_error(self):
        """Test VWAP execution with error."""
        handler = ExecutionHandler(order_manager=Mock(), market_data_provider=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        with patch.object(handler, '_get_volume_profile', side_effect=Exception("Data error")):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_vwap(plan))
            loop.close()

        assert plan.status == "error"

    def test_execute_iceberg_partial_fill(self):
        """Test Iceberg execution with partial fills."""
        manager = Mock()
        handler = ExecutionHandler(order_manager=manager)

        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Mock child order that doesn't fill
        unfilled_child = Order(
            order_id="child",
            symbol="AAPL",
            quantity=100,
            status=OrderStatus.CANCELLED,
            filled_quantity=0
        )
        manager.create_order = AsyncMock(return_value=unfilled_child)
        manager.submit_order = AsyncMock()

        async def mock_wait(order, timeout):
            # Simulate order not filling
            order.status = OrderStatus.CANCELLED

        with patch.object(handler, '_wait_for_order_completion', side_effect=mock_wait):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_iceberg(plan))
            loop.close()

        assert plan.status == "partial"
        assert plan.total_executed == 0.0

    def test_execute_iceberg_cancelled(self):
        """Test Iceberg execution cancellation."""
        manager = Mock()
        manager.create_order = AsyncMock()
        handler = ExecutionHandler(order_manager=manager)
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_iceberg(plan))
        loop.close()

        # Should exit early without creating orders
        manager.create_order.assert_not_called()

    def test_execute_iceberg_error(self):
        """Test Iceberg execution with error."""
        manager = Mock()
        manager.create_order = AsyncMock(side_effect=Exception("Create error"))

        handler = ExecutionHandler(order_manager=manager)
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_iceberg(plan))
        loop.close()

        assert plan.status == "error"

    def test_execute_pov_no_market_data(self):
        """Test POV execution without market data provider."""
        handler = ExecutionHandler(order_manager=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_pov(plan))
            loop.close()

            mock_twap.assert_called_once_with(plan)

    def test_execute_pov_no_volume(self):
        """Test POV execution with no market volume."""
        handler = ExecutionHandler(order_manager=Mock(), market_data_provider=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_vol:
            mock_vol.return_value = 0.0

            # Create counter to break loop
            call_count = 0
            async def mock_sleep(delay):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    plan.status = "stopped"

            with patch('asyncio.sleep', mock_sleep):
                loop = asyncio.new_event_loop()
                loop.run_until_complete(handler._execute_pov(plan))
                loop.close()

    def test_execute_pov_cancelled(self):
        """Test POV execution cancellation."""
        manager = Mock()
        manager.create_order = AsyncMock()
        handler = ExecutionHandler(order_manager=manager, market_data_provider=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._execute_pov(plan))
        loop.close()

        # Should exit early without creating orders
        manager.create_order.assert_not_called()

    def test_execute_pov_error(self):
        """Test POV execution with error."""
        handler = ExecutionHandler(order_manager=Mock(), market_data_provider=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        with patch.object(handler, '_get_market_volume', side_effect=Exception("Volume error")):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_pov(plan))
            loop.close()

        assert plan.status == "error"

    def test_execute_smart_order(self):
        """Test smart order execution."""
        handler = ExecutionHandler(order_manager=Mock())
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Smart order delegates to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler._execute_smart_order(plan))
            loop.close()

            mock_twap.assert_called_once_with(plan)

    def test_calculate_slippage_legacy_no_fill_price(self):
        """Test legacy slippage calculation with no fill price."""
        handler = ExecutionHandler()

        # No fill price provided
        slippage = handler.calculate_slippage(100.0, side='BUY')
        assert slippage == -1.0

        slippage = handler.calculate_slippage(100.0, side='SELL')
        assert slippage == 1.0

    def test_calculate_slippage_dict_zero_price(self):
        """Test dict slippage calculation with zero expected price."""
        handler = ExecutionHandler()

        order = {'expected_price': 0, 'side': 'BUY'}
        fill = {'avg_fill_price': 100.0, 'filled_quantity': 10}

        slippage = handler.calculate_slippage(order, fill)
        assert slippage['amount'] == -100.0
        assert slippage['percentage'] == 0  # Division by zero protection
        assert slippage['cost'] == -1000.0

    def test_calculate_slippage_dict_price_fallback(self):
        """Test dict slippage calculation with price fallback."""
        handler = ExecutionHandler()

        # No expected_price, use price field
        order = {'price': 100.0, 'side': 'BUY'}
        fill = {'avg_fill_price': 101.0, 'filled_quantity': 10}

        slippage = handler.calculate_slippage(order, fill)
        assert slippage['amount'] == -1.0

    def test_smart_route_order_no_venues(self):
        """Test smart routing with no venues."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(handler.smart_route_order(order, []))
        loop.close()

        assert result['status'] == 'FILLED'

    def test_smart_route_order_no_set_venue(self):
        """Test smart routing when executor has no set_venue."""
        executor = Mock(spec=['place_order'])  # No set_venue method
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}
        venues = ['NASDAQ']

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(handler.smart_route_order(order, venues))
        loop.close()

        assert result['status'] == 'FILLED'

    def test_get_execution_stats_not_found(self):
        """Test getting stats for non-existent execution."""
        handler = ExecutionHandler()
        stats = handler.get_execution_stats("non_existent")
        assert stats is None

    def test_get_execution_stats_no_fills(self):
        """Test getting stats with no filled orders."""
        handler = ExecutionHandler()

        parent = Order(order_id="parent", symbol="AAPL", quantity=1000)
        child = Order(order_id="child", status=OrderStatus.PENDING)

        plan = ExecutionPlan(
            parent_order=parent,
            child_orders=[child],
            total_executed=0.0
        )

        handler.execution_plans["parent"] = plan

        stats = handler.get_execution_stats("parent")
        assert stats["executed_quantity"] == 0.0
        assert stats["average_price"] == 0.0
        assert stats["completion_time"] is None
        assert stats["duration_seconds"] is None

    def test_execute_order_legacy_max_retries(self):
        """Test legacy execution exceeding max retries."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=Exception("Persistent error"))

        handler = ExecutionHandler(executor=executor)
        handler.retry_delay = 0.01
        order = {'order_id': '123'}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            with pytest.raises(Exception, match="Persistent error"):
                loop.run_until_complete(handler._execute_order_legacy(order))
            loop.close()

        assert executor.place_order.call_count == 3

    def test_execute_with_retry_no_custom_retries(self):
        """Test execute_with_retry without custom retry count."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(handler.execute_with_retry(order))
        loop.close()

        assert result['status'] == 'FILLED'

    def test_execute_with_retry_custom_retries(self):
        """Test execute_with_retry with custom retry count."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=[
            Exception("Error 1"),
            {'status': 'FILLED'}
        ])

        handler = ExecutionHandler(executor=executor)
        handler.retry_delay = 0.01
        order = {'order_id': '123'}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(handler.execute_with_retry(order, max_retries=2))
            loop.close()

        assert result['status'] == 'FILLED'
        assert handler.max_retries == 3  # Restored to original

    def test_cancel_execution_not_found(self):
        """Test cancelling non-existent execution."""
        handler = ExecutionHandler(order_manager=Mock())

        loop = asyncio.new_event_loop()
        success = loop.run_until_complete(handler.cancel_execution("not_found"))
        loop.close()

        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
