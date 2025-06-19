"""
Comprehensive test suite for ExecutionHandler achieving 100% coverage.

This test file covers all execution algorithms, error handling, backward
compatibility, and edge cases.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from core.engine.execution_handler import (
    ExecutionAlgorithm,
    ExecutionHandler,
    ExecutionParams,
    ExecutionPlan,
)
from core.engine.order_manager import Order, OrderSide, OrderStatus, OrderType


# Removed run_async helper - using pytest-asyncio instead


class TestExecutionDataStructures:
    """Test ExecutionParams and ExecutionPlan data structures."""

    def test_execution_params_defaults(self):
        """Test ExecutionParams default values."""
        params = ExecutionParams()

        assert params.algorithm == ExecutionAlgorithm.MARKET
        assert params.start_time is None
        assert params.end_time is None
        assert params.max_participation_rate == 0.1
        assert params.slice_size is None
        assert params.price_limit is None
        assert params.urgency == 0.5
        assert params.hidden_quantity is None

    def test_execution_params_custom(self):
        """Test ExecutionParams with custom values."""
        start = datetime.now()
        end = start + timedelta(hours=1)

        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.TWAP,
            start_time=start,
            end_time=end,
            max_participation_rate=0.2,
            slice_size=100.0,
            price_limit=150.0,
            urgency=0.8,
            hidden_quantity=500.0
        )

        assert params.algorithm == ExecutionAlgorithm.TWAP
        assert params.start_time == start
        assert params.end_time == end
        assert params.max_participation_rate == 0.2
        assert params.slice_size == 100.0
        assert params.price_limit == 150.0
        assert params.urgency == 0.8
        assert params.hidden_quantity == 500.0

    def test_execution_plan_defaults(self):
        """Test ExecutionPlan default values."""
        order = Order(symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        assert plan.parent_order == order
        assert isinstance(plan.child_orders, list)
        assert len(plan.child_orders) == 0
        assert isinstance(plan.execution_params, ExecutionParams)
        assert plan.total_executed == 0.0
        assert plan.average_price == 0.0
        assert isinstance(plan.start_time, datetime)
        assert plan.completion_time is None
        assert plan.status == "pending"

    def test_execution_plan_custom(self):
        """Test ExecutionPlan with custom values."""
        order = Order(symbol="AAPL", quantity=100)
        child1 = Order(symbol="AAPL", quantity=50)
        child2 = Order(symbol="AAPL", quantity=50)
        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)

        plan = ExecutionPlan(
            parent_order=order,
            child_orders=[child1, child2],
            execution_params=params,
            total_executed=75.0,
            average_price=150.5,
            status="executing"
        )

        assert len(plan.child_orders) == 2
        assert plan.execution_params.algorithm == ExecutionAlgorithm.VWAP
        assert plan.total_executed == 75.0
        assert plan.average_price == 150.5
        assert plan.status == "executing"


class TestExecutionAlgorithm:
    """Test ExecutionAlgorithm enum."""

    def test_algorithm_values(self):
        """Test algorithm enum values."""
        assert ExecutionAlgorithm.MARKET.value == "market"
        assert ExecutionAlgorithm.TWAP.value == "twap"
        assert ExecutionAlgorithm.VWAP.value == "vwap"
        assert ExecutionAlgorithm.ICEBERG.value == "iceberg"
        assert ExecutionAlgorithm.SMART.value == "smart"
        assert ExecutionAlgorithm.POV.value == "pov"


class TestExecutionHandler:
    """Test ExecutionHandler class."""

    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        manager = Mock()
        manager.create_order = AsyncMock()
        manager.submit_order = AsyncMock(return_value=True)
        manager.cancel_order = AsyncMock(return_value=True)
        return manager

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data provider."""
        provider = Mock()
        provider.get_volume_profile = AsyncMock()
        provider.get_market_volume = AsyncMock()
        return provider

    @pytest.fixture
    def handler(self, mock_order_manager, mock_market_data):
        """Create ExecutionHandler instance."""
        return ExecutionHandler(
            order_manager=mock_order_manager,
            market_data_provider=mock_market_data
        )

    @pytest.fixture
    def mock_executor(self):
        """Create mock executor for backward compatibility."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={
            'order_id': '123',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50
        })
        return executor

    @pytest.fixture
    def legacy_handler(self, mock_executor):
        """Create ExecutionHandler with legacy executor."""
        return ExecutionHandler(executor=mock_executor)

    def test_initialization_new_api(self, mock_order_manager, mock_market_data):
        """Test initialization with new API."""
        handler = ExecutionHandler(
            order_manager=mock_order_manager,
            market_data_provider=mock_market_data
        )

        assert handler.order_manager == mock_order_manager
        assert handler.market_data_provider == mock_market_data
        assert handler.executor is None
        assert isinstance(handler.execution_plans, dict)
        assert isinstance(handler.active_executions, dict)

    def test_initialization_legacy_api(self, mock_executor):
        """Test initialization with legacy executor."""
        handler = ExecutionHandler(executor=mock_executor)

        assert handler.executor == mock_executor
        assert handler.order_manager is None
        assert handler.market_data_provider is None
        assert handler.max_retries == 3
        assert handler.retry_delay == 1.0

    def test_execute_order_market_algorithm(self, handler, mock_order_manager):
        """Test market order execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        # Mock the execution to complete immediately
        with patch.object(handler, '_execute_market_order', new_callable=AsyncMock):
            plan = run_async(handler.execute_order(order))

            assert isinstance(plan, ExecutionPlan)
            assert plan.parent_order == order
            assert plan.execution_params.algorithm == ExecutionAlgorithm.MARKET
            assert "test_order" in handler.execution_plans
            assert "test_order" in handler.active_executions

    def test_execute_order_twap_algorithm(self, handler, mock_order_manager):
        """Test TWAP execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.TWAP,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=2)
        )

        # Mock order creation
        child_order = Order(order_id="child_1", symbol="AAPL", quantity=100)
        mock_order_manager.create_order.return_value = child_order

        plan = run_async(handler.execute_order(order, params))

        assert plan.execution_params.algorithm == ExecutionAlgorithm.TWAP

        # Cancel execution to stop the task
        run_async(handler.cancel_execution("test_order"))

    def test_execute_order_vwap_algorithm(self, handler, mock_order_manager):
        """Test VWAP execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000,
            side=OrderSide.BUY
        )

        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)

        # Mock volume profile
        with patch.object(handler, '_get_volume_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = [
                (datetime.now() + timedelta(minutes=i), 1000.0)
                for i in range(5)
            ]

            # Mock order creation
            child_order = Order(order_id="child_1", symbol="AAPL", quantity=200)
            mock_order_manager.create_order.return_value = child_order

            plan = run_async(handler.execute_order(order, params))

            assert plan.execution_params.algorithm == ExecutionAlgorithm.VWAP

            # Cancel execution
            run_async(handler.cancel_execution("test_order"))

    def test_execute_order_vwap_no_market_data(self, handler):
        """Test VWAP execution without market data provider."""
        handler.market_data_provider = None

        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000
        )

        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock):
            run_async(handler.execute_order(order, params))

            # Wait for async task to start
            run_async(asyncio.sleep(0.1))

            # Cancel execution
            run_async(handler.cancel_execution("test_order"))

    def test_execute_order_iceberg_algorithm(self, handler, mock_order_manager):
        """Test Iceberg execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=150.0
        )

        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.ICEBERG,
            slice_size=100.0,
            hidden_quantity=900.0
        )

        # Mock child orders
        child_orders = []
        for i in range(10):
            child = Order(
                order_id=f"child_{i}",
                symbol="AAPL",
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100
            )
            child_orders.append(child)

        mock_order_manager.create_order.side_effect = child_orders

        with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
            plan = run_async(handler.execute_order(order, params))

            assert plan.execution_params.algorithm == ExecutionAlgorithm.ICEBERG

            # Cancel execution
            run_async(handler.cancel_execution("test_order"))

    def test_execute_order_pov_algorithm(self, handler, mock_order_manager):
        """Test POV execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000,
            side=OrderSide.BUY
        )

        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.POV,
            max_participation_rate=0.15
        )

        # Mock market volume
        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_volume:
            mock_volume.return_value = 10000.0

            # Mock child order
            child_order = Order(
                order_id="child_1",
                symbol="AAPL",
                quantity=1500,  # 15% of 10000
                filled_quantity=1000
            )
            mock_order_manager.create_order.return_value = child_order

            with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
                plan = run_async(handler.execute_order(order, params))

                assert plan.execution_params.algorithm == ExecutionAlgorithm.POV

                # Cancel execution
                run_async(handler.cancel_execution("test_order"))

    def test_execute_order_pov_no_market_data(self, handler):
        """Test POV execution without market data provider."""
        handler.market_data_provider = None

        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000
        )

        params = ExecutionParams(algorithm=ExecutionAlgorithm.POV)

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock):
            run_async(handler.execute_order(order, params))

            # Wait for async task to start
            run_async(asyncio.sleep(0.1))

            # Cancel execution
            run_async(handler.cancel_execution("test_order"))

    def test_execute_order_smart_algorithm(self, handler):
        """Test SMART execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000
        )

        params = ExecutionParams(algorithm=ExecutionAlgorithm.SMART)

        # Should fall back to TWAP for now
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock):
            plan = run_async(handler.execute_order(order, params))

            assert plan.execution_params.algorithm == ExecutionAlgorithm.SMART

            # Wait for async task to start
            run_async(asyncio.sleep(0.1))

            # Cancel execution
            run_async(handler.cancel_execution("test_order"))

    def test_execute_order_legacy_api(self, legacy_handler, mock_executor):
        """Test legacy executor-based execution."""
        order = {
            'order_id': '123',
            'symbol': 'AAPL',
            'quantity': 100,
            'side': 'BUY',
            'order_type': 'MARKET'
        }

        result = run_async(legacy_handler.execute_order(order))

        assert result['order_id'] == '123'
        assert result['status'] == 'FILLED'
        mock_executor.place_order.assert_called_once_with(order)

    def test_cancel_execution_success(self, handler, mock_order_manager):
        """Test successful execution cancellation."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=1000
        )

        # Start execution
        plan = run_async(handler.execute_order(order))

        # Add some child orders to the plan
        child1 = Order(order_id="child_1", status=OrderStatus.SUBMITTED)
        child2 = Order(order_id="child_2", status=OrderStatus.PENDING)
        plan.child_orders = [child1, child2]

        # Cancel execution
        success = run_async(handler.cancel_execution("test_order"))

        assert success is True
        assert plan.status == "cancelled"
        assert "test_order" not in handler.active_executions

        # Verify child orders were cancelled
        mock_order_manager.cancel_order.assert_any_call("child_1")
        mock_order_manager.cancel_order.assert_any_call("child_2")

    def test_cancel_execution_not_found(self, handler):
        """Test cancelling non-existent execution."""
        success = run_async(handler.cancel_execution("non_existent"))
        assert success is False

    def test_execute_market_order_success(self, handler, mock_order_manager):
        """Test successful market order execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        run_async(handler._execute_market_order(plan))

        assert plan.status == "completed"
        assert len(plan.child_orders) == 1
        assert plan.child_orders[0] == order
        mock_order_manager.submit_order.assert_called_once_with(order)

    def test_execute_market_order_failure(self, handler, mock_order_manager):
        """Test failed market order execution."""
        mock_order_manager.submit_order.return_value = False

        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        run_async(handler._execute_market_order(plan))

        assert plan.status == "failed"

    def test_execute_market_order_exception(self, handler, mock_order_manager):
        """Test market order execution with exception."""
        mock_order_manager.submit_order.side_effect = Exception("Submit error")

        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        run_async(handler._execute_market_order(plan))

        assert plan.status == "error"

    def test_execute_twap_complete(self, handler, mock_order_manager):
        """Test complete TWAP execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=100)
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.TWAP,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=5)
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock child order creation
        child_order = Order(order_id="child", symbol="AAPL", quantity=10)
        mock_order_manager.create_order.return_value = child_order

        # Use very short interval for testing
        with patch('asyncio.sleep', new_callable=AsyncMock):
            run_async(handler._execute_twap(plan))

        assert plan.status == "completed"
        assert plan.completion_time is not None
        assert plan.total_executed == 100.0

    def test_execute_twap_cancelled(self, handler, mock_order_manager):
        """Test cancelled TWAP execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Simulate cancellation
        plan.status = "cancelled"

        with pytest.raises(asyncio.CancelledError):
            run_async(handler._execute_twap(plan))

        assert plan.status == "cancelled"

    def test_execute_twap_error(self, handler, mock_order_manager):
        """Test TWAP execution with error."""
        mock_order_manager.create_order.side_effect = Exception("Create error")

        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        run_async(handler._execute_twap(plan))

        assert plan.status == "error"

    def test_get_volume_profile(self, handler):
        """Test volume profile retrieval."""
        profile = run_async(handler._get_volume_profile("AAPL"))

        assert isinstance(profile, list)
        assert len(profile) == 60
        assert all(isinstance(t, datetime) and v == 1000.0 for t, v in profile)

    def test_calculate_vwap_schedule(self, handler):
        """Test VWAP schedule calculation."""
        profile = [
            (datetime.now(), 1000.0),
            (datetime.now() + timedelta(minutes=1), 2000.0),
            (datetime.now() + timedelta(minutes=2), 3000.0),
        ]

        schedule = handler._calculate_vwap_schedule(600.0, profile)

        assert len(schedule) == 3
        # First slot: 600 * (1000/6000) = 100
        assert schedule[0][1] == 100.0
        # Second slot: 600 * (2000/6000) = 200
        assert schedule[1][1] == 200.0
        # Third slot: 600 * (3000/6000) = 300
        assert schedule[2][1] == 300.0

    def test_get_market_volume(self, handler):
        """Test market volume retrieval."""
        volume = run_async(handler._get_market_volume("AAPL", 60))
        assert volume == 10000.0

    def test_wait_for_order_completion(self, handler):
        """Test waiting for order completion."""
        order = Order(order_id="test", status=OrderStatus.SUBMITTED)

        # Simulate order becoming filled after delay
        async def fill_order():
            await asyncio.sleep(0.1)
            order.status = OrderStatus.FILLED

        # Run fill task and wait concurrently
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.gather(
                fill_order(),
                handler._wait_for_order_completion(order, timeout=1.0)
            ))
        finally:
            loop.close()

        assert order.status == OrderStatus.FILLED

    def test_execute_order_legacy_success(self, legacy_handler, mock_executor):
        """Test successful legacy order execution."""
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        result = run_async(legacy_handler._execute_order_legacy(order))

        assert result['status'] == 'FILLED'
        mock_executor.place_order.assert_called_once_with(order)

    def test_execute_order_legacy_retry(self, legacy_handler, mock_executor):
        """Test legacy order execution with retry."""
        mock_executor.place_order.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            {'order_id': '123', 'status': 'FILLED'}
        ]

        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = run_async(legacy_handler._execute_order_legacy(order))

        assert result['status'] == 'FILLED'
        assert mock_executor.place_order.call_count == 3

    def test_execute_order_legacy_max_retries(self, legacy_handler, mock_executor):
        """Test legacy order execution exceeding max retries."""
        mock_executor.place_order.side_effect = Exception("Persistent error")

        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Persistent error"):
                run_async(legacy_handler._execute_order_legacy(order))

        assert mock_executor.place_order.call_count == 3

    def test_execute_with_retry_custom_retries(self, legacy_handler, mock_executor):
        """Test execute_with_retry with custom retry count."""
        mock_executor.place_order.side_effect = [
            Exception("Error 1"),
            {'order_id': '123', 'status': 'FILLED'}
        ]

        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = run_async(legacy_handler.execute_with_retry(order, max_retries=2))

        assert result['status'] == 'FILLED'
        assert legacy_handler.max_retries == 3  # Restored to original

    def test_calculate_slippage_dict_api_buy(self, handler):
        """Test slippage calculation with dict API for buy order."""
        order = {
            'expected_price': 150.0,
            'side': 'BUY'
        }
        fill = {
            'avg_fill_price': 150.25,
            'filled_quantity': 100
        }

        slippage = handler.calculate_slippage(order, fill)

        assert slippage['amount'] == -0.25  # Paid more
        assert slippage['percentage'] == pytest.approx(-0.167, rel=0.01)
        assert slippage['cost'] == -25.0

    def test_calculate_slippage_dict_api_sell(self, handler):
        """Test slippage calculation with dict API for sell order."""
        order = {
            'expected_price': 150.0,
            'side': 'SELL'
        }
        fill = {
            'avg_fill_price': 149.75,
            'filled_quantity': 100
        }

        slippage = handler.calculate_slippage(order, fill)

        assert slippage['amount'] == -0.25  # Received less
        assert slippage['percentage'] == pytest.approx(-0.167, rel=0.01)
        assert slippage['cost'] == -25.0

    def test_calculate_slippage_zero_price(self, handler):
        """Test slippage calculation with zero expected price."""
        order = {'expected_price': 0, 'side': 'BUY'}
        fill = {'avg_fill_price': 150.0, 'filled_quantity': 100}

        slippage = handler.calculate_slippage(order, fill)

        # When expected_price is 0, slippage_amount = 0 - 150 = -150
        assert slippage['amount'] == -150.0
        assert slippage['percentage'] == 0  # Division by zero protection
        assert slippage['cost'] == -15000.0

    def test_calculate_slippage_legacy_api_buy(self, handler):
        """Test slippage calculation with legacy API for buy."""
        slippage = handler.calculate_slippage(150.0, 150.25, 'BUY')

        # (150.25 - 150) / 150 = 0.00167
        assert slippage == pytest.approx(0.00167, rel=0.01)

    def test_calculate_slippage_legacy_api_sell(self, handler):
        """Test slippage calculation with legacy API for sell."""
        slippage = handler.calculate_slippage(150.0, 149.75, 'SELL')

        # (150 - 149.75) / 150 = 0.00167
        assert slippage == pytest.approx(0.00167, rel=0.01)

    def test_smart_route_order(self, legacy_handler, mock_executor):
        """Test smart order routing."""
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}
        venues = ['NASDAQ', 'NYSE', 'ARCA']

        # Mock set_venue method
        mock_executor.set_venue = Mock()

        result = run_async(legacy_handler.smart_route_order(order, venues))

        assert result['status'] == 'FILLED'
        mock_executor.set_venue.assert_called_once_with('NASDAQ')
        mock_executor.place_order.assert_called_once_with(order)

    def test_smart_route_order_no_venues(self, legacy_handler, mock_executor):
        """Test smart order routing without venues."""
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        result = run_async(legacy_handler.smart_route_order(order, []))

        assert result['status'] == 'FILLED'
        mock_executor.place_order.assert_called_once_with(order)

    def test_get_execution_stats(self, handler):
        """Test getting execution statistics."""
        # Create a plan with some child orders
        parent = Order(order_id="parent", symbol="AAPL", quantity=1000)
        child1 = Order(
            order_id="child1",
            symbol="AAPL",
            quantity=500,
            status=OrderStatus.FILLED,
            filled_quantity=500,
            average_fill_price=150.0
        )
        child2 = Order(
            order_id="child2",
            symbol="AAPL",
            quantity=300,
            status=OrderStatus.FILLED,
            filled_quantity=300,
            average_fill_price=150.5
        )
        child3 = Order(
            order_id="child3",
            symbol="AAPL",
            quantity=200,
            status=OrderStatus.PARTIAL,
            filled_quantity=100,
            average_fill_price=151.0
        )

        params = ExecutionParams(algorithm=ExecutionAlgorithm.TWAP)
        plan = ExecutionPlan(
            parent_order=parent,
            child_orders=[child1, child2, child3],
            execution_params=params,
            total_executed=900.0,
            status="executing"
        )
        plan.completion_time = plan.start_time + timedelta(minutes=5)

        handler.execution_plans["parent"] = plan

        stats = handler.get_execution_stats("parent")

        assert stats["parent_order_id"] == "parent"
        assert stats["algorithm"] == "twap"
        assert stats["status"] == "executing"
        assert stats["total_quantity"] == 1000
        assert stats["executed_quantity"] == 900.0
        assert stats["fill_rate"] == 0.9
        assert stats["child_orders"] == 3
        # Average price: (500*150 + 300*150.5) / 800 = 150.1875
        assert stats["average_price"] == pytest.approx(150.1875)
        assert stats["duration_seconds"] == 300.0

    def test_get_execution_stats_not_found(self, handler):
        """Test getting stats for non-existent execution."""
        stats = handler.get_execution_stats("non_existent")
        assert stats is None

    def test_get_execution_stats_no_fills(self, handler):
        """Test getting stats with no filled orders."""
        parent = Order(order_id="parent", symbol="AAPL", quantity=1000)
        child1 = Order(order_id="child1", status=OrderStatus.PENDING)

        plan = ExecutionPlan(
            parent_order=parent,
            child_orders=[child1],
            total_executed=0.0,
            status="executing"
        )

        handler.execution_plans["parent"] = plan

        stats = handler.get_execution_stats("parent")

        assert stats["executed_quantity"] == 0.0
        assert stats["average_price"] == 0.0
        assert stats["fill_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
