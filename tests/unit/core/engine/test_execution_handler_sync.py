"""
Synchronous test suite for ExecutionHandler achieving 100% coverage.

This test file properly mocks all async operations to avoid hanging.
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


def run_async(coro):
    """Helper to run async code in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


class TestExecutionHandlerSync:
    """Test ExecutionHandler with synchronous test methods."""

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

    def test_initialization_new_api(self):
        """Test initialization with new API."""
        manager = Mock()
        provider = Mock()
        handler = ExecutionHandler(order_manager=manager, market_data_provider=provider)

        assert handler.order_manager == manager
        assert handler.market_data_provider == provider
        assert handler.executor is None
        assert isinstance(handler.execution_plans, dict)
        assert isinstance(handler.active_executions, dict)

    def test_initialization_legacy_api(self):
        """Test initialization with legacy executor."""
        executor = Mock()
        handler = ExecutionHandler(executor=executor)

        assert handler.executor == executor
        assert handler.order_manager is None
        assert handler.market_data_provider is None
        assert handler.max_retries == 3
        assert handler.retry_delay == 1.0

    def test_execute_order_market_algorithm(self, handler):
        """Test market order execution."""
        order = Order(
            order_id="test_order",
            symbol="AAPL",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        # Mock the async execution task
        async def mock_task():
            pass

        with patch('asyncio.create_task') as mock_create_task:
            mock_create_task.return_value = Mock()

            plan = run_async(handler.execute_order(order))

            assert isinstance(plan, ExecutionPlan)
            assert plan.parent_order == order
            assert plan.execution_params.algorithm == ExecutionAlgorithm.MARKET
            assert "test_order" in handler.execution_plans
            assert "test_order" in handler.active_executions

    def test_execute_order_all_algorithms(self, handler):
        """Test all execution algorithms."""
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

            with patch('asyncio.create_task') as mock_create_task:
                mock_create_task.return_value = Mock()

                plan = run_async(handler.execute_order(order, params))

                assert plan.execution_params.algorithm == algo
                assert mock_create_task.called

    def test_execute_order_legacy_api(self):
        """Test legacy executor-based execution."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={
            'order_id': '123',
            'status': 'FILLED',
            'filled_quantity': 100,
            'avg_fill_price': 150.50
        })

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        result = run_async(handler.execute_order(order))

        assert result['order_id'] == '123'
        assert result['status'] == 'FILLED'
        executor.place_order.assert_called_once_with(order)

    def test_cancel_execution_success(self, handler, mock_order_manager):
        """Test successful execution cancellation."""
        # Setup execution plan
        order = Order(order_id="test_order", symbol="AAPL", quantity=1000)
        child1 = Order(order_id="child_1", status=OrderStatus.SUBMITTED)
        child2 = Order(order_id="child_2", status=OrderStatus.PENDING)

        plan = ExecutionPlan(parent_order=order, child_orders=[child1, child2])
        handler.execution_plans["test_order"] = plan

        # Mock active task
        mock_task = Mock()
        handler.active_executions["test_order"] = mock_task

        # Cancel execution
        success = run_async(handler.cancel_execution("test_order"))

        assert success is True
        assert plan.status == "cancelled"
        assert "test_order" not in handler.active_executions
        mock_task.cancel.assert_called_once()

        # Verify child orders were cancelled
        assert mock_order_manager.cancel_order.call_count == 2

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

    def test_execute_twap_cancelled(self, handler):
        """Test cancelled TWAP execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        with pytest.raises(asyncio.CancelledError):
            run_async(handler._execute_twap(plan))

    def test_execute_twap_error(self, handler, mock_order_manager):
        """Test TWAP execution with error."""
        mock_order_manager.create_order.side_effect = Exception("Create error")

        order = Order(order_id="test", symbol="AAPL", quantity=100)
        plan = ExecutionPlan(parent_order=order)

        run_async(handler._execute_twap(plan))

        assert plan.status == "error"

    def test_execute_vwap_complete(self, handler, mock_order_manager):
        """Test complete VWAP execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(algorithm=ExecutionAlgorithm.VWAP)
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock volume profile
        with patch.object(handler, '_get_volume_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = [
                (datetime.now() + timedelta(seconds=i), 1000.0)
                for i in range(3)
            ]

            # Mock order creation
            child_order = Order(order_id="child", symbol="AAPL", quantity=333.33)
            mock_order_manager.create_order.return_value = child_order

            with patch('asyncio.sleep', new_callable=AsyncMock):
                run_async(handler._execute_vwap(plan))

            assert plan.status == "completed"
            assert plan.completion_time is not None

    def test_execute_vwap_no_market_data(self, handler):
        """Test VWAP execution without market data provider."""
        handler.market_data_provider = None

        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order, execution_params=ExecutionParams(algorithm=ExecutionAlgorithm.VWAP))

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            run_async(handler._execute_vwap(plan))
            mock_twap.assert_called_once_with(plan)

    def test_execute_vwap_cancelled(self, handler):
        """Test cancelled VWAP execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        with pytest.raises(asyncio.CancelledError):
            run_async(handler._execute_vwap(plan))

    def test_execute_vwap_error(self, handler):
        """Test VWAP execution with error."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        with patch.object(handler, '_get_volume_profile', side_effect=Exception("Profile error")):
            run_async(handler._execute_vwap(plan))

        assert plan.status == "error"

    def test_execute_iceberg_complete(self, handler, mock_order_manager):
        """Test complete Iceberg execution."""
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

        # Mock child orders that get filled
        filled_orders = []
        for i in range(10):
            child = Order(
                order_id=f"child_{i}",
                symbol="AAPL",
                quantity=100,
                status=OrderStatus.FILLED,
                filled_quantity=100
            )
            filled_orders.append(child)

        mock_order_manager.create_order.side_effect = filled_orders

        with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
            run_async(handler._execute_iceberg(plan))

        assert plan.status == "completed"
        assert plan.total_executed == 1000.0
        assert len(plan.child_orders) == 10

    def test_execute_iceberg_partial(self, handler, mock_order_manager):
        """Test partial Iceberg execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Mock child order that doesn't fill
        child = Order(
            order_id="child_1",
            symbol="AAPL",
            quantity=100,
            status=OrderStatus.CANCELLED,
            filled_quantity=0
        )
        mock_order_manager.create_order.return_value = child

        with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
            run_async(handler._execute_iceberg(plan))

        assert plan.status == "partial"
        assert plan.total_executed == 0.0

    def test_execute_iceberg_cancelled(self, handler):
        """Test cancelled Iceberg execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        with pytest.raises(asyncio.CancelledError):
            run_async(handler._execute_iceberg(plan))

    def test_execute_iceberg_error(self, handler, mock_order_manager):
        """Test Iceberg execution with error."""
        mock_order_manager.create_order.side_effect = Exception("Create error")

        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        run_async(handler._execute_iceberg(plan))

        assert plan.status == "error"

    def test_execute_pov_complete(self, handler, mock_order_manager):
        """Test complete POV execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.POV,
            max_participation_rate=0.1
        )
        plan = ExecutionPlan(parent_order=order, execution_params=params)

        # Mock market volume
        with patch.object(handler, '_get_market_volume', new_callable=AsyncMock) as mock_volume:
            mock_volume.return_value = 10000.0  # Will create 1000 share order (10%)

            # Mock child order that fills completely
            child = Order(
                order_id="child_1",
                symbol="AAPL",
                quantity=1000,
                filled_quantity=1000
            )
            mock_order_manager.create_order.return_value = child

            with patch.object(handler, '_wait_for_order_completion', new_callable=AsyncMock):
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    run_async(handler._execute_pov(plan))

            assert plan.status == "completed"
            assert plan.total_executed == 1000.0

    def test_execute_pov_no_market_data(self, handler):
        """Test POV execution without market data provider."""
        handler.market_data_provider = None

        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order, execution_params=ExecutionParams(algorithm=ExecutionAlgorithm.POV))

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            run_async(handler._execute_pov(plan))
            mock_twap.assert_called_once_with(plan)

    def test_execute_pov_cancelled(self, handler):
        """Test cancelled POV execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)
        plan.status = "cancelled"

        with pytest.raises(asyncio.CancelledError):
            run_async(handler._execute_pov(plan))

    def test_execute_pov_error(self, handler):
        """Test POV execution with error."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        with patch.object(handler, '_get_market_volume', side_effect=Exception("Volume error")):
            run_async(handler._execute_pov(plan))

        assert plan.status == "error"

    def test_execute_smart_order(self, handler):
        """Test smart order execution."""
        order = Order(order_id="test", symbol="AAPL", quantity=1000)
        plan = ExecutionPlan(parent_order=order)

        # Should fall back to TWAP
        with patch.object(handler, '_execute_twap', new_callable=AsyncMock) as mock_twap:
            run_async(handler._execute_smart_order(plan))
            mock_twap.assert_called_once_with(plan)

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
        assert schedule[0][1] == 100.0  # 600 * (1000/6000)
        assert schedule[1][1] == 200.0  # 600 * (2000/6000)
        assert schedule[2][1] == 300.0  # 600 * (3000/6000)

    def test_get_market_volume(self, handler):
        """Test market volume retrieval."""
        volume = run_async(handler._get_market_volume("AAPL", 60))
        assert volume == 10000.0

    def test_wait_for_order_completion(self, handler):
        """Test waiting for order completion."""
        order = Order(order_id="test", status=OrderStatus.SUBMITTED)

        # Mock order becoming filled
        async def mock_wait():
            order.status = OrderStatus.FILLED

        with patch('asyncio.sleep', new_callable=AsyncMock):
            run_async(handler._wait_for_order_completion(order, timeout=60))

        # Order should remain submitted since we're just mocking sleep
        assert order.status == OrderStatus.SUBMITTED

    def test_execute_order_legacy_success(self):
        """Test successful legacy order execution."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        result = run_async(handler._execute_order_legacy(order))

        assert result['status'] == 'FILLED'
        executor.place_order.assert_called_once_with(order)

    def test_execute_order_legacy_retry(self):
        """Test legacy order execution with retry."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=[
            Exception("Connection error"),
            Exception("Timeout"),
            {'order_id': '123', 'status': 'FILLED'}
        ])

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = run_async(handler._execute_order_legacy(order))

        assert result['status'] == 'FILLED'
        assert executor.place_order.call_count == 3

    def test_execute_order_legacy_max_retries(self):
        """Test legacy order execution exceeding max retries."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=Exception("Persistent error"))

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Persistent error"):
                run_async(handler._execute_order_legacy(order))

        assert executor.place_order.call_count == 3

    def test_execute_with_retry_custom_retries(self):
        """Test execute_with_retry with custom retry count."""
        executor = Mock()
        executor.place_order = AsyncMock(side_effect=[
            Exception("Error 1"),
            {'order_id': '123', 'status': 'FILLED'}
        ])

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = run_async(handler.execute_with_retry(order, max_retries=2))

        assert result['status'] == 'FILLED'
        assert handler.max_retries == 3  # Restored to original

    def test_execute_with_retry_no_custom_retries(self):
        """Test execute_with_retry without custom retry count."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}

        result = run_async(handler.execute_with_retry(order))

        assert result['status'] == 'FILLED'

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

    def test_calculate_slippage_dict_api_zero_price(self, handler):
        """Test slippage calculation with zero expected price."""
        order = {'expected_price': 0, 'side': 'BUY'}
        fill = {'avg_fill_price': 150.0, 'filled_quantity': 100}

        slippage = handler.calculate_slippage(order, fill)

        assert slippage['amount'] == 0
        assert slippage['percentage'] == 0
        assert slippage['cost'] == 0

    def test_calculate_slippage_dict_api_price_fallback(self, handler):
        """Test slippage calculation with price fallback."""
        order = {'price': 150.0, 'side': 'BUY'}  # No expected_price
        fill = {'avg_fill_price': 150.25, 'filled_quantity': 100}

        slippage = handler.calculate_slippage(order, fill)

        assert slippage['amount'] == -0.25

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

    def test_calculate_slippage_legacy_api_no_fill(self, handler):
        """Test slippage calculation with no fill price."""
        slippage = handler.calculate_slippage(150.0)

        # Should use 0 as actual price
        assert slippage == -1.0  # (0 - 150) / 150

    def test_smart_route_order_with_venues(self):
        """Test smart order routing with venues."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})
        executor.set_venue = Mock()

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}
        venues = ['NASDAQ', 'NYSE', 'ARCA']

        result = run_async(handler.smart_route_order(order, venues))

        assert result['status'] == 'FILLED'
        executor.set_venue.assert_called_once_with('NASDAQ')

    def test_smart_route_order_no_venues(self):
        """Test smart order routing without venues."""
        executor = Mock()
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}

        result = run_async(handler.smart_route_order(order, []))

        assert result['status'] == 'FILLED'
        executor.place_order.assert_called_once_with(order)

    def test_smart_route_order_no_set_venue(self):
        """Test smart order routing when executor has no set_venue."""
        executor = Mock(spec=['place_order'])
        executor.place_order = AsyncMock(return_value={'status': 'FILLED'})

        handler = ExecutionHandler(executor=executor)
        order = {'order_id': '123'}
        venues = ['NASDAQ']

        result = run_async(handler.smart_route_order(order, venues))

        assert result['status'] == 'FILLED'

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
        assert stats["completion_time"] is None
        assert stats["duration_seconds"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
