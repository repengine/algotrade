"""
Tests for executor framework.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from adapters.ibkr_executor import IBKRExecutor
from adapters.paper_executor import PaperExecutor
from core.executor import (
    ExecutionCallback,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)


class TestCallback(ExecutionCallback):
    """Test callback implementation."""

    def __init__(self):
        self.order_updates = []
        self.fills = []
        self.errors = []

    def on_order_status(self, order: Order) -> None:
        self.order_updates.append(order)

    def on_fill(self, fill: Fill) -> None:
        self.fills.append(fill)

    def on_error(self, error: Exception, order: Optional[Order] = None) -> None:
        self.errors.append((error, order))


@pytest.fixture
def paper_executor():
    """Create paper executor fixture."""
    config = {
        "initial_capital": 100000,
        "commission": 1.0,
        "slippage": 0.0001,
        "fill_delay": 0.01,  # Fast fills for testing
    }
    return PaperExecutor(config)


@pytest.fixture
def test_callback():
    """Create test callback fixture."""
    return TestCallback()


class TestPaperExecutor:
    """Test paper trading executor."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, paper_executor):
        """Test connection lifecycle."""
        # Test connection
        assert await paper_executor.connect() is True
        assert paper_executor.is_connected is True

        # Test disconnection
        await paper_executor.disconnect()
        assert paper_executor.is_connected is False

    @pytest.mark.asyncio
    async def test_submit_market_order(self, paper_executor, test_callback):
        """Test market order submission."""
        await paper_executor.connect()
        try:
            paper_executor.register_callback(test_callback)

            # Set price data
            paper_executor.update_price("AAPL", 150.0)

            # Create and submit order
            order = Order(
                order_id="TEST-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )

            order_id = await paper_executor.submit_order(order)
            assert order_id == "TEST-001"

            # Wait for fill
            await asyncio.sleep(0.1)

            # Check callbacks
            assert len(test_callback.order_updates) >= 2  # Submitted + Filled
            assert test_callback.order_updates[-1].status == OrderStatus.FILLED
            assert len(test_callback.fills) == 1

            # Check fill details
            fill = test_callback.fills[0]
            assert fill.symbol == "AAPL"
            assert fill.quantity == 100
            assert fill.price == pytest.approx(150.0 * 1.0001)  # With slippage
        finally:
            await paper_executor.disconnect()

    @pytest.mark.asyncio
    async def test_submit_limit_order(self, paper_executor, test_callback):
        """Test limit order submission."""
        await paper_executor.connect()
        try:
            paper_executor.register_callback(test_callback)

            # Set price data
            paper_executor.update_price("AAPL", 150.0)

            # Create limit order above market
            order = Order(
                order_id="TEST-002",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=151.0,  # Above market, should fill immediately
            )

            await paper_executor.submit_order(order)

            # Wait for fill
            await asyncio.sleep(0.2)  # Increase wait time to ensure fill

            # Check fill at limit price or better
            assert len(test_callback.fills) == 1
            fill = test_callback.fills[0]
            assert fill.price <= 151.0  # Should fill at market price (150) or better
        finally:
            await paper_executor.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_order(self, paper_executor, test_callback):
        """Test order cancellation."""
        await paper_executor.connect()
        try:
            paper_executor.register_callback(test_callback)

            # Set price data
            paper_executor.update_price("AAPL", 150.0)

            # Create limit order that won't fill immediately
            order = Order(
                order_id="TEST-003",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=140.0,  # Far below market
            )

            await paper_executor.submit_order(order)

            # Cancel before fill
            success = await paper_executor.cancel_order("TEST-003")
            assert success is True

            # Check status
            assert len(test_callback.order_updates) >= 2
            assert test_callback.order_updates[-1].status == OrderStatus.CANCELLED
        finally:
            await paper_executor.disconnect()

    @pytest.mark.asyncio
    async def test_insufficient_buying_power(self, paper_executor, test_callback):
        """Test order rejection due to insufficient buying power."""
        await paper_executor.connect()
        try:
            paper_executor.register_callback(test_callback)

            # Set price data
            paper_executor.update_price("AAPL", 150.0)

            # Try to buy more than we can afford
            order = Order(
                order_id="TEST-004",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10000,  # Would cost $1.5M
                order_type=OrderType.MARKET,
            )

            with pytest.raises(ValueError, match="Insufficient buying power"):
                await paper_executor.submit_order(order)
        finally:
            await paper_executor.disconnect()

    @pytest.mark.asyncio
    async def test_position_tracking(self, paper_executor):
        """Test position tracking after fills."""
        await paper_executor.connect()
        try:
            # Set price data
            paper_executor.update_price("AAPL", 150.0)

            # Buy 100 shares
            buy_order = Order(
                order_id="TEST-005",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
            )

            await paper_executor.submit_order(buy_order)
            await asyncio.sleep(0.1)

            # Check position
            positions = await paper_executor.get_positions()
            assert "AAPL" in positions
            assert positions["AAPL"].quantity == 100
            assert positions["AAPL"].average_cost == pytest.approx(150.0 * 1.0001)

            # Sell 50 shares
            sell_order = Order(
                order_id="TEST-006",
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=50,
                order_type=OrderType.MARKET,
            )

            await paper_executor.submit_order(sell_order)
            await asyncio.sleep(0.1)

            # Check updated position
            positions = await paper_executor.get_positions()
            assert positions["AAPL"].quantity == 50
        finally:
            await paper_executor.disconnect()

    @pytest.mark.asyncio
    async def test_account_info(self, paper_executor):
        """Test account information."""
        await paper_executor.connect()

        account_info = await paper_executor.get_account_info()

        assert account_info["initial_capital"] == 100000
        assert account_info["cash"] == 100000
        assert account_info["total_value"] == 100000
        assert account_info["is_paper"] is True

        await paper_executor.disconnect()


class TestIBKRExecutor:
    """Test IBKR executor."""

    @pytest.fixture
    def mock_ibkr_adapter(self):
        """Create mock IBKR adapter."""
        adapter = AsyncMock()
        adapter.connect = AsyncMock(return_value=True)
        adapter.disconnect = AsyncMock()
        adapter.get_accounts = AsyncMock(return_value=["DU123456"])
        adapter.selected_account = "DU123456"
        adapter.search_contracts = AsyncMock(
            return_value=[MagicMock(conid=12345, symbol="AAPL")]
        )
        adapter.place_order = AsyncMock(return_value={"order_id": "IBKR-001"})
        adapter.cancel_order = AsyncMock(return_value=True)
        adapter.get_positions = AsyncMock(return_value=[])
        adapter.get_account_info = AsyncMock(
            return_value=MagicMock(
                account_id="DU123456",
                net_liquidation=100000,
                buying_power=100000,
            )
        )
        adapter.subscribe_orders = AsyncMock(return_value=True)
        adapter.keep_alive = AsyncMock(return_value=True)
        return adapter

    @pytest.mark.asyncio
    async def test_connect_with_mock(self, mock_ibkr_adapter):
        """Test IBKR executor connection."""
        with patch(
            "adapters.ibkr_executor.IBKRAdapter",
            return_value=mock_ibkr_adapter,
        ):
            executor = IBKRExecutor({"account": "DU123456"})

            assert await executor.connect() is True
            assert executor.is_connected is True

            mock_ibkr_adapter.connect.assert_called_once()
            mock_ibkr_adapter.get_accounts.assert_called_once()

            await executor.disconnect()

    @pytest.mark.asyncio
    async def test_submit_order_with_mock(self, mock_ibkr_adapter, test_callback):
        """Test order submission with mock."""
        with patch(
            "adapters.ibkr_executor.IBKRAdapter",
            return_value=mock_ibkr_adapter,
        ):
            executor = IBKRExecutor({"account": "DU123456"})
            executor.register_callback(test_callback)

            await executor.connect()
            try:
                # Submit order
                order = Order(
                    order_id="TEST-007",
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    quantity=100,
                    order_type=OrderType.MARKET,
                )

                order_id = await executor.submit_order(order)
                assert order_id == "TEST-007"

                # Check adapter was called
                mock_ibkr_adapter.place_order.assert_called_once()

                # Check order status updated
                assert order.status == OrderStatus.SUBMITTED
                assert len(test_callback.order_updates) == 1
            finally:
                await executor.disconnect()


class TestOrderValidation:
    """Test order validation."""

    def test_validate_market_order(self, paper_executor):
        """Test market order validation."""
        order = Order(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert paper_executor.validate_order(order) is True

    def test_validate_limit_order_missing_price(self, paper_executor):
        """Test limit order without price."""
        order = Order(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=None,
        )

        with pytest.raises(ValueError, match="Limit order requires limit price"):
            paper_executor.validate_order(order)

    def test_validate_stop_order_missing_price(self, paper_executor):
        """Test stop order without price."""
        order = Order(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=None,
        )

        with pytest.raises(ValueError, match="Stop order requires stop price"):
            paper_executor.validate_order(order)

    def test_validate_invalid_quantity(self, paper_executor):
        """Test order with invalid quantity."""
        order = Order(
            order_id="TEST",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.MARKET,
        )

        with pytest.raises(ValueError, match="Invalid quantity"):
            paper_executor.validate_order(order)
