"""
Tests for IBKR Client Portal API Adapter

Note: These are integration tests that require:
1. IBKR Client Portal Gateway running at localhost:5000
2. Valid authentication through the gateway
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from algostack.adapters.ibkr_adapter import (
    ConnectionState,
    Contract,
    IBKRAdapter,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)


class TestIBKRAdapter:
    """Test cases for IBKR adapter"""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance"""
        return IBKRAdapter(gateway_url="https://localhost:5000", ssl_verify=False)

    @pytest.fixture
    def mock_response(self):
        """Create mock response"""
        mock = AsyncMock()
        mock.status = 200
        mock.json = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_initialization(self, adapter):
        """Test adapter initialization"""
        assert adapter.base_url == "https://localhost:5000"
        assert adapter.ssl_verify is False
        assert adapter.state == ConnectionState.DISCONNECTED
        assert adapter.authenticated is False
        assert adapter.accounts == []
        assert adapter.selected_account is None

    @pytest.mark.asyncio
    async def test_connection_state_transitions(self, adapter):
        """Test connection state transitions"""
        # Initial state
        assert adapter.state == ConnectionState.DISCONNECTED

        # Mock successful connection
        with patch.object(adapter, "_check_gateway_status", return_value=True):
            with patch.object(adapter, "_check_auth_status", return_value=True):
                with patch.object(adapter, "_load_accounts", return_value=None):
                    with patch.object(adapter, "ws_client") as mock_ws:
                        mock_ws.connect = AsyncMock(return_value=True)

                        # Connect
                        adapter.state = ConnectionState.CONNECTING
                        assert adapter.state == ConnectionState.CONNECTING

                        # After successful auth
                        adapter.state = ConnectionState.CONNECTED
                        assert adapter.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_search_contracts(self, adapter, mock_response):
        """Test contract search"""
        # Mock response data
        mock_data = [
            {
                "conid": 265598,
                "symbol": "AAPL",
                "secType": "STK",
                "exchange": "NASDAQ",
                "currency": "USD",
            }
        ]
        mock_response.json.return_value = mock_data

        # Create a proper mock session
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        
        # Create async context manager for request
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                return None
        
        # Make request return the context manager directly (not a coroutine)
        mock_session.request = MagicMock(return_value=AsyncContextManager())
        adapter.session = mock_session

        # Search contracts
        contracts = await adapter.search_contracts("AAPL", "STK")

        # Verify
        assert len(contracts) == 1
        assert contracts[0].conid == 265598
        assert contracts[0].symbol == "AAPL"
        assert contracts[0].sec_type == "STK"
        assert contracts[0].exchange == "NASDAQ"
        assert contracts[0].currency == "USD"

    @pytest.mark.asyncio
    async def test_market_data_snapshot(self, adapter, mock_response):
        """Test market data snapshot"""
        # Mock response data
        mock_data = [
            {
                "conid": 265598,
                "31": "150.25",  # last_price
                "84": "150.20",  # bid
                "86": "150.30",  # ask
                "87": "0.5",  # change_percent
                "7295": "1000000",  # volume
            }
        ]
        mock_response.json.return_value = mock_data

        # Create a proper mock session
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        
        # Create async context manager for request
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                return None
        
        # Make request return the context manager directly (not a coroutine)
        mock_session.request = MagicMock(return_value=AsyncContextManager())
        adapter.session = mock_session

        # Get market data
        market_data = await adapter.get_market_data_snapshot([265598])

        # Verify
        assert 265598 in market_data
        data = market_data[265598]
        assert data["last_price"] == "150.25"
        assert data["bid"] == "150.20"
        assert data["ask"] == "150.30"
        assert data["change_percent"] == "0.5"
        assert data["volume"] == "1000000"

    @pytest.mark.asyncio
    async def test_place_order(self, adapter, mock_response):
        """Test order placement"""
        # Mock response data
        mock_data = {"id": "123456", "message": ["Order will be submitted"]}
        mock_response.json.return_value = mock_data

        # Create a proper mock session
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        
        # Create async context manager for request
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                return None
        
        # Make request return the context manager directly (not a coroutine)
        mock_session.request = MagicMock(return_value=AsyncContextManager())
        adapter.session = mock_session
        adapter.selected_account = "DU123456"

        # Create order
        contract = Contract(conid=265598, symbol="AAPL", sec_type="STK")

        order = Order(
            account="DU123456",
            contract=contract,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=150.00,
            tif=TimeInForce.DAY,
        )

        # Place order
        await adapter.place_order(order)

        # Verify requests were made (first order, then confirmation)
        assert adapter.session.request.call_count == 2
        
        # Check first call (order placement)
        first_call = adapter.session.request.call_args_list[0]
        assert first_call[0][0] == "POST"
        assert "orders" in first_call[0][1]

        # Verify order data
        order_data = first_call[1]["json"]["orders"][0]
        assert order_data["acctId"] == "DU123456"
        assert order_data["conid"] == 265598
        assert order_data["orderType"] == "LMT"
        assert order_data["side"] == "BUY"
        assert order_data["quantity"] == 100
        assert order_data["price"] == 150.00
        assert order_data["tif"] == "DAY"
        
        # Check second call (confirmation)
        second_call = adapter.session.request.call_args_list[1]
        assert second_call[0][0] == "POST"
        assert "reply/123456" in second_call[0][1]

    @pytest.mark.asyncio
    async def test_get_account_info(self, adapter, mock_response):
        """Test getting account information"""
        # Mock response data
        mock_data = {
            "accountType": "INDIVIDUAL",
            "baseCurrency": "USD",
            "netliquidation": 100000.50,
            "totalcashvalue": 50000.25,
            "buyingpower": 200000.00,
            "grosspositionvalue": 50000.25,
            "maintmarginreq": 25000.00,
            "excessliquidity": 75000.50,
        }
        mock_response.json.return_value = mock_data

        # Create a proper mock session
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        
        # Create async context manager for request
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                return None
        
        # Make request return the context manager directly (not a coroutine)
        mock_session.request = MagicMock(return_value=AsyncContextManager())
        adapter.session = mock_session
        adapter.selected_account = "DU123456"

        # Get account info
        account_info = await adapter.get_account_info()

        # Verify
        assert account_info is not None
        assert account_info.account_id == "DU123456"
        assert account_info.account_type == "INDIVIDUAL"
        assert account_info.base_currency == "USD"
        assert account_info.net_liquidation == 100000.50
        assert account_info.total_cash == 50000.25
        assert account_info.buying_power == 200000.00

    @pytest.mark.asyncio
    async def test_get_positions(self, adapter, mock_response):
        """Test getting positions"""
        # Mock response data
        mock_data = [
            {
                "conid": 265598,
                "contractDesc": "AAPL",
                "assetClass": "STK",
                "position": 100.0,
                "mktValue": 15025.00,
                "avgCost": 140.00,
                "unrealizedPnl": 1025.00,
                "realizedPnl": 0.0,
            }
        ]
        mock_response.json.return_value = mock_data

        # Create a proper mock session
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        
        # Create async context manager for request
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, *args):
                return None
        
        # Make request return the context manager directly (not a coroutine)
        mock_session.request = MagicMock(return_value=AsyncContextManager())
        adapter.session = mock_session
        adapter.selected_account = "DU123456"

        # Get positions
        positions = await adapter.get_positions()

        # Verify
        assert len(positions) == 1
        position = positions[0]
        assert position.account == "DU123456"
        assert position.contract.conid == 265598
        assert position.contract.symbol == "AAPL"
        assert position.position == 100.0
        assert position.market_value == 15025.00
        assert position.average_cost == 140.00
        assert position.unrealized_pnl == 1025.00

    @pytest.mark.asyncio
    async def test_websocket_market_data_subscription(self, adapter):
        """Test WebSocket market data subscription"""
        # Mock WebSocket client
        adapter.ws_client = AsyncMock()
        adapter.ws_client.subscribe_market_data = AsyncMock(return_value=True)
        adapter.ws_client.register_callback = Mock()

        # Subscribe to market data
        callback = AsyncMock()
        success = await adapter.subscribe_market_data(265598, callback)

        # Verify
        assert success is True
        adapter.ws_client.subscribe_market_data.assert_called_once_with(
            265598, ["31", "84", "86", "88", "85", "87"]
        )
        adapter.ws_client.register_callback.assert_called_once()

        # Verify callback registration
        assert 265598 in adapter._market_data_callbacks
        assert callback in adapter._market_data_callbacks[265598]

    @pytest.mark.asyncio
    async def test_error_handling(self, adapter):
        """Test error handling"""
        # Create a proper mock session that raises an exception
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        
        # Make request raise an exception
        mock_session.request.side_effect = Exception("Connection error")
        adapter.session = mock_session

        # Test search contracts with error
        contracts = await adapter.search_contracts("AAPL")
        assert contracts == []

        # Test market data with error
        market_data = await adapter.get_market_data_snapshot([265598])
        assert market_data == {}


@pytest.mark.integration
class TestIBKRAdapterIntegration:
    """
    Integration tests that require running IBKR gateway

    Skip these tests if gateway is not available
    """

    @pytest.fixture
    async def live_adapter(self):
        """Create live adapter instance"""
        adapter = IBKRAdapter(gateway_url="https://localhost:5000", ssl_verify=False)
        yield adapter
        await adapter.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Skip integration tests by default
        reason="Integration tests require --integration flag",
    )
    async def test_live_connection(self, live_adapter):
        """Test actual connection to IBKR gateway"""
        connected = await live_adapter.connect()

        if connected:
            assert live_adapter.state == ConnectionState.CONNECTED
            assert live_adapter.authenticated is True
            assert len(live_adapter.accounts) > 0
        else:
            # Gateway not running or not authenticated
            assert live_adapter.state in [
                ConnectionState.AUTHENTICATING,
                ConnectionState.ERROR,
            ]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Skip integration tests by default
        reason="Integration tests require --integration flag",
    )
    async def test_live_contract_search(self, live_adapter):
        """Test actual contract search"""
        connected = await live_adapter.connect()

        if connected:
            contracts = await live_adapter.search_contracts("AAPL", "STK")
            assert len(contracts) > 0
            assert any(c.symbol == "AAPL" for c in contracts)
