"""
Unit tests for IBKR WebSocket reconnection functionality
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from adapters.ibkr_adapter import ConnectionState, IBKRWebSocketClient


class TestIBKRWebSocketReconnection:
    """Test IBKR WebSocket reconnection logic"""

    @pytest.fixture
    def ws_client(self):
        """Create WebSocket client instance"""
        return IBKRWebSocketClient(
            base_url="https://localhost:5000",
            ssl_verify=False,
            max_reconnect_attempts=3,
            initial_reconnect_delay=0.1,
            max_reconnect_delay=1.0
        )

    @pytest.mark.asyncio
    async def test_connection_state_tracking(self, ws_client):
        """Test connection state tracking"""
        import ssl

        assert ws_client.connection_state == ConnectionState.DISCONNECTED

        # Mock at the module level where it's imported
        with patch('adapters.ibkr_adapter.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_ws = AsyncMock()
            mock_ws.closed = False
            mock_ws.close = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            mock_session.close = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock SSL context creation
            mock_ssl_context = MagicMock(spec=ssl.SSLContext)
            with patch('ssl.create_default_context', return_value=mock_ssl_context):
                # Connect
                result = await ws_client.connect()
                assert result is True
                assert ws_client.connection_state == ConnectionState.CONNECTED

                # Disconnect
                await ws_client.disconnect()
                assert ws_client.connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnection_on_error(self, ws_client):
        """Test automatic reconnection on WebSocket error"""
        reconnect_attempts = []

        async def mock_connect():
            reconnect_attempts.append(len(reconnect_attempts) + 1)
            if len(reconnect_attempts) < 2:
                # Fail first attempt
                ws_client._connection_state = ConnectionState.ERROR
                return False
            else:
                # Succeed on second attempt
                ws_client._connection_state = ConnectionState.CONNECTED
                return True

        ws_client.connect = mock_connect
        ws_client._running = True

        # Trigger reconnection
        await ws_client._handle_reconnection()

        # Wait for reconnection to complete
        await asyncio.sleep(0.5)

        assert len(reconnect_attempts) >= 2
        assert ws_client.connection_state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, ws_client):
        """Test exponential backoff for reconnection attempts"""
        ws_client._reconnect_delay = ws_client.initial_reconnect_delay
        delays = []

        # Mock sleep to capture delays
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            delays.append(delay)
            await original_sleep(0.01)  # Short delay for testing

        with patch('asyncio.sleep', mock_sleep):
            # Simulate multiple failed reconnection attempts
            ws_client._running = True
            ws_client.connect = AsyncMock(return_value=False)

            await ws_client._reconnect_with_backoff()

        # Verify exponential backoff
        assert len(delays) == ws_client.max_reconnect_attempts
        assert delays[0] == 0.1  # initial_reconnect_delay
        assert delays[1] == 0.2  # 0.1 * 2
        assert delays[2] == 0.4  # 0.2 * 2

    @pytest.mark.asyncio
    async def test_max_reconnection_attempts(self, ws_client):
        """Test max reconnection attempts limit"""
        ws_client._running = True
        ws_client.connect = AsyncMock(return_value=False)

        await ws_client._reconnect_with_backoff()

        assert ws_client._reconnect_attempts == ws_client.max_reconnect_attempts
        assert ws_client.connection_state == ConnectionState.ERROR
        assert not ws_client._running

    @pytest.mark.asyncio
    async def test_subscription_restoration(self, ws_client):
        """Test subscription restoration after reconnection"""
        # Set up subscriptions
        ws_client._market_data_subscriptions = {
            12345: ["31", "84", "86"],
            67890: ["31", "88"]
        }
        ws_client._order_subscription = True
        ws_client._pnl_subscription = True

        # Mock WebSocket
        mock_ws = AsyncMock()
        ws_client.ws = mock_ws
        ws_client._reconnect_attempts = 1  # Indicate this is a reconnection

        # Restore subscriptions
        await ws_client._restore_subscriptions()

        # Verify all subscriptions were restored
        calls = mock_ws.send_str.call_args_list
        assert len(calls) == 4  # 2 market data + 1 order + 1 P&L

        # Check market data subscriptions
        call_strs = [call[0][0] for call in calls]
        assert any('smd+12345' in s for s in call_strs)
        assert any('smd+67890' in s for s in call_strs)
        assert any('sor+{}' in s for s in call_strs)
        assert any('spl+{}' in s for s in call_strs)

    @pytest.mark.asyncio
    async def test_connection_status_callbacks(self, ws_client):
        """Test connection status callbacks"""
        status_updates = []

        async def status_callback(data):
            status_updates.append(data)

        ws_client.register_callback("connection_status", status_callback)

        # Trigger status notifications
        ws_client._notify_connection_status("connecting")
        ws_client._notify_connection_status("connected")
        ws_client._notify_connection_status("disconnected")

        # Wait for callbacks
        await asyncio.sleep(0.1)

        assert len(status_updates) == 3
        assert status_updates[0]["status"] == "connecting"
        assert status_updates[1]["status"] == "connected"
        assert status_updates[2]["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_heartbeat_triggers_reconnection(self, ws_client):
        """Test heartbeat loop triggers reconnection on error"""
        ws_client._running = True

        # Mock WebSocket that raises error
        mock_ws = MagicMock()
        mock_ws.send_str = AsyncMock(side_effect=Exception("Connection lost"))
        mock_ws.closed = False
        ws_client.ws = mock_ws

        # Mock reconnection handler
        ws_client._handle_reconnection = AsyncMock()

        # Run heartbeat loop once
        heartbeat_task = asyncio.create_task(ws_client._heartbeat_loop())
        await asyncio.sleep(0.1)
        heartbeat_task.cancel()

        # Verify reconnection was triggered
        ws_client._handle_reconnection.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_messages_handles_disconnection(self, ws_client):
        """Test message receiver handles disconnection properly"""
        ws_client._running = True

        # Mock WebSocket that returns closed message
        mock_ws = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.CLOSED
        mock_ws.receive = AsyncMock(return_value=mock_msg)
        mock_ws.closed = False
        ws_client.ws = mock_ws

        # Mock reconnection handler
        ws_client._handle_reconnection = AsyncMock()

        # Run receive loop
        asyncio.create_task(ws_client._receive_messages())
        await asyncio.sleep(0.1)

        # Verify reconnection was triggered
        ws_client._handle_reconnection.assert_called_once()

    @pytest.mark.asyncio
    async def test_prevent_multiple_reconnection_tasks(self, ws_client):
        """Test that multiple reconnection tasks are prevented"""
        ws_client._running = True

        # Create a long-running reconnection task
        async def slow_reconnect():
            await asyncio.sleep(1)

        ws_client._reconnect_with_backoff = slow_reconnect

        # Trigger multiple reconnection attempts
        await ws_client._handle_reconnection()
        await ws_client._handle_reconnection()
        await ws_client._handle_reconnection()

        # Only one reconnection task should be created
        assert ws_client._reconnect_task is not None
        assert not ws_client._reconnect_task.done()
