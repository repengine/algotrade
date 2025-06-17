"""
Interactive Brokers Client Portal API Adapter

This module provides integration with IBKR's Client Portal Gateway API
for real-time market data, order execution, and account management.
"""

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union
from urllib.parse import urljoin

import aiohttp
import pandas as pd
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiohttp.client_exceptions import ClientError

from algostack.utils.logging import setup_logger

logger = setup_logger(__name__)


class ConnectionState(Enum):
    """Connection states for IBKR gateway"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    ERROR = "error"


class OrderStatus(Enum):
    """Order status types"""

    PENDING_SUBMIT = "PendingSubmit"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    INACTIVE = "Inactive"


class OrderSide(Enum):
    """Order side types"""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types"""

    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"


class TimeInForce(Enum):
    """Time in force types"""

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


@dataclass
class MarketDataField:
    """Market data field mapping"""

    field_id: str
    name: str
    description: str


# Common market data fields
MARKET_DATA_FIELDS = {
    "31": MarketDataField("31", "last_price", "Last traded price"),
    "84": MarketDataField("84", "bid", "Bid price"),
    "85": MarketDataField("85", "ask_size", "Ask size"),
    "86": MarketDataField("86", "ask", "Ask price"),
    "88": MarketDataField("88", "bid_size", "Bid size"),
    "7295": MarketDataField("7295", "volume", "Volume"),
    "7296": MarketDataField("7296", "open", "Open price"),
    "7291": MarketDataField("7291", "close", "Close price"),
    "7293": MarketDataField("7293", "high", "High price"),
    "7294": MarketDataField("7294", "low", "Low price"),
    "83": MarketDataField("83", "change", "Price change"),
    "87": MarketDataField("87", "change_percent", "Price change %"),
}


@dataclass
class Contract:
    """IBKR Contract definition"""

    conid: int
    symbol: str
    sec_type: str = "STK"
    exchange: Optional[str] = None
    currency: str = "USD"
    multiplier: float = 1.0


@dataclass
class Order:
    """IBKR Order definition"""

    account: str
    contract: Contract
    order_type: OrderType
    side: OrderSide
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    order_id: Optional[str] = None
    parent_id: Optional[str] = None
    oca_group: Optional[str] = None
    transmit: bool = True


@dataclass
class Position:
    """Account position"""

    account: str
    contract: Contract
    position: float
    market_value: float
    average_cost: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class AccountInfo:
    """Account information"""

    account_id: str
    account_type: str
    base_currency: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    maintenance_margin: float
    excess_liquidity: float


class IBKRWebSocketClient:
    """WebSocket client for IBKR streaming data with robust reconnection logic"""

    def __init__(
        self, 
        base_url: str, 
        ssl_verify: bool = False,
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0
    ):
        self.base_url = base_url.replace("https://", "wss://")
        self.ws_url = f"{self.base_url}/v1/api/ws"
        self.ssl_verify = ssl_verify
        self.session: Optional[ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.logger = setup_logger(self.__class__.__name__)
        self._subscriptions: dict[str, set[str]] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._running = False
        self._heartbeat_task = None
        
        # Reconnection parameters
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self._reconnect_attempts = 0
        self._reconnect_delay = initial_reconnect_delay
        
        # Connection state tracking
        self._connection_state = ConnectionState.DISCONNECTED
        self._reconnect_task = None
        self._receive_task = None
        
        # Store subscriptions for re-subscription after reconnect
        self._market_data_subscriptions: dict[int, list[str]] = {}
        self._order_subscription = False
        self._pnl_subscription = False
    
    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state"""
        return self._connection_state

    async def connect(self) -> bool:
        """Connect to WebSocket with automatic reconnection"""
        try:
            # Update state
            self._connection_state = ConnectionState.CONNECTING
            self._notify_connection_status("connecting")
            
            if self.session:
                await self.session.close()

            # Create SSL context that doesn't verify certificates for localhost
            ssl_context = None
            if not self.ssl_verify:
                import ssl

                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            self.session = ClientSession()
            self.ws = await self.session.ws_connect(self.ws_url, ssl=ssl_context)

            self._running = True
            self._connection_state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._reconnect_delay = self.initial_reconnect_delay

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Start message receiver
            self._receive_task = asyncio.create_task(self._receive_messages())
            
            # Notify connection established
            self._notify_connection_status("connected")

            self.logger.info("WebSocket connected")
            
            # Re-subscribe to previous subscriptions if this is a reconnection
            await self._restore_subscriptions()
            
            return True

        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            self._connection_state = ConnectionState.ERROR
            self._notify_connection_status("error", str(e))
            
            # Start reconnection if not already disconnecting
            if self._running:
                asyncio.create_task(self._handle_reconnection())
            
            return False

    async def disconnect(self):
        """Disconnect WebSocket"""
        self._running = False
        self._connection_state = ConnectionState.DISCONNECTED

        # Cancel reconnection if in progress
        if self._reconnect_task:
            self._reconnect_task.cancel()
            
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            
        if self._receive_task:
            self._receive_task.cancel()

        if self.ws:
            await self.ws.close()

        if self.session:
            await self.session.close()
            
        self._notify_connection_status("disconnected")
        self.logger.info("WebSocket disconnected")

    async def subscribe_market_data(self, conid: int, fields: list[str]) -> bool:
        """Subscribe to market data"""
        try:
            # Store subscription for reconnection
            self._market_data_subscriptions[conid] = fields
            
            if not self.ws or self.ws.closed:
                self.logger.warning("WebSocket not connected, subscription will be restored on reconnection")
                return False
                
            field_list = ",".join(fields)
            message = f'smd+{conid}+{{"fields":[{field_list}]}}'
            await self.ws.send_str(message)

            if conid not in self._subscriptions:
                self._subscriptions[conid] = set()
            self._subscriptions[conid].update(fields)

            self.logger.info(f"Subscribed to market data for {conid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to subscribe to market data: {e}")
            # Trigger reconnection if send failed
            if self._running:
                asyncio.create_task(self._handle_reconnection())
            return False

    async def unsubscribe_market_data(self, conid: int) -> bool:
        """Unsubscribe from market data"""
        try:
            # Remove from stored subscriptions
            if conid in self._market_data_subscriptions:
                del self._market_data_subscriptions[conid]
                
            if not self.ws or self.ws.closed:
                return True
                
            message = f"umd+{conid}+{{}}"
            await self.ws.send_str(message)

            if conid in self._subscriptions:
                del self._subscriptions[conid]

            self.logger.info(f"Unsubscribed from market data for {conid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from market data: {e}")
            return False

    async def subscribe_orders(self) -> bool:
        """Subscribe to order updates"""
        try:
            # Store subscription state
            self._order_subscription = True
            
            if not self.ws or self.ws.closed:
                self.logger.warning("WebSocket not connected, subscription will be restored on reconnection")
                return False
                
            await self.ws.send_str("sor+{}")
            self.logger.info("Subscribed to order updates")
            return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe to orders: {e}")
            if self._running:
                asyncio.create_task(self._handle_reconnection())
            return False

    async def subscribe_pnl(self) -> bool:
        """Subscribe to P&L updates"""
        try:
            # Store subscription state
            self._pnl_subscription = True
            
            if not self.ws or self.ws.closed:
                self.logger.warning("WebSocket not connected, subscription will be restored on reconnection")
                return False
                
            await self.ws.send_str("spl+{}")
            self.logger.info("Subscribed to P&L updates")
            return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe to P&L: {e}")
            if self._running:
                asyncio.create_task(self._handle_reconnection())
            return False

    def register_callback(self, topic: str, callback: Callable):
        """Register callback for topic"""
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        self._callbacks[topic].append(callback)

    async def _receive_messages(self):
        """Receive and process WebSocket messages"""
        while self._running:
            try:
                if not self.ws or self.ws.closed:
                    self.logger.warning("WebSocket closed unexpectedly")
                    if self._running:
                        await self._handle_reconnection()
                    break
                    
                msg = await self.ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    topic = data.get("topic", "")

                    # Notify callbacks
                    if topic in self._callbacks:
                        for callback in self._callbacks[topic]:
                            try:
                                await callback(data)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {msg.data}")
                    if self._running:
                        await self._handle_reconnection()
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.warning("WebSocket closed")
                    if self._running:
                        await self._handle_reconnection()
                    break

            except Exception as e:
                self.logger.error(f"Message receive error: {e}")
                if self._running:
                    await self._handle_reconnection()
                break

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                if not self.ws or self.ws.closed:
                    self.logger.warning("WebSocket closed during heartbeat")
                    if self._running:
                        await self._handle_reconnection()
                    break
                    
                await self.ws.send_str("ech+hb")
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                if self._running:
                    await self._handle_reconnection()
                break
    
    async def _handle_reconnection(self):
        """Handle WebSocket reconnection with exponential backoff"""
        # Prevent multiple reconnection tasks
        if self._reconnect_task and not self._reconnect_task.done():
            return
            
        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())
        
    async def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff"""
        self._connection_state = ConnectionState.DISCONNECTED
        self._notify_connection_status("disconnected")
        
        # Clean up existing connection
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
                
        while self._running and self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            self._connection_state = ConnectionState.CONNECTING
            self._notify_connection_status("reconnecting", 
                                         f"Attempt {self._reconnect_attempts}/{self.max_reconnect_attempts}")
            
            self.logger.info(f"Reconnection attempt {self._reconnect_attempts} after {self._reconnect_delay}s delay")
            
            # Wait before reconnecting
            await asyncio.sleep(self._reconnect_delay)
            
            # Try to reconnect
            try:
                success = await self.connect()
                if success:
                    self.logger.info("Reconnection successful")
                    return
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")
            
            # Exponential backoff
            self._reconnect_delay = min(self._reconnect_delay * 2, self.max_reconnect_delay)
        
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self._connection_state = ConnectionState.ERROR
            self._notify_connection_status("max_reconnect_attempts_reached")
            self._running = False
    
    async def _restore_subscriptions(self):
        """Restore all subscriptions after reconnection"""
        if not self._reconnect_attempts:  # Skip on initial connection
            return
            
        self.logger.info("Restoring subscriptions after reconnection")
        
        # Restore market data subscriptions
        for conid, fields in self._market_data_subscriptions.items():
            try:
                field_list = ",".join(fields)
                message = f'smd+{conid}+{{"fields":[{field_list}]}}'
                await self.ws.send_str(message)
                self.logger.info(f"Restored market data subscription for {conid}")
            except Exception as e:
                self.logger.error(f"Failed to restore market data subscription for {conid}: {e}")
        
        # Restore order subscription
        if self._order_subscription:
            try:
                await self.ws.send_str("sor+{}")
                self.logger.info("Restored order subscription")
            except Exception as e:
                self.logger.error(f"Failed to restore order subscription: {e}")
        
        # Restore P&L subscription
        if self._pnl_subscription:
            try:
                await self.ws.send_str("spl+{}")
                self.logger.info("Restored P&L subscription")
            except Exception as e:
                self.logger.error(f"Failed to restore P&L subscription: {e}")
    
    def _notify_connection_status(self, status: str, details: str = ""):
        """Notify callbacks about connection status changes"""
        if "connection_status" in self._callbacks:
            status_data = {
                "topic": "connection_status",
                "status": status,
                "state": self._connection_state.value,
                "details": details,
                "reconnect_attempts": self._reconnect_attempts,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            for callback in self._callbacks["connection_status"]:
                try:
                    asyncio.create_task(callback(status_data))
                except Exception as e:
                    self.logger.error(f"Connection status callback error: {e}")


class IBKRAdapter:
    """
    Interactive Brokers Client Portal API Adapter

    Provides unified interface for IBKR trading operations including:
    - Authentication and session management
    - Real-time market data
    - Order placement and management
    - Account and position queries
    """

    def __init__(
        self,
        gateway_url: str = "https://localhost:5000",
        ssl_verify: bool = False,
        timeout: int = 30,
        ws_max_reconnect_attempts: int = 10,
        ws_initial_reconnect_delay: float = 1.0,
        ws_max_reconnect_delay: float = 60.0,
    ):
        """
        Initialize IBKR adapter

        Args:
            gateway_url: URL of the Client Portal Gateway
            ssl_verify: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            ws_max_reconnect_attempts: Maximum WebSocket reconnection attempts
            ws_initial_reconnect_delay: Initial delay between reconnection attempts
            ws_max_reconnect_delay: Maximum delay between reconnection attempts
        """
        self.base_url = gateway_url
        self.ssl_verify = ssl_verify
        self.timeout = ClientTimeout(total=timeout)
        self.logger = setup_logger(self.__class__.__name__)

        self.session: Optional[ClientSession] = None
        self.ws_client: Optional[IBKRWebSocketClient] = None
        self.state = ConnectionState.DISCONNECTED
        self.authenticated = False
        self.accounts: list[str] = []
        self.selected_account: Optional[str] = None
        
        # WebSocket reconnection parameters
        self.ws_max_reconnect_attempts = ws_max_reconnect_attempts
        self.ws_initial_reconnect_delay = ws_initial_reconnect_delay
        self.ws_max_reconnect_delay = ws_max_reconnect_delay

        # Callbacks
        self._market_data_callbacks: dict[int, list[Callable]] = {}
        self._order_callbacks: list[Callable] = []
        self._pnl_callbacks: list[Callable] = []

    async def connect(self) -> bool:
        """
        Connect to IBKR gateway

        Returns:
            True if connection successful
        """
        try:
            self.state = ConnectionState.CONNECTING

            # Create SSL context for self-signed certificates
            ssl_context = None
            if not self.ssl_verify:
                import ssl

                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            # Create HTTP session
            connector = TCPConnector(ssl=ssl_context)
            self.session = ClientSession(connector=connector, timeout=self.timeout)

            # Check if gateway is running
            status = await self._check_gateway_status()
            if not status:
                self.logger.error("Gateway is not running")
                self.state = ConnectionState.ERROR
                return False

            # Check authentication status
            auth_status = await self._check_auth_status()
            if auth_status:
                self.authenticated = True
                self.state = ConnectionState.CONNECTED

                # Get accounts
                await self._load_accounts()

                # Initialize WebSocket with reconnection support
                self.ws_client = IBKRWebSocketClient(
                    self.base_url, 
                    self.ssl_verify,
                    max_reconnect_attempts=self.ws_max_reconnect_attempts,
                    initial_reconnect_delay=self.ws_initial_reconnect_delay,
                    max_reconnect_delay=self.ws_max_reconnect_delay
                )
                await self.ws_client.connect()
                
                # Register connection status callback for monitoring
                self.ws_client.register_callback("connection_status", self._on_ws_connection_status)

                self.logger.info("Connected to IBKR gateway")
                return True
            else:
                self.state = ConnectionState.AUTHENTICATING
                self.logger.warning("Not authenticated. Please login via browser.")
                return False

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self):
        """Disconnect from IBKR gateway"""
        if self.ws_client:
            await self.ws_client.disconnect()

        if self.session:
            await self.session.close()

        self.state = ConnectionState.DISCONNECTED
        self.authenticated = False
        self.logger.info("Disconnected from IBKR gateway")

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Optional[Union[dict, list]]:
        """Make HTTP request to gateway"""
        url = urljoin(self.base_url, endpoint)

        try:
            async with self.session.request(
                method, url, json=data, params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    self.logger.error(f"Request failed: {response.status} - {text}")
                    return None

        except (ClientError, Exception) as e:
            self.logger.error(f"Request error: {e}")
            return None

    async def _check_gateway_status(self) -> bool:
        """Check if gateway is running"""
        try:
            # Try to access the root endpoint
            async with self.session.get(self.base_url) as response:
                return response.status in [200, 302, 401]
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return False

    async def _check_auth_status(self) -> bool:
        """Check authentication status"""
        result = await self._request("GET", "/v1/api/iserver/auth/status")
        if result:
            return result.get("authenticated", False)
        return False

    async def _load_accounts(self):
        """Load available accounts"""
        result = await self._request("GET", "/v1/api/portfolio/accounts")
        if result:
            self.accounts = result
            if self.accounts and not self.selected_account:
                self.selected_account = self.accounts[0]
                self.logger.info(f"Selected account: {self.selected_account}")

    async def reauthenticate(self) -> bool:
        """
        Trigger reauthentication

        Note: This will open the login page in the default browser
        """
        result = await self._request("POST", "/v1/api/iserver/reauthenticate")
        if result:
            self.logger.info(
                "Reauthentication triggered. Please complete login in browser."
            )
            return True
        return False

    # Market Data Methods

    async def search_contracts(
        self, symbol: str, sec_type: Optional[str] = None
    ) -> list[Contract]:
        """
        Search for contracts by symbol

        Args:
            symbol: Symbol to search for
            sec_type: Security type filter (STK, OPT, FUT, etc.)

        Returns:
            List of matching contracts
        """
        params = {"symbol": symbol}
        if sec_type:
            params["sec_type"] = sec_type

        result = await self._request(
            "GET", "/v1/api/iserver/secdef/search", params=params
        )

        contracts = []
        if result:
            for item in result:
                contracts.append(
                    Contract(
                        conid=item["conid"],
                        symbol=item["symbol"],
                        sec_type=item.get("secType", "STK"),
                        exchange=item.get("exchange"),
                        currency=item.get("currency", "USD"),
                    )
                )

        return contracts

    async def get_contract_details(self, conid: int) -> Optional[dict]:
        """Get detailed contract information"""
        result = await self._request("GET", f"/v1/api/iserver/contract/{conid}/info")
        return result

    async def get_market_data_snapshot(
        self, conids: list[int], fields: Optional[list[str]] = None
    ) -> dict[int, dict]:
        """
        Get market data snapshot

        Args:
            conids: List of contract IDs
            fields: List of field IDs (default: common fields)

        Returns:
            Dictionary mapping conid to market data
        """
        if not fields:
            fields = list(MARKET_DATA_FIELDS.keys())

        # Request market data
        params = {"conids": ",".join(map(str, conids)), "fields": ",".join(fields)}

        result = await self._request(
            "GET", "/v1/api/iserver/marketdata/snapshot", params=params
        )

        if not result:
            return {}

        # Parse response
        market_data = {}
        for item in result:
            conid = item.get("conid")
            if conid:
                data = {}
                for field_id, field_info in MARKET_DATA_FIELDS.items():
                    if field_id in item:
                        data[field_info.name] = item[field_id]
                market_data[conid] = data

        return market_data

    async def subscribe_market_data(
        self, conid: int, callback: Callable, fields: Optional[list[str]] = None
    ) -> bool:
        """
        Subscribe to real-time market data

        Args:
            conid: Contract ID
            callback: Callback function for data updates
            fields: List of field IDs to subscribe to

        Returns:
            True if subscription successful
        """
        if not self.ws_client:
            self.logger.error("WebSocket not connected")
            return False

        if not fields:
            fields = ["31", "84", "86", "88", "85", "87"]  # Common fields

        # Register callback
        if conid not in self._market_data_callbacks:
            self._market_data_callbacks[conid] = []
        self._market_data_callbacks[conid].append(callback)

        # Subscribe via WebSocket
        success = await self.ws_client.subscribe_market_data(conid, fields)

        if success:
            # Register WebSocket callback
            async def ws_callback(data):
                if data.get("topic", "").startswith(f"smd+{conid}"):
                    for cb in self._market_data_callbacks.get(conid, []):
                        await cb(data)

            self.ws_client.register_callback(f"smd+{conid}", ws_callback)

        return success

    async def unsubscribe_market_data(self, conid: int) -> bool:
        """Unsubscribe from market data"""
        if not self.ws_client:
            return False

        # Remove callbacks
        if conid in self._market_data_callbacks:
            del self._market_data_callbacks[conid]

        return await self.ws_client.unsubscribe_market_data(conid)

    # Order Management Methods

    async def place_order(self, order: Order) -> Optional[dict]:
        """
        Place an order

        Args:
            order: Order object

        Returns:
            Order confirmation or None
        """
        # Build order payload
        payload = {
            "acctId": order.account,
            "conid": order.contract.conid,
            "orderType": order.order_type.value,
            "side": order.side.value,
            "quantity": order.quantity,
            "tif": order.tif.value,
        }

        if order.limit_price is not None:
            payload["price"] = order.limit_price

        if order.stop_price is not None:
            payload["auxPrice"] = order.stop_price

        # Place order
        result = await self._request(
            "POST",
            f"/v1/api/iserver/account/{order.account}/orders",
            data={"orders": [payload]},
        )

        if result:
            # Check for confirmations
            if "id" in result:
                # Confirm order
                confirm_result = await self._request(
                    "POST",
                    f"/v1/api/iserver/reply/{result['id']}",
                    data={"confirmed": True},
                )
                return confirm_result
            else:
                return result

        return None

    async def cancel_order(self, account: str, order_id: str) -> bool:
        """Cancel an order"""
        result = await self._request(
            "DELETE", f"/v1/api/iserver/account/{account}/order/{order_id}"
        )
        return result is not None

    async def get_orders(self, account: Optional[str] = None) -> list[dict]:
        """Get all orders"""
        if not account:
            account = self.selected_account

        result = await self._request("GET", "/v1/api/iserver/account/orders")
        return result.get("orders", []) if result else []

    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get order status"""
        result = await self._request(
            "GET", f"/v1/api/iserver/account/order/status/{order_id}"
        )
        return result

    async def subscribe_orders(self, callback: Callable) -> bool:
        """Subscribe to order updates"""
        if not self.ws_client:
            return False

        self._order_callbacks.append(callback)

        # Subscribe via WebSocket
        success = await self.ws_client.subscribe_orders()

        if success:
            # Register WebSocket callback
            async def ws_callback(data):
                if data.get("topic") == "sor":
                    for cb in self._order_callbacks:
                        await cb(data.get("args", []))

            self.ws_client.register_callback("sor", ws_callback)

        return success

    # Account Management Methods

    async def get_accounts(self) -> list[str]:
        """Get list of accounts"""
        if not self.accounts:
            await self._load_accounts()
        return self.accounts

    async def get_account_info(
        self, account: Optional[str] = None
    ) -> Optional[AccountInfo]:
        """Get account information"""
        if not account:
            account = self.selected_account

        result = await self._request("GET", f"/v1/api/portfolio/{account}/summary")

        if result:
            return AccountInfo(
                account_id=account,
                account_type=result.get("accountType", ""),
                base_currency=result.get("baseCurrency", "USD"),
                net_liquidation=result.get("netliquidation", 0),
                total_cash=result.get("totalcashvalue", 0),
                buying_power=result.get("buyingpower", 0),
                gross_position_value=result.get("grosspositionvalue", 0),
                maintenance_margin=result.get("maintmarginreq", 0),
                excess_liquidity=result.get("excessliquidity", 0),
            )

        return None

    async def get_positions(self, account: Optional[str] = None) -> list[Position]:
        """Get account positions"""
        if not account:
            account = self.selected_account

        result = await self._request("GET", f"/v1/api/portfolio/{account}/positions/0")

        positions = []
        if result:
            for item in result:
                positions.append(
                    Position(
                        account=account,
                        contract=Contract(
                            conid=item["conid"],
                            symbol=item.get("contractDesc", ""),
                            sec_type=item.get("assetClass", "STK"),
                        ),
                        position=item.get("position", 0),
                        market_value=item.get("mktValue", 0),
                        average_cost=item.get("avgCost", 0),
                        unrealized_pnl=item.get("unrealizedPnl", 0),
                        realized_pnl=item.get("realizedPnl", 0),
                    )
                )

        return positions

    async def subscribe_pnl(self, callback: Callable) -> bool:
        """Subscribe to P&L updates"""
        if not self.ws_client:
            return False

        self._pnl_callbacks.append(callback)

        # Subscribe via WebSocket
        success = await self.ws_client.subscribe_pnl()

        if success:
            # Register WebSocket callback
            async def ws_callback(data):
                if data.get("topic") == "spl":
                    for cb in self._pnl_callbacks:
                        await cb(data.get("args", {}))

            self.ws_client.register_callback("spl", ws_callback)

        return success

    # Historical Data Methods

    async def get_historical_data(
        self,
        conid: int,
        period: str = "1d",
        bar_size: str = "1min",
        data_type: str = "TRADES",
        outside_rth: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical market data

        Args:
            conid: Contract ID
            period: Time period (1d, 1w, 1m, 1y)
            bar_size: Bar size (1min, 5min, 1h, 1d)
            data_type: TRADES, MIDPOINT, BID, ASK
            outside_rth: Include data outside regular trading hours

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            "conid": conid,
            "period": period,
            "bar": bar_size,
            "outsideRth": outside_rth,
        }

        result = await self._request(
            "GET", "/v1/api/iserver/marketdata/history", params=params
        )

        if result and "data" in result:
            data = result["data"]
            df = pd.DataFrame(data)

            # Convert timestamp to datetime
            if "t" in df.columns:
                df["datetime"] = pd.to_datetime(df["t"], unit="ms")
                df.set_index("datetime", inplace=True)

            # Rename columns
            column_map = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
            df.rename(columns=column_map, inplace=True)

            return df

        return None

    # Utility Methods

    async def keep_alive(self):
        """Send keep-alive request"""
        result = await self._request("POST", "/v1/api/tickle")
        return result is not None

    async def get_server_info(self) -> Optional[dict]:
        """Get server information"""
        return await self._request("GET", "/v1/api/one/user")
    
    async def _on_ws_connection_status(self, data: dict):
        """Handle WebSocket connection status changes"""
        status = data.get("status", "")
        state = data.get("state", "")
        details = data.get("details", "")
        
        self.logger.info(f"WebSocket connection status: {status} (state={state}, details={details})")
        
        # Update adapter state based on WebSocket state
        if state == "connected":
            self.state = ConnectionState.CONNECTED
        elif state == "connecting" or state == "reconnecting":
            self.state = ConnectionState.CONNECTING
        elif state == "disconnected":
            self.state = ConnectionState.DISCONNECTED
        elif state == "error":
            self.state = ConnectionState.ERROR
            
        # Notify any registered callbacks about connection state changes
        if status == "max_reconnect_attempts_reached":
            self.logger.error("WebSocket failed to reconnect after maximum attempts")
            # Could trigger additional recovery logic here
