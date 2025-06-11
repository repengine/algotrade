"""
IBKR Live Trading Executor.

This module provides live trading execution through Interactive Brokers
using the Client Portal Gateway API.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from adapters.ibkr_adapter import (
    Contract as IBKRContract,
)
from adapters.ibkr_adapter import (
    IBKRAdapter,
)
from adapters.ibkr_adapter import (
    Order as IBKROrder,
)
from adapters.ibkr_adapter import (
    OrderSide as IBKROrderSide,
)
from adapters.ibkr_adapter import (
    OrderType as IBKROrderType,
)
from adapters.ibkr_adapter import (
    TimeInForce as IBKRTimeInForce,
)
from core.executor import (
    BaseExecutor,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class IBKRExecutor(BaseExecutor):
    """
    Interactive Brokers executor for live trading.

    This executor connects to IBKR via the Client Portal Gateway
    and provides order execution and position management.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize IBKR executor.

        Args:
            config: Configuration with:
                - gateway_url: URL of IBKR gateway (default: https://localhost:5000)
                - ssl_verify: Whether to verify SSL certificates (default: False)
                - account: IBKR account ID (optional, uses first available)
                - contract_mappings: Dict mapping symbols to contract IDs
        """
        super().__init__(config)

        # Initialize IBKR adapter
        self.adapter = IBKRAdapter(
            gateway_url=config.get("gateway_url", "https://localhost:5000"),
            ssl_verify=config.get("ssl_verify", False),
            timeout=config.get("timeout", 30),
        )

        # Configuration
        self.account = config.get("account")
        self.contract_mappings = config.get("contract_mappings", {})
        self._symbol_to_conid: dict[str, int] = {}
        self._conid_to_symbol: dict[int, str] = {}

        # Order tracking
        self._ibkr_order_map: dict[str, str] = {}  # Our order ID -> IBKR order ID
        self._order_contracts: dict[str, IBKRContract] = {}  # Order ID -> Contract

        # Connection monitoring
        self._keep_alive_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to IBKR gateway."""
        try:
            # Connect to gateway
            connected = await self.adapter.connect()
            if not connected:
                logger.error("Failed to connect to IBKR gateway")
                return False

            # Get accounts
            accounts = await self.adapter.get_accounts()
            if not accounts:
                logger.error("No accounts available")
                return False

            # Select account
            if self.account:
                if self.account not in accounts:
                    logger.error(f"Account {self.account} not found")
                    return False
                self.adapter.selected_account = self.account
            else:
                self.account = accounts[0]
                self.adapter.selected_account = self.account

            logger.info(f"Connected to IBKR with account: {self.account}")

            # Load contract mappings
            await self._load_contract_mappings()

            # Subscribe to order updates
            await self.adapter.subscribe_orders(self._handle_order_update)

            # Start keep-alive task
            self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from IBKR gateway."""
        # Cancel keep-alive task
        if self._keep_alive_task:
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except asyncio.CancelledError:
                pass

        # Disconnect adapter
        await self.adapter.disconnect()

        self.is_connected = False
        logger.info("Disconnected from IBKR")

    async def submit_order(self, order: Order) -> str:
        """
        Submit order to IBKR.

        Args:
            order: Order to submit

        Returns:
            Order ID
        """
        if not self.is_connected:
            raise RuntimeError("Executor not connected")

        # Validate order
        self.validate_order(order)

        # Get contract
        contract = await self._get_contract(order.symbol)
        if not contract:
            order.status = OrderStatus.REJECTED
            self._notify_order_status(order)
            raise ValueError(f"No contract found for symbol: {order.symbol}")

        # Convert order
        ibkr_order = self._convert_order(order, contract)

        # Submit to IBKR
        try:
            result = await self.adapter.place_order(ibkr_order)

            if result and "order_id" in result:
                # Map order IDs
                ibkr_order_id = str(result["order_id"])
                self._ibkr_order_map[order.order_id] = ibkr_order_id
                self._order_contracts[order.order_id] = contract

                # Update order
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now()
                order.metadata["ibkr_order_id"] = ibkr_order_id
                self._orders[order.order_id] = order

                # Notify submission
                self._notify_order_status(order)

                logger.info(
                    f"Order submitted: {order.order_id} -> IBKR {ibkr_order_id} - "
                    f"{order.side.value} {order.quantity} {order.symbol}"
                )

                return order.order_id
            else:
                # Order rejected
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reason"] = (
                    result.get("error", "Unknown error") if result else "No response"
                )
                self._notify_order_status(order)
                raise ValueError(
                    f"Order rejected: {order.metadata['rejection_reason']}"
                )

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = str(e)
            self._notify_order_status(order)
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False

        ibkr_order_id = self._ibkr_order_map.get(order_id)
        if not ibkr_order_id:
            logger.error(f"No IBKR order ID for: {order_id}")
            return False

        try:
            success = await self.adapter.cancel_order(self.account, ibkr_order_id)

            if success:
                order.status = OrderStatus.CANCELLED
                self._notify_order_status(order)
                logger.info(f"Order cancelled: {order_id} (IBKR: {ibkr_order_id})")

            return success

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        order = self._orders.get(order_id)
        if not order:
            return None

        # Get latest status from IBKR
        ibkr_order_id = self._ibkr_order_map.get(order_id)
        if ibkr_order_id:
            try:
                status = await self.adapter.get_order_status(ibkr_order_id)
                if status:
                    # Update order status
                    self._update_order_from_ibkr(order, status)
            except Exception as e:
                logger.error(f"Failed to get order status: {e}")

        return order

    async def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        try:
            ibkr_positions = await self.adapter.get_positions(self.account)

            positions = {}
            for ibkr_pos in ibkr_positions:
                # Map contract to symbol
                symbol = self._conid_to_symbol.get(ibkr_pos.contract.conid)
                if not symbol:
                    # Try to get symbol from contract description
                    symbol = ibkr_pos.contract.symbol

                if symbol:
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=int(ibkr_pos.position),
                        average_cost=ibkr_pos.average_cost,
                        current_price=(
                            ibkr_pos.market_value / ibkr_pos.position
                            if ibkr_pos.position != 0
                            else 0
                        ),
                        unrealized_pnl=ibkr_pos.unrealized_pnl,
                        realized_pnl=ibkr_pos.realized_pnl,
                        market_value=ibkr_pos.market_value,
                        last_updated=datetime.now(),
                    )

            self._positions = positions
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information."""
        try:
            info = await self.adapter.get_account_info(self.account)

            if info:
                return {
                    "account_id": info.account_id,
                    "account_type": info.account_type,
                    "base_currency": info.base_currency,
                    "net_liquidation": info.net_liquidation,
                    "total_cash": info.total_cash,
                    "buying_power": info.buying_power,
                    "gross_position_value": info.gross_position_value,
                    "maintenance_margin": info.maintenance_margin,
                    "excess_liquidity": info.excess_liquidity,
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    # Private methods

    async def _load_contract_mappings(self) -> None:
        """Load contract mappings from config or search."""
        for symbol, conid in self.contract_mappings.items():
            self._symbol_to_conid[symbol] = conid
            self._conid_to_symbol[conid] = symbol

        logger.info(f"Loaded {len(self._symbol_to_conid)} contract mappings")

    async def _get_contract(self, symbol: str) -> Optional[IBKRContract]:
        """Get contract for symbol."""
        # Check cache
        conid = self._symbol_to_conid.get(symbol)
        if conid:
            return IBKRContract(conid=conid, symbol=symbol)

        # Search for contract
        try:
            contracts = await self.adapter.search_contracts(symbol, "STK")
            if contracts:
                # Use first match
                contract = contracts[0]

                # Cache mapping
                self._symbol_to_conid[symbol] = contract.conid
                self._conid_to_symbol[contract.conid] = symbol

                return contract
            else:
                logger.error(f"No contracts found for symbol: {symbol}")
                return None

        except Exception as e:
            logger.error(f"Contract search failed: {e}")
            return None

    def _convert_order(self, order: Order, contract: IBKRContract) -> IBKROrder:
        """Convert our order to IBKR order."""
        # Map order type
        order_type_map = {
            OrderType.MARKET: IBKROrderType.MARKET,
            OrderType.LIMIT: IBKROrderType.LIMIT,
            OrderType.STOP: IBKROrderType.STOP,
            OrderType.STOP_LIMIT: IBKROrderType.STOP_LIMIT,
        }
        ibkr_order_type = order_type_map.get(order.order_type, IBKROrderType.MARKET)

        # Map side
        ibkr_side = (
            IBKROrderSide.BUY if order.side == OrderSide.BUY else IBKROrderSide.SELL
        )

        # Map time in force
        tif_map = {
            TimeInForce.DAY: IBKRTimeInForce.DAY,
            TimeInForce.GTC: IBKRTimeInForce.GTC,
            TimeInForce.IOC: IBKRTimeInForce.IOC,
            TimeInForce.FOK: IBKRTimeInForce.FOK,
        }
        ibkr_tif = tif_map.get(order.time_in_force, IBKRTimeInForce.DAY)

        return IBKROrder(
            account=self.account,
            contract=contract,
            order_type=ibkr_order_type,
            side=ibkr_side,
            quantity=float(order.quantity),
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            tif=ibkr_tif,
        )

    def _convert_order_status(self, ibkr_status: str) -> OrderStatus:
        """Convert IBKR order status to our status."""
        status_map = {
            "PendingSubmit": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.CANCELLED,
            "ApiPending": OrderStatus.PENDING,
            "ApiCancelled": OrderStatus.CANCELLED,
        }
        return status_map.get(ibkr_status, OrderStatus.PENDING)

    def _update_order_from_ibkr(self, order: Order, ibkr_data: dict[str, Any]) -> None:
        """Update order from IBKR data."""
        # Update status
        if "status" in ibkr_data:
            order.status = self._convert_order_status(ibkr_data["status"])

        # Update fill information
        if "filledQuantity" in ibkr_data:
            order.filled_quantity = int(ibkr_data["filledQuantity"])

        if "avgFillPrice" in ibkr_data:
            order.average_fill_price = float(ibkr_data["avgFillPrice"])

        # Update timestamp
        if order.status == OrderStatus.FILLED and not order.filled_at:
            order.filled_at = datetime.now()

    async def _handle_order_update(self, updates: list) -> None:
        """Handle order updates from WebSocket."""
        for update in updates:
            try:
                ibkr_order_id = str(update.get("orderId", ""))

                # Find our order
                our_order_id = None
                for oid, ioid in self._ibkr_order_map.items():
                    if ioid == ibkr_order_id:
                        our_order_id = oid
                        break

                if our_order_id and our_order_id in self._orders:
                    order = self._orders[our_order_id]

                    # Update order
                    self._update_order_from_ibkr(order, update)

                    # Check for fills
                    if "lastFillTime" in update and update.get("filledQuantity", 0) > 0:
                        # Create fill notification
                        fill = Fill(
                            fill_id=f"FILL-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                            order_id=our_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=int(
                                update.get("lastFillQuantity", order.filled_quantity)
                            ),
                            price=float(
                                update.get("lastFillPrice", order.average_fill_price)
                            ),
                            commission=float(update.get("commission", 0)),
                            timestamp=datetime.now(),
                        )
                        self._notify_fill(fill)

                    # Notify status change
                    self._notify_order_status(order)

            except Exception as e:
                logger.error(f"Error handling order update: {e}")

    async def _keep_alive_loop(self) -> None:
        """Send periodic keep-alive requests."""
        while self.is_connected:
            try:
                await asyncio.sleep(60)  # Every minute
                await self.adapter.keep_alive()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Keep-alive error: {e}")
