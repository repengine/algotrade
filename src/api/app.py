"""
FastAPI Monitoring Dashboard for AlgoStack.

This module provides a web-based monitoring interface for the trading system
with real-time updates via WebSocket.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.models import (
    AlertInfo,
    OrderCommand,
    OrderInfo,
    PerformanceMetrics,
    PositionInfo,
    RiskMetrics,
    SignalInfo,
    StrategyCommand,
    StrategyInfo,
    SystemCommand,
    SystemInfo,
    SystemStatus,
    TradeInfo,
    WSMessage,
    WSSubscription,
)
from core.live_engine import LiveTradingEngine

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[WebSocket, set] = {}

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected: {websocket.client}")

    async def send_message(self, message: WSMessage, websocket: WebSocket):
        """Send message to specific WebSocket."""
        await websocket.send_json(message.dict())

    async def broadcast(self, message: WSMessage, channel: str):
        """Broadcast message to all subscribed connections."""
        disconnected = []

        for connection in self.active_connections:
            if channel in self.subscriptions.get(connection, set()):
                try:
                    await connection.send_json(message.dict())
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    def subscribe(self, websocket: WebSocket, channels: list[str]):
        """Subscribe WebSocket to channels."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(channels)

    def unsubscribe(self, websocket: WebSocket, channels: list[str]):
        """Unsubscribe WebSocket from channels."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].difference_update(channels)


class MonitoringAPI:
    """FastAPI monitoring application."""

    def __init__(self, engine: Optional[LiveTradingEngine] = None):
        self.app = FastAPI(
            title="AlgoStack Monitoring Dashboard",
            description="Real-time trading system monitoring and control",
            version="1.0.0",
        )

        self.engine = engine
        self.manager = ConnectionManager()
        self._setup_middleware()
        self._setup_routes()
        self._background_tasks = []

        # Cache for performance
        self._cache = {
            "system_info": None,
            "strategies": {},
            "positions": {},
            "metrics": None,
            "last_update": datetime.now(),
        }

    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        # System endpoints
        self.app.get("/api/system/info", response_model=SystemInfo)(
            self.get_system_info
        )
        self.app.post("/api/system/control")(self.control_system)

        # Strategy endpoints
        self.app.get("/api/strategies", response_model=list[StrategyInfo])(
            self.get_strategies
        )
        self.app.get("/api/strategies/{strategy_id}", response_model=StrategyInfo)(
            self.get_strategy
        )
        self.app.post("/api/strategies/control")(self.control_strategy)

        # Position endpoints
        self.app.get("/api/positions", response_model=list[PositionInfo])(
            self.get_positions
        )
        self.app.get("/api/positions/{symbol}", response_model=PositionInfo)(
            self.get_position
        )

        # Order endpoints
        self.app.get("/api/orders", response_model=list[OrderInfo])(self.get_orders)
        self.app.get("/api/orders/{order_id}", response_model=OrderInfo)(self.get_order)
        self.app.post("/api/orders")(self.create_order)
        self.app.delete("/api/orders/{order_id}")(self.cancel_order)

        # Performance endpoints
        self.app.get("/api/performance", response_model=PerformanceMetrics)(
            self.get_performance
        )
        self.app.get("/api/risk", response_model=RiskMetrics)(self.get_risk_metrics)

        # Alert endpoints
        self.app.get("/api/alerts", response_model=list[AlertInfo])(self.get_alerts)
        self.app.put("/api/alerts/{alert_id}/acknowledge")(self.acknowledge_alert)

        # Signal endpoints
        self.app.get("/api/signals", response_model=list[SignalInfo])(self.get_signals)

        # Trade history endpoints
        self.app.get("/api/trades", response_model=list[TradeInfo])(self.get_trades)
        self.app.get("/api/trades/export")(self.export_trades)

        # WebSocket endpoint
        self.app.websocket("/ws")(self.websocket_endpoint)

        # Dashboard UI
        self.app.get("/", response_class=HTMLResponse)(self.get_dashboard)

    # System endpoints

    async def get_system_info(self) -> SystemInfo:
        """Get system information."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        uptime = 0
        if self.engine.stats["engine_start"]:
            uptime = (
                datetime.now() - self.engine.stats["engine_start"]
            ).total_seconds()

        return SystemInfo(
            status=(
                SystemStatus.RUNNING if self.engine.is_running else SystemStatus.OFFLINE
            ),
            mode=self.engine.mode.value,
            uptime_seconds=uptime,
            start_time=self.engine.stats["engine_start"],
        )

    async def control_system(self, command: SystemCommand):
        """Control system state."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        if not command.confirm:
            raise HTTPException(status_code=400, detail="Confirmation required")

        try:
            if command.action == "start":
                asyncio.create_task(self.engine.start())
            elif command.action == "stop":
                await self.engine.stop()
            elif command.action == "pause":
                self.engine.is_trading_hours = False
            elif command.action == "resume":
                self.engine.is_trading_hours = True
            elif command.action == "emergency_stop":
                await self.engine._cancel_all_orders()
                await self.engine.stop()

            return {"status": "success", "action": command.action}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Strategy endpoints

    async def get_strategies(self) -> list[StrategyInfo]:
        """Get all strategies."""
        if not self.engine:
            return []

        strategies = []
        for strategy_id, strategy in self.engine.strategies.items():
            info = StrategyInfo(
                id=strategy_id,
                name=strategy.__class__.__name__,
                status="active",  # TODO: Track actual status
                symbols=getattr(strategy, "symbols", [getattr(strategy, "symbol", "")]),
                parameters={},  # TODO: Extract parameters
                signals_generated=0,  # TODO: Track signals
                orders_placed=len(
                    self.engine.order_manager.get_strategy_orders(strategy_id)
                ),
                last_signal_time=None,
            )
            strategies.append(info)

        return strategies

    async def get_strategy(self, strategy_id: str) -> StrategyInfo:
        """Get specific strategy."""
        strategies = await self.get_strategies()
        for strategy in strategies:
            if strategy.id == strategy_id:
                return strategy
        raise HTTPException(status_code=404, detail="Strategy not found")

    async def control_strategy(self, command: StrategyCommand):
        """Control strategy state."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        if command.strategy_id not in self.engine.strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # TODO: Implement strategy control
        return {"status": "success", "action": command.action}

    # Position endpoints

    async def get_positions(self) -> list[PositionInfo]:
        """Get all positions."""
        if not self.engine:
            return []

        try:
            positions = await self.engine.order_manager.get_positions()

            position_list = []
            for symbol, pos in positions.items():
                pnl_pct = (
                    pos.unrealized_pnl / (pos.average_cost * pos.quantity) * 100
                    if pos.quantity != 0
                    else 0
                )

                info = PositionInfo(
                    symbol=symbol,
                    quantity=pos.quantity,
                    average_cost=pos.average_cost,
                    current_price=pos.current_price,
                    market_value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                    pnl_percentage=pnl_pct,
                )
                position_list.append(info)

            return position_list

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def get_position(self, symbol: str) -> PositionInfo:
        """Get specific position."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        raise HTTPException(status_code=404, detail="Position not found")

    # Order endpoints

    async def get_orders(self, active_only: bool = True) -> list[OrderInfo]:
        """Get orders."""
        if not self.engine:
            return []

        if active_only:
            orders = self.engine.order_manager.get_active_orders()
        else:
            orders = list(self.engine.order_manager._orders.values())

        order_list = []
        for order in orders:
            info = OrderInfo(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                status=order.status.value.lower(),
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                filled_quantity=order.filled_quantity,
                average_fill_price=order.average_fill_price,
                submitted_at=order.submitted_at or datetime.now(),
                filled_at=order.filled_at,
                strategy_id=order.metadata.get("strategy_id"),
            )
            order_list.append(info)

        return order_list

    async def get_order(self, order_id: str) -> OrderInfo:
        """Get specific order."""
        orders = await self.get_orders(active_only=False)
        for order in orders:
            if order.order_id == order_id:
                return order
        raise HTTPException(status_code=404, detail="Order not found")

    async def create_order(self, command: OrderCommand):
        """Create manual order."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        try:
            # Convert to internal types
            from core.executor import OrderSide, OrderType

            side = OrderSide.BUY if command.side == "buy" else OrderSide.SELL
            order_type_map = {
                "market": OrderType.MARKET,
                "limit": OrderType.LIMIT,
                "stop": OrderType.STOP,
                "stop_limit": OrderType.STOP_LIMIT,
            }
            order_type = order_type_map[command.order_type]

            # Create order
            order = await self.engine.order_manager.create_order(
                symbol=command.symbol,
                side=side,
                quantity=command.quantity,
                order_type=order_type,
                limit_price=command.limit_price,
                stop_price=command.stop_price,
                strategy_id="MANUAL",
            )

            # Submit order
            success = await self.engine.order_manager.submit_order(order)

            if success:
                return {"status": "success", "order_id": order.order_id}
            else:
                raise HTTPException(status_code=400, detail="Order submission failed")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def cancel_order(self, order_id: str):
        """Cancel order."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        success = await self.engine.order_manager.cancel_order(order_id)

        if success:
            return {"status": "success", "order_id": order_id}
        else:
            raise HTTPException(status_code=400, detail="Order cancellation failed")

    # Performance endpoints

    async def get_performance(self) -> PerformanceMetrics:
        """Get performance metrics."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        portfolio = self.engine.portfolio_engine
        metrics_collector = self.engine.metrics_collector

        # Update current portfolio value in metrics
        metrics_collector.update_portfolio_value(portfolio.total_value)

        # Get comprehensive metrics
        perf_metrics = metrics_collector.get_performance_metrics()

        # Calculate daily P&L
        today_trades = [
            t
            for t in metrics_collector.trades
            if t.exit_time.date() == datetime.now().date()
        ]
        daily_pnl = sum(t.pnl for t in today_trades)
        daily_pnl_pct = (
            (daily_pnl / portfolio.initial_capital) * 100
            if portfolio.initial_capital > 0
            else 0
        )

        return PerformanceMetrics(
            total_value=portfolio.total_value,
            cash=portfolio.cash,
            positions_value=portfolio.total_value - portfolio.cash,
            daily_pnl=daily_pnl,
            daily_pnl_percentage=daily_pnl_pct,
            total_pnl=perf_metrics.get("total_pnl", 0),
            total_pnl_percentage=(
                perf_metrics.get("total_pnl", 0) / portfolio.initial_capital * 100
            ),
            sharpe_ratio=perf_metrics.get("sharpe_ratio"),
            max_drawdown=perf_metrics.get("max_drawdown", 0),
            win_rate=perf_metrics.get("win_rate", 0),
            profit_factor=perf_metrics.get("profit_factor", 1.0),
            trades_today=len(today_trades),
            trades_total=perf_metrics.get("total_trades", 0),
        )

    async def get_risk_metrics(self) -> RiskMetrics:
        """Get risk metrics."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        # TODO: Implement actual risk calculations
        return RiskMetrics(
            current_leverage=1.0,
            max_leverage=1.0,
            var_95=0.0,
            position_concentration={},
            sector_exposure={},
            correlation_risk=0.0,
            margin_usage=0.0,
            buying_power=self.engine.portfolio_engine.cash,
        )

    # Alert endpoints

    async def get_alerts(self, limit: int = 100) -> list[AlertInfo]:
        """Get recent alerts."""
        # TODO: Implement alert storage
        return []

    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert."""
        # TODO: Implement alert acknowledgment
        return {"status": "success", "alert_id": alert_id}

    # Signal endpoints

    async def get_signals(self, limit: int = 100) -> list[SignalInfo]:
        """Get recent signals."""
        # TODO: Implement signal storage
        return []

    # Trade history endpoints

    async def get_trades(self, limit: int = 100) -> list[TradeInfo]:
        """Get completed trades."""
        if not self.engine:
            return []

        trades = self.engine.metrics_collector.trades[-limit:]

        trade_list = []
        for trade in trades:
            duration = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

            info = TradeInfo(
                symbol=trade.symbol,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                side=trade.side,
                pnl=trade.pnl,
                pnl_percentage=trade.pnl_percentage,
                commission=trade.commission,
                strategy_id=trade.strategy_id,
                duration_minutes=duration,
            )
            trade_list.append(info)

        return trade_list

    async def export_trades(self, format: str = "csv"):
        """Export trade history."""
        if not self.engine:
            raise HTTPException(status_code=503, detail="Engine not connected")

        trades = await self.get_trades(limit=10000)  # Get all trades

        if format == "csv":
            # Convert to CSV
            import csv
            import io

            output = io.StringIO()
            if trades:
                writer = csv.DictWriter(output, fieldnames=trades[0].dict().keys())
                writer.writeheader()
                for trade in trades:
                    writer.writerow(trade.dict())

            from fastapi.responses import Response

            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=trades.csv"},
            )
        else:
            # Return as JSON
            return trades

    # WebSocket endpoint

    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket connection handler."""
        await self.manager.connect(websocket)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle subscription
                if message.get("action") in ["subscribe", "unsubscribe"]:
                    sub = WSSubscription(**message)
                    if sub.action == "subscribe":
                        self.manager.subscribe(websocket, sub.channels)
                    else:
                        self.manager.unsubscribe(websocket, sub.channels)

        except WebSocketDisconnect:
            self.manager.disconnect(websocket)

    # Dashboard UI

    async def get_dashboard(self) -> HTMLResponse:
        """Serve dashboard HTML."""
        # For now, return a simple placeholder
        # In production, this would serve a React/Vue/Angular app
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlgoStack Monitoring Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .info { background: #f0f0f0; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>AlgoStack Monitoring Dashboard</h1>
            <div class="info">
                <p>API Documentation: <a href="/docs">/docs</a></p>
                <p>WebSocket Endpoint: ws://localhost:8000/ws</p>
            </div>
            <div id="app">
                <p>Dashboard UI coming soon...</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    # Background tasks

    async def start_background_tasks(self):
        """Start background update tasks."""
        self._background_tasks.append(asyncio.create_task(self._broadcast_updates()))

    async def stop_background_tasks(self):
        """Stop background tasks."""
        for task in self._background_tasks:
            task.cancel()

    async def _broadcast_updates(self):
        """Broadcast updates to WebSocket clients."""
        while True:
            try:
                # Broadcast position updates
                positions = await self.get_positions()
                await self.manager.broadcast(
                    WSMessage(type="positions", data=positions), "positions"
                )

                # Broadcast order updates
                orders = await self.get_orders()
                await self.manager.broadcast(
                    WSMessage(type="orders", data=orders), "orders"
                )

                # Broadcast performance metrics
                metrics = await self.get_performance()
                await self.manager.broadcast(
                    WSMessage(type="metrics", data=metrics), "metrics"
                )

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(5)


def create_app(engine: Optional[LiveTradingEngine] = None) -> FastAPI:
    """Create FastAPI application."""
    api = MonitoringAPI(engine)

    @api.app.on_event("startup")
    async def startup_event():
        await api.start_background_tasks()

    @api.app.on_event("shutdown")
    async def shutdown_event():
        await api.stop_background_tasks()

    return api.app


# Module-level instances for test compatibility
app = None
ws_manager = ConnectionManager()


def get_trading_engine():
    """Get trading engine instance (for dependency injection)."""
    # This would be properly implemented with FastAPI dependency injection
    return None


def get_portfolio_engine():
    """Get portfolio engine instance (for dependency injection)."""
    # This would be properly implemented with FastAPI dependency injection
    return None

