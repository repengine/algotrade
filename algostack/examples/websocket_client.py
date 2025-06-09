"""
WebSocket Client Example for AlgoStack Monitoring.

This example shows how to connect to the monitoring dashboard
and receive real-time updates.
"""

import asyncio
import json
import logging
from datetime import datetime

import websockets


async def monitor_trading():
    """Connect to monitoring WebSocket and display updates."""
    uri = "ws://localhost:8000/ws"

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to AlgoStack monitoring at {uri}")

            # Subscribe to all channels
            subscription = {
                "action": "subscribe",
                "channels": ["positions", "orders", "signals", "alerts", "metrics"],
            }
            await websocket.send(json.dumps(subscription))
            print("Subscribed to all channels")

            # Listen for updates
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Parse and display update
                    msg_type = data.get("type")
                    timestamp = data.get("timestamp", datetime.now().isoformat())

                    print(f"\n[{timestamp}] {msg_type.upper()} Update:")

                    if msg_type == "positions":
                        positions = data.get("data", [])
                        if positions:
                            print("Current Positions:")
                            for pos in positions:
                                symbol = pos.get("symbol")
                                qty = pos.get("quantity")
                                pnl = pos.get("unrealized_pnl", 0)
                                pnl_pct = pos.get("pnl_percentage", 0)
                                print(
                                    f"  {symbol}: {qty} shares, P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                                )
                        else:
                            print("  No open positions")

                    elif msg_type == "orders":
                        orders = data.get("data", [])
                        if orders:
                            print("Active Orders:")
                            for order in orders:
                                oid = order.get("order_id")
                                symbol = order.get("symbol")
                                side = order.get("side")
                                qty = order.get("quantity")
                                status = order.get("status")
                                print(f"  {oid}: {side} {qty} {symbol} - {status}")
                        else:
                            print("  No active orders")

                    elif msg_type == "metrics":
                        metrics = data.get("data", {})
                        total_value = metrics.get("total_value", 0)
                        cash = metrics.get("cash", 0)
                        total_pnl = metrics.get("total_pnl", 0)
                        total_pnl_pct = metrics.get("total_pnl_percentage", 0)

                        print("Performance Metrics:")
                        print(f"  Total Value: ${total_value:,.2f}")
                        print(f"  Cash: ${cash:,.2f}")
                        print(f"  Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")

                    elif msg_type == "signals":
                        signals = data.get("data", [])
                        if signals:
                            print("Recent Signals:")
                            for signal in signals[-5:]:  # Show last 5
                                strategy = signal.get("strategy_id")
                                symbol = signal.get("symbol")
                                direction = (
                                    "BUY" if signal.get("direction", 0) > 0 else "SELL"
                                )
                                strength = signal.get("strength", 0)
                                print(
                                    f"  {strategy}: {direction} {symbol} (strength: {strength:.2f})"
                                )

                    elif msg_type == "alerts":
                        alerts = data.get("data", [])
                        if alerts:
                            print("Recent Alerts:")
                            for alert in alerts[-3:]:  # Show last 3
                                level = alert.get("level")
                                message = alert.get("message")
                                print(f"  [{level.upper()}] {message}")

                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
                except Exception as e:
                    print(f"Error processing message: {e}")

    except Exception as e:
        print(f"Connection error: {e}")


async def send_test_order():
    """Send a test order via REST API."""
    import aiohttp

    url = "http://localhost:8000/api/orders"
    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 150.0,
        "time_in_force": "day",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=order) as response:
            result = await response.json()
            print(f"Order result: {result}")


async def main():
    """Run WebSocket monitoring client."""
    print("AlgoStack WebSocket Monitoring Client")
    print("=" * 40)
    print("Connecting to monitoring dashboard...")
    print("Press Ctrl+C to exit\n")

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Run monitoring
        await monitor_trading()
    except KeyboardInterrupt:
        print("\nDisconnecting...")


if __name__ == "__main__":
    asyncio.run(main())
