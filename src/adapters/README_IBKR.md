# IBKR Client Portal API Adapter

This adapter provides integration with Interactive Brokers' Client Portal Gateway API for trading operations.

## Prerequisites

1. **IBKR Client Portal Gateway**: Download and extract the gateway from IBKR
2. **Java 8+**: Required to run the gateway
3. **IBKR Account**: Valid Interactive Brokers account with API access enabled

## Setup

### 1. Start the Gateway

Navigate to your `clientportal.gw` directory and run:

```bash
# Linux/Mac
bin/run.sh root/conf.yaml

# Windows
bin\run.bat root\conf.yaml
```

The gateway will start on `https://localhost:5000` by default.

### 2. Authenticate

Open your browser and navigate to: `https://localhost:5000`

Log in with your IBKR credentials. Once authenticated, you can close the browser.

### 3. Use the Adapter

```python
from algostack.adapters import IBKRAdapter

# Initialize adapter
adapter = IBKRAdapter(
    gateway_url="https://localhost:5000",
    ssl_verify=False  # For self-signed certificates
)

# Connect
await adapter.connect()

# Check connection status
print(f"Connected: {adapter.authenticated}")
print(f"Accounts: {adapter.accounts}")
```

## Features

### Market Data

```python
# Search for contracts
contracts = await adapter.search_contracts("AAPL", "STK")
apple = contracts[0]

# Get snapshot
data = await adapter.get_market_data_snapshot([apple.conid])

# Subscribe to real-time data
async def on_market_data(data):
    print(f"Price update: {data}")

await adapter.subscribe_market_data(apple.conid, on_market_data)

# Historical data
df = await adapter.get_historical_data(
    apple.conid,
    period="1d",
    bar_size="5min"
)
```

### Order Management

```python
from algostack.adapters import Order, OrderType, OrderSide

# Create order
order = Order(
    account=adapter.selected_account,
    contract=apple,
    order_type=OrderType.LIMIT,
    side=OrderSide.BUY,
    quantity=100,
    limit_price=150.00
)

# Place order
result = await adapter.place_order(order)

# Monitor orders
async def on_order_update(orders):
    for order in orders:
        print(f"Order {order['orderId']}: {order['status']}")

await adapter.subscribe_orders(on_order_update)

# Get all orders
orders = await adapter.get_orders()

# Cancel order
await adapter.cancel_order(account, order_id)
```

### Account Management

```python
# Account info
info = await adapter.get_account_info()
print(f"Net Liquidation: ${info.net_liquidation:,.2f}")
print(f"Buying Power: ${info.buying_power:,.2f}")

# Positions
positions = await adapter.get_positions()
for pos in positions:
    print(f"{pos.contract.symbol}: {pos.position} @ ${pos.average_cost}")
    print(f"Unrealized P&L: ${pos.unrealized_pnl:,.2f}")

# Subscribe to P&L updates
async def on_pnl(data):
    print(f"P&L Update: {data}")

await adapter.subscribe_pnl(on_pnl)
```

## WebSocket Streaming

The adapter supports real-time streaming for:

- **Market Data**: Price quotes, trades, volume
- **Orders**: Status updates, fills, cancellations  
- **P&L**: Real-time profit/loss updates

## Error Handling

```python
try:
    await adapter.connect()
except Exception as e:
    print(f"Connection failed: {e}")

# Check connection state
if adapter.state == ConnectionState.CONNECTED:
    # Ready to trade
    pass
elif adapter.state == ConnectionState.AUTHENTICATING:
    # Need to authenticate via browser
    pass
```

## Security Notes

1. The gateway uses self-signed certificates for localhost
2. Set `ssl_verify=False` for local development
3. Never expose the gateway port to external networks
4. Keep your gateway and credentials secure

## Rate Limits

IBKR has various rate limits:

- Market data: 100 simultaneous subscriptions
- Orders: 50 orders per second
- Historical data: Limited by data type and duration

## Troubleshooting

### Gateway Won't Start
- Check Java version: `java -version` (needs 1.8+)
- Check port 5000 is not in use
- Check gateway logs in the console

### Authentication Issues
- Clear browser cookies for localhost:5000
- Try incognito/private browsing mode
- Check 2FA settings in IBKR account

### Connection Errors
- Verify gateway is running: `https://localhost:5000`
- Check firewall settings
- Ensure API permissions are enabled in IBKR account

### Market Data Issues
- Verify market data subscriptions in IBKR account
- Check if markets are open
- Some data requires specific subscriptions

## Example Scripts

See `algostack/examples/ibkr_example.py` for a complete working example.

## References

- [IBKR Client Portal API Documentation](https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/)
- [WebSocket Streaming Guide](https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/#ws-connection-1)
- [API Endpoints Reference](https://www.interactivebrokers.com/api/doc.html)