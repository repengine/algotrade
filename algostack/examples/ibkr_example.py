"""
Example usage of IBKR Client Portal API Adapter

This script demonstrates how to:
1. Connect to IBKR gateway
2. Search for contracts
3. Subscribe to real-time market data
4. Place and manage orders
5. Query account information
"""

import asyncio
import logging
from datetime import datetime

from algostack.adapters.ibkr_adapter import (
    IBKRAdapter,
    Contract,
    Order,
    OrderType,
    OrderSide,
    TimeInForce
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def market_data_callback(data):
    """Callback for market data updates"""
    logger.info(f"Market data update: {data}")


async def order_callback(orders):
    """Callback for order updates"""
    for order in orders:
        logger.info(f"Order update: {order}")


async def pnl_callback(pnl_data):
    """Callback for P&L updates"""
    logger.info(f"P&L update: {pnl_data}")


async def main():
    """Main example function"""
    
    # Initialize adapter
    adapter = IBKRAdapter(
        gateway_url="https://localhost:5000",
        ssl_verify=False  # For self-signed certificates
    )
    
    try:
        # 1. Connect to gateway
        logger.info("Connecting to IBKR gateway...")
        connected = await adapter.connect()
        
        if not connected:
            logger.error("Failed to connect. Please ensure gateway is running and you're authenticated.")
            return
            
        logger.info(f"Connected! State: {adapter.state}")
        logger.info(f"Authenticated: {adapter.authenticated}")
        logger.info(f"Accounts: {adapter.accounts}")
        
        # 2. Search for a contract (Apple stock)
        logger.info("\nSearching for AAPL contract...")
        contracts = await adapter.search_contracts("AAPL", "STK")
        
        if contracts:
            apple_contract = contracts[0]
            logger.info(f"Found contract: {apple_contract}")
            
            # 3. Get contract details
            details = await adapter.get_contract_details(apple_contract.conid)
            logger.info(f"Contract details: {details}")
            
            # 4. Get market data snapshot
            logger.info("\nGetting market data snapshot...")
            market_data = await adapter.get_market_data_snapshot([apple_contract.conid])
            logger.info(f"Market data: {market_data}")
            
            # 5. Subscribe to real-time market data
            logger.info("\nSubscribing to real-time market data...")
            await adapter.subscribe_market_data(
                apple_contract.conid,
                market_data_callback
            )
            
            # 6. Subscribe to order updates
            logger.info("\nSubscribing to order updates...")
            await adapter.subscribe_orders(order_callback)
            
            # 7. Get account information
            logger.info("\nGetting account information...")
            account_info = await adapter.get_account_info()
            if account_info:
                logger.info(f"Account: {account_info.account_id}")
                logger.info(f"Net Liquidation: ${account_info.net_liquidation:,.2f}")
                logger.info(f"Buying Power: ${account_info.buying_power:,.2f}")
                logger.info(f"Total Cash: ${account_info.total_cash:,.2f}")
            
            # 8. Get positions
            logger.info("\nGetting positions...")
            positions = await adapter.get_positions()
            for position in positions:
                logger.info(f"Position: {position.contract.symbol} - "
                          f"Qty: {position.position}, "
                          f"Value: ${position.market_value:,.2f}, "
                          f"Unrealized P&L: ${position.unrealized_pnl:,.2f}")
            
            # 9. Get historical data
            logger.info("\nGetting historical data...")
            hist_data = await adapter.get_historical_data(
                apple_contract.conid,
                period="1d",
                bar_size="5min"
            )
            if hist_data is not None:
                logger.info(f"Historical data shape: {hist_data.shape}")
                logger.info(f"Latest bars:\n{hist_data.tail()}")
            
            # 10. Place a limit order (commented out for safety)
            # logger.info("\nPlacing a limit order...")
            # order = Order(
            #     account=adapter.selected_account,
            #     contract=apple_contract,
            #     order_type=OrderType.LIMIT,
            #     side=OrderSide.BUY,
            #     quantity=1,
            #     limit_price=150.00,  # Set your limit price
            #     tif=TimeInForce.DAY
            # )
            # order_result = await adapter.place_order(order)
            # logger.info(f"Order result: {order_result}")
            
            # 11. Get all orders
            logger.info("\nGetting all orders...")
            orders = await adapter.get_orders()
            for order in orders:
                logger.info(f"Order: {order}")
            
            # Subscribe to P&L updates
            logger.info("\nSubscribing to P&L updates...")
            await adapter.subscribe_pnl(pnl_callback)
            
            # Keep the script running to receive updates
            logger.info("\nListening for updates (press Ctrl+C to stop)...")
            await asyncio.sleep(60)  # Run for 60 seconds
            
        else:
            logger.error("No contracts found for AAPL")
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        
    finally:
        # Disconnect
        await adapter.disconnect()
        logger.info("Disconnected from IBKR gateway")


if __name__ == "__main__":
    asyncio.run(main())