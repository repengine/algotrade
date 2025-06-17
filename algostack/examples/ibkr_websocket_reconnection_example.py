"""
Example demonstrating IBKR WebSocket reconnection functionality

This example shows how the IBKRAdapter handles WebSocket disconnections
and automatically reconnects with exponential backoff.
"""

import asyncio
import logging
import signal
from datetime import datetime

from algostack.adapters.ibkr_adapter import IBKRAdapter

logger = logging.getLogger(__name__)


class IBKRWebSocketMonitor:
    """Monitor IBKR WebSocket connection with reconnection handling"""
    
    def __init__(self):
        self.adapter = IBKRAdapter(
            gateway_url="https://localhost:5000",
            ssl_verify=False
        )
        self.running = True
        self.market_data_received = 0
        self.connection_events = []
        
    async def on_market_data(self, data):
        """Handle market data updates"""
        self.market_data_received += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Market data update #{self.market_data_received}: {data}")
        
    async def on_connection_status(self, data):
        """Handle connection status changes"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = data.get("status", "")
        state = data.get("state", "")
        details = data.get("details", "")
        attempts = data.get("reconnect_attempts", 0)
        
        event = f"[{timestamp}] Connection {status}: state={state}, attempts={attempts}, details={details}"
        self.connection_events.append(event)
        logger.info(event)
        
    async def simulate_network_interruption(self):
        """Simulate network interruption after 30 seconds"""
        await asyncio.sleep(30)
        logger.warning("\n🔴 SIMULATING NETWORK INTERRUPTION - Closing WebSocket connection...")
        
        if self.adapter.ws_client and self.adapter.ws_client.ws:
            # Force close the WebSocket to simulate network failure
            await self.adapter.ws_client.ws.close()
            logger.warning("🔴 WebSocket forcibly closed - reconnection should start automatically\n")
            
    async def run(self):
        """Run the monitoring example"""
        logger.info("Starting IBKR WebSocket Reconnection Example")
        logger.info("=" * 50)
        
        # Connect to IBKR
        logger.info("Connecting to IBKR Gateway...")
        connected = await self.adapter.connect()
        
        if not connected:
            logger.error("❌ Failed to connect to IBKR Gateway")
            logger.error("Please ensure:")
            logger.error("1. IBKR Gateway is running on https://localhost:5000")
            logger.error("2. You are logged in via the web interface")
            return
            
        logger.info("✅ Connected to IBKR Gateway")
        
        # Register connection status callback
        if self.adapter.ws_client:
            self.adapter.ws_client.register_callback("connection_status", self.on_connection_status)
        
        # Search for a test contract (AAPL)
        logger.info("\nSearching for AAPL contract...")
        contracts = await self.adapter.search_contracts("AAPL", "STK")
        
        if contracts:
            contract = contracts[0]
            logger.info(f"✅ Found contract: {contract.symbol} (conid: {contract.conid})")
            
            # Subscribe to market data
            logger.info(f"\nSubscribing to market data for {contract.symbol}...")
            success = await self.adapter.subscribe_market_data(
                contract.conid,
                self.on_market_data,
                fields=["31", "84", "86"]  # Last, Bid, Ask
            )
            
            if success:
                logger.info("✅ Subscribed to market data")
                
                # Start network interruption simulation
                asyncio.create_task(self.simulate_network_interruption())
                
                # Monitor for a while
                logger.info("\nMonitoring connection and market data...")
                logger.info("The WebSocket will be forcibly closed after 30 seconds to test reconnection")
                logger.info("Press Ctrl+C to stop\n")
                
                start_time = datetime.now()
                while self.running:
                    await asyncio.sleep(1)
                    
                    # Print summary every 10 seconds
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                        logger.info(f"\n📊 Status after {int(elapsed)}s:")
                        logger.info(f"   Market data updates received: {self.market_data_received}")
                        logger.info(f"   Connection state: {self.adapter.state.value}")
                        if self.adapter.ws_client:
                            logger.info(f"   WebSocket state: {self.adapter.ws_client.connection_state.value}")
                        logger.info("")
                        
            else:
                logger.error("❌ Failed to subscribe to market data")
                
        else:
            logger.error("❌ No contracts found for AAPL")
            
        # Disconnect
        logger.info("\nDisconnecting...")
        await self.adapter.disconnect()
        
        # Print connection event summary
        logger.info("\n📋 Connection Event Summary:")
        logger.info("=" * 50)
        for event in self.connection_events:
            logger.info(event)
            
        logger.info(f"\nTotal market data updates received: {self.market_data_received}")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False


async def main():
    """Main entry point"""
    monitor = IBKRWebSocketMonitor()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("\nStopping monitor...")
        monitor.stop()
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await monitor.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.exception("Exception details:")


if __name__ == "__main__":
    asyncio.run(main())