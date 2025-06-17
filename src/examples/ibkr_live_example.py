"""
IBKR Live Trading Example

This example demonstrates how to set up and run the AlgoStack
live trading engine with Interactive Brokers.

IMPORTANT: This example is for demonstration purposes only.
Always test thoroughly with paper trading before using real money.

Prerequisites:
1. IBKR Client Portal Gateway running on localhost:5000
2. Valid IBKR account with appropriate permissions
3. Authenticated session in the gateway
"""

import asyncio
import logging

from core.live_engine import LiveTradingEngine, TradingMode
from strategies.mean_reversion_equity import MeanReversionEquityStrategy

logger = logging.getLogger(__name__)


async def main():
    """Run IBKR live trading example."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # IMPORTANT: These are example contract IDs - you need to look up the actual IDs
    CONTRACT_MAPPINGS = {
        "SPY": 756733,  # SPDR S&P 500 ETF
        "QQQ": 320227571,  # Invesco QQQ Trust
        "IWM": 9579970,  # iShares Russell 2000 ETF
        "AAPL": 265598,  # Apple Inc.
        "MSFT": 272093,  # Microsoft Corp.
    }

    # Configure trading engine for live trading
    config = {
        "mode": TradingMode.LIVE,  # LIVE mode - be careful!
        # Conservative strategy for live trading
        "strategies": [
            {
                "class": MeanReversionEquityStrategy,
                "id": "mean_reversion_spy_live",
                "params": {
                    "symbol": "SPY",
                    "lookback": 20,
                    "entry_threshold": 2.5,  # More conservative
                    "exit_threshold": 0.5,
                    "stop_loss": 0.02,  # Tighter stop loss
                },
            }
        ],
        # Portfolio configuration
        "portfolio_config": {
            "initial_capital": 25000,  # Minimum for pattern day trading
        },
        # Conservative risk management for live trading
        "risk_config": {
            "max_position_size": 0.1,  # 10% max per position
            "max_leverage": 1.0,  # No leverage
            "max_drawdown": 0.05,  # 5% max drawdown - very conservative
            "max_daily_loss": 0.02,  # 2% max daily loss
        },
        # Executor configuration
        "executor_config": {
            # Paper executor as backup
            "paper": {
                "initial_capital": 25000,
                "commission": 1.0,
                "slippage": 0.0001,
            },
            # IBKR executor for live trading
            "ibkr": {
                "gateway_url": "https://localhost:5000",
                "ssl_verify": False,  # Self-signed certificate
                "account": None,  # Will use first available account
                "contract_mappings": CONTRACT_MAPPINGS,
                "timeout": 30,
            },
        },
        # Trading schedule (Eastern Time)
        "schedule": {
            "market_open": "09:30",
            "market_close": "16:00",
            "pre_market": "09:15",  # Start 15 min before open
            "post_market": "16:15",  # End 15 min after close
            "timezone": "US/Eastern",
        },
        # Conservative engine settings for live trading
        "update_interval": 5.0,  # Update every 5 seconds
        "min_signal_strength": 0.8,  # High confidence signals only
        "risk_per_trade": 0.01,  # Risk only 1% per trade
        "max_position_size": 100,  # Max 100 shares per position
    }

    # Create engine
    engine = LiveTradingEngine(config)

    logger.warning("=" * 60)
    logger.warning("AlgoStack IBKR Live Trading Example")
    logger.warning("=" * 60)
    logger.warning("WARNING: This is LIVE TRADING with REAL MONEY!")
    logger.warning("=" * 60)
    logger.info(f"Mode: {engine.mode}")
    logger.info(f"Strategies: {list(engine.strategies.keys())}")
    logger.info(f"Active Symbols: {engine._active_symbols}")
    logger.info(f"Initial Capital: ${config['portfolio_config']['initial_capital']:,.2f}")
    logger.info("=" * 60)

    # Confirmation prompt
    response = input(
        "\nAre you sure you want to start LIVE TRADING? (type 'YES' to confirm): "
    )
    if response != "YES":
        logger.info("Live trading cancelled.")
        return

    logger.info("\nChecking IBKR connection...")

    try:
        # Test IBKR connection first
        ibkr_executor = engine.order_manager.executors.get("ibkr")
        if ibkr_executor:
            connected = await ibkr_executor.connect()
            if not connected:
                logger.error("ERROR: Failed to connect to IBKR. Make sure:")
                logger.error("1. Client Portal Gateway is running")
                logger.error("2. You are authenticated")
                logger.error("3. Gateway URL is correct")
                return

            # Get account info
            account_info = await ibkr_executor.get_account_info()
            logger.info(f"\nConnected to IBKR Account: {account_info.get('account_id')}")
            logger.info(f"Net Liquidation: ${account_info.get('net_liquidation', 0):,.2f}")
            logger.info(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")

            await ibkr_executor.disconnect()

        logger.info("\nStarting live trading engine...")
        logger.info("Press Ctrl+C to stop\n")

        # Run engine
        await engine.start()

    except KeyboardInterrupt:
        logger.info("\n\nShutting down trading engine...")
        await engine.stop()

        # Print final statistics
        logger.info("\n" + "=" * 60)
        logger.info("Live Trading Session Summary")
        logger.info("=" * 60)
        logger.info(f"Total Signals: {engine.stats['total_signals']}")
        logger.info(f"Total Orders: {engine.stats['total_orders']}")
        logger.info(f"Total Fills: {engine.stats['total_fills']}")
        logger.info(f"Errors: {engine.stats['errors']}")

        # Get final positions
        positions = await engine.order_manager.get_positions()
        if positions:
            logger.info("\nFinal Positions:")
            for symbol, pos in positions.items():
                logger.info(f"  {symbol}: {pos.quantity} shares @ ${pos.average_cost:.2f}")
                logger.info(f"    Unrealized P&L: ${pos.unrealized_pnl:,.2f}")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\nERROR: {e}")
        await engine.stop()


if __name__ == "__main__":
    # Additional safety check
    logger.warning("\n" + "!" * 60)
    logger.warning("! WARNING: This script will execute LIVE TRADES!")
    logger.warning("! Only run this if you understand the risks!")
    logger.warning("!" * 60 + "\n")

    asyncio.run(main())
