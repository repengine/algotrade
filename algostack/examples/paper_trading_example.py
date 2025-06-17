"""
Paper Trading Example

This example demonstrates how to set up and run the AlgoStack
live trading engine in paper trading mode.
"""

import asyncio
import logging

from algostack.core.live_engine import LiveTradingEngine, TradingMode
from algostack.strategies.mean_reversion_equity import MeanReversionEquityStrategy
from algostack.strategies.trend_following_multi import TrendFollowingMultiStrategy

logger = logging.getLogger(__name__)


async def main():
    """Run paper trading example."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configure trading engine
    config = {
        "mode": TradingMode.PAPER,
        # Strategies to run
        "strategies": [
            {
                "class": MeanReversionEquityStrategy,
                "id": "mean_reversion_spy",
                "params": {
                    "symbol": "SPY",
                    "lookback": 20,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss": 0.03,
                },
            },
            {
                "class": TrendFollowingMultiStrategy,
                "id": "trend_following",
                "params": {
                    "symbols": ["QQQ", "IWM"],
                    "fast_ma": 10,
                    "slow_ma": 30,
                    "atr_period": 14,
                    "atr_multiplier": 2.0,
                },
            },
        ],
        # Portfolio configuration
        "portfolio_config": {
            "initial_capital": 100000,
        },
        # Risk management
        "risk_config": {
            "max_position_size": 0.2,  # 20% max per position
            "max_leverage": 1.0,  # No leverage
            "max_drawdown": 0.15,  # 15% max drawdown
        },
        # Executor configuration
        "executor_config": {
            "paper": {
                "initial_capital": 100000,
                "commission": 1.0,  # $1 per trade
                "slippage": 0.0001,  # 0.01% slippage
                "fill_delay": 0.1,  # 100ms fill delay
            }
        },
        # Trading schedule (Eastern Time)
        "schedule": {
            "market_open": "09:30",
            "market_close": "16:00",
            "pre_market": "09:00",
            "post_market": "16:30",
            "timezone": "US/Eastern",
        },
        # Engine settings
        "update_interval": 1.0,  # Update every second
        "min_signal_strength": 0.6,  # Minimum signal strength to trade
        "risk_per_trade": 0.02,  # Risk 2% per trade
        "max_position_size": 1000,  # Max 1000 shares per position
    }

    # Create and start engine
    engine = LiveTradingEngine(config)

    logger.info("=" * 60)
    logger.info("AlgoStack Paper Trading Example")
    logger.info("=" * 60)
    logger.info(f"Mode: {engine.mode}")
    logger.info(f"Strategies: {list(engine.strategies.keys())}")
    logger.info(f"Active Symbols: {engine._active_symbols}")
    logger.info(f"Initial Capital: ${config['portfolio_config']['initial_capital']:,.2f}")
    logger.info("=" * 60)
    logger.info("\nStarting trading engine...")
    logger.info("Press Ctrl+C to stop\n")

    try:
        # Run engine
        await engine.start()

    except KeyboardInterrupt:
        logger.info("\n\nShutting down trading engine...")
        await engine.stop()

        # Print final statistics
        logger.info("\n" + "=" * 60)
        logger.info("Final Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Signals: {engine.stats['total_signals']}")
        logger.info(f"Total Orders: {engine.stats['total_orders']}")
        logger.info(f"Total Fills: {engine.stats['total_fills']}")
        logger.info(f"Errors: {engine.stats['errors']}")

        # Get final portfolio value
        portfolio_value = engine.portfolio_engine.total_value
        initial_capital = engine.portfolio_engine.initial_capital
        pnl = portfolio_value - initial_capital
        pnl_pct = (pnl / initial_capital) * 100

        logger.info(f"\nPortfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Total P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
