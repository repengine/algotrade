"""Main trading engine orchestrating strategies, data, and execution."""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from .data_handler import DataHandler
from .portfolio import Portfolio
from .risk import EnhancedRiskManager as RiskManager

logger = logging.getLogger(__name__)


class EngineConfig(BaseModel):
    """Configuration for the trading engine."""

    mode: str = "paper"  # paper, live
    data_providers: list[str] = ["yfinance"]
    strategies: list[str] = []
    risk_params: dict = {}


class TradingEngine:
    """Core event-driven trading engine."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.data_handler = DataHandler(config.data_providers)
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager(config.risk_params)
        self.strategies = {}
        self.running = False

    async def start(self) -> None:
        """Start the trading engine."""
        logger.info(f"Starting trading engine in {self.config.mode} mode")
        self.running = True

        # Initialize components
        await self.data_handler.initialize()
        await self.portfolio.initialize()

        # Main event loop
        while self.running:
            try:
                # Fetch latest data
                market_data = await self.data_handler.get_latest()

                # Update portfolio with latest prices
                self.portfolio.update_prices(market_data)

                # Generate signals from each strategy
                signals = {}
                for name, strategy in self.strategies.items():
                    if self.risk_manager.can_trade(strategy):
                        signals[name] = strategy.generate_signals(market_data)

                # Risk-adjusted sizing
                sized_orders = self.risk_manager.size_orders(signals, self.portfolio)

                # Execute orders
                if sized_orders:
                    await self.portfolio.execute_orders(sized_orders)

                # Sleep until next tick
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Engine error: {e}")
                if self.config.mode == "live":
                    self.risk_manager.trigger_kill_switch()

    def stop(self) -> None:
        """Stop the trading engine."""
        logger.info("Stopping trading engine")
        self.running = False

    def add_strategy(self, name: str, strategy: Any) -> None:
        """Add a strategy to the engine."""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
