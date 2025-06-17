"""
Trading Engine Module

This module contains the main trading engine that coordinates all trading activities,
including strategy execution, order management, and system orchestration.
"""

import asyncio
import queue
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from utils.logging import setup_logger


class EngineState(Enum):
    """Trading engine states"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineConfig:
    """Trading engine configuration"""

    name: str = "AlgoStack Trading Engine"
    tick_interval: float = 1.0  # seconds
    max_strategies: int = 10
    enable_paper_trading: bool = True
    enable_risk_checks: bool = True
    log_level: str = "INFO"
    data_buffer_size: int = 1000
    order_timeout: float = 30.0  # seconds


class TradingEngine:
    """
    Main trading engine class that orchestrates all trading operations.

    This class manages:
    - Strategy lifecycle
    - Order execution
    - Risk management
    - Data flow
    - System monitoring
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        # Component-based API for backward compatibility
        portfolio: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        data_handler: Optional[Any] = None,
        executor: Optional[Any] = None,
        strategies: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trading engine.

        Args:
            config: Engine configuration object (new API)
            portfolio: Portfolio manager (backward compatibility)
            risk_manager: Risk manager (backward compatibility)
            data_handler: Data handler (backward compatibility)
            executor: Order executor (backward compatibility)
            strategies: Trading strategies (backward compatibility)
        """
        # Handle backward compatibility
        if portfolio is not None:
            # Old API - component-based initialization
            self.portfolio = portfolio
            self.risk_manager = risk_manager
            self.data_handler = data_handler
            self.executor = executor
            self.strategies = strategies or {}
            self.config = EngineConfig()
            self.logger = setup_logger(__name__, self.config.log_level)
            self.is_running = False  # For backward compatibility
            self.order_queue = queue.Queue()  # For backward compatibility
        else:
            # New API - config-based initialization
            self.config = config or EngineConfig()
            self.logger = setup_logger(__name__, self.config.log_level)
            self.strategies: dict[str, Any] = {}
            self.portfolio = None
            self.risk_manager = None
            self.data_handler = None
            self.executor = None
            self.is_running = False
            self.order_queue = queue.Queue()

        self.state = EngineState.STOPPED
        self.active_orders: dict[str, Any] = {}
        self.position_manager = None
        self.data_manager = None
        self._main_loop_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the trading engine"""
        if self.state != EngineState.STOPPED:
            raise RuntimeError(f"Cannot start engine in state: {self.state}")

        self.logger.info("Starting trading engine...")
        self.state = EngineState.STARTING

        try:
            # Initialize components
            await self._initialize_components()

            # Connect executor if available
            if self.executor and hasattr(self.executor, 'connect'):
                await self.executor.connect()

            # Start main event loop
            self._main_loop_task = asyncio.create_task(self._main_loop())

            self.state = EngineState.RUNNING
            self.is_running = True  # For backward compatibility
            self.logger.info("Trading engine started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start engine: {e}")
            self.state = EngineState.ERROR
            self.is_running = False
            raise

    async def stop(self) -> None:
        """Stop the trading engine"""
        if self.state not in [EngineState.RUNNING, EngineState.PAUSED]:
            return

        self.logger.info("Stopping trading engine...")
        self.state = EngineState.STOPPING

        try:
            # Cancel main loop
            if self._main_loop_task:
                self._main_loop_task.cancel()
                await asyncio.gather(self._main_loop_task, return_exceptions=True)

            # Cleanup components
            await self._cleanup_components()

            # Disconnect executor if available
            if self.executor and hasattr(self.executor, 'disconnect'):
                await self.executor.disconnect()

            self.state = EngineState.STOPPED
            self.is_running = False  # For backward compatibility
            self.logger.info("Trading engine stopped")

        except Exception as e:
            self.logger.error(f"Error stopping engine: {e}")
            self.state = EngineState.ERROR
            self.is_running = False

    async def pause(self) -> None:
        """Pause the trading engine"""
        if self.state != EngineState.RUNNING:
            raise RuntimeError(f"Cannot pause engine in state: {self.state}")

        self.state = EngineState.PAUSED
        self.logger.info("Trading engine paused")

    async def resume(self) -> None:
        """Resume the trading engine"""
        if self.state != EngineState.PAUSED:
            raise RuntimeError(f"Cannot resume engine in state: {self.state}")

        self.state = EngineState.RUNNING
        self.logger.info("Trading engine resumed")

    def add_strategy(self, strategy_id: str, strategy: Any) -> None:
        """Add a strategy to the engine"""
        if len(self.strategies) >= self.config.max_strategies:
            raise ValueError("Maximum number of strategies reached")

        self.strategies[strategy_id] = strategy
        self.logger.info(f"Added strategy: {strategy_id}")

    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the engine"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.logger.info(f"Removed strategy: {strategy_id}")

    def enable_strategy(self, strategy_id: str) -> None:
        """Enable a strategy."""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            if hasattr(strategy, 'enabled'):
                strategy.enabled = True
            self.logger.info(f"Enabled strategy: {strategy_id}")

    def disable_strategy(self, strategy_id: str) -> None:
        """Disable a strategy."""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            if hasattr(strategy, 'enabled'):
                strategy.enabled = False
            self.logger.info(f"Disabled strategy: {strategy_id}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics from portfolio."""
        if self.portfolio and hasattr(self.portfolio, 'get_performance_metrics'):
            return self.portfolio.get_performance_metrics()
        return {}

    async def _main_loop(self) -> None:
        """Main trading engine event loop"""
        while self.state in [EngineState.RUNNING, EngineState.PAUSED]:
            try:
                if self.state == EngineState.RUNNING:
                    # Process market data
                    await self._process_market_data()

                    # Execute strategies
                    await self._execute_strategies()

                    # Check risk limits
                    await self._check_risk_limits()

                    # Process orders
                    await self._process_orders()

                await asyncio.sleep(self.config.tick_interval)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")

    async def _initialize_components(self) -> None:
        """Initialize engine components"""
        # TODO: Initialize position manager, risk manager, data manager
        pass

    async def _cleanup_components(self) -> None:
        """Cleanup engine components"""
        # TODO: Cleanup all components
        pass

    async def _process_market_data(self) -> None:
        """Process incoming market data"""
        # TODO: Implement market data processing
        pass

    async def _execute_strategies(self) -> None:
        """Execute trading strategies"""
        # TODO: Implement strategy execution
        pass

    async def _check_risk_limits(self) -> None:
        """Check risk limits and constraints"""
        # TODO: Implement risk checks
        pass

    async def _process_orders(self) -> None:
        """Process pending orders"""
        # TODO: Implement order processing
        pass

    async def _process_signals(self) -> None:
        """Process trading signals from strategies."""
        if not self.strategies or not self.data_handler:
            return

        try:
            # Get latest market data
            market_data = await self.data_handler.get_latest()

            # Process each strategy
            for _strategy_id, strategy in self.strategies.items():
                if hasattr(strategy, 'next'):
                    # Get signal from strategy
                    signal = strategy.next(market_data)

                    if signal and self.risk_manager:
                        # Check with risk manager
                        risk_check = self.risk_manager.pre_trade_check(signal)

                        if risk_check.get('approved', False):
                            # Convert signal to order and add to queue
                            order = self._signal_to_order(signal)
                            self.order_queue.put(order)

        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")

    async def _execute_orders(self) -> None:
        """Execute orders from the queue."""
        if not self.executor or self.order_queue.empty():
            return

        try:
            while not self.order_queue.empty():
                order = self.order_queue.get()

                # Execute order
                result = await self.executor.place_order(order)

                # Update portfolio if filled
                if result and result.get('status') == 'FILLED' and self.portfolio:
                    self.portfolio.process_fill(result)

        except Exception as e:
            self.logger.error(f"Error executing orders: {e}")

    async def _update_positions(self) -> None:
        """Update portfolio positions with current market prices."""
        if not self.portfolio or not self.data_handler:
            return

        try:
            # Get current market prices
            market_data = await self.data_handler.get_latest()

            # Extract prices from market data
            prices = {}
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'close' in data:
                    prices[symbol] = data['close']
                elif hasattr(data, 'get'):
                    prices[symbol] = data.get('close', 0)

            # Update portfolio positions
            if hasattr(self.portfolio, 'update_positions'):
                self.portfolio.update_positions(prices)
            else:
                # Fallback: Update each position individually
                for symbol, position in getattr(self.portfolio, 'positions', {}).items():
                    if symbol in prices and hasattr(position, 'update_price'):
                        position.update_price(prices[symbol])

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def _signal_to_order(self, signal: dict) -> dict:
        """Convert a trading signal to an order."""
        return {
            'symbol': signal.get('symbol'),
            'side': 'BUY' if signal.get('direction') == 'LONG' else 'SELL',
            'quantity': 100,  # Default quantity, should be calculated based on position sizing
            'order_type': 'MARKET',
            'timestamp': signal.get('timestamp')
        }

    def get_status(self) -> dict[str, Any]:
        """Get engine status"""
        status = {
            "state": self.state.value,
            "strategies": list(self.strategies.keys()),
            "active_orders": len(self.active_orders),
            "uptime": self._calculate_uptime(),
            "config": {
                "name": self.config.name,
                "tick_interval": self.config.tick_interval,
                "paper_trading": self.config.enable_paper_trading,
            },
        }

        # Add backward compatibility fields
        status["is_running"] = self.is_running
        status["active_strategies"] = len(self.strategies)
        status["timestamp"] = datetime.now()

        # Add portfolio info if available
        if self.portfolio:
            if hasattr(self.portfolio, 'get_portfolio_value'):
                status["portfolio_value"] = self.portfolio.get_portfolio_value()
            if hasattr(self.portfolio, 'positions'):
                status["position_count"] = len(self.portfolio.positions)

        return status

    def _calculate_uptime(self) -> float:
        """Calculate engine uptime in seconds"""
        # TODO: Implement uptime calculation
        return 0.0
