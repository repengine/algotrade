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

            # Start main event loop
            self._main_loop_task = asyncio.create_task(self._main_loop())

            self.state = EngineState.RUNNING
            self.is_running = True  # For backward compatibility
            self.start_time = datetime.now()  # Track engine start time
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
            if hasattr(strategy, "enabled"):
                strategy.enabled = True
            self.logger.info(f"Enabled strategy: {strategy_id}")

    def disable_strategy(self, strategy_id: str) -> None:
        """Disable a strategy."""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            if hasattr(strategy, "enabled"):
                strategy.enabled = False
            self.logger.info(f"Disabled strategy: {strategy_id}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics from portfolio."""
        if self.portfolio and hasattr(self.portfolio, "get_performance_metrics"):
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
        # Initialize components if they have async initialization
        if self.data_handler and hasattr(self.data_handler, "initialize"):
            if asyncio.iscoroutinefunction(self.data_handler.initialize):
                await self.data_handler.initialize()
            else:
                self.data_handler.initialize()

        if self.executor and hasattr(self.executor, "connect"):
            if asyncio.iscoroutinefunction(self.executor.connect):
                await self.executor.connect()
            else:
                self.executor.connect()

        if self.risk_manager and hasattr(self.risk_manager, "initialize"):
            if asyncio.iscoroutinefunction(self.risk_manager.initialize):
                await self.risk_manager.initialize()
            else:
                self.risk_manager.initialize()

        # Initialize strategies
        for strategy_id, strategy in self.strategies.items():
            if hasattr(strategy, "initialize"):
                if asyncio.iscoroutinefunction(strategy.initialize):
                    await strategy.initialize()
                else:
                    strategy.initialize()
                self.logger.info(f"Initialized strategy: {strategy_id}")

    async def _cleanup_components(self) -> None:
        """Cleanup engine components"""
        # Cleanup strategies first
        for strategy_id, strategy in self.strategies.items():
            if hasattr(strategy, "cleanup"):
                try:
                    if asyncio.iscoroutinefunction(strategy.cleanup):
                        await strategy.cleanup()
                    else:
                        strategy.cleanup()
                    self.logger.info(f"Cleaned up strategy: {strategy_id}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up strategy {strategy_id}: {e}")

        # Cleanup components
        if self.executor and hasattr(self.executor, "disconnect"):
            try:
                if asyncio.iscoroutinefunction(self.executor.disconnect):
                    await self.executor.disconnect()
                else:
                    self.executor.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting executor: {e}")

        if self.data_handler and hasattr(self.data_handler, "cleanup"):
            try:
                if asyncio.iscoroutinefunction(self.data_handler.cleanup):
                    await self.data_handler.cleanup()
                else:
                    self.data_handler.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up data handler: {e}")

        if self.risk_manager and hasattr(self.risk_manager, "cleanup"):
            try:
                if asyncio.iscoroutinefunction(self.risk_manager.cleanup):
                    await self.risk_manager.cleanup()
                else:
                    self.risk_manager.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up risk manager: {e}")

    async def _process_market_data(self) -> None:
        """Process incoming market data"""
        if not self.data_handler:
            return

        try:
            # Get latest market data
            market_data = await self.data_handler.get_latest()

            if market_data:
                # Update portfolio with latest prices
                if self.portfolio:
                    prices = {}
                    for symbol, data in market_data.items():
                        if isinstance(data, dict) and "close" in data:
                            prices[symbol] = data["close"]
                        elif hasattr(data, "get"):
                            prices[symbol] = data.get("close", 0)

                    if prices and hasattr(self.portfolio, "update_prices"):
                        self.portfolio.update_prices(prices)

                # Update strategies with market data
                for strategy_id, strategy in self.strategies.items():
                    if hasattr(strategy, "on_market_data"):
                        await strategy.on_market_data(market_data)

        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")

    async def _execute_strategies(self) -> None:
        """Execute trading strategies"""
        if not self.strategies or not self.data_handler:
            return

        try:
            # Get latest market data
            market_data = await self.data_handler.get_latest()

            if not market_data:
                return

            # Execute each active strategy
            for strategy_id, strategy in self.strategies.items():
                try:
                    # Skip disabled strategies
                    if hasattr(strategy, "enabled") and not strategy.enabled:
                        continue

                    # Generate signals
                    if hasattr(strategy, "generate_signals"):
                        signals = await strategy.generate_signals(market_data)

                        # Process each signal
                        if signals:
                            for signal in signals:
                                # Validate signal with risk manager
                                if self.risk_manager:
                                    risk_check = (
                                        await self.risk_manager.validate_signal(signal)
                                    )
                                    if not risk_check.get("approved", False):
                                        self.logger.warning(
                                            f"Signal rejected by risk manager: {risk_check.get('reason', 'Unknown')}"
                                        )
                                        continue

                                # Convert signal to order
                                order = self._signal_to_order(signal)
                                order["strategy_id"] = strategy_id

                                # Add to order queue
                                self.order_queue.put(order)
                                self.logger.info(
                                    f"Generated order from strategy {strategy_id}: {order}"
                                )

                except Exception as e:
                    self.logger.error(f"Error executing strategy {strategy_id}: {e}")

        except Exception as e:
            self.logger.error(f"Error in strategy execution: {e}")

    async def _check_risk_limits(self) -> None:
        """Check risk limits and constraints"""
        if not self.risk_manager or not self.portfolio:
            return

        try:
            # Get current portfolio state
            portfolio_state = {
                "total_equity": getattr(self.portfolio, "total_equity", 0),
                "positions": getattr(self.portfolio, "positions", {}),
                "cash": getattr(self.portfolio, "cash", 0),
                "buying_power": getattr(self.portfolio, "buying_power", 0),
            }

            # Check portfolio-level risk limits
            if hasattr(self.risk_manager, "check_portfolio_limits"):
                risk_status = await self.risk_manager.check_portfolio_limits(
                    portfolio_state
                )

                if risk_status.get("breached", False):
                    self.logger.warning(
                        f"Risk limits breached: {risk_status.get('reasons', [])}"
                    )

                    # Handle risk breach
                    if risk_status.get("action") == "HALT_TRADING":
                        self.logger.error("HALTING TRADING due to risk breach")
                        self.state = EngineState.PAUSED
                        # Cancel all pending orders
                        while not self.order_queue.empty():
                            self.order_queue.get()

                    elif risk_status.get("action") == "REDUCE_POSITIONS":
                        self.logger.warning("Reducing positions due to risk breach")
                        # Queue position reduction orders
                        if hasattr(self.risk_manager, "generate_reduction_orders"):
                            reduction_orders = (
                                await self.risk_manager.generate_reduction_orders(
                                    portfolio_state
                                )
                            )
                            for order in reduction_orders:
                                self.order_queue.put(order)

            # Check individual position limits
            for symbol, position in portfolio_state["positions"].items():
                if hasattr(self.risk_manager, "check_position_limits"):
                    position_status = await self.risk_manager.check_position_limits(
                        symbol, position
                    )

                    if position_status.get("breached", False):
                        self.logger.warning(
                            f"Position limits breached for {symbol}: {position_status.get('reason')}"
                        )

                        # Generate exit order if needed
                        if position_status.get("action") == "EXIT_POSITION":
                            exit_order = {
                                "symbol": symbol,
                                "side": "SELL"
                                if position.get("side") == "LONG"
                                else "BUY",
                                "quantity": abs(position.get("quantity", 0)),
                                "order_type": "MARKET",
                                "reason": "RISK_LIMIT_BREACH",
                            }
                            self.order_queue.put(exit_order)

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")

    async def _process_orders(self) -> None:
        """Process pending orders"""
        if not self.executor or self.order_queue.empty():
            return

        orders_to_process = []

        # Collect all pending orders
        while not self.order_queue.empty():
            try:
                order = self.order_queue.get_nowait()
                orders_to_process.append(order)
            except queue.Empty:
                break

        # Process each order
        for order in orders_to_process:
            try:
                # Final risk check before execution
                if self.risk_manager and hasattr(self.risk_manager, "pre_trade_check"):
                    risk_check = await self.risk_manager.pre_trade_check(order)
                    if not risk_check.get("approved", False):
                        self.logger.warning(
                            f"Order rejected by pre-trade risk check: {risk_check.get('reason', 'Unknown')}"
                        )
                        continue

                # Execute the order
                self.logger.info(f"Executing order: {order}")
                result = await self.executor.place_order(order)

                # Handle execution result
                if result:
                    if result.get("status") == "FILLED":
                        self.logger.info(f"Order filled: {result}")

                        # Update portfolio
                        if self.portfolio and hasattr(self.portfolio, "process_fill"):
                            await self.portfolio.process_fill(result)

                        # Notify risk manager
                        if self.risk_manager and hasattr(
                            self.risk_manager, "on_order_filled"
                        ):
                            await self.risk_manager.on_order_filled(result)

                        # Notify strategy
                        strategy_id = order.get("strategy_id")
                        if strategy_id and strategy_id in self.strategies:
                            strategy = self.strategies[strategy_id]
                            if hasattr(strategy, "on_order_filled"):
                                await strategy.on_order_filled(result)

                    elif result.get("status") == "REJECTED":
                        self.logger.error(
                            f"Order rejected: {result.get('reason', 'Unknown')}"
                        )

                    elif result.get("status") in ["PENDING", "SUBMITTED"]:
                        self.logger.info(f"Order submitted: {result}")
                        # Track active orders
                        if hasattr(self, "active_orders"):
                            self.active_orders[result.get("order_id")] = result
                else:
                    self.logger.error(f"Failed to execute order: {order}")

            except Exception as e:
                self.logger.error(f"Error processing order {order}: {e}")
                # Put order back in queue for retry if appropriate
                if order.get("retry_count", 0) < 3:
                    order["retry_count"] = order.get("retry_count", 0) + 1
                    self.order_queue.put(order)

    async def _process_signals(self) -> None:
        """Process trading signals from strategies."""
        if not self.strategies or not self.data_handler:
            return

        try:
            # Get latest market data
            market_data = await self.data_handler.get_latest()

            # Process each strategy
            for _strategy_id, strategy in self.strategies.items():
                if hasattr(strategy, "next"):
                    # Get signal from strategy
                    signal = strategy.next(market_data)

                    if signal and self.risk_manager:
                        # Check with risk manager
                        risk_check = self.risk_manager.pre_trade_check(signal)

                        if risk_check.get("approved", False):
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
                if result and result.get("status") == "FILLED" and self.portfolio:
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
                if isinstance(data, dict) and "close" in data:
                    prices[symbol] = data["close"]
                elif hasattr(data, "get"):
                    prices[symbol] = data.get("close", 0)

            # Update portfolio positions
            if hasattr(self.portfolio, "update_positions"):
                self.portfolio.update_positions(prices)
            else:
                # Fallback: Update each position individually
                for symbol, position in getattr(
                    self.portfolio, "positions", {}
                ).items():
                    if symbol in prices and hasattr(position, "update_price"):
                        position.update_price(prices[symbol])

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def _signal_to_order(self, signal: dict) -> dict:
        """Convert a trading signal to an order."""
        return {
            "symbol": signal.get("symbol"),
            "side": "BUY" if signal.get("direction") == "LONG" else "SELL",
            "quantity": 100,  # Default quantity, should be calculated based on position sizing
            "order_type": "MARKET",
            "timestamp": signal.get("timestamp"),
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
            if hasattr(self.portfolio, "get_portfolio_value"):
                status["portfolio_value"] = self.portfolio.get_portfolio_value()
            if hasattr(self.portfolio, "positions"):
                status["position_count"] = len(self.portfolio.positions)

        return status

    def _calculate_uptime(self) -> float:
        """Calculate engine uptime in seconds"""
        if hasattr(self, "start_time") and self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
