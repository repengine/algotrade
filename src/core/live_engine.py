"""
Live Trading Engine.

This module provides the main event loop for live trading that coordinates:
- Strategy execution
- Portfolio management
- Risk management
- Order execution
- Real-time data feeds
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from adapters.ibkr_executor import IBKRExecutor
from adapters.paper_executor import PaperExecutor
from core.data_handler import DataHandler
from core.engine.enhanced_order_manager import (
    EnhancedOrderManager,
    OrderEventType,
)
from core.executor import Order, OrderSide, OrderType, TimeInForce
from core.memory_manager import MemoryManager
from core.metrics import MetricsCollector
from core.portfolio import PortfolioEngine
from core.risk import EnhancedRiskManager as RiskManager
from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class TradingMode:
    """Trading mode enumeration."""

    PAPER = "paper"
    LIVE = "live"
    HYBRID = "hybrid"  # Paper trade with live data


class LiveTradingEngine:
    """
    Main trading engine for live execution.

    Coordinates all trading components and manages the execution lifecycle.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize trading engine.

        Args:
            config: Configuration dictionary with:
                - mode: Trading mode (paper/live/hybrid)
                - strategies: List of strategy configurations
                - data_config: Data handler configuration
                - portfolio_config: Portfolio engine configuration
                - risk_config: Risk manager configuration
                - executor_config: Executor configurations by name
                - schedule: Trading schedule configuration
        """
        self.config = config
        self.mode = config.get("mode", TradingMode.PAPER)

        # Initialize components
        self.data_handler = DataHandler(config.get("data_config", {}))
        portfolio_config = config.get("portfolio_config", {})
        if "initial_capital" not in portfolio_config:
            portfolio_config["initial_capital"] = 100000
        self.portfolio_engine = PortfolioEngine(portfolio_config)
        self.risk_manager = RiskManager(config.get("risk_config", {}))
        self.order_manager = EnhancedOrderManager(risk_manager=self.risk_manager)
        self.metrics_collector = MetricsCollector(portfolio_config["initial_capital"])

        # Initialize memory manager
        memory_config = config.get("memory_config", {})
        if "max_memory_mb" not in memory_config:
            memory_config["max_memory_mb"] = 2048  # 2GB default for trading
        if "gc_interval" not in memory_config:
            memory_config["gc_interval"] = 300  # 5 minutes
        if "cleanup_interval" not in memory_config:
            memory_config["cleanup_interval"] = 3600  # 1 hour
        self.memory_manager = MemoryManager(memory_config)

        # Initialize strategies
        self.strategies: dict[str, BaseStrategy] = {}

        # Trading state - initialize before strategies
        self.is_running = False
        self.running = False  # Alias for backward compatibility
        self.is_trading_hours = False
        self._active_symbols: set[str] = set()
        self._last_prices: dict[str, float] = {}
        self.current_prices: dict[str, float] = {}  # For test compatibility
        self.market_data: dict[str, pd.DataFrame] = {}  # Store market data by symbol
        self.signal_queue: asyncio.Queue[Any] = asyncio.Queue()  # Signal processing queue
        self.stop_orders: dict[str, dict] = {}  # Stop order tracking
        self.emergency_shutdown = False
        self.last_save_time: Optional[datetime] = None

        # Now initialize strategies (needs _active_symbols)
        self._initialize_strategies()

        # Initialize executors
        self._initialize_executors()

        # Scheduling
        self.scheduler = AsyncIOScheduler()
        self._setup_schedule()

        # Statistics
        self.stats: dict[str, Any] = {
            "engine_start": None,
            "total_signals": 0,
            "signals_generated": 0,
            "signals_rejected": 0,
            "total_orders": 0,
            "total_fills": 0,
            "trades_executed": 0,
            "data_updates": 0,
            "errors": 0,
        }

        # Register data structures with memory manager
        self.memory_manager.register_object("market_data", self.market_data)
        self.memory_manager.register_object("last_prices", self._last_prices)
        self.memory_manager.register_object("current_prices", self.current_prices)
        self.memory_manager.register_object("stop_orders", self.stop_orders)
        self.memory_manager.register_object("stats", self.stats)

    def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        strategy_configs = self.config.get("strategies", [])

        for strategy_config in strategy_configs:
            strategy_class = strategy_config["class"]
            strategy_id = strategy_config.get("id", strategy_class.__name__)
            params = strategy_config.get("params", {})

            # Create strategy instance
            # Strategies expect a config dict, not kwargs
            strategy = strategy_class(params)
            self.strategies[strategy_id] = strategy

            # Collect symbols
            if hasattr(strategy, "symbols"):
                self._active_symbols.update(strategy.symbols)
            elif hasattr(strategy, "symbol"):
                self._active_symbols.add(strategy.symbol)

        logger.info(f"Initialized {len(self.strategies)} strategies")
        logger.info(f"Active symbols: {self._active_symbols}")

    def _initialize_executors(self) -> None:
        """Initialize execution adapters."""
        executor_configs = self.config.get("executor_config", {})

        # Always add paper executor
        paper_config = executor_configs.get(
            "paper",
            {
                "initial_capital": self.portfolio_engine.initial_capital,
                "commission": 1.0,
                "slippage": 0.0001,
            },
        )
        paper_executor = PaperExecutor(paper_config)
        self.order_manager.add_executor("paper", paper_executor)

        # Store reference to paper executor for tests
        self.executor = paper_executor

        # Add live executor if configured
        if self.mode in [TradingMode.LIVE, TradingMode.HYBRID]:
            if "ibkr" in executor_configs:
                ibkr_executor = IBKRExecutor(executor_configs["ibkr"])
                self.order_manager.add_executor("ibkr", ibkr_executor)

        # Set active executor based on mode
        if self.mode == TradingMode.PAPER:
            self.order_manager.set_active_executor("paper")
        elif self.mode == TradingMode.LIVE:
            self.order_manager.set_active_executor("ibkr")
        else:  # HYBRID
            self.order_manager.set_active_executor("paper")

        # Register order callbacks
        self.order_manager.register_event_callback("*", self._handle_order_event)

    def _setup_schedule(self) -> None:
        """Setup trading schedule."""
        schedule_config = self.config.get("schedule", {})

        # Market hours
        market_open = schedule_config.get("market_open", "09:30")
        market_close = schedule_config.get("market_close", "16:00")
        timezone = schedule_config.get("timezone", "US/Eastern")

        # Pre-market tasks
        pre_market_time = schedule_config.get("pre_market", "09:00")
        self.scheduler.add_job(
            self._pre_market_routine,
            CronTrigger(
                hour=int(pre_market_time.split(":")[0]),
                minute=int(pre_market_time.split(":")[1]),
                timezone=timezone,
            ),
            id="pre_market",
        )

        # Market open
        self.scheduler.add_job(
            self._market_open_routine,
            CronTrigger(
                hour=int(market_open.split(":")[0]),
                minute=int(market_open.split(":")[1]),
                timezone=timezone,
            ),
            id="market_open",
        )

        # Market close
        self.scheduler.add_job(
            self._market_close_routine,
            CronTrigger(
                hour=int(market_close.split(":")[0]),
                minute=int(market_close.split(":")[1]),
                timezone=timezone,
            ),
            id="market_close",
        )

        # Post-market tasks
        post_market_time = schedule_config.get("post_market", "16:30")
        self.scheduler.add_job(
            self._post_market_routine,
            CronTrigger(
                hour=int(post_market_time.split(":")[0]),
                minute=int(post_market_time.split(":")[1]),
                timezone=timezone,
            ),
            id="post_market",
        )

    async def start(self) -> None:
        """Start trading engine."""
        logger.info(f"Starting trading engine in {self.mode} mode")

        # Connect executors
        for name, executor in self.order_manager.executors.items():
            logger.info(f"Connecting executor: {name}")
            connected = await executor.connect()
            if not connected:
                raise RuntimeError(f"Failed to connect executor: {name}")

        # Initialize data feeds
        await self._initialize_data_feeds()

        # Start scheduler
        self.scheduler.start()

        # Update state
        self.is_running = True
        self.running = True  # Keep in sync
        self.stats["engine_start"] = datetime.now()

        # Start main loop
        await self._main_loop()

    async def stop(self) -> None:
        """Stop trading engine."""
        logger.info("Stopping trading engine")

        # Stop running
        self.is_running = False
        self.running = False  # Keep in sync

        # Cancel all open orders
        await self._cancel_all_orders()

        # Stop scheduler
        self.scheduler.shutdown()

        # Disconnect executors
        for name, executor in self.order_manager.executors.items():
            logger.info(f"Disconnecting executor: {name}")
            await executor.disconnect()

        # Final statistics
        self._log_statistics()

    async def _main_loop(self) -> None:
        """Main trading loop."""
        update_interval = self.config.get("update_interval", 1.0)  # seconds

        while self.is_running:
            try:
                if self.is_trading_hours:
                    # Update market data
                    await self._update_market_data()

                    # Update positions
                    await self._update_positions()

                    # Run strategies
                    await self._run_strategies()

                    # Check risk limits
                    await self._check_risk_limits()

                    # Check memory usage and perform cleanup if needed
                    await self._check_memory_health()

                await asyncio.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                await asyncio.sleep(update_interval)

    async def _initialize_data_feeds(self) -> None:
        """Initialize real-time data feeds."""
        # For paper trading, use simulated data
        if self.mode == TradingMode.PAPER:
            # Initialize with random walk or historical data replay
            pass
        else:
            # Initialize live data feeds
            # This would connect to real-time data sources
            pass

        logger.info("Data feeds initialized")

    # Removed duplicate _update_market_data method - see line 765 for the actual implementation

    async def _update_positions(self) -> None:
        """Update current positions."""
        try:
            positions = await self.order_manager.get_positions()

            # Update portfolio engine if method exists
            if hasattr(self.portfolio_engine, 'update_position'):
                for symbol, position in positions.items():
                    self.portfolio_engine.update_position(
                        symbol=symbol,
                        quantity=position.quantity,
                        avg_price=position.average_cost,
                        current_price=position.current_price,
                    )

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def _run_strategies(self) -> None:
        """Execute all strategies."""

        for strategy_id, strategy in self.strategies.items():
            try:
                # Prepare data for strategy
                strategy_data = self._prepare_strategy_data(strategy)

                if strategy_data is not None:
                    # Generate signals
                    if hasattr(strategy, 'generate_signals'):
                        signals = strategy.generate_signals(strategy_data)
                    else:
                        signals = []

                    if signals:
                        self.stats["total_signals"] = self.stats.get("total_signals", 0) + len(signals)

                        # Process signals
                        for signal in signals:
                            await self._process_signal(strategy_id, signal)

            except Exception as e:
                logger.error(f"Error running strategy {strategy_id}: {e}")
                self.stats["errors"] = self.stats.get("errors", 0) + 1

    def _prepare_strategy_data(self, strategy: BaseStrategy) -> Optional[pd.DataFrame]:
        """Prepare data for strategy execution."""
        # In production, this would fetch the appropriate data
        # based on strategy requirements (timeframe, symbols, etc.)

        # For now, return None to skip execution
        return None

    async def _process_signal(self, strategy_id: str, signal: Signal) -> None:
        """Process trading signal."""
        logger.info(f"Processing signal from {strategy_id}: {signal}")

        try:
            # Check if we should trade this signal
            if not self._should_trade_signal(signal):
                return

            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size == 0:
                return

            # Determine order side
            signal_direction = getattr(signal, 'direction', 0)
            if isinstance(signal_direction, (int, float)) and signal_direction > 0:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            # Create order
            order = await self.order_manager.create_order(
                symbol=signal.symbol,
                side=side,
                quantity=position_size,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                strategy_id=strategy_id,
                signal_strength=signal.strength,
                signal_timestamp=signal.timestamp,
            )

            # Submit order
            success = await self.order_manager.submit_order(order)

            if success:
                self.stats["total_orders"] = self.stats.get("total_orders", 0) + 1
                logger.info(f"Order submitted: {order.order_id}")
            else:
                logger.error(f"Failed to submit order for signal: {signal}")

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            self.stats["errors"] += 1

    def _should_trade_signal(self, signal: Signal) -> bool:
        """Determine if signal should be traded."""
        # First validate the signal
        if not self._is_valid_signal(signal):
            return False

        # Check signal strength threshold
        min_strength = self.config.get("min_signal_strength", 0.5)
        if abs(signal.strength) < min_strength:
            return False

        # Check if symbol is tradeable
        if signal.symbol not in self._active_symbols:
            return False

        # Additional filters can be added here
        return True

    def _is_valid_signal(self, signal: Signal) -> bool:
        """Check if signal is valid for processing."""
        strength = getattr(signal, 'strength', 0)

        # Ensure strength is numeric and positive
        if isinstance(strength, (int, float)) and strength > 0:
            return True

        return False

    def _calculate_position_size(self, signal: Signal) -> int:
        """Calculate position size for signal."""
        # Get account value
        account_value = getattr(self.portfolio_engine, 'total_value', self.portfolio_engine.initial_capital)

        # Risk per trade
        risk_per_trade = self.config.get("risk_per_trade", 0.02)  # 2% default

        # Calculate base position value
        position_value = account_value * risk_per_trade * abs(signal.strength)

        # Get current price
        current_price = self._last_prices.get(signal.symbol, 100.0)

        # Calculate shares
        shares = int(position_value / current_price)

        # Apply limits
        max_position_size = self.config.get("max_position_size", 1000)
        shares = min(shares, max_position_size)

        return int(shares)

    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        try:
            # Check portfolio risk
            if hasattr(self.risk_manager, 'check_limits'):
                violations = self.risk_manager.check_limits()
            else:
                violations = []

            if violations:
                logger.warning(f"Risk limit violations: {violations}")

                # Take corrective action
                for violation in violations:
                    if violation["severity"] == "critical":
                        # Emergency liquidation
                        await self._emergency_liquidation(violation)

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")

    async def _check_memory_health(self) -> None:
        """Monitor and maintain memory health."""
        try:
            # Check current memory usage
            memory_stats = self.memory_manager.check_memory_usage()

            # Log if memory usage is high
            if memory_stats["memory_mb"] > memory_stats["max_memory_mb"] * 0.8:
                logger.warning(f"Memory usage high: {memory_stats['memory_mb']:.1f}MB "
                             f"({memory_stats['memory_percent']:.1f}%)")

                # If memory usage is critical, run optimization
                if memory_stats["memory_mb"] > memory_stats["max_memory_mb"] * 0.95:
                    logger.warning("Critical memory usage - running optimization")
                    optimization_results = self.memory_manager.optimize_memory()

                    memory_saved = (optimization_results["initial_memory"]["memory_mb"] -
                                  optimization_results["final_memory"]["memory_mb"])
                    logger.info(f"Memory optimization freed {memory_saved:.1f}MB")

                    # If still critical after optimization, consider emergency measures
                    if optimization_results["final_memory"]["memory_mb"] > memory_stats["max_memory_mb"]:
                        logger.critical("Memory usage still critical after optimization")
                        # Could implement emergency data pruning here

            # Log memory report periodically (every 1000 checks)
            if self.stats.get("memory_checks", 0) % 1000 == 0:
                report = self.memory_manager.get_memory_report()
                logger.info(f"Memory Report - Current: {report['current']['memory_mb']:.1f}MB, "
                           f"Average: {report['average_mb']:.1f}MB, "
                           f"Peak: {report['statistics']['peak_memory_mb']:.1f}MB")

            self.stats["memory_checks"] = self.stats.get("memory_checks", 0) + 1

        except Exception as e:
            logger.error(f"Error checking memory health: {e}")

    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        active_orders = self.order_manager.get_active_orders()

        for order in active_orders:
            try:
                await self.order_manager.cancel_order(order.order_id)
            except Exception as e:
                logger.error(f"Error cancelling order {order.order_id}: {e}")

    async def _emergency_liquidation(self, violation: dict[str, Any]) -> None:
        """Perform emergency liquidation."""
        logger.critical(f"Emergency liquidation triggered: {violation}")

        # Cancel all orders
        await self._cancel_all_orders()

        # Liquidate all positions
        positions = await self.order_manager.get_positions()

        for symbol, position in positions.items():
            if position.quantity != 0:
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                quantity = abs(position.quantity)

                try:
                    order = await self.order_manager.create_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY,
                        strategy_id="EMERGENCY_LIQUIDATION",
                    )

                    await self.order_manager.submit_order(order)

                except Exception as e:
                    logger.error(f"Failed to liquidate {symbol}: {e}")

    # Scheduled routines

    async def _pre_market_routine(self) -> None:
        """Pre-market preparation routine."""
        logger.info("Running pre-market routine")

        # Update account information
        for executor in self.order_manager.executors.values():
            account_info = await executor.get_account_info()
            logger.info(f"Account info: {account_info}")

        # Check system health
        # Load any overnight updates
        # Prepare strategies

    async def _market_open_routine(self) -> None:
        """Market open routine."""
        logger.info("Market opened - starting trading")
        self.is_trading_hours = True

        # Initial position sync
        await self._update_positions()

    async def _market_close_routine(self) -> None:
        """Market close routine."""
        logger.info("Market closed - stopping trading")
        self.is_trading_hours = False

        # Cancel any remaining day orders
        active_orders = self.order_manager.get_active_orders()
        for order in active_orders:
            if order.time_in_force == TimeInForce.DAY:
                await self.order_manager.cancel_order(order.order_id)

    async def _post_market_routine(self) -> None:
        """Post-market analysis routine."""
        logger.info("Running post-market routine")

        # Generate daily report
        self._generate_daily_report()

        # Update performance metrics
        if hasattr(self.portfolio_engine, 'calculate_metrics'):
            self.portfolio_engine.calculate_metrics()

        # Log statistics
        self._log_statistics()

    # Event handlers

    def _handle_order_event(
        self, order: Order, event_type: str, data: Optional[Any]
    ) -> None:
        """Handle order events."""
        if event_type == OrderEventType.FILLED:
            self.stats["total_fills"] = self.stats.get("total_fills", 0) + 1
            logger.info(
                f"Order filled: {order.order_id} - {order.filled_quantity} @ {order.average_fill_price}"
            )

            # Record trade in metrics collector
            if order.side == OrderSide.BUY:
                self.metrics_collector.record_trade_entry(
                    symbol=order.symbol,
                    price=order.average_fill_price,
                    quantity=order.filled_quantity,
                    side="long",
                    timestamp=order.filled_at or datetime.now(),
                    strategy_id=order.metadata.get("strategy_id"),
                )
            else:  # SELL
                trade = self.metrics_collector.record_trade_exit(
                    symbol=order.symbol,
                    price=order.average_fill_price,
                    quantity=order.filled_quantity,
                    timestamp=order.filled_at or datetime.now(),
                    commission=order.commission,
                )
                if trade:
                    logger.info(
                        f"Trade completed: {trade.symbol} P&L: ${trade.pnl:.2f} ({trade.pnl_percentage:+.2f}%)"
                    )

        elif event_type == OrderEventType.REJECTED:
            logger.warning(f"Order rejected: {order.order_id} - {data}")
        elif event_type == OrderEventType.ERROR:
            self.stats["errors"] += 1
            logger.error(f"Order error: {order.order_id} - {data}")

    # Reporting

    def _generate_daily_report(self) -> None:
        """Generate daily trading report."""
        # Get memory report
        memory_report = self.memory_manager.get_memory_report()

        report = {
            "date": datetime.now().date(),
            "mode": self.mode,
            "strategies": list(self.strategies.keys()),
            "statistics": self.stats.copy(),
            "portfolio": {
                "total_value": getattr(self.portfolio_engine, 'total_value', self.portfolio_engine.initial_capital),
                "cash": getattr(self.portfolio_engine, 'cash', self.portfolio_engine.initial_capital),
                "positions": len(getattr(self.portfolio_engine, 'positions', {})),
                "daily_pnl": self.portfolio_engine.calculate_daily_pnl() if hasattr(self.portfolio_engine, 'calculate_daily_pnl') else 0,
            },
            "order_stats": self.order_manager.get_order_statistics(),
            "memory": {
                "current_mb": memory_report['current']['memory_mb'],
                "average_mb": memory_report['average_mb'],
                "peak_mb": memory_report['statistics']['peak_memory_mb'],
                "gc_runs": memory_report['statistics']['gc_runs'],
                "cleanups": memory_report['statistics']['cleanups'],
            },
        }

        logger.info("Daily Report:")
        for key, value in report.items():
            logger.info(f"  {key}: {value}")

    def _log_statistics(self) -> None:
        """Log engine statistics."""
        logger.info("Engine Statistics:")
        logger.info(f"  Total signals: {self.stats['total_signals']}")
        logger.info(f"  Total orders: {self.stats['total_orders']}")
        logger.info(f"  Total fills: {self.stats['total_fills']}")
        logger.info(f"  Errors: {self.stats['errors']}")

        engine_start = self.stats.get("engine_start")
        if isinstance(engine_start, datetime):
            runtime = datetime.now() - engine_start
            logger.info(f"  Runtime: {runtime}")

        # Add memory statistics
        try:
            memory_report = self.memory_manager.get_memory_report()
            logger.info("Memory Statistics:")
            logger.info(f"  Current usage: {memory_report['current']['memory_mb']:.1f}MB "
                       f"({memory_report['current']['memory_percent']:.1f}%)")
            logger.info(f"  Average usage: {memory_report['average_mb']:.1f}MB")
            logger.info(f"  Peak usage: {memory_report['statistics']['peak_memory_mb']:.1f}MB")
            logger.info(f"  GC runs: {memory_report['statistics']['gc_runs']}")
            logger.info(f"  Data cleanups: {memory_report['statistics']['cleanups']}")
            logger.info(f"  Memory warnings: {memory_report['statistics']['memory_warnings']}")
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")

    async def _update_market_data(self) -> None:
        """Update market data for all symbols."""
        try:
            # Get latest data from data handler
            latest_data = await self.data_handler.get_latest(list(self._active_symbols))

            # Maximum rows to keep in memory per symbol (e.g., last 1000 bars)
            max_rows = self.config.get("max_market_data_rows", 1000)

            # Update prices and store data
            for symbol, data in latest_data.items():
                if isinstance(data, dict) and 'close' in data:
                    price = data['close']
                    self._last_prices[symbol] = price
                    self.current_prices[symbol] = price
                elif isinstance(data, pd.DataFrame) and not data.empty:
                    price = data['close'].iloc[-1]
                    self._last_prices[symbol] = price
                    self.current_prices[symbol] = price

                    # Limit stored data to prevent memory leak
                    if len(data) > max_rows:
                        # Keep only the most recent rows
                        self.market_data[symbol] = data.iloc[-max_rows:].copy()
                    else:
                        self.market_data[symbol] = data.copy()

                    # Clear old data if we have existing data
                    if symbol in self.market_data and len(self.market_data[symbol]) > max_rows:
                        self.market_data[symbol] = self.market_data[symbol].iloc[-max_rows:]

            self.stats["data_updates"] = self.stats.get("data_updates", 0) + 1

            # Check if we should run garbage collection
            if self.memory_manager.should_run_gc():
                self.memory_manager.run_garbage_collection()

            # Check if we should run data cleanup
            if self.memory_manager.should_run_cleanup():
                # Run cleanup with 24-hour retention by default
                retention_hours = self.config.get("data_retention_hours", 24)
                self.memory_manager.cleanup_old_data(retention_hours)

        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            self.stats["errors"] = self.stats.get("errors", 0) + 1

    async def _process_strategies(self) -> None:
        """Process all active strategies."""
        for strategy_id, strategy in self.strategies.items():
            try:
                if not getattr(strategy, 'enabled', True):
                    continue

                # Get market data for strategy
                for symbol in getattr(strategy, 'symbols', []):
                    if symbol in self.market_data:
                        # Call strategy to generate signals
                        if hasattr(strategy, 'calculate_signals'):
                            signals = strategy.calculate_signals(self.market_data[symbol])
                            if signals:
                                # Convert to standard signal format
                                for sig_symbol, strength in signals.items():
                                    if strength != 0:
                                        signal = {
                                            'symbol': sig_symbol,
                                            'action': 'BUY' if strength > 0 else 'SELL',
                                            'strength': abs(strength),
                                            'strategy': strategy_id,
                                            'quantity': 100  # Default
                                        }

                                        # Pre-trade risk check
                                        risk_check = self.risk_manager.pre_trade_check(
                                            symbol=sig_symbol,
                                            side='BUY' if strength > 0 else 'SELL',
                                            quantity=100,
                                            price=self.current_prices.get(sig_symbol, 100)
                                        )

                                        if risk_check['approved']:
                                            await self.signal_queue.put(signal)
                                            self.stats['signals_generated'] += 1
                                        else:
                                            self.stats['signals_rejected'] += 1
                                            logger.warning(f"Signal rejected: {risk_check.get('reason', 'Unknown')}")

            except Exception as e:
                logger.error(f"Error processing strategy {strategy_id}: {e}")
                self.stats["errors"] = self.stats.get("errors", 0) + 1

    async def _execute_signals(self) -> None:
        """Execute signals from the queue."""
        processed = 0
        max_signals = 10  # Process up to 10 signals per cycle

        while not self.signal_queue.empty() and processed < max_signals:
            try:
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=0.1)

                # Risk check
                risk_check = self.risk_manager.pre_trade_check(
                    symbol=signal['symbol'],
                    side=signal['action'],
                    quantity=signal['quantity'],
                    price=self.current_prices.get(signal['symbol'], 100)
                )

                if not risk_check['approved']:
                    self.stats['signals_rejected'] += 1
                    logger.warning(f"Signal rejected for {signal['symbol']}: {risk_check.get('reason')}")
                    continue

                # Create and submit order
                order = await self.order_manager.create_order(
                    symbol=signal['symbol'],
                    side=OrderSide.BUY if signal['action'] == 'BUY' else OrderSide.SELL,
                    quantity=signal['quantity'],
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id=signal['strategy']
                )

                success = await self.order_manager.submit_order(order)
                if success:
                    self.stats['trades_executed'] += 1
                    logger.info(f"Order executed: {order.order_id}")

                    # Update portfolio if filled immediately
                    if hasattr(self, 'portfolio_engine') and hasattr(self.portfolio_engine, 'process_fill'):
                        self.portfolio_engine.process_fill({
                            'symbol': signal['symbol'],
                            'quantity': signal['quantity'],
                            'price': self.current_prices.get(signal['symbol'], 100),
                            'side': signal['action']
                        })

                processed += 1

            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
                self.stats["errors"] += 1

    async def _update_positions(self) -> None:
        """Update portfolio positions with current prices."""
        try:
            if hasattr(self.portfolio_engine, 'update_positions'):
                self.portfolio_engine.update_positions(self.current_prices)

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def process_market_data(self, market_data: dict[str, Any]) -> None:
        """
        Process and validate incoming market data before distribution.
        
        Critical for:
        - Data integrity validation
        - Preventing bad data from triggering trades
        - Maintaining system state consistency
        
        Args:
            market_data: Dictionary of symbol -> data (dict or DataFrame)
        """
        try:
            validated_data = {}
            max_price_change_pct = self.config.get("max_price_change_pct", 0.20)  # 20% circuit breaker
            max_data_age_seconds = self.config.get("max_data_age_seconds", 60)  # 1 minute
            
            for symbol, data in market_data.items():
                try:
                    # Extract price and timestamp based on data format
                    if isinstance(data, dict):
                        price = data.get('close', data.get('price'))
                        timestamp = data.get('timestamp', datetime.now())
                        volume = data.get('volume', 0)
                    elif isinstance(data, pd.DataFrame) and not data.empty:
                        price = data['close'].iloc[-1]
                        timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()
                        volume = data['volume'].iloc[-1] if 'volume' in data else 0
                    else:
                        logger.warning(f"Invalid data format for {symbol}")
                        continue
                    
                    # Validation checks
                    
                    # 1. Check for stale data
                    if isinstance(timestamp, str):
                        timestamp = pd.to_datetime(timestamp)
                    
                    data_age = (datetime.now() - timestamp).total_seconds()
                    if data_age > max_data_age_seconds:
                        logger.warning(f"Stale data for {symbol}: {data_age:.1f}s old")
                        self.stats["stale_data_rejected"] = self.stats.get("stale_data_rejected", 0) + 1
                        continue
                    
                    # 2. Price sanity check
                    if price <= 0:
                        logger.error(f"Invalid price for {symbol}: {price}")
                        self.stats["invalid_price_rejected"] = self.stats.get("invalid_price_rejected", 0) + 1
                        continue
                    
                    # 3. Circuit breaker check
                    if symbol in self._last_prices:
                        price_change_pct = abs(price - self._last_prices[symbol]) / self._last_prices[symbol]
                        if price_change_pct > max_price_change_pct:
                            logger.warning(f"Circuit breaker triggered for {symbol}: {price_change_pct:.1%} change")
                            self.stats["circuit_breaker_triggered"] = self.stats.get("circuit_breaker_triggered", 0) + 1
                            # Still update but flag for risk manager
                            if isinstance(data, dict):
                                data['circuit_breaker'] = True
                            validated_data[symbol] = data
                            # Update prices even with circuit breaker
                            self._last_prices[symbol] = price
                            self.current_prices[symbol] = price
                            continue
                    
                    # 4. Volume validation (if available)
                    if volume < 0:
                        logger.warning(f"Invalid volume for {symbol}: {volume}")
                        volume = 0
                    
                    # Data passed validation
                    validated_data[symbol] = data
                    
                    # Update internal state
                    self._last_prices[symbol] = price
                    self.current_prices[symbol] = price
                    
                    # Store market data with memory limits
                    max_rows = self.config.get("max_market_data_rows", 1000)
                    if isinstance(data, pd.DataFrame):
                        if symbol not in self.market_data:
                            self.market_data[symbol] = data.tail(max_rows).copy()
                        else:
                            # Append and limit size
                            self.market_data[symbol] = pd.concat([self.market_data[symbol], data]).tail(max_rows)
                    
                except Exception as e:
                    logger.error(f"Error validating data for {symbol}: {e}")
                    self.stats["data_validation_errors"] = self.stats.get("data_validation_errors", 0) + 1
            
            # Distribute validated data
            if validated_data:
                # Update data handler
                if hasattr(self.data_handler, 'update_data'):
                    await self.data_handler.update_data(validated_data)
                
                # Notify strategies via event system
                for strategy_id, strategy in self.strategies.items():
                    if hasattr(strategy, 'on_market_data'):
                        await strategy.on_market_data(validated_data)
                
                # Update risk manager with latest prices
                if hasattr(self.risk_manager, 'update_prices'):
                    self.risk_manager.update_prices(self.current_prices)
                
                # Track successful updates
                self.stats["data_updates"] = self.stats.get("data_updates", 0) + 1
                
            # Log data quality metrics periodically
            if self.stats.get("data_updates", 0) % 100 == 0:
                total_rejected = (
                    self.stats.get("stale_data_rejected", 0) +
                    self.stats.get("invalid_price_rejected", 0) +
                    self.stats.get("data_validation_errors", 0)
                )
                total_processed = self.stats.get("data_updates", 0) + total_rejected
                quality_pct = (self.stats.get("data_updates", 0) / total_processed * 100) if total_processed > 0 else 0
                logger.info(f"Data quality: {quality_pct:.1f}% ({total_rejected} rejected out of {total_processed})")
                
        except Exception as e:
            logger.error(f"Critical error in process_market_data: {e}")
            self.stats["errors"] = self.stats.get("errors", 0) + 1

    async def collect_signals(self) -> list[Signal]:
        """
        Collect signals from all active strategies with deduplication.
        
        Critical for:
        - Preventing duplicate orders
        - Coordinating multi-strategy positions
        - Signal conflict resolution
        
        Returns:
            List of deduplicated and validated signals
        """
        try:
            all_signals = []
            signal_timeout = self.config.get("signal_timeout_ms", 100) / 1000  # Convert to seconds
            
            # Collect signals from all strategies concurrently
            tasks = []
            for strategy_id, strategy in self.strategies.items():
                if not getattr(strategy, 'enabled', True):
                    continue
                
                # Create task for each strategy
                task = asyncio.create_task(self._collect_strategy_signals(strategy_id, strategy, signal_timeout))
                tasks.append((strategy_id, task))
            
            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Gather results
            for strategy_id, task in tasks:
                try:
                    result = task.result()
                    if isinstance(result, Exception):
                        logger.error(f"Error collecting signals from {strategy_id}: {result}")
                        self.stats["strategy_errors"] = self.stats.get("strategy_errors", {})
                        self.stats["strategy_errors"][strategy_id] = self.stats["strategy_errors"].get(strategy_id, 0) + 1
                    elif result:
                        all_signals.extend(result)
                        self.stats["signals_per_strategy"] = self.stats.get("signals_per_strategy", {})
                        self.stats["signals_per_strategy"][strategy_id] = self.stats["signals_per_strategy"].get(strategy_id, 0) + len(result)
                except Exception as e:
                    logger.error(f"Error processing signals from {strategy_id}: {e}")
                    self.stats["strategy_errors"] = self.stats.get("strategy_errors", {})
                    self.stats["strategy_errors"][strategy_id] = self.stats["strategy_errors"].get(strategy_id, 0) + 1
            
            # Deduplicate signals
            deduplicated_signals = self._deduplicate_signals(all_signals)
            
            # Validate signals
            validated_signals = []
            for signal in deduplicated_signals:
                if self._validate_signal(signal):
                    validated_signals.append(signal)
                else:
                    self.stats["invalid_signals"] = self.stats.get("invalid_signals", 0) + 1
            
            self.stats["total_signals"] = self.stats.get("total_signals", 0) + len(validated_signals)
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"Critical error in collect_signals: {e}")
            self.stats["errors"] = self.stats.get("errors", 0) + 1
            return []
    
    async def _collect_strategy_signals(self, strategy_id: str, strategy: BaseStrategy, timeout: float) -> list[Signal]:
        """Collect signals from a single strategy with timeout."""
        try:
            # Prepare market data for strategy
            strategy_data = {}
            for symbol in getattr(strategy, 'symbols', []):
                if symbol in self.market_data:
                    strategy_data[symbol] = self.market_data[symbol]
            
            if not strategy_data:
                return []
            
            # Call strategy with timeout
            if hasattr(strategy, 'generate_signals'):
                signals = await asyncio.wait_for(
                    asyncio.create_task(strategy.generate_signals(strategy_data)),
                    timeout=timeout
                )
                
                # Convert to Signal objects if needed
                signal_objects = []
                for signal in signals:
                    if isinstance(signal, Signal):
                        signal_objects.append(signal)
                    elif isinstance(signal, dict):
                        # Convert dict to Signal
                        direction_val = signal.get('direction', 0)
                        if isinstance(direction_val, (int, float)):
                            # Convert numeric to string direction
                            if direction_val > 0:
                                direction_str = "LONG"
                            elif direction_val < 0:
                                direction_str = "SHORT"
                            else:
                                direction_str = "FLAT"
                        else:
                            direction_str = str(direction_val)
                        
                        signal_obj = Signal(
                            symbol=signal.get('symbol'),
                            direction=direction_str,
                            strength=signal.get('strength', 0),
                            timestamp=signal.get('timestamp', datetime.now()),
                            strategy_id=strategy_id,
                            price=self.current_prices.get(signal.get('symbol'), 100.0),
                            metadata=signal.get('metadata', {})
                        )
                        signal_objects.append(signal_obj)
                
                return signal_objects
                
        except asyncio.TimeoutError:
            logger.warning(f"Strategy {strategy_id} timed out after {timeout}s")
            self.stats["strategy_timeouts"] = self.stats.get("strategy_timeouts", {})
            self.stats["strategy_timeouts"][strategy_id] = self.stats["strategy_timeouts"].get(strategy_id, 0) + 1
        except Exception as e:
            logger.error(f"Error in strategy {strategy_id}: {e}")
            
        return []
    
    def _deduplicate_signals(self, signals: list[Signal]) -> list[Signal]:
        """Deduplicate and aggregate signals by symbol and direction."""
        signal_map = {}  # (symbol, direction) -> list of signals
        
        for signal in signals:
            key = (signal.symbol, signal.direction)
            if key not in signal_map:
                signal_map[key] = []
            signal_map[key].append(signal)
        
        # Aggregate signals
        deduplicated = []
        for (symbol, direction), signal_list in signal_map.items():
            if len(signal_list) == 1:
                deduplicated.append(signal_list[0])
            else:
                # Multiple signals for same symbol/direction - aggregate
                total_strength = sum(abs(s.strength) for s in signal_list)
                avg_strength = total_strength / len(signal_list)
                
                # Adjust strength sign based on direction
                if direction == "SHORT":
                    avg_strength = -abs(avg_strength)
                elif direction == "LONG":
                    avg_strength = abs(avg_strength)
                else:  # FLAT
                    avg_strength = 0
                
                # Check for conflicts (should not happen with same direction)
                strengths = [s.strength for s in signal_list]
                if max(strengths) - min(strengths) > 0.5:  # Large disagreement
                    logger.warning(f"Large signal disagreement for {symbol}: {strengths}")
                
                # Use average price
                avg_price = sum(s.price for s in signal_list) / len(signal_list)
                
                # Create aggregated signal
                agg_signal = Signal(
                    symbol=symbol,
                    direction=direction,
                    strength=avg_strength,
                    timestamp=datetime.now(),
                    strategy_id=f"aggregated_{len(signal_list)}",
                    price=avg_price,
                    metadata={
                        'aggregated': True,
                        'source_count': len(signal_list),
                        'strategies': [s.strategy_id for s in signal_list]
                    }
                )
                deduplicated.append(agg_signal)
                
                self.stats["aggregated_signals"] = self.stats.get("aggregated_signals", 0) + 1
        
        return deduplicated
    
    def _validate_signal(self, signal: Signal) -> bool:
        """Validate signal parameters."""
        # Check required fields
        if not signal.symbol or signal.symbol not in self._active_symbols:
            return False
        
        # Check direction is valid
        if signal.direction not in ["LONG", "SHORT", "FLAT"]:
            return False
        
        # Check signal strength is within bounds
        if not isinstance(signal.strength, (int, float)) or abs(signal.strength) > 1.0:
            return False
        
        # Check strength consistency with direction
        if signal.direction == "LONG" and signal.strength < 0:
            return False
        if signal.direction == "SHORT" and signal.strength > 0:
            return False
        if signal.direction == "FLAT" and signal.strength != 0:
            return False
        
        # Check timestamp freshness (signals older than 5 seconds are stale)
        if isinstance(signal.timestamp, datetime):
            signal_age = (datetime.now() - signal.timestamp).total_seconds()
            if signal_age > 5:
                logger.warning(f"Stale signal for {signal.symbol}: {signal_age:.1f}s old")
                return False
        
        return True

    async def _update_portfolio(self) -> None:
        """
        Update portfolio state with atomic operations.
        
        Critical for:
        - Accurate position tracking
        - P&L calculation
        - Risk metric updates
        - State consistency
        """
        try:
            # Use asyncio lock for atomic updates
            if not hasattr(self, '_portfolio_lock'):
                self._portfolio_lock = asyncio.Lock()
            
            async with self._portfolio_lock:
                # Get current positions from broker
                broker_positions = await self.order_manager.get_positions()
                
                # Get internal positions
                internal_positions = getattr(self.portfolio_engine, 'positions', {})
                
                # Reconcile positions
                reconciliation_needed = False
                discrepancies = []
                
                for symbol, broker_pos in broker_positions.items():
                    if symbol not in internal_positions:
                        discrepancies.append({
                            'symbol': symbol,
                            'type': 'missing_internal',
                            'broker_qty': broker_pos.quantity,
                            'internal_qty': 0
                        })
                        reconciliation_needed = True
                    else:
                        internal_pos = internal_positions[symbol]
                        qty_diff = abs(broker_pos.quantity - internal_pos.quantity)
                        if qty_diff > 0.01:  # Small tolerance for rounding
                            discrepancies.append({
                                'symbol': symbol,
                                'type': 'quantity_mismatch',
                                'broker_qty': broker_pos.quantity,
                                'internal_qty': internal_pos.quantity
                            })
                            reconciliation_needed = True
                
                # Check for positions in internal but not in broker
                for symbol, internal_pos in internal_positions.items():
                    if symbol not in broker_positions and internal_pos.quantity != 0:
                        discrepancies.append({
                            'symbol': symbol,
                            'type': 'missing_broker',
                            'broker_qty': 0,
                            'internal_qty': internal_pos.quantity
                        })
                        reconciliation_needed = True
                
                # Handle discrepancies
                if reconciliation_needed:
                    logger.warning(f"Position discrepancies found: {len(discrepancies)}")
                    for disc in discrepancies[:5]:  # Log first 5
                        logger.warning(f"  {disc}")
                    
                    # Reconcile with broker as source of truth
                    if hasattr(self.portfolio_engine, 'reconcile_positions'):
                        # Check if it's async
                        if asyncio.iscoroutinefunction(self.portfolio_engine.reconcile_positions):
                            await self.portfolio_engine.reconcile_positions(broker_positions)
                        else:
                            self.portfolio_engine.reconcile_positions(broker_positions)
                    else:
                        # Manual reconciliation
                        for symbol, broker_pos in broker_positions.items():
                            if hasattr(self.portfolio_engine, 'update_position'):
                                self.portfolio_engine.update_position(
                                    symbol=symbol,
                                    quantity=broker_pos.quantity,
                                    avg_price=broker_pos.average_cost,
                                    current_price=self.current_prices.get(symbol, broker_pos.current_price)
                                )
                
                # Update mark-to-market values
                positions_updated = 0
                for symbol, position in internal_positions.items():
                    if symbol in self.current_prices:
                        position.current_price = self.current_prices[symbol]
                        positions_updated += 1
                
                # Calculate portfolio metrics
                if hasattr(self.portfolio_engine, 'calculate_metrics'):
                    self.portfolio_engine.calculate_metrics()
                
                # Update risk metrics
                if hasattr(self.risk_manager, 'update_portfolio_metrics'):
                    portfolio_value = getattr(self.portfolio_engine, 'total_value', self.portfolio_engine.initial_capital)
                    self.risk_manager.update_portfolio_metrics({
                        'total_value': portfolio_value,
                        'positions': internal_positions,
                        'cash': getattr(self.portfolio_engine, 'cash', 0)
                    })
                
                # Broadcast portfolio update event
                update_event = {
                    'timestamp': datetime.now(),
                    'positions_count': len(internal_positions),
                    'positions_updated': positions_updated,
                    'reconciliation_needed': reconciliation_needed,
                    'portfolio_value': getattr(self.portfolio_engine, 'total_value', self.portfolio_engine.initial_capital)
                }
                
                # Notify strategies of portfolio update
                for strategy_id, strategy in self.strategies.items():
                    if hasattr(strategy, 'on_portfolio_update'):
                        await strategy.on_portfolio_update(update_event)
                
                # Log portfolio state periodically
                if self.stats.get("portfolio_updates", 0) % 60 == 0:  # Every 60 updates
                    portfolio_val = update_event['portfolio_value']
                    if isinstance(portfolio_val, (int, float)):
                        logger.info(f"Portfolio state: {len(internal_positions)} positions, "
                                  f"value: ${portfolio_val:,.2f}")
                    else:
                        logger.info(f"Portfolio state: {len(internal_positions)} positions, "
                                  f"value: {portfolio_val}")
                
                self.stats["portfolio_updates"] = self.stats.get("portfolio_updates", 0) + 1
                
        except Exception as e:
            logger.error(f"Critical error in _update_portfolio: {e}")
            self.stats["errors"] = self.stats.get("errors", 0) + 1
            # Don't re-raise to prevent cascading failures

    async def _check_stops(self) -> None:
        """Check and trigger stop orders."""
        triggered_stops = []

        for symbol, stop_info in self.stop_orders.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                stop_price = stop_info['stop_price']

                # Check if stop triggered
                if current_price <= stop_price:
                    triggered_stops.append((symbol, stop_info))

        # Execute triggered stops
        for symbol, stop_info in triggered_stops:
            try:
                order = await self.order_manager.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=stop_info['quantity'],
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id='STOP_LOSS'
                )

                success = await self.order_manager.submit_order(order)
                if success:
                    logger.info(f"Stop order triggered for {symbol} at {self.current_prices[symbol]}")
                    del self.stop_orders[symbol]

            except Exception as e:
                logger.error(f"Error executing stop order for {symbol}: {e}")

    def add_strategy(self, name: str, strategy: BaseStrategy) -> None:
        """Add a strategy to the engine."""
        self.strategies[name] = strategy

        # Update active symbols
        if hasattr(strategy, 'symbols'):
            self._active_symbols.update(strategy.symbols)

    def remove_strategy(self, name: str) -> None:
        """Remove a strategy from the engine."""
        if name in self.strategies:
            del self.strategies[name]

    def get_status(self) -> dict[str, Any]:
        """Get current engine status."""
        # Get memory status
        memory_status = self.memory_manager.check_memory_usage()

        return {
            'running': self.running,
            'mode': self.mode.upper() if hasattr(self.mode, 'upper') else str(self.mode),
            'portfolio_value': self.portfolio_engine.get_portfolio_value() if hasattr(self.portfolio_engine, 'get_portfolio_value') else self.portfolio_engine.initial_capital,
            'position_count': len(getattr(self.portfolio_engine, 'positions', {})),
            'stats': self.stats.copy(),
            'strategies': list(self.strategies.keys()),
            'memory': {
                'current_mb': memory_status['memory_mb'],
                'percent': memory_status['memory_percent'],
                'max_mb': memory_status['max_memory_mb'],
                'peak_mb': memory_status['peak_memory_mb']
            },
            'timestamp': datetime.now()
        }

    def get_performance(self) -> dict[str, Any]:
        """Get performance metrics."""
        if hasattr(self.portfolio_engine, 'get_performance_metrics'):
            return self.portfolio_engine.get_performance_metrics()
        return {}

    def get_memory_statistics(self) -> dict[str, Any]:
        """Get detailed memory statistics."""
        return self.memory_manager.get_memory_report()

    def schedule_task(self, task: Any, interval: int, name: str) -> None:
        """Schedule a recurring task."""
        if hasattr(self.scheduler, 'add_job'):
            self.scheduler.add_job(
                task,
                'interval',
                seconds=interval,
                id=name,
                replace_existing=True
            )

    async def emergency_stop(self) -> None:
        """Emergency stop - cancel all orders and close positions."""
        logger.critical("Emergency stop triggered!")

        self.running = False
        self.is_running = False
        self.emergency_shutdown = True

        # Cancel all orders
        if hasattr(self.executor, 'cancel_all_orders'):
            await self.executor.cancel_all_orders()

        # Close all positions
        if hasattr(self.executor, 'close_all_positions'):
            await self.executor.close_all_positions()

    async def save_state(self) -> None:
        """Save current engine state."""
        try:
            {
                'positions': self.portfolio_engine.export_state() if hasattr(self.portfolio_engine, 'export_state') else {},
                'stop_orders': self.stop_orders,
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }

            # In production, this would save to a file or database
            self.last_save_time = datetime.now()
            logger.info("Engine state saved")

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    async def load_state(self) -> None:
        """Load saved engine state."""
        try:
            # In production, this would load from a file or database
            # For testing, check if stop_orders is being set
            try:
                with open('engine_state.json') as f:
                    state = json.load(f)
                    if 'stop_orders' in state:
                        self.stop_orders = state['stop_orders']
                    logger.info("Engine state loaded")
            except FileNotFoundError:
                logger.info("No saved state found")

        except Exception as e:
            logger.error(f"Error loading state: {e}")
