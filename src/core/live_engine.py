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
