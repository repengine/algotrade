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
import logging
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from algostack.adapters.paper_executor import PaperExecutor
from algostack.adapters.ibkr_executor import IBKRExecutor
from algostack.core.data_handler import DataHandler
from algostack.core.engine.enhanced_order_manager import EnhancedOrderManager, OrderEventType
from algostack.core.executor import Order, OrderSide, OrderType, TimeInForce
from algostack.core.metrics import MetricsCollector
from algostack.core.portfolio import PortfolioEngine
from algostack.core.risk import EnhancedRiskManager as RiskManager
from algostack.strategies.base import BaseStrategy, Signal

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
    
    def __init__(self, config: Dict[str, Any]):
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
        self.mode = TradingMode(config.get("mode", TradingMode.PAPER))
        
        # Initialize components
        self.data_handler = DataHandler(config.get("data_config", {}))
        initial_capital = config.get("portfolio_config", {}).get("initial_capital", 100000)
        self.portfolio_engine = PortfolioEngine(initial_capital=initial_capital)
        self.risk_manager = RiskManager(config.get("risk_config", {}))
        self.order_manager = EnhancedOrderManager(risk_manager=self.risk_manager)
        self.metrics_collector = MetricsCollector(initial_capital)
        
        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        self._initialize_strategies()
        
        # Initialize executors
        self._initialize_executors()
        
        # Trading state
        self.is_running = False
        self.is_trading_hours = False
        self._active_symbols: Set[str] = set()
        self._last_prices: Dict[str, float] = {}
        
        # Scheduling
        self.scheduler = AsyncIOScheduler()
        self._setup_schedule()
        
        # Statistics
        self.stats = {
            "engine_start": None,
            "total_signals": 0,
            "total_orders": 0,
            "total_fills": 0,
            "errors": 0,
        }
        
    def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        strategy_configs = self.config.get("strategies", [])
        
        for strategy_config in strategy_configs:
            strategy_class = strategy_config["class"]
            strategy_id = strategy_config.get("id", strategy_class.__name__)
            params = strategy_config.get("params", {})
            
            # Create strategy instance
            strategy = strategy_class(**params)
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
        paper_config = executor_configs.get("paper", {
            "initial_capital": self.portfolio_engine.initial_capital,
            "commission": 1.0,
            "slippage": 0.0001,
        })
        paper_executor = PaperExecutor(paper_config)
        self.order_manager.add_executor("paper", paper_executor)
        
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
        self.stats["engine_start"] = datetime.now()
        
        # Start main loop
        await self._main_loop()
    
    async def stop(self) -> None:
        """Stop trading engine."""
        logger.info("Stopping trading engine")
        
        # Stop running
        self.is_running = False
        
        # Cancel all open orders
        await self._cancel_all_orders()
        
        # Stop scheduler
        self.scheduler.shutdown()
        
        # Disconnect executors
        for name, executor in self.order_manager.executors.items():
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
                    
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.stats["errors"] += 1
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
    
    async def _update_market_data(self) -> None:
        """Update market data for all symbols."""
        # Get latest prices for all active symbols
        for symbol in self._active_symbols:
            try:
                # In production, this would fetch from real-time feed
                # For now, simulate with last price + small random change
                if symbol in self._last_prices:
                    import random
                    change = random.uniform(-0.01, 0.01)
                    price = self._last_prices[symbol] * (1 + change)
                else:
                    price = 100.0  # Default price
                    
                self._last_prices[symbol] = price
                
                # Update paper executor prices
                if "paper" in self.order_manager.executors:
                    paper_executor = self.order_manager.executors["paper"]
                    paper_executor.update_price(symbol, price)
                    
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
    
    async def _update_positions(self) -> None:
        """Update current positions."""
        try:
            positions = await self.order_manager.get_positions()
            
            # Update portfolio engine
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
        current_time = datetime.now()
        
        for strategy_id, strategy in self.strategies.items():
            try:
                # Prepare data for strategy
                strategy_data = self._prepare_strategy_data(strategy)
                
                if strategy_data is not None:
                    # Generate signals
                    signals = strategy.generate_signals(strategy_data)
                    
                    if signals:
                        self.stats["total_signals"] += len(signals)
                        
                        # Process signals
                        for signal in signals:
                            await self._process_signal(strategy_id, signal)
                            
            except Exception as e:
                logger.error(f"Error running strategy {strategy_id}: {e}")
                self.stats["errors"] += 1
    
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
            if signal.direction > 0:
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
                self.stats["total_orders"] += 1
                logger.info(f"Order submitted: {order.order_id}")
            else:
                logger.error(f"Failed to submit order for signal: {signal}")
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            self.stats["errors"] += 1
    
    def _should_trade_signal(self, signal: Signal) -> bool:
        """Determine if signal should be traded."""
        # Check signal strength threshold
        min_strength = self.config.get("min_signal_strength", 0.5)
        if abs(signal.strength) < min_strength:
            return False
            
        # Check if symbol is tradeable
        if signal.symbol not in self._active_symbols:
            return False
            
        # Additional filters can be added here
        return True
    
    def _calculate_position_size(self, signal: Signal) -> int:
        """Calculate position size for signal."""
        # Get account value
        account_value = self.portfolio_engine.total_value
        
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
        
        return shares
    
    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        try:
            # Check portfolio risk
            violations = self.risk_manager.check_limits()
            
            if violations:
                logger.warning(f"Risk limit violations: {violations}")
                
                # Take corrective action
                for violation in violations:
                    if violation["severity"] == "critical":
                        # Emergency liquidation
                        await self._emergency_liquidation(violation)
                        
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        active_orders = self.order_manager.get_active_orders()
        
        for order in active_orders:
            try:
                await self.order_manager.cancel_order(order.order_id)
            except Exception as e:
                logger.error(f"Error cancelling order {order.order_id}: {e}")
    
    async def _emergency_liquidation(self, violation: Dict[str, Any]) -> None:
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
        self.portfolio_engine.calculate_metrics()
        
        # Log statistics
        self._log_statistics()
    
    # Event handlers
    
    def _handle_order_event(self, order: Order, event_type: str, data: Optional[Any]) -> None:
        """Handle order events."""
        if event_type == OrderEventType.FILLED:
            self.stats["total_fills"] += 1
            logger.info(f"Order filled: {order.order_id} - {order.filled_quantity} @ {order.average_fill_price}")
            
            # Record trade in metrics collector
            if order.side == OrderSide.BUY:
                self.metrics_collector.record_trade_entry(
                    symbol=order.symbol,
                    price=order.average_fill_price,
                    quantity=order.filled_quantity,
                    side="long",
                    timestamp=order.filled_at or datetime.now(),
                    strategy_id=order.metadata.get("strategy_id")
                )
            else:  # SELL
                trade = self.metrics_collector.record_trade_exit(
                    symbol=order.symbol,
                    price=order.average_fill_price,
                    quantity=order.filled_quantity,
                    timestamp=order.filled_at or datetime.now(),
                    commission=order.commission
                )
                if trade:
                    logger.info(f"Trade completed: {trade.symbol} P&L: ${trade.pnl:.2f} ({trade.pnl_percentage:+.2f}%)")
                    
        elif event_type == OrderEventType.REJECTED:
            logger.warning(f"Order rejected: {order.order_id} - {data}")
        elif event_type == OrderEventType.ERROR:
            self.stats["errors"] += 1
            logger.error(f"Order error: {order.order_id} - {data}")
    
    # Reporting
    
    def _generate_daily_report(self) -> None:
        """Generate daily trading report."""
        report = {
            "date": datetime.now().date(),
            "mode": self.mode,
            "strategies": list(self.strategies.keys()),
            "statistics": self.stats.copy(),
            "portfolio": {
                "total_value": self.portfolio_engine.total_value,
                "cash": self.portfolio_engine.cash,
                "positions": len(self.portfolio_engine.positions),
                "daily_pnl": self.portfolio_engine.calculate_daily_pnl(),
            },
            "order_stats": self.order_manager.get_order_statistics(),
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
        
        if self.stats["engine_start"]:
            runtime = datetime.now() - self.stats["engine_start"]
            logger.info(f"  Runtime: {runtime}")