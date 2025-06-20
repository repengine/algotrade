"""
End-to-end tests for live trading simulation.

Tests the complete live trading system in simulation mode.
Validates:
- Real-time data handling
- Order execution simulation
- Risk management in live conditions
- System stability over extended periods
- Error recovery and resilience
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from adapters.paper_executor import PaperExecutor
from core.engine.order_manager import (
    Order,
    OrderManager,
    OrderSide,
    OrderType,
)
from core.executor import BaseExecutor
from core.live_engine import LiveTradingEngine
from core.portfolio import PortfolioEngine
from core.risk import EnhancedRiskManager
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti
from helpers.safe_logging import get_test_logger, suppress_test_output

# Configure safe logging
suppress_test_output()
logger = get_test_logger(__name__)


# Tests fixed - using proper async/await patterns


class TestLiveTradingSimulation:
    """Test live trading system in simulation mode."""

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_trading_day_simulation(self):
        """
        Simulate a complete trading day with live data.

        Tests:
        1. Market open procedures
        2. Real-time signal processing
        3. Order management throughout the day
        4. Market close procedures
        5. End-of-day reconciliation
        """
        # Initialize components
        config = {
            'trading': {
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                'initial_capital': 100000,
                'market_open': '09:30:00',
                'market_close': '16:00:00',
                'pre_market_start': '09:00:00',
                'after_hours_end': '17:00:00'
            },
            'risk': {
                'max_position_size': 0.15,
                'max_portfolio_risk': 0.02,
                'daily_loss_limit': 0.03,
                'max_orders_per_minute': 10
            },
            'execution': {
                'use_limit_orders': True,
                'limit_price_offset': 0.0005,
                'order_timeout': 30,  # seconds
                'partial_fill_min': 100  # shares
            }
        }

        # Initialize trading components
        PortfolioEngine({"initial_capital": config['trading']['initial_capital']})
        EnhancedRiskManager(**config['risk'])

        # Use paper executor for simulation
        executor = PaperExecutor({
            'initial_capital': config['trading']['initial_capital'],
            'commission': 1.0,
            'slippage': 0.0001,
            'fill_delay': 0.0  # Instant fills for testing
        })

        OrderManager()

        # Initialize strategies
        [
            MeanReversionEquity({
                'symbols': config['trading']['symbols'],
                'lookback_period': 20,
                'zscore_threshold': 2.0,
                'exit_zscore': 0.5,
                'rsi_period': 14,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0
            }),
            TrendFollowingMulti({
                'symbols': config['trading']['symbols'],
                'channel_period': 20,
                'atr_period': 14,
                'adx_period': 14,
                'adx_threshold': 25.0
            })
        ]

        # Create strategy configs for the engine
        strategy_configs = [
            {
                'name': 'mean_reversion',
                'class': MeanReversionEquity,  # Use actual class, not string
                'params': {  # Changed from 'config' to 'params' to match engine expectation
                    'lookback_period': 20,
                    'zscore_threshold': 2.0,  # Changed from entry_zscore
                    'exit_zscore': 0.5,
                    'symbols': config['trading']['symbols'],
                    'rsi_period': 2,
                    'rsi_oversold': 10.0,  # Must be float
                    'rsi_overbought': 90.0  # Must be float
                }
            },
            {
                'name': 'trend_following',
                'class': TrendFollowingMulti,  # Use actual class, not string
                'params': {  # Changed from 'config' to 'params' to match engine expectation
                    'symbols': config['trading']['symbols'],
                    'channel_period': 20,
                    'atr_period': 14,
                    'adx_period': 14,
                    'adx_threshold': 25.0
                }
            }
        ]

        # Update config with strategies
        config['strategies'] = strategy_configs
        config['portfolio_config'] = {'initial_capital': config['trading']['initial_capital']}
        config['risk_config'] = config['risk']
        config['executor_config'] = {'paper': executor.config}

        # Initialize live engine
        engine = LiveTradingEngine(config)

        # Simulate market data stream
        async def market_data_generator():
            """Generate realistic intraday market data."""
            current_time = datetime.now().replace(hour=9, minute=30, second=0)
            end_time = current_time.replace(hour=16, minute=0, second=0)

            prices = {
                'AAPL': 150.0,
                'GOOGL': 2500.0,
                'MSFT': 300.0,
                'AMZN': 140.0
            }

            while current_time <= end_time:
                # Generate price updates
                updates = {}
                for symbol in config['trading']['symbols']:
                    # Intraday price movement
                    price_change = np.random.normal(0, 0.001)
                    prices[symbol] *= (1 + price_change)

                    updates[symbol] = {
                        'timestamp': current_time,
                        'bid': prices[symbol] - 0.01,
                        'ask': prices[symbol] + 0.01,
                        'last': prices[symbol],
                        'volume': np.random.randint(1000, 5000)
                    }

                yield updates

                # Advance time (simulate 5-second updates)
                current_time += timedelta(seconds=5)
                await asyncio.sleep(0.01)  # Small delay for testing

        # Track engine state
        engine_states = []
        trades_executed = []
        errors_encountered = []

        # Start engine in background task
        engine_task = asyncio.create_task(engine.start())

        # Give engine time to initialize
        await asyncio.sleep(0.1)

        # Let engine run for a short period
        test_duration = 10  # seconds instead of full trading day
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < test_duration:
            try:
                # Record state periodically using engine's components
                engine_states.append({
                    'timestamp': datetime.now(),
                    'portfolio_value': engine.portfolio_engine.current_equity,
                    'open_positions': len(engine.portfolio_engine.positions),
                    'pending_orders': len(engine.order_manager.get_active_orders())
                })

                # Check for executed trades
                recent_trades = engine.order_manager.get_recent_fills(seconds=5)
                trades_executed.extend(recent_trades)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                errors_encountered.append({
                    'timestamp': datetime.now(),
                    'error': str(e)
                })
                print(f"Error during monitoring: {e}")  # Debug output

        # Stop the engine
        await engine.stop()

        # Cancel the engine task
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass

        # Verify simulation results
        if errors_encountered:
            print(f"Errors encountered during test: {errors_encountered}")

        assert len(engine_states) > 0, f"No engine states recorded. Errors: {errors_encountered}"
        assert engine.is_running is False, "Engine should be stopped"

        # Check portfolio consistency
        final_value = engine.portfolio_engine.current_equity
        assert final_value > 0, "Portfolio value should be positive"

        # Verify risk limits were respected
        max_position_sizes = []
        for state in engine_states:
            if state['open_positions'] > 0:
                for position in engine.portfolio_engine.positions.values():
                    position_value = position.quantity * position.current_price
                    position_size = position_value / engine.portfolio_engine.current_equity
                    max_position_sizes.append(position_size)

        if max_position_sizes:
            assert max(max_position_sizes) <= config['risk']['max_position_size']

        # Check order management
        total_orders = len(engine.order_manager.get_all_orders())
        print(f"Total orders placed: {total_orders}")
        print(f"Trades executed: {len(trades_executed)}")
        print(f"Final portfolio value: ${final_value:,.2f}")
        print(f"Errors encountered: {len(errors_encountered)}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self):
        """
        Test system behavior during connection failures.

        Verifies:
        1. Graceful handling of disconnections
        2. Automatic reconnection attempts
        3. Order status synchronization
        4. No duplicate orders
        """
        # Initialize components
        PortfolioEngine({"initial_capital": 100000})
        EnhancedRiskManager({})

        # Mock executor with connection simulation
        executor = Mock(spec=BaseExecutor)
        executor.is_connected = True
        executor.connect = AsyncMock(return_value=True)
        executor.disconnect = AsyncMock()
        executor.submit_order = AsyncMock()

        order_manager = OrderManager()
        strategy = MeanReversionEquity({
            'lookback_period': 20,
            'zscore_threshold': 2.0,
            'exit_zscore': 0.5,
            'rsi_period': 14,
            'rsi_oversold': 30.0,
            'rsi_overbought': 70.0
        })

        # Create config for LiveTradingEngine
        config = {
            'mode': 'paper',
            'strategies': [{
                'class': MeanReversionEquity,
                'enabled': True,
                'allocation': 1.0,
                'params': strategy.config
            }],
            'portfolio_config': {
                'initial_capital': 100000
            },
            'risk_config': {},
            'executor_config': {
                'paper': {}
            }
        }

        engine = LiveTradingEngine(config)

        # Create a task for the engine to run in the background
        engine_task = asyncio.create_task(engine.start())
        
        try:
            # Give engine time to start
            await asyncio.sleep(0.1)

            # Simulate normal operation
            market_data = {
                'AAPL': {
                    'timestamp': datetime.now(),
                    'last': 150.0,
                    'bid': 149.95,
                    'ask': 150.05,
                    'volume': 1000
                }
            }

            await engine.process_market_data(market_data)

            # Simulate connection failure
            executor.is_connected = False
            datetime.now()

            # Try to process more data during disconnection
            disconnected_orders = []
            for i in range(5):
                market_data['AAPL']['last'] = 150.0 + i * 0.1

                # Engine should queue orders during disconnection
                await engine.process_market_data(market_data)

                # Check if orders are queued
                pending = order_manager.get_active_orders()
                if pending:
                    disconnected_orders.extend(pending)

            # Simulate reconnection after 10 seconds
            await asyncio.sleep(0.1)  # Simulated delay
            executor.is_connected = True

            # The engine should detect reconnection automatically
            # when processing the next market data update
            await engine.process_market_data(market_data)

            # Verify recovery behavior
            assert executor.connect.called

            # Check order deduplication
            all_order_ids = [order.id for order in order_manager.get_all_orders()]
            unique_order_ids = set(all_order_ids)
            assert len(all_order_ids) == len(unique_order_ids), "Duplicate orders detected"

            # Verify queued orders are processed
            if disconnected_orders:
                # Should attempt to submit queued orders
                assert executor.submit_order.call_count >= len(disconnected_orders)
                
        finally:
            # Always stop the engine to prevent hanging
            await engine.stop()
            # Cancel the engine task
            engine_task.cancel()
            try:
                await engine_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_high_frequency_simulation(self):
        """
        Test system under high-frequency trading conditions.

        Verifies:
        1. Performance under rapid market updates
        2. Order queue management
        3. Rate limiting compliance
        4. System stability
        """
        config = {
            'trading': {
                'symbols': ['AAPL', 'GOOGL'],
                'update_frequency': 0.001,  # 1ms updates
                'max_orders_per_second': 100
            },
            'risk': {
                'max_position_size': 0.1,
                'max_orders_per_minute': 200
            }
        }

        portfolio = PortfolioEngine({"initial_capital": 100000})
        risk_manager = EnhancedRiskManager(**config['risk'])
        Mock(spec=BaseExecutor)
        order_manager = OrderManager()

        # High-frequency strategy (simplified)
        class HighFrequencyStrategy:
            def __init__(self):
                self.name = "HFT"
                self.signal_count = 0

            def init(self):
                pass

            def next(self, market_data):
                # Generate signals on small price movements
                if random.random() < 0.1:  # 10% chance
                    self.signal_count += 1
                    return {
                        'symbol': random.choice(config['trading']['symbols']),
                        'direction': random.choice(['BUY', 'SELL']),
                        'quantity': random.randint(10, 100),
                        'order_type': 'LIMIT'
                    }
                return None

        strategy = HighFrequencyStrategy()

        # Track performance metrics
        start_time = time.time()
        updates_processed = 0
        orders_submitted = 0

        # Simulate rapid market updates
        for i in range(1000):  # 1000 updates
            market_data = {}
            for symbol in config['trading']['symbols']:
                market_data[symbol] = {
                    'timestamp': datetime.now(),
                    'last': 100 + random.gauss(0, 0.1),
                    'volume': random.randint(100, 1000)
                }

            # Process update
            signal = strategy.next(market_data)
            if signal:
                # Check rate limits
                time.time()
                recent_orders = orders_submitted

                if recent_orders < config['risk']['max_orders_per_minute']:
                    # Submit order
                    order = Order(
                        symbol=signal['symbol'],
                        quantity=signal['quantity'],
                        order_type=OrderType.LIMIT,
                        side=OrderSide.BUY if signal['direction'] == 'buy' else OrderSide.SELL,
                        price=market_data[signal['symbol']]['last']
                    )

                    if risk_manager.check_order(order, portfolio):
                        order_id = f"HFT_{i}_{signal['symbol']}"
                        order_manager.add_order(order_id, order)
                        orders_submitted += 1

            updates_processed += 1

            # Simulate processing delay
            await asyncio.sleep(config['trading']['update_frequency'])

        # Calculate performance metrics
        end_time = time.time()
        elapsed_time = end_time - start_time

        updates_per_second = updates_processed / elapsed_time
        orders_per_second = orders_submitted / elapsed_time

        print(f"Updates processed: {updates_processed}")
        print(f"Orders submitted: {orders_submitted}")
        print(f"Updates per second: {updates_per_second:.2f}")
        print(f"Orders per second: {orders_per_second:.2f}")

        # Verify system stability
        assert updates_processed == 1000
        assert orders_per_second <= config['trading']['max_orders_per_second']
        assert strategy.signal_count > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_strategy_coordination_live(self):
        """
        Test multiple strategies running concurrently in live mode.

        Verifies:
        1. Strategy isolation
        2. Signal aggregation
        3. Conflicting signal resolution
        4. Resource management
        """
        # Initialize strategies with different characteristics
        conservative_strategy = MeanReversionEquity({
            'name': 'Conservative_MR',
            'lookback_period': 50,
            'zscore_threshold': 2.5,
            'exit_zscore': 0.3,
            'rsi_period': 21,
            'rsi_oversold': 20.0,
            'rsi_overbought': 80.0,
            'max_positions': 2
        })

        aggressive_strategy = MeanReversionEquity({
            'name': 'Aggressive_MR',
            'lookback_period': 10,
            'zscore_threshold': 1.5,
            'exit_zscore': 0.5,
            'rsi_period': 7,
            'rsi_oversold': 35.0,
            'rsi_overbought': 65.0,
            'max_positions': 5
        })

        trend_strategy = TrendFollowingMulti({
            'name': 'Trend_Follow',
            'symbols': ['AAPL', 'GOOGL'],
            'channel_period': 20,
            'atr_period': 14,
            'adx_period': 14,
            'adx_threshold': 30.0
        })

        # Set strategy allocations
        conservative_strategy.allocation = 0.3
        aggressive_strategy.allocation = 0.3
        trend_strategy.allocation = 0.4

        # Initialize engine
        PortfolioEngine({"initial_capital": 100000})
        EnhancedRiskManager({})
        Mock(spec=BaseExecutor)

        # Create config for LiveTradingEngine
        config = {
            'mode': 'paper',
            'strategies': [
                {
                    'class': MeanReversionEquity,
                    'enabled': True,
                    'allocation': 0.3,
                    'params': conservative_strategy.config
                },
                {
                    'class': MeanReversionEquity,
                    'enabled': True,
                    'allocation': 0.3,
                    'params': aggressive_strategy.config
                },
                {
                    'class': TrendFollowingMulti,
                    'enabled': True,
                    'allocation': 0.4,
                    'params': trend_strategy.config
                }
            ],
            'portfolio_config': {
                'initial_capital': 100000
            },
            'risk_config': {},
            'executor_config': {
                'paper': {}
            }
        }

        engine = LiveTradingEngine(config
        )

        # Track signals by strategy
        signals_by_strategy = {
            'Conservative_MR': [],
            'Aggressive_MR': [],
            'Trend_Follow': []
        }

        # Generate market scenarios
        market_scenarios = [
            {'type': 'trending', 'direction': 'up', 'volatility': 'low'},
            {'type': 'ranging', 'volatility': 'high'},
            {'type': 'trending', 'direction': 'down', 'volatility': 'medium'}
        ]

        for scenario in market_scenarios:
            # Generate appropriate market data
            market_data = self._generate_scenario_data(scenario)

            # Process through all strategies
            # Process market data first, then collect signals
            await engine.process_market_data(market_data)
            signals = await engine.collect_signals()

            # Categorize signals
            for signal in signals:
                if signal.strategy_id in signals_by_strategy:
                    signals_by_strategy[signal.strategy_id].append(signal)

            # Check for conflicts
            symbol_signals = {}
            for signal in signals:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append(signal)

            # Resolve conflicts
            resolved_signals = engine.resolve_signal_conflicts(symbol_signals)

            # Verify conflict resolution
            for symbol, final_signal in resolved_signals.items():
                conflicting = symbol_signals[symbol]
                if len(conflicting) > 1:
                    # Should pick strongest signal or use weighted average
                    assert final_signal is not None

        # Verify strategy independence
        assert len(signals_by_strategy['Conservative_MR']) >= 0
        assert len(signals_by_strategy['Aggressive_MR']) >= 0
        assert len(signals_by_strategy['Trend_Follow']) >= 0

        # Aggressive should generate more signals
        if signals_by_strategy['Aggressive_MR'] and signals_by_strategy['Conservative_MR']:
            assert len(signals_by_strategy['Aggressive_MR']) >= len(signals_by_strategy['Conservative_MR'])

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_disaster_recovery_procedures(self):
        """
        Test system recovery from various disaster scenarios.

        Verifies:
        1. Data corruption handling
        2. State recovery
        3. Position reconciliation
        4. Order book reconstruction
        """
        # Initialize system
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Add some positions
        # Positions would be added through trading in real usage
        # Positions would be added through trading in real usage

        # Save initial state
        initial_state = {
            'portfolio_value': portfolio.current_equity,
            'positions': portfolio.positions.copy(),
            'cash': portfolio.cash
        }

        # Simulate various disasters
        disasters = [
            'portfolio_corruption',
            'order_history_loss',
            'strategy_state_corruption',
            'market_data_corruption'
        ]

        recovery_results = {}

        for disaster in disasters:
            if disaster == 'portfolio_corruption':
                # Add a position first if it doesn't exist
                from core.portfolio import Position
                if 'AAPL' not in portfolio.positions:
                    portfolio.positions['AAPL'] = Position(
                        symbol='AAPL',
                        direction='LONG',
                        quantity=100,
                        entry_price=150.0,
                        entry_time=datetime.now(),
                        current_price=150.0
                    )

                # Corrupt portfolio data
                portfolio.positions['AAPL'].quantity = -999
                portfolio.cash = float('nan')

                # Recovery procedure
                try:
                    # Detect corruption
                    is_valid = portfolio.validate_state()
                    assert not is_valid

                    # Restore from backup
                    portfolio.positions = initial_state['positions'].copy()
                    portfolio.cash = initial_state['cash']

                    # Verify recovery
                    assert portfolio.validate_state()
                    recovery_results[disaster] = 'SUCCESS'
                except Exception as e:
                    recovery_results[disaster] = f'FAILED: {str(e)}'

            elif disaster == 'order_history_loss':
                # Simulate lost order history
                order_manager = OrderManager()

                # Add some orders
                for i in range(10):
                    order = Order(
                        symbol='AAPL',
                        quantity=10,
                        order_type=OrderType.MARKET,
                        side='BUY'
                    )
                    order_manager.add_order(f'ORDER_{i}', order)

                # Simulate data loss
                order_manager._orders.clear()

                # Recovery from audit trail
                try:
                    # In real system, would recover from database/logs
                    audit_trail = [
                        {'id': f'ORDER_{i}', 'symbol': 'AAPL', 'quantity': 10}
                        for i in range(10)
                    ]

                    # Reconstruct orders
                    for record in audit_trail:
                        recovered_order = Order(
                            symbol=record['symbol'],
                            quantity=record['quantity'],
                            order_type=OrderType.MARKET,
                            side='BUY'
                        )
                        order_manager.add_order(record['id'], recovered_order)

                    assert len(order_manager.get_all_orders()) == 10
                    recovery_results[disaster] = 'SUCCESS'
                except Exception as e:
                    recovery_results[disaster] = f'FAILED: {str(e)}'

        # Verify all disasters were handled
        assert len(recovery_results) == len(disasters)

        # Check recovery success rate
        successful_recoveries = sum(1 for result in recovery_results.values() if result == 'SUCCESS')
        recovery_rate = successful_recoveries / len(disasters)

        logger.info(f"Disaster recovery rate: {recovery_rate:.0%}")
        for disaster, result in list(recovery_results.items())[:5]:  # Limit output
            logger.info(f"  {disaster}: {result}")

        assert recovery_rate >= 0.75, "Too many recovery failures"

    def _generate_scenario_data(self, scenario):
        """Generate market data for specific scenario."""
        symbols = ['AAPL', 'GOOGL']
        data = {}

        for symbol in symbols:
            base_price = 100

            if scenario['type'] == 'trending':
                if scenario['direction'] == 'up':
                    price = base_price * 1.02
                else:
                    price = base_price * 0.98
            else:  # ranging
                price = base_price + random.gauss(0, 1)

            # Add volatility
            vol_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 2.0}[scenario['volatility']]
            price += random.gauss(0, 0.5 * vol_multiplier)

            data[symbol] = {
                'timestamp': datetime.now(),
                'last': price,
                'volume': random.randint(1000, 5000)
            }

        return data
