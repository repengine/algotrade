"""
Performance benchmark tests for AlgoStack.

Tests system performance under various loads.
Benchmarks:
- Data processing throughput
- Strategy calculation speed
- Order processing latency
- Portfolio update performance
- Risk calculation overhead
"""

import pytest
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statistics
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

from core.portfolio import PortfolioEngine
from core.risk import EnhancedRiskManager
from core.data_handler import DataHandler
from core.trading_engine_main import TradingEngine
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti
from core.metrics import MetricsCollector
from core.engine.order_manager import OrderManager, Order, OrderType


class TestPerformanceBenchmarks:
    """Benchmark tests for system performance."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_data_processing_throughput(self, benchmark):
        """
        Benchmark data processing speed.
        
        Measures:
        - Rows per second
        - Memory usage
        - CPU utilization
        """
        # Generate large dataset
        n_symbols = 100
        n_days = 252 * 5  # 5 years
        n_rows = n_symbols * n_days
        
        print(f"\nTesting with {n_rows:,} total rows ({n_symbols} symbols × {n_days} days)")
        
        # Create data
        symbols = [f"STOCK_{i}" for i in range(n_symbols)]
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        def generate_symbol_data(symbol):
            """Generate data for one symbol."""
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_days)))
            return pd.DataFrame({
                'symbol': symbol,
                'open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_days))),
                'close': prices,
                'volume': np.random.lognormal(14, 0.5, n_days).astype(int)
            }, index=dates)
        
        # Benchmark data generation
        start_time = time.time()
        
        all_data = {}
        for symbol in symbols:
            all_data[symbol] = generate_symbol_data(symbol)
        
        generation_time = time.time() - start_time
        
        # Benchmark data validation
        data_handler = DataHandler()
        
        validation_start = time.time()
        valid_count = 0
        
        for symbol, data in all_data.items():
            if data_handler.validate_data(data):
                valid_count += 1
        
        validation_time = time.time() - validation_start
        
        # Calculate metrics
        generation_rate = n_rows / generation_time
        validation_rate = n_rows / validation_time
        
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"Data generation: {generation_rate:,.0f} rows/second")
        print(f"Data validation: {validation_rate:,.0f} rows/second")
        print(f"Memory usage: {memory_mb:.1f} MB")
        print(f"Valid datasets: {valid_count}/{n_symbols}")
        
        # Assert performance thresholds
        assert generation_rate > 100000, "Data generation too slow"
        assert validation_rate > 500000, "Data validation too slow"
        assert memory_mb < 2000, "Excessive memory usage"
    
    @pytest.mark.benchmark
    def test_strategy_calculation_speed(self, benchmark):
        """
        Benchmark strategy signal generation speed.
        
        Measures calculation time for various strategies.
        """
        # Generate test data
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(1000).cumsum(),
            'high': 101 + np.random.randn(1000).cumsum(),
            'low': 99 + np.random.randn(1000).cumsum(),
            'close': 100 + np.random.randn(1000).cumsum(),
            'volume': np.random.lognormal(14, 0.5, 1000).astype(int)
        }, index=dates)
        
        # Fix OHLC relationships
        test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
        test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
        test_data.attrs['symbol'] = 'TEST'
        
        # Initialize strategies
        strategies = {
            'mean_reversion': MeanReversionEquity({
                'lookback_period': 20,
                'zscore_threshold': 2.0,
                'exit_zscore': 0.5,
                'rsi_period': 14,
                'rsi_oversold': 30.0,
                'rsi_overbought': 70.0
            }),
            'trend_following': TrendFollowingMulti({
                'symbols': ['TEST'],
                'channel_period': 20,
                'atr_period': 14,
                'adx_period': 14,
                'adx_threshold': 25.0
            })
        }
        
        # Benchmark each strategy
        results = {}
        
        for name, strategy in strategies.items():
            strategy.init()
            
            # Warm-up run
            _ = strategy.next(test_data)
            
            # Benchmark runs
            times = []
            for _ in range(100):
                start = time.perf_counter()
                signal = strategy.next(test_data)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to milliseconds
            
            results[name] = {
                'mean_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'std_ms': statistics.stdev(times),
                'min_ms': min(times),
                'max_ms': max(times)
            }
        
        # Print results
        print("\nStrategy Calculation Performance:")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Mean: {metrics['mean_ms']:.3f} ms")
            print(f"  Median: {metrics['median_ms']:.3f} ms")
            print(f"  Std Dev: {metrics['std_ms']:.3f} ms")
            print(f"  Range: {metrics['min_ms']:.3f} - {metrics['max_ms']:.3f} ms")
        
        # Assert performance requirements
        for name, metrics in results.items():
            assert metrics['mean_ms'] < 10, f"{name} calculation too slow"
            assert metrics['max_ms'] < 50, f"{name} has excessive outliers"
    
    @pytest.mark.benchmark
    def test_order_processing_latency(self, benchmark):
        """
        Benchmark order processing speed.
        
        Measures time from order creation to execution.
        """
        portfolio = PortfolioEngine({"initial_capital": 1000000})
        risk_manager = EnhancedRiskManager({})
        order_manager = OrderManager()
        
        # Create many orders
        n_orders = 10000
        orders = []
        
        for i in range(n_orders):
            order = Order(
                symbol=f"STOCK_{i % 100}",
                quantity=np.random.randint(10, 1000),
                order_type=OrderType.MARKET if i % 2 == 0 else OrderType.LIMIT,
                side='BUY' if i % 3 != 0 else 'SELL',
                limit_price=100.0 + np.random.uniform(-5, 5) if i % 2 != 0 else None
            )
            orders.append(order)
        
        # Benchmark order validation
        validation_times = []
        
        for order in orders[:1000]:  # Sample
            start = time.perf_counter()
            is_valid = risk_manager.validate_order(order)
            elapsed = time.perf_counter() - start
            validation_times.append(elapsed * 1000000)  # Microseconds
        
        # Benchmark order management
        management_times = []
        
        for i, order in enumerate(orders):
            start = time.perf_counter()
            order_manager.add_order(f"ORDER_{i}", order)
            elapsed = time.perf_counter() - start
            management_times.append(elapsed * 1000000)
        
        # Benchmark order lookup
        lookup_times = []
        sample_ids = [f"ORDER_{i}" for i in range(0, n_orders, 100)]
        
        for order_id in sample_ids:
            start = time.perf_counter()
            order = order_manager.get_order(order_id)
            elapsed = time.perf_counter() - start
            lookup_times.append(elapsed * 1000000)
        
        # Calculate statistics
        print("\nOrder Processing Performance:")
        print(f"Validation: {statistics.mean(validation_times):.1f} µs (avg)")
        print(f"Management: {statistics.mean(management_times):.1f} µs (avg)")
        print(f"Lookup: {statistics.mean(lookup_times):.1f} µs (avg)")
        print(f"Total orders managed: {n_orders:,}")
        
        # Assert latency requirements
        assert statistics.mean(validation_times) < 100, "Order validation too slow"
        assert statistics.mean(management_times) < 50, "Order management too slow"
        assert statistics.mean(lookup_times) < 10, "Order lookup too slow"
    
    @pytest.mark.benchmark
    def test_portfolio_update_performance(self, benchmark):
        """
        Benchmark portfolio update operations.
        
        Measures position updates and P&L calculations.
        """
        portfolio = PortfolioEngine({"initial_capital": 1000000})
        
        # Generate positions
        n_positions = 500
        positions = []
        
        for i in range(n_positions):
            symbol = f"STOCK_{i}"
            quantity = np.random.randint(100, 10000)
            price = np.random.uniform(10, 500)
            positions.append((symbol, quantity, price))
        
        # Benchmark position additions
        add_times = []
        
        for symbol, quantity, price in positions:
            start = time.perf_counter()
            # Positions would be added through trading in real usage
            elapsed = time.perf_counter() - start
            add_times.append(elapsed * 1000000)
        
        # Benchmark position updates
        update_times = []
        new_prices = {pos[0]: pos[2] * np.random.uniform(0.95, 1.05) for pos in positions}
        
        for symbol, new_price in new_prices.items():
            start = time.perf_counter()
            portfolio.update_position_price(symbol, new_price)
            elapsed = time.perf_counter() - start
            update_times.append(elapsed * 1000000)
        
        # Benchmark P&L calculations
        pnl_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            total_pnl = portfolio.calculate_total_pnl()
            elapsed = time.perf_counter() - start
            pnl_times.append(elapsed * 1000)
        
        # Benchmark total equity calculation
        equity_times = []
        
        for _ in range(1000):
            start = time.perf_counter()
            equity = portfolio.total_equity
            elapsed = time.perf_counter() - start
            equity_times.append(elapsed * 1000000)
        
        print("\nPortfolio Performance:")
        print(f"Position additions: {statistics.mean(add_times):.1f} µs (avg)")
        print(f"Price updates: {statistics.mean(update_times):.1f} µs (avg)")
        print(f"P&L calculation: {statistics.mean(pnl_times):.3f} ms (avg)")
        print(f"Equity calculation: {statistics.mean(equity_times):.1f} µs (avg)")
        print(f"Positions tracked: {n_positions}")
        
        # Assert performance requirements
        assert statistics.mean(add_times) < 100, "Position addition too slow"
        assert statistics.mean(update_times) < 50, "Price update too slow"
        assert statistics.mean(pnl_times) < 5, "P&L calculation too slow"
        assert statistics.mean(equity_times) < 10, "Equity calculation too slow"
    
    @pytest.mark.benchmark
    def test_risk_calculation_overhead(self, benchmark):
        """
        Benchmark risk management calculations.
        
        Measures various risk metrics computation time.
        """
        portfolio = PortfolioEngine({"initial_capital": 1000000})
        risk_manager = EnhancedRiskManager({"max_position_size": 0.1, "max_portfolio_risk": 0.02, "max_correlation": 0.7
        })
        
        # Add diverse positions
        for i in range(50):
            portfolio.add_position_method(
                f"STOCK_{i}",
                np.random.randint(100, 1000),
                np.random.uniform(50, 200)
            )
        
        # Generate historical data for risk calculations
        n_days = 252
        returns_data = pd.DataFrame(
            np.random.normal(0, 0.02, (n_days, 50)),
            columns=[f"STOCK_{i}" for i in range(50)]
        )
        
        # Benchmark VaR calculation
        var_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            var_95 = risk_manager.calculate_var(portfolio, returns_data, confidence=0.95)
            elapsed = time.perf_counter() - start
            var_times.append(elapsed * 1000)
        
        # Benchmark correlation matrix
        corr_times = []
        
        for _ in range(50):
            start = time.perf_counter()
            corr_matrix = risk_manager.calculate_correlation_matrix(returns_data)
            elapsed = time.perf_counter() - start
            corr_times.append(elapsed * 1000)
        
        # Benchmark portfolio risk
        portfolio_risk_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            risk_metrics = risk_manager.calculate_portfolio_risk(portfolio)
            elapsed = time.perf_counter() - start
            portfolio_risk_times.append(elapsed * 1000)
        
        print("\nRisk Calculation Performance:")
        print(f"VaR (95%): {statistics.mean(var_times):.2f} ms (avg)")
        print(f"Correlation matrix: {statistics.mean(corr_times):.2f} ms (avg)")
        print(f"Portfolio risk: {statistics.mean(portfolio_risk_times):.2f} ms (avg)")
        
        # Assert performance requirements
        assert statistics.mean(var_times) < 10, "VaR calculation too slow"
        assert statistics.mean(corr_times) < 50, "Correlation calculation too slow"
        assert statistics.mean(portfolio_risk_times) < 5, "Portfolio risk calculation too slow"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_concurrent_backtest_performance(self):
        """
        Benchmark parallel backtest execution.
        
        Tests system scalability with multiple concurrent backtests.
        """
        # Create test configurations
        n_backtests = 10
        backtest_configs = []
        
        for i in range(n_backtests):
            config = {
                'name': f'Backtest_{i}',
                'symbols': [f'STOCK_{j}' for j in range(i*5, (i+1)*5)],
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 100000,
                'strategy': {
                    'lookback_period': 20 + i,
                    'zscore_threshold': 2.0,
                    'exit_zscore': 0.5,
                    'rsi_period': 14,
                    'rsi_oversold': 30.0,
                    'rsi_overbought': 70.0
                }
            }
            backtest_configs.append(config)
        
        # Generate market data
        all_symbols = set()
        for config in backtest_configs:
            all_symbols.update(config['symbols'])
        
        market_data = {}
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        for symbol in all_symbols:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
            market_data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.uniform(-0.002, 0.002, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.003, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.003, len(dates)))),
                'close': prices,
                'volume': np.random.lognormal(14, 0.5, len(dates)).astype(int)
            }, index=dates)
        
        def run_single_backtest(config):
            """Run one backtest."""
            portfolio = PortfolioEngine({"initial_capital": config['initial_capital']})
            strategy = MeanReversionEquity(config['strategy'])
            
            engine = TradingEngine(
                strategies=[strategy],
                portfolio=portfolio
            )
            
            # Filter market data for this backtest
            backtest_data = {s: market_data[s] for s in config['symbols']}
            
            start_time = time.time()
            results = engine.run(market_data=backtest_data)
            elapsed = time.time() - start_time
            
            return {
                'name': config['name'],
                'elapsed_time': elapsed,
                'total_return': results['metrics']['total_return'],
                'total_trades': results['metrics'].get('total_trades', 0)
            }
        
        # Run backtests sequentially
        sequential_start = time.time()
        sequential_results = []
        
        for config in backtest_configs:
            result = run_single_backtest(config)
            sequential_results.append(result)
        
        sequential_time = time.time() - sequential_start
        
        # Run backtests in parallel (threads)
        thread_start = time.time()
        
        with ThreadPoolBaseExecutor(max_workers=4) as executor:
            thread_results = list(executor.map(run_single_backtest, backtest_configs))
        
        thread_time = time.time() - thread_start
        
        # Calculate speedup
        speedup = sequential_time / thread_time
        
        print(f"\nConcurrent Backtest Performance:")
        print(f"Sequential execution: {sequential_time:.2f} seconds")
        print(f"Parallel execution (4 threads): {thread_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Average time per backtest: {sequential_time/n_backtests:.2f} seconds")
        
        # Verify results consistency
        for seq, par in zip(sequential_results, thread_results):
            assert seq['name'] == par['name']
            assert abs(seq['total_return'] - par['total_return']) < 0.0001
        
        # Assert performance improvement
        assert speedup > 2.0, "Insufficient parallel speedup"
        assert thread_time < sequential_time * 0.6, "Parallel execution not efficient"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_memory_efficiency_large_dataset(self):
        """
        Test memory efficiency with large datasets.
        
        Verifies system can handle large data without excessive memory use.
        """
        # Monitor initial memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset progressively
        n_symbols = 1000
        n_days = 252 * 10  # 10 years
        
        print(f"\nTesting with {n_symbols:,} symbols × {n_days:,} days")
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Process in chunks to test streaming capability
        chunk_size = 100  # symbols per chunk
        memory_readings = []
        
        for chunk_start in range(0, n_symbols, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_symbols)
            chunk_symbols = [f"STOCK_{i}" for i in range(chunk_start, chunk_end)]
            
            # Generate chunk data
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
            chunk_data = {}
            
            for symbol in chunk_symbols:
                prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_days)))
                chunk_data[symbol] = pd.DataFrame({
                    'close': prices,
                    'volume': np.random.lognormal(14, 0.5, n_days).astype(int)
                }, index=dates)
            
            # Process chunk (simulate strategy calculations)
            for symbol, data in chunk_data.items():
                # Simple calculations to simulate processing
                sma_20 = data['close'].rolling(20).mean()
                rsi = self._calculate_rsi(data['close'], 14)
            
            # Clear chunk data
            del chunk_data
            
            # Force garbage collection
            gc.collect()
            
            # Record memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
        
        # Analyze memory usage
        max_memory = max(memory_readings)
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        
        print(f"Peak memory: {max_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory growth: {memory_growth:.1f} MB")
        print(f"Memory per symbol: {memory_growth/n_symbols:.3f} MB")
        
        # Assert memory efficiency
        assert max_memory < initial_memory + 1000, "Excessive peak memory usage"
        assert memory_growth < 500, "Excessive memory growth"
        assert memory_growth / n_symbols < 0.5, "Inefficient memory per symbol"
    
    def _calculate_rsi(self, prices, period):
        """Simple RSI calculation for testing."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi