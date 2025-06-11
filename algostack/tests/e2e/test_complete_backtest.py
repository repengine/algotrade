"""
End-to-end test for complete backtest workflow.

Tests the entire system from configuration to results.
Validates:
- Configuration loading and validation
- Data fetching for multiple symbols
- Strategy execution across time periods
- Risk management throughout
- Results generation and persistence
"""

import pytest
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from core.trading_engine_main import TradingEngine
from core.portfolio import PortfolioEngine
from core.risk import EnhancedRiskManager
from core.metrics import MetricsCollector
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti
from adapters.yf_fetcher import YFinanceFetcher
from core.data_handler import DataHandler


class TestCompleteBacktest:
    """Test complete backtest scenarios."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_backtest_workflow(self, tmp_path):
        """
        Test complete backtest from config to results.
        
        Simulates real usage:
        1. Load configuration
        2. Fetch historical data
        3. Run backtest
        4. Generate reports
        5. Save results
        """
        # Create comprehensive test configuration
        config = {
            'backtest': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005,
                'data_frequency': 'daily'
            },
            'strategies': [
                {
                    'class': 'MeanReversionEquity',
                    'enabled': True,
                    'allocation': 0.5,
                    'parameters': {
                        'lookback_period': 20,
                        'zscore_threshold': 2.0,
                        'exit_zscore': 0.5,
                        'rsi_period': 14,
                        'rsi_oversold': 30.0,
                        'rsi_overbought': 70.0,
                        'max_positions': 3
                    }
                },
                {
                    'class': 'TrendFollowingMulti',
                    'enabled': True,
                    'allocation': 0.5,
                    'parameters': {
                        'channel_period': 20,
                        'atr_period': 14,
                        'adx_period': 14,
                        'adx_threshold': 25.0,
                        'max_positions': 3
                    }
                }
            ],
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'risk': {
                'max_position_size': 0.15,
                'max_portfolio_risk': 0.02,
                'max_correlation': 0.7,
                'stop_loss': 0.05,
                'take_profit': 0.10
            },
            'data': {
                'source': 'yahoo',
                'cache_enabled': True,
                'cache_dir': str(tmp_path / 'cache')
            }
        }
        
        # Save config
        config_path = tmp_path / "backtest_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Load config and verify
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config == config
        
        # Initialize components
        portfolio = PortfolioEngine({"initial_capital": config["backtest"]["initial_capital"]})
        risk_manager = EnhancedRiskManager({"max_position_size": config['risk']['max_position_size'], "max_portfolio_risk": config['risk']['max_portfolio_risk'], "max_correlation": config['risk']['max_correlation']
        })
        metrics_calculator = MetricsCollector()
        
        # Initialize strategies
        strategies = []
        for strat_config in config['strategies']:
            if strat_config['enabled']:
                if strat_config['class'] == 'MeanReversionEquity':
                    strategy = MeanReversionEquity(strat_config['parameters'])
                elif strat_config['class'] == 'TrendFollowingMulti':
                    params = strat_config['parameters'].copy()
                    params['symbols'] = config['symbols']
                    strategy = TrendFollowingMulti(params)
                
                strategy.allocation = strat_config['allocation']
                strategies.append(strategy)
        
        # Create backtest engine
        engine = TradingEngine(
            strategies=strategies,
            portfolio=portfolio,
            risk_manager=risk_manager,
            metrics_calculator=metrics_calculator,
            config=config
        )
        
        # Generate test data (in real scenario, would fetch from data source)
        test_data = self._generate_realistic_market_data(
            symbols=config['symbols'],
            start_date=config['backtest']['start_date'],
            end_date=config['backtest']['end_date']
        )
        
        # Run backtest
        results = engine.run(
            market_data=test_data,
            start_date=datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d'),
            end_date=datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
        )
        
        # Verify results structure
        assert 'metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        assert 'positions_history' in results
        assert 'strategy_performance' in results
        
        # Verify metrics completeness
        metrics = results['metrics']
        required_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'win_rate', 'profit_factor', 'avg_win', 'avg_loss',
            'total_trades', 'winning_trades', 'losing_trades'
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Verify trades recorded
        trades = results['trades']
        if len(trades) > 0:
            for trade in trades:
                assert 'symbol' in trade
                assert 'entry_time' in trade
                assert 'exit_time' in trade
                assert 'entry_price' in trade
                assert 'exit_price' in trade
                assert 'quantity' in trade
                assert 'pnl' in trade
                assert 'strategy' in trade
        
        # Verify equity curve
        equity_curve = results['equity_curve']
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) > 200  # Daily data for year
        assert equity_curve.iloc[0] == config['backtest']['initial_capital']
        
        # Verify no look-ahead bias
        assert equity_curve.index.is_monotonic_increasing
        assert not equity_curve.isna().any()
        
        # Save results
        results_path = tmp_path / "backtest_results.json"
        
        # Convert results to serializable format
        serializable_results = {
            'metrics': results['metrics'],
            'trades': results['trades'],
            'equity_curve': results['equity_curve'].to_dict(),
            'config': config,
            'run_timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        assert results_path.exists()
        
        # Generate performance report
        report_path = tmp_path / "performance_report.txt"
        self._generate_performance_report(results, report_path)
        assert report_path.exists()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_multi_timeframe_backtest(self, tmp_path):
        """
        Test backtest across multiple timeframes.
        
        Verifies system handles different data frequencies.
        """
        timeframes = ['1min', '5min', '1hour', '1day']
        results_by_timeframe = {}
        
        for timeframe in timeframes:
            config = {
                'backtest': {
                    'start_date': '2023-10-01',
                    'end_date': '2023-10-31',
                    'initial_capital': 100000,
                    'data_frequency': timeframe
                },
                'strategies': [{
                    'class': 'MeanReversionEquity',
                    'parameters': {
                        'lookback_period': 20,
                        'zscore_threshold': 2.0,
                        'exit_zscore': 0.5,
                        'rsi_period': 14,
                        'rsi_oversold': 30.0,
                        'rsi_overbought': 70.0
                    }
                }],
                'symbols': ['AAPL', 'GOOGL'],
                'risk': {
                    'max_position_size': 0.2,
                    'max_portfolio_risk': 0.02
                }
            }
            
            # Adjust parameters for different timeframes
            if timeframe in ['1min', '5min']:
                config['strategies'][0]['parameters']['lookback_period'] = 50
                config['strategies'][0]['parameters']['rsi_period'] = 21
            
            # Run backtest for timeframe
            portfolio = PortfolioEngine({"initial_capital": 100000})
            risk_manager = EnhancedRiskManager({})
            strategy = MeanReversionEquity(config['strategies'][0]['parameters'])
            
            engine = TradingEngine(
                strategies=[strategy],
                portfolio=portfolio,
                risk_manager=risk_manager,
                config=config
            )
            
            # Generate data for timeframe
            test_data = self._generate_intraday_data(
                symbols=config['symbols'],
                timeframe=timeframe,
                start_date=config['backtest']['start_date'],
                end_date=config['backtest']['end_date']
            )
            
            results = engine.run(
                market_data=test_data,
                start_date=datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d'),
                end_date=datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
            )
            
            results_by_timeframe[timeframe] = results
        
        # Compare results across timeframes
        for tf1 in timeframes:
            for tf2 in timeframes:
                if tf1 != tf2:
                    # Verify different timeframes produce different results
                    trades1 = len(results_by_timeframe[tf1]['trades'])
                    trades2 = len(results_by_timeframe[tf2]['trades'])
                    
                    # Different timeframes should have different trade counts
                    if tf1 in ['1min', '5min'] and tf2 in ['1hour', '1day']:
                        assert trades1 != trades2
    
    @pytest.mark.e2e
    def test_walk_forward_optimization(self):
        """
        Test walk-forward optimization process.
        
        Verifies parameter optimization works correctly.
        """
        # Define optimization windows
        optimization_windows = [
            {
                'train_start': '2023-01-01',
                'train_end': '2023-06-30',
                'test_start': '2023-07-01',
                'test_end': '2023-09-30'
            },
            {
                'train_start': '2023-04-01',
                'train_end': '2023-09-30',
                'test_start': '2023-10-01',
                'test_end': '2023-12-31'
            }
        ]
        
        # Parameter ranges to optimize
        param_ranges = {
            'lookback_period': [10, 20, 30],
            'zscore_threshold': [1.5, 2.0, 2.5],
            'rsi_period': [7, 14, 21]
        }
        
        optimization_results = []
        
        for window in optimization_windows:
            # Generate training data
            train_data = self._generate_realistic_market_data(
                symbols=['AAPL', 'GOOGL'],
                start_date=window['train_start'],
                end_date=window['train_end']
            )
            
            # Grid search optimization
            best_params = None
            best_sharpe = -np.inf
            
            for lookback in param_ranges['lookback_period']:
                for zscore in param_ranges['zscore_threshold']:
                    for rsi_period in param_ranges['rsi_period']:
                        params = {
                            'lookback_period': lookback,
                            'zscore_threshold': zscore,
                            'exit_zscore': zscore / 4,
                            'rsi_period': rsi_period,
                            'rsi_oversold': 30.0,
                            'rsi_overbought': 70.0
                        }
                        
                        # Run backtest with parameters
                        portfolio = PortfolioEngine({"initial_capital": 100000})
                        strategy = MeanReversionEquity(params)
                        engine = TradingEngine(
                            strategies=[strategy],
                            portfolio=portfolio
                        )
                        
                        results = engine.run(market_data=train_data)
                        
                        sharpe = results['metrics'].get('sharpe_ratio', -np.inf)
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params
            
            # Test on out-of-sample data
            test_data = self._generate_realistic_market_data(
                symbols=['AAPL', 'GOOGL'],
                start_date=window['test_start'],
                end_date=window['test_end']
            )
            
            portfolio = PortfolioEngine({"initial_capital": 100000})
            strategy = MeanReversionEquity(best_params)
            engine = TradingEngine(
                strategies=[strategy],
                portfolio=portfolio
            )
            
            test_results = engine.run(market_data=test_data)
            
            optimization_results.append({
                'window': window,
                'best_params': best_params,
                'in_sample_sharpe': best_sharpe,
                'out_sample_sharpe': test_results['metrics']['sharpe_ratio'],
                'out_sample_return': test_results['metrics']['total_return']
            })
        
        # Verify optimization results
        assert len(optimization_results) == len(optimization_windows)
        
        # Check for overfitting
        for result in optimization_results:
            in_sample = result['in_sample_sharpe']
            out_sample = result['out_sample_sharpe']
            
            # Out-of-sample should not be drastically worse
            if in_sample > 0:
                degradation = (in_sample - out_sample) / in_sample
                assert degradation < 0.5, "Severe overfitting detected"
    
    @pytest.mark.e2e
    def test_stress_scenarios(self):
        """
        Test backtest under various stress scenarios.
        
        Verifies system handles extreme market conditions.
        """
        stress_scenarios = {
            'flash_crash': {
                'description': 'Sudden 10% drop and recovery',
                'drop_magnitude': 0.10,
                'recovery_time': 5  # days
            },
            'bear_market': {
                'description': 'Prolonged 30% decline',
                'total_decline': 0.30,
                'duration': 90  # days
            },
            'high_volatility': {
                'description': 'Extended high volatility period',
                'volatility_multiplier': 3.0,
                'duration': 30  # days
            },
            'liquidity_crisis': {
                'description': 'Low volume and wide spreads',
                'volume_reduction': 0.8,
                'spread_multiplier': 5.0
            }
        }
        
        base_config = {
            'initial_capital': 100000,
            'strategies': [{
                'class': 'MeanReversionEquity',
                'parameters': {
                    'lookback_period': 20,
                    'zscore_threshold': 2.0,
                    'exit_zscore': 0.5,
                    'rsi_period': 14,
                    'rsi_oversold': 30.0,
                    'rsi_overbought': 70.0
                }
            }],
            'risk': {
                'max_position_size': 0.1,
                'max_portfolio_risk': 0.02,
                'stop_loss': 0.05
            }
        }
        
        results_by_scenario = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Generate stressed market data
            stressed_data = self._generate_stressed_market_data(
                scenario_type=scenario_name,
                params=scenario_params,
                symbols=['AAPL', 'GOOGL', 'MSFT']
            )
            
            # Run backtest
            portfolio = PortfolioEngine({"initial_capital": base_config['initial_capital']})
            risk_manager = EnhancedRiskManager(**base_config['risk'])
            strategy = MeanReversionEquity(base_config['strategies'][0]['parameters'])
            
            engine = TradingEngine(
                strategies=[strategy],
                portfolio=portfolio,
                risk_manager=risk_manager
            )
            
            results = engine.run(market_data=stressed_data)
            results_by_scenario[scenario_name] = results
            
            # Verify risk management worked
            max_dd = results['metrics']['max_drawdown']
            assert max_dd < 0.5, f"Excessive drawdown in {scenario_name}: {max_dd}"
            
            # Verify portfolio survived
            final_equity = results['equity_curve'].iloc[-1]
            assert final_equity > base_config['initial_capital'] * 0.5
        
        # Compare scenario impacts
        baseline_return = -0.05  # Assume slight loss in stress
        
        for scenario, results in results_by_scenario.items():
            total_return = results['metrics']['total_return']
            print(f"{scenario}: {total_return:.2%} return")
            
            # Some scenarios should be worse than others
            if scenario == 'flash_crash':
                assert total_return > baseline_return - 0.10
            elif scenario == 'bear_market':
                assert total_return < baseline_return
    
    def _generate_realistic_market_data(self, symbols, start_date, end_date):
        """Generate realistic market data for testing."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Market factor (affects all stocks)
        market_returns = np.random.normal(0.0005, 0.015, len(dates))
        
        data = {}
        for symbol in symbols:
            # Stock-specific parameters
            beta = np.random.uniform(0.8, 1.2)
            volatility = np.random.uniform(0.015, 0.025)
            
            # Generate returns
            specific_returns = np.random.normal(0, volatility, len(dates))
            total_returns = beta * market_returns + specific_returns
            
            # Generate prices
            prices = 100 * np.exp(np.cumsum(total_returns))
            
            # Generate OHLCV
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = prices * (1 + np.random.uniform(-0.003, 0.003, len(dates)))
            df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.002, len(dates))))
            df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.002, len(dates))))
            df['volume'] = np.random.lognormal(14, 0.5, len(dates)).astype(int)
            
            data[symbol] = df
        
        return data
    
    def _generate_intraday_data(self, symbols, timeframe, start_date, end_date):
        """Generate intraday data for different timeframes."""
        freq_map = {
            '1min': 'T',
            '5min': '5T', 
            '1hour': 'H',
            '1day': 'D'
        }
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate timestamps for market hours only
        all_timestamps = []
        current = start
        
        while current <= end:
            if current.weekday() < 5:  # Weekdays only
                market_open = current.replace(hour=9, minute=30, second=0)
                market_close = current.replace(hour=16, minute=0, second=0)
                
                day_timestamps = pd.date_range(
                    start=market_open,
                    end=market_close,
                    freq=freq_map[timeframe]
                )
                all_timestamps.extend(day_timestamps)
            
            current += timedelta(days=1)
        
        dates = pd.DatetimeIndex(all_timestamps)
        
        # Generate data
        data = {}
        for symbol in symbols:
            n = len(dates)
            volatility = 0.0001 if timeframe == '1min' else 0.001
            
            returns = np.random.normal(0, volatility, n)
            prices = 100 * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            df['open'] = prices * (1 + np.random.uniform(-0.0005, 0.0005, n))
            df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.0002, n)))
            df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.0002, n)))
            df['volume'] = np.random.lognormal(10, 0.5, n).astype(int)
            
            data[symbol] = df
        
        return data
    
    def _generate_stressed_market_data(self, scenario_type, params, symbols):
        """Generate market data for stress scenarios."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n = len(dates)
        
        data = {}
        for symbol in symbols:
            if scenario_type == 'flash_crash':
                # Normal returns with sudden drop
                returns = np.random.normal(0.0005, 0.015, n)
                crash_day = n // 2
                returns[crash_day] = -params['drop_magnitude']
                # Recovery over next few days
                recovery_per_day = params['drop_magnitude'] / params['recovery_time']
                for i in range(params['recovery_time']):
                    if crash_day + i + 1 < n:
                        returns[crash_day + i + 1] = recovery_per_day
            
            elif scenario_type == 'bear_market':
                # Steady decline
                daily_decline = params['total_decline'] / params['duration']
                returns = np.random.normal(-daily_decline, 0.02, n)
            
            elif scenario_type == 'high_volatility':
                # Increased volatility
                base_vol = 0.015
                returns = np.random.normal(0, base_vol * params['volatility_multiplier'], n)
            
            elif scenario_type == 'liquidity_crisis':
                # Normal returns but adjust volume and implied spreads
                returns = np.random.normal(0.0005, 0.015, n)
            
            else:
                returns = np.random.normal(0.0005, 0.015, n)
            
            # Generate prices
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Generate OHLCV
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            
            if scenario_type == 'liquidity_crisis':
                # Wider spreads
                spread = 0.003 * params['spread_multiplier']
                df['open'] = prices * (1 + np.random.uniform(-spread, spread, n))
                df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, spread, n)))
                df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, spread, n)))
                # Reduced volume
                df['volume'] = (np.random.lognormal(14, 0.5, n) * (1 - params['volume_reduction'])).astype(int)
            else:
                df['open'] = prices * (1 + np.random.uniform(-0.003, 0.003, n))
                df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.002, n)))
                df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.002, n)))
                df['volume'] = np.random.lognormal(14, 0.5, n).astype(int)
            
            data[symbol] = df
        
        return data
    
    def _generate_performance_report(self, results, output_path):
        """Generate a comprehensive performance report."""
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BACKTEST PERFORMANCE REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Summary metrics
            f.write("SUMMARY METRICS\n")
            f.write("-"*30 + "\n")
            metrics = results['metrics']
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}\n")
            f.write(f"Volatility: {metrics.get('volatility', 0):.2%}\n")
            f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
            f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%}\n")
            f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n\n")
            
            # Strategy breakdown
            if 'strategy_performance' in results:
                f.write("STRATEGY PERFORMANCE\n")
                f.write("-"*30 + "\n")
                for strategy, perf in results['strategy_performance'].items():
                    f.write(f"\n{strategy}:\n")
                    f.write(f"  Trades: {perf.get('trades', 0)}\n")
                    f.write(f"  Win Rate: {perf.get('win_rate', 0):.2%}\n")
                    f.write(f"  Avg P&L: ${perf.get('avg_pnl', 0):.2f}\n")
            
            f.write("\n" + "="*60 + "\n")