"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backtests.run_backtests import BacktestEngine, AlgoStackStrategy
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.base import Signal, RiskContext


class TestBacktestEngine:
    """Test the backtesting engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        n = len(dates)
        
        # Generate trending data with some volatility
        np.random.seed(42)
        trend = np.linspace(100, 120, n)
        noise = np.random.normal(0, 2, n)
        close = trend + noise
        
        data = pd.DataFrame({
            'open': close * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': close * (1 + np.random.uniform(0, 0.02, n)),
            'low': close * (1 - np.random.uniform(0, 0.02, n)),
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def strategy(self):
        """Create a test strategy."""
        config = {
            'symbols': ['TEST'],
            'rsi_period': 2,
            'rsi_oversold': 10,
            'rsi_overbought': 90,
            'atr_period': 14,
            'lookback_period': 30
        }
        return MeanReversionEquity(config)
    
    def test_backtest_engine_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(initial_capital=10000)
        assert engine.initial_capital == 10000
        assert engine.results == {}
    
    @patch('backtests.run_backtests.DataHandler')
    def test_run_backtest_basic(self, mock_data_handler, strategy, sample_data):
        """Test basic backtest execution."""
        # Mock data handler
        mock_handler = Mock()
        mock_handler.get_historical.return_value = sample_data
        mock_data_handler.return_value = mock_handler
        
        # Run backtest
        engine = BacktestEngine(initial_capital=5000)
        metrics = engine.run_backtest(
            strategy,
            ['TEST'],
            '2022-01-01',
            '2022-12-31',
            commission=0.0,
            slippage=0.0
        )
        
        # Check metrics structure
        assert 'initial_capital' in metrics
        assert 'final_value' in metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        
        # Check reasonable values
        assert metrics['initial_capital'] == 5000
        assert metrics['final_value'] > 0
        assert -100 < metrics['total_return'] < 1000  # Reasonable return range
        assert metrics['max_drawdown'] <= 0  # Drawdown should be negative
    
    def test_metrics_extraction(self):
        """Test performance metrics calculation."""
        engine = BacktestEngine()
        
        # Create mock cerebro and results
        mock_cerebro = Mock()
        mock_cerebro.broker.getvalue.return_value = 6000
        
        mock_results = Mock()
        mock_results.analyzers.sharpe.get_analysis.return_value = {'sharperatio': 1.2}
        mock_results.analyzers.drawdown.get_analysis.return_value = {
            'max': {'drawdown': -15.5}
        }
        mock_results.analyzers.returns.get_analysis.return_value = {
            'start': '2022-01-01 00:00:00',
            'end': '2022-12-31 00:00:00'
        }
        mock_results.analyzers.trades.get_analysis.return_value = {
            'total': {'total': 20},
            'won': {'total': 12, 'pnl': {'total': 2000}},
            'lost': {'total': 8, 'pnl': {'total': -800}},
            'pnl': {'average': 60}
        }
        
        # Extract metrics
        metrics = engine._extract_metrics(mock_cerebro, mock_results)
        
        # Verify calculations
        assert metrics['final_value'] == 6000
        assert metrics['total_return'] == 20.0  # (6000-5000)/5000 * 100
        assert metrics['sharpe_ratio'] == 1.2
        assert metrics['max_drawdown'] == -15.5
        assert metrics['total_trades'] == 20
        assert metrics['winning_trades'] == 12
        assert metrics['losing_trades'] == 8
        assert metrics['win_rate'] == 0.6  # 12/20
        assert metrics['profit_factor'] == 2.5  # 2000/800
    
    def test_save_and_load_results(self, tmp_path):
        """Test saving results to file."""
        engine = BacktestEngine()
        
        # Add mock results
        engine.results['test_strategy'] = {
            'metrics': {
                'total_return': 25.5,
                'sharpe_ratio': 1.1,
                'max_drawdown': -12.3
            },
            'signals': [],
            'trades': []
        }
        
        # Save results
        output_file = tmp_path / "test_results.json"
        engine.save_results(str(output_file))
        
        # Verify file exists and contains correct data
        assert output_file.exists()
        
        import json
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'test_strategy' in loaded_results
        assert loaded_results['test_strategy']['metrics']['total_return'] == 25.5
    
    def test_algostack_strategy_adapter(self, strategy, sample_data):
        """Test the Backtrader strategy adapter."""
        # Create mock Backtrader environment
        mock_data = Mock()
        mock_data.open = sample_data['open'].values
        mock_data.high = sample_data['high'].values
        mock_data.low = sample_data['low'].values
        mock_data.close = sample_data['close'].values
        mock_data.volume = sample_data['volume'].values
        mock_data._name = 'TEST'
        mock_data.datetime.datetime.return_value = datetime(2022, 6, 1)
        
        mock_broker = Mock()
        mock_broker.getvalue.return_value = 5000
        mock_broker.positions = {}
        
        # Create adapter
        risk_context = RiskContext(
            account_equity=5000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.15
        )
        
        adapter = AlgoStackStrategy()
        adapter.algostack_strategy = strategy
        adapter.risk_context = risk_context
        adapter.data = mock_data
        adapter.broker = mock_broker
        
        # Initialize
        adapter.__init__()
        
        # Verify strategy was initialized
        assert len(adapter.signals_history) == 0
        assert len(adapter.trades_history) == 0