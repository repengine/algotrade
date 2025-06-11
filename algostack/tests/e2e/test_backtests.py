"""Comprehensive test suite for the backtests module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtests.run_backtests import BacktestEngine, AlgoStackStrategy
from strategies.base import BaseStrategy
from core.data_handler import DataHandler


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MockStrategy"
        self.signals_called = 0
    
    def calculate_signals(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Return simple signals for testing."""
        self.signals_called += 1
        # Return alternating buy/sell signals
        signals = pd.Series([0] * len(data), index=data.index)
        for i in range(1, len(data), 4):
            if i < len(data):
                signals.iloc[i] = 1  # Buy
            if i + 2 < len(data):
                signals.iloc[i + 2] = -1  # Sell
        return signals
    
    def validate_config(self, config: dict) -> dict:
        """Simple validation."""
        return config


class TestBacktestEngine:
    """Test suite for BacktestEngine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        close_prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(len(dates)) * 0.001),
            'high': close_prices * (1 + np.abs(np.random.randn(len(dates)) * 0.005)),
            'low': close_prices * (1 - np.abs(np.random.randn(len(dates)) * 0.005)),
            'close': close_prices,
            'volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)
        
        return data
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(initial_capital=50000)
        
        assert engine.initial_capital == 50000
        assert engine.results == {}
    
    def test_initialization_defaults(self):
        """Test default initialization values."""
        engine = BacktestEngine()
        
        assert engine.initial_capital == 5000.0
    
    @patch('backtests.run_backtests.DataHandler')
    @patch('backtests.run_backtests.bt')
    def test_run_backtest_basic(self, mock_bt, mock_data_handler, sample_data):
        """Test basic backtest execution."""
        # Setup mock data handler
        mock_handler_instance = Mock()
        mock_handler_instance.get_historical.return_value = sample_data
        mock_data_handler.return_value = mock_handler_instance
        
        # Setup mock backtrader
        mock_cerebro = Mock()
        mock_broker = Mock()
        mock_cerebro.broker = mock_broker
        mock_bt.Cerebro.return_value = mock_cerebro
        
        # Mock run results
        mock_cerebro.run.return_value = [Mock()]
        mock_broker.getvalue.return_value = 6000.0
        
        # Create engine and strategy
        engine = BacktestEngine(initial_capital=5000)
        strategy = MockStrategy({'lookback_period': 20})
        
        # Run backtest
        results = engine.run_backtest(
            strategy=strategy,
            symbols=['TEST'],
            start_date='2023-01-01',
            end_date='2023-03-31',
            commission=0.001,
            slippage=0.0005
        )
        
        # Verify backtrader setup
        mock_bt.Cerebro.assert_called_once()
        mock_broker.setcash.assert_called_with(5000.0)
        mock_broker.setcommission.assert_called_with(commission=0.001)
        mock_broker.set_slippage_perc.assert_called_with(0.0005)
        
        # Verify data handler was called
        mock_data_handler.assert_called_once_with(['yfinance'], premium_av=True)
        mock_handler_instance.get_historical.assert_called()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'initial_capital' in results
        assert 'final_value' in results
    
    @patch('backtests.run_backtests.DataHandler')
    def test_run_backtest_empty_data(self, mock_data_handler):
        """Test backtest with empty data."""
        # Setup mock to return empty dataframe
        mock_handler_instance = Mock()
        mock_handler_instance.get_historical.return_value = pd.DataFrame()
        mock_data_handler.return_value = mock_handler_instance
        
        engine = BacktestEngine()
        strategy = MockStrategy({})
        
        # Should handle empty data gracefully
        results = engine.run_backtest(
            strategy=strategy,
            symbols=['NODATA'],
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        
        # Should return results even with no data
        assert isinstance(results, dict)
    
    def test_print_summary(self, capsys):
        """Test summary printing."""
        engine = BacktestEngine()
        
        # Set up some mock results
        engine.results = {
            'initial_capital': 5000,
            'final_value': 6000,
            'total_return': 20.0,
            'sharpe_ratio': 1.5,
            'max_drawdown': -10.0,
            'total_trades': 50,
            'winning_trades': 30,
            'win_rate': 0.6
        }
        
        engine.print_summary()
        
        captured = capsys.readouterr()
        assert "Initial Capital: $5,000.00" in captured.out
        assert "Final Value: $6,000.00" in captured.out
        assert "Total Return: 20.00%" in captured.out
        assert "Sharpe Ratio: 1.50" in captured.out
    
    def test_export_results(self, tmp_path):
        """Test result export functionality."""
        engine = BacktestEngine()
        
        # Set up some results
        engine.results = {
            'initial_capital': 5000,
            'final_value': 6000,
            'total_return': 20.0,
            'trades': [
                {'date': '2023-01-01', 'action': 'BUY', 'quantity': 100},
                {'date': '2023-02-01', 'action': 'SELL', 'quantity': 100}
            ]
        }
        
        # Export to file
        output_file = tmp_path / "test_results.json"
        engine.export_results(str(output_file))
        
        # Verify file exists and contains data
        assert output_file.exists()
        
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert data['initial_capital'] == 5000
            assert data['final_value'] == 6000
            assert len(data['trades']) == 2


class TestAlgoStackStrategy:
    """Test suite for the AlgoStack strategy adapter."""
    
    def test_algostack_strategy_initialization(self):
        """Test AlgoStackStrategy initialization."""
        mock_strategy = Mock()
        mock_strategy.init = Mock()
        mock_risk = Mock()
        
        # This would normally be done by backtrader
        class TestableAlgoStackStrategy(AlgoStackStrategy):
            params = (
                ("algostack_strategy", mock_strategy),
                ("risk_context", mock_risk),
            )
        
        strategy = TestableAlgoStackStrategy()
        
        # Verify initialization
        assert strategy.algostack_strategy == mock_strategy
        assert strategy.risk_context == mock_risk
        mock_strategy.init.assert_called_once()
    
    def test_algostack_strategy_next(self):
        """Test strategy next() method."""
        # Create a mock strategy
        mock_strategy = Mock()
        mock_strategy.config = {'lookback_period': 10}
        mock_strategy.calculate_signals = Mock(return_value=pd.Series([1]))
        
        # Create mock data
        mock_data = Mock()
        mock_data.open = [100] * 20
        mock_data.high = [101] * 20
        mock_data.low = [99] * 20
        mock_data.close = [100.5] * 20
        mock_data.volume = [1000000] * 20
        mock_data.datetime = Mock()
        mock_data.datetime.datetime.return_value = datetime.now()
        
        # Create testable strategy
        class TestableAlgoStackStrategy(AlgoStackStrategy):
            params = (
                ("algostack_strategy", mock_strategy),
                ("risk_context", None),
            )
            
            def __init__(self):
                self.data = mock_data
                super().__init__()
            
            def buy(self, *args, **kwargs):
                return Mock()
            
            def sell(self, *args, **kwargs):
                return Mock()
            
            def close(self, *args, **kwargs):
                return Mock()
        
        strategy = TestableAlgoStackStrategy()
        
        # Test next() execution
        strategy.position = Mock()
        strategy.position.size = 0
        
        strategy.next()
        
        # Verify signal calculation was called
        mock_strategy.calculate_signals.assert_called()


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    @patch('backtests.run_backtests.DataHandler')
    @patch('backtests.run_backtests.bt')
    def test_multi_symbol_backtest(self, mock_bt, mock_data_handler):
        """Test backtesting with multiple symbols."""
        # Create sample data for multiple symbols
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        
        data_dict = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            data_dict[symbol] = pd.DataFrame({
                'open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
                'high': 102 + np.random.randn(len(dates)).cumsum() * 0.5,
                'low': 98 + np.random.randn(len(dates)).cumsum() * 0.5,
                'close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
                'volume': np.random.randint(1000000, 2000000, len(dates))
            }, index=dates)
        
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_instance.get_historical.side_effect = lambda s, *args, **kwargs: data_dict.get(s, pd.DataFrame())
        mock_data_handler.return_value = mock_handler_instance
        
        mock_cerebro = Mock()
        mock_cerebro.broker = Mock()
        mock_cerebro.run.return_value = [Mock()]
        mock_bt.Cerebro.return_value = mock_cerebro
        
        # Create engine and strategy
        engine = BacktestEngine()
        strategy = MockStrategy({'symbols': ['AAPL', 'GOOGL', 'MSFT']})
        
        # Run backtest
        results = engine.run_backtest(
            strategy=strategy,
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        
        # Verify data was fetched for all symbols
        assert mock_handler_instance.get_historical.call_count == 3
        
        # Verify results
        assert isinstance(results, dict)
    
    @patch('backtests.run_backtests.DataHandler')
    def test_different_data_providers(self, mock_data_handler):
        """Test backtest with different data providers."""
        # Setup mock
        mock_handler_instance = Mock()
        mock_handler_instance.get_historical.return_value = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99], 
            'close': [100], 'volume': [1000000]
        }, index=[datetime(2023, 1, 1)])
        mock_data_handler.return_value = mock_handler_instance
        
        engine = BacktestEngine()
        strategy = MockStrategy({})
        
        # Test with Alpha Vantage
        results = engine.run_backtest(
            strategy=strategy,
            symbols=['TEST'],
            start_date='2023-01-01',
            end_date='2023-01-31',
            data_provider='alphavantage'
        )
        
        # Verify correct provider was used
        mock_data_handler.assert_called_with(['alphavantage'], premium_av=True)


def test_run_walk_forward_optimization():
    """Test the walk-forward optimization function."""
    from backtests.run_backtests import run_walk_forward_optimization
    
    # This would need proper mocking of the entire backtest flow
    # For now, we'll just verify the function exists
    assert callable(run_walk_forward_optimization)