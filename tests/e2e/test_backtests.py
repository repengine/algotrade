"""Comprehensive test suite for the backtests module."""

import logging
from datetime import datetime
from typing import Optional
from unittest.mock import Mock, patch

import backtrader as bt
import numpy as np
import pandas as pd
import pytest
from backtests.run_backtests import AlgoStackStrategy, BacktestEngine
from strategies.base import BaseStrategy, RiskContext, Signal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MockStrategy"
        self.signals_called = 0

    def init(self) -> None:
        """Initialize strategy state."""
        pass

    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process new data and generate trading signal."""
        # For testing, return None
        return None

    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """Calculate position size."""
        # Simple sizing: 100 shares, no stop loss
        return (100.0, 0.0)

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

        # Mock run results with proper metrics
        mock_strategy_result = Mock()
        mock_analyzer = Mock()

        # Mock trade analyzer
        mock_trade_analyzer = Mock()
        mock_trade_analyzer.get_analysis.return_value = {
            'total': {
                'closed': 10,
                'total': 10
            },
            'won': {
                'total': 6,
                'pnl': {'total': 1200}
            },
            'lost': {
                'total': 4,
                'pnl': {'total': -400}
            }
        }

        # Mock returns analyzer
        mock_returns_analyzer = Mock()
        mock_returns_analyzer.get_analysis.return_value = {
            'start': '2023-01-01 00:00:00',
            'end': '2023-03-31 00:00:00'
        }

        # Mock drawdown analyzer
        mock_drawdown_analyzer = Mock()
        mock_drawdown_analyzer.get_analysis.return_value = {
            'max': {'drawdown': 5.2}
        }

        # Mock sharpe analyzer
        mock_sharpe_analyzer = Mock()
        mock_sharpe_analyzer.get_analysis.return_value = {
            'sharpe': 1.5
        }

        # Set up analyzer returns
        mock_analyzer.get_analysis.return_value = {}
        mock_strategy_result.analyzers.returns = mock_returns_analyzer
        mock_strategy_result.analyzers.drawdown = mock_drawdown_analyzer
        mock_strategy_result.analyzers.sharpe = mock_sharpe_analyzer
        mock_strategy_result.analyzers.trades = mock_trade_analyzer

        mock_cerebro.run.return_value = [mock_strategy_result]
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
        assert 'metrics' in results
        assert 'initial_capital' in results['metrics']
        assert 'final_value' in results['metrics']

    @patch('backtests.run_backtests.DataHandler')
    @patch('backtests.run_backtests.bt')
    def test_run_backtest_empty_data(self, mock_bt, mock_data_handler):
        """Test backtest with empty data."""
        # Setup mock to return empty dataframe
        mock_handler_instance = Mock()
        mock_handler_instance.get_historical.return_value = pd.DataFrame()
        mock_data_handler.return_value = mock_handler_instance

        engine = BacktestEngine()
        strategy = MockStrategy({})

        # Mock backtrader setup
        mock_cerebro = Mock()
        mock_broker = Mock()
        mock_cerebro.broker = mock_broker
        mock_bt.Cerebro.return_value = mock_cerebro

        # Return empty results when no data
        mock_cerebro.run.return_value = []
        mock_broker.getvalue.return_value = 5000.0  # Initial capital unchanged

        # Should handle empty data gracefully
        results = engine.run_backtest(
            strategy=strategy,
            symbols=['NODATA'],
            start_date='2023-01-01',
            end_date='2023-03-31'
        )

        # Should return results with no trades
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert results['metrics'].get('total_trades', 0) == 0
        assert results['metrics']['final_value'] == results['metrics']['initial_capital']
        assert results['metrics']['total_return'] == 0.0

    def test_print_summary(self, caplog):
        """Test summary printing."""
        engine = BacktestEngine()

        # Set up some mock results with correct structure
        engine.results = {
            'TestStrategy': {
                'metrics': {
                    'initial_capital': 5000,
                    'final_value': 6000,
                    'total_return': 20.0,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -10.0,
                    'total_trades': 50,
                    'winning_trades': 30,
                    'win_rate': 0.6,
                    'annual_return': 25.0,
                    'profit_factor': 2.5
                }
            }
        }

        with caplog.at_level(logging.INFO):
            engine.print_summary()

        assert "Initial Capital: $5,000.00" in caplog.text
        assert "Final Value: $6,000.00" in caplog.text
        assert "Total Return: 20.00%" in caplog.text
        assert "Sharpe Ratio: 1.50" in caplog.text

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
        engine.save_results(str(output_file))

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
        # Create mock strategy and risk context
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.init = Mock()
        mock_strategy.config = {'lookback_period': 10}
        mock_risk = Mock(spec=RiskContext)

        # Use backtrader properly
        cerebro = bt.Cerebro()

        # Add strategy with our mocks
        cerebro.addstrategy(
            AlgoStackStrategy,
            algostack_strategy=mock_strategy,
            risk_context=mock_risk
        )

        # Create a minimal data feed with enough history
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        data = bt.feeds.PandasData(
            dataname=pd.DataFrame({
                'open': [100] * 15,
                'high': [101] * 15,
                'low': [99] * 15,
                'close': [100] * 15,
                'volume': [1000] * 15
            }, index=dates)
        )
        cerebro.adddata(data)

        # Run to trigger initialization
        cerebro.run()

        # Verify init was called on the wrapped strategy
        mock_strategy.init.assert_called_once()

    def test_algostack_strategy_next(self):
        """Test strategy next() method."""
        # Create a mock strategy
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.init = Mock()
        mock_strategy.config = {'lookback_period': 10}
        mock_strategy.next = Mock(return_value=Signal(
            timestamp=datetime.now(),
            symbol="TEST",
            direction="LONG",
            strength=0.8,
            strategy_id="mock_strategy",
            price=100.5,
            metadata={}
        ))
        mock_strategy.size = Mock(return_value=(100, 95.0))

        mock_risk = Mock(spec=RiskContext)

        # Set up backtrader
        cerebro = bt.Cerebro()

        # Create test data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        test_data = pd.DataFrame({
            'open': [100 + i*0.1 for i in range(20)],
            'high': [101 + i*0.1 for i in range(20)],
            'low': [99 + i*0.1 for i in range(20)],
            'close': [100.5 + i*0.1 for i in range(20)],
            'volume': [1000000] * 20
        }, index=dates)

        data = bt.feeds.PandasData(dataname=test_data)
        cerebro.adddata(data, name="TEST")

        # Add strategy
        cerebro.addstrategy(
            AlgoStackStrategy,
            algostack_strategy=mock_strategy,
            risk_context=mock_risk
        )

        # Run backtest
        results = cerebro.run()

        # Verify next was called (at least for the last 10 bars since lookback is 10)
        assert mock_strategy.next.call_count >= 10

        # Check that signals were generated
        strategy = results[0]
        assert len(strategy.signals_history) > 0


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
        mock_cerebro.broker.getvalue.return_value = 110000.0  # 10% gain

        # Mock strategy results with analyzers
        mock_strategy_result = Mock()
        mock_strategy_result.analyzers = Mock()
        mock_strategy_result.analyzers.sharpe.get_analysis.return_value = {'sharperatio': 1.5}
        mock_strategy_result.analyzers.drawdown.get_analysis.return_value = {'max': {'drawdown': 5.0}}
        mock_strategy_result.analyzers.returns.get_analysis.return_value = {'rtot': 0.1}
        mock_strategy_result.analyzers.trades.get_analysis.return_value = {
            'total': {'total': 10},
            'won': {'total': 6},
            'lost': {'total': 4},
            'pnl': {'average': 100.0}
        }

        mock_cerebro.run.return_value = [mock_strategy_result]
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
        # Create enough data points to avoid IndexError
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        mock_handler_instance.get_historical.return_value = pd.DataFrame({
            'open': [100] * len(dates),
            'high': [101] * len(dates),
            'low': [99] * len(dates),
            'close': [100] * len(dates),
            'volume': [1000000] * len(dates)
        }, index=dates)
        mock_data_handler.return_value = mock_handler_instance

        engine = BacktestEngine()
        strategy = MockStrategy({'lookback_period': 10})  # Small lookback for test

        # Test with Alpha Vantage
        engine.run_backtest(
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
