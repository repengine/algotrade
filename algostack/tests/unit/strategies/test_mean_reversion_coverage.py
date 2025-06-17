"""
Test coverage for mean_reversion_equity.py focusing on missing coverage areas.
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import pytest

# Mock talib import to test fallback
sys.modules['talib'] = None

from algostack.strategies.mean_reversion_equity import MeanReversionEquity
from algostack.strategies.base import Signal, RiskContext


class TestMeanReversionCoverage:
    """Tests for missing coverage in MeanReversionEquity."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        return data
    
    def test_talib_import_fallback(self):
        """Test that pandas_indicators is used when talib is not available."""
        # The import already happened at module level with talib mocked to None
        # Check that the strategy can be instantiated without talib
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_factor": 1.5,
            "max_position_size": 0.1,
            "stop_loss_atr": 2.0,
            "take_profit_atr": 3.0,
            "holding_period": 5
        }
        
        strategy = MeanReversionEquity(config)
        assert strategy is not None
        assert hasattr(strategy, 'init')
        assert hasattr(strategy, 'next')
    
    def test_signal_generation_edge_cases(self, mock_data):
        """Test edge cases in signal generation logic."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_factor": 1.5,
            "max_position_size": 0.1,
            "stop_loss_atr": 2.0,
            "take_profit_atr": 3.0,
            "holding_period": 5
        }
        
        strategy = MeanReversionEquity(config)
        
        # Initialize strategy
        strategy.init()
        
        # Set up data for oversold condition
        symbol = "AAPL"
        strategy.data[symbol] = mock_data.copy()
        
        # Manually set indicators to trigger oversold signal
        strategy.data[symbol]['rsi'] = 25  # Below oversold threshold
        strategy.data[symbol]['close'] = 95  # Price
        strategy.data[symbol]['lower_band'] = 100  # Below lower band
        strategy.data[symbol]['upper_band'] = 110
        strategy.data[symbol]['sma'] = 105
        strategy.data[symbol]['atr'] = 2.0
        strategy.data[symbol]['volume'] = 1500000
        strategy.data[symbol]['volume_sma'] = 1000000
        strategy.data[symbol]['volume_ratio'] = 1.5
        
        # Generate signal
        signal = strategy.generate_signal(symbol)
        
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.symbol == symbol
        assert 0 < signal.strength <= 1.0
        
        # Check metadata
        assert 'reason' in signal.metadata
        assert signal.metadata['reason'] == 'mean_reversion_entry'
        assert 'rsi' in signal.metadata
        assert 'band_distance' in signal.metadata
        assert 'volume_ratio' in signal.metadata
        
        # Check position tracking
        assert symbol in strategy.positions
        assert 'entry_price' in strategy.positions[symbol]
        assert 'entry_atr' in strategy.positions[symbol]
        assert 'entry_time' in strategy.positions[symbol]
    
    def test_signal_strength_calculation_edge_case(self, mock_data):
        """Test signal strength calculation when rsi_oversold is 0."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 0.0,  # Edge case
            "rsi_overbought": 70.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_factor": 1.5,
            "max_position_size": 0.1,
            "stop_loss_atr": 2.0,
            "take_profit_atr": 3.0,
            "holding_period": 5
        }
        
        strategy = MeanReversionEquity(config)
        strategy.init()
        
        symbol = "AAPL"
        strategy.data[symbol] = mock_data.copy()
        
        # Set up oversold condition
        strategy.data[symbol]['rsi'] = -10  # Very oversold
        strategy.data[symbol]['close'] = 95
        strategy.data[symbol]['lower_band'] = 100
        strategy.data[symbol]['upper_band'] = 110
        strategy.data[symbol]['sma'] = 105
        strategy.data[symbol]['atr'] = 2.0
        strategy.data[symbol]['volume'] = 1500000
        strategy.data[symbol]['volume_sma'] = 1000000
        strategy.data[symbol]['volume_ratio'] = 1.5
        
        signal = strategy.generate_signal(symbol)
        
        # When rsi_oversold is 0, strength should default to 0.5
        assert signal is not None
        assert signal.strength == 0.5
    
    def test_exit_conditions_holding_period(self, mock_data):
        """Test exit signal generation based on holding period."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_factor": 1.5,
            "max_position_size": 0.1,
            "stop_loss_atr": 2.0,
            "take_profit_atr": 3.0,
            "holding_period": 2  # 2 days
        }
        
        strategy = MeanReversionEquity(config)
        strategy.init()
        
        symbol = "AAPL"
        strategy.data[symbol] = mock_data.copy()
        
        # Set up existing position that exceeds holding period
        entry_time = datetime.now() - timedelta(days=3)
        strategy.positions[symbol] = {
            "entry_price": 100,
            "entry_atr": 2.0,
            "entry_time": entry_time
        }
        
        # Set current data
        strategy.data[symbol]['close'] = 102  # Small profit
        strategy.data[symbol]['atr'] = 2.0
        
        signal = strategy.generate_signal(symbol)
        
        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.metadata['reason'] == 'holding_period_exit'
        assert symbol not in strategy.positions  # Position should be cleared
    
    def test_get_performance_metrics_empty_trades(self):
        """Test performance metrics calculation with empty trades."""
        config = {
            "name": "test_strategy",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }
        strategy = MeanReversionEquity(config)
        
        empty_trades = pd.DataFrame()
        metrics = strategy.get_performance_metrics(empty_trades)
        
        assert metrics == {}
    
    def test_get_performance_metrics_with_trades(self):
        """Test performance metrics calculation with actual trades."""
        config = {
            "name": "test_strategy",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }
        strategy = MeanReversionEquity(config)
        
        # Create sample trades
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150, 0, -20],
            'symbol': ['AAPL'] * 7,
            'exit_time': pd.date_range('2024-01-01', periods=7, freq='D')
        })
        
        metrics = strategy.get_performance_metrics(trades)
        
        assert 'win_rate' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'profit_factor' in metrics
        assert 'trade_frequency' in metrics
        
        # Check calculations
        assert metrics['win_rate'] == 3/7  # 3 wins out of 7 trades
        assert metrics['avg_win'] == 150.0  # (100 + 200 + 150) / 3
        assert metrics['avg_loss'] == 33.333333333333336  # (50 + 30 + 20) / 3
        assert metrics['profit_factor'] == 4.5  # 450 / 100
    
    def test_get_performance_metrics_all_wins(self):
        """Test performance metrics when all trades are winners."""
        config = {
            "name": "test_strategy",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }
        strategy = MeanReversionEquity(config)
        
        trades = pd.DataFrame({
            'pnl': [100, 200, 150],
            'symbol': ['AAPL'] * 3,
            'exit_time': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        
        metrics = strategy.get_performance_metrics(trades)
        
        assert metrics['win_rate'] == 1.0
        assert metrics['avg_win'] == 150.0
        assert metrics['avg_loss'] == 1.0  # Default when no losses
        assert metrics['profit_factor'] == float('inf')  # No losses
    
    def test_get_performance_metrics_all_losses(self):
        """Test performance metrics when all trades are losers."""
        config = {
            "name": "test_strategy",
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }
        strategy = MeanReversionEquity(config)
        
        trades = pd.DataFrame({
            'pnl': [-100, -200, -150],
            'symbol': ['AAPL'] * 3,
            'exit_time': pd.date_range('2024-01-01', periods=3, freq='D')
        })
        
        metrics = strategy.get_performance_metrics(trades)
        
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_win'] == 0.0
        assert metrics['avg_loss'] == 150.0
        assert metrics['profit_factor'] == 0.0  # No wins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])