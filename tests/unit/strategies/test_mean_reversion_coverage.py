"""
Test coverage for mean_reversion_equity.py focusing on missing coverage areas.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Mock talib import to test fallback
sys.modules['talib'] = None

from strategies.mean_reversion_equity import MeanReversionEquity


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
        test_data = mock_data.copy()

        # Prepare test data without calling calculate_indicators
        # since next() will call it internally
        test_data.loc[test_data.index[-1], 'close'] = 95  # Price

        # Set symbol attribute on data
        test_data.attrs['symbol'] = symbol

        # Generate signal
        signal = strategy.next(test_data)

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
        test_data = mock_data.copy()
        test_data.attrs['symbol'] = symbol

        # Calculate indicators first
        test_data = strategy.calculate_indicators(test_data)

        # Set up oversold condition
        test_data.loc[test_data.index[-1], 'rsi'] = -10  # Very oversold
        test_data.loc[test_data.index[-1], 'close'] = 95
        test_data.loc[test_data.index[-1], 'lower_band'] = 100
        test_data.loc[test_data.index[-1], 'upper_band'] = 110
        test_data.loc[test_data.index[-1], 'sma_20'] = 105
        test_data.loc[test_data.index[-1], 'atr'] = 2.0
        test_data.loc[test_data.index[-1], 'volume'] = 1500000
        test_data.loc[test_data.index[-1], 'volume_sma'] = 1000000
        test_data.loc[test_data.index[-1], 'volume_ratio'] = 1.5

        signal = strategy.next(test_data)

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
        test_data = mock_data.copy()
        test_data.attrs['symbol'] = symbol

        # Calculate indicators first
        test_data = strategy.calculate_indicators(test_data)

        # Set up existing position that exceeds holding period
        entry_time = datetime.now() - timedelta(days=3)
        strategy.positions[symbol] = {
            "entry_price": 100,
            "entry_atr": 2.0,
            "entry_time": entry_time
        }

        # Set current data
        test_data.loc[test_data.index[-1], 'close'] = 102  # Small profit
        test_data.loc[test_data.index[-1], 'atr'] = 2.0
        test_data.loc[test_data.index[-1], 'sma_exit'] = 101  # Above exit MA
        test_data.loc[test_data.index[-1], 'rsi'] = 50  # Neutral RSI

        signal = strategy.next(test_data)

        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.metadata['reason'] == 'holding_period_exit'
        assert symbol not in strategy.positions  # Position should be cleared

    # Performance metrics tests removed - get_performance_metrics functionality moved to MetricsCollector
    # The strategy class no longer has this method, so these tests are obsolete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
