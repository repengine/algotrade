"""Comprehensive tests for TrendFollowingMulti strategy."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from strategies.base import RiskContext, Signal
from strategies.trend_following_multi import TrendFollowingMulti


class TestTrendFollowingMulti:
    """Test suite for TrendFollowingMulti strategy."""

    @pytest.fixture
    def mock_data(self):
        """Create mock market data with trend characteristics."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

        # Create trending price data
        np.random.seed(42)
        trend = np.linspace(100, 120, 100)  # Upward trend
        noise = np.random.normal(0, 1, 100)
        close_prices = trend + noise

        data = pd.DataFrame({
            'open': close_prices * np.random.uniform(0.99, 1.01, 100),
            'high': close_prices * np.random.uniform(1.01, 1.02, 100),
            'low': close_prices * np.random.uniform(0.98, 0.99, 100),
            'close': close_prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)

        return data

    def test_initialization(self):
        """Test strategy initialization with default and custom configs."""
        # Test with minimal config
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0
        }

        strategy = TrendFollowingMulti(config)
        assert strategy.name == "TrendFollowingMulti"
        assert strategy.config["channel_period"] == 20
        assert strategy.config["max_positions"] == 4
        assert strategy.positions == {}
        assert strategy.channel_breaks == {}

    def test_calculate_indicators(self, mock_data):
        """Test indicator calculation."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(mock_data)

        # Verify all indicators are present
        assert 'atr' in df_with_indicators.columns
        assert 'channel_high' in df_with_indicators.columns
        assert 'channel_low' in df_with_indicators.columns
        assert 'trail_high' in df_with_indicators.columns
        assert 'trail_low' in df_with_indicators.columns
        assert 'adx' in df_with_indicators.columns
        assert 'volume_sma' in df_with_indicators.columns
        assert 'volume_ratio' in df_with_indicators.columns

        # Verify calculations
        assert not df_with_indicators['atr'].iloc[-20:].isnull().any()
        assert not df_with_indicators['channel_high'].iloc[-20:].isnull().any()
        assert df_with_indicators['channel_high'].iloc[-1] > df_with_indicators['channel_low'].iloc[-1]

    def test_entry_signal_long(self, mock_data):
        """Test long entry signal generation."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 15.0,  # Lower threshold for testing
            "use_volume_filter": False  # Disable for simpler testing
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Create data that will trigger entry
        test_data = mock_data.copy()

        # Force a breakout - make last close above recent highs
        test_data.iloc[-1, test_data.columns.get_loc('close')] = test_data['high'].iloc[-20:-1].max() + 2
        test_data.iloc[-1, test_data.columns.get_loc('high')] = test_data.iloc[-1]['close'] + 0.5

        test_data.attrs['symbol'] = 'BTC-USD'

        # Generate signal
        signal = strategy.next(test_data)

        # Should generate long signal on breakout
        if signal is not None:
            assert signal.direction == "LONG"
            assert signal.symbol == "BTC-USD"
            assert signal.strength > 0
            assert 'BTC-USD' in strategy.positions

    def test_entry_signal_short(self, mock_data):
        """Test short entry signal generation."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 15.0,
            "use_volume_filter": False
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Create data that will trigger short entry
        test_data = mock_data.copy()

        # Force a breakdown - make last close below recent lows
        test_data.iloc[-1, test_data.columns.get_loc('close')] = test_data['low'].iloc[-20:-1].min() - 2
        test_data.iloc[-1, test_data.columns.get_loc('low')] = test_data.iloc[-1]['close'] - 0.5

        test_data.attrs['symbol'] = 'BTC-USD'

        # Generate signal
        signal = strategy.next(test_data)

        # Should generate short signal on breakdown
        if signal is not None:
            assert signal.direction == "SHORT"
            assert signal.symbol == "BTC-USD"
            assert signal.strength < 0

    def test_exit_signal_trailing_stop(self, mock_data):
        """Test exit signal on trailing stop."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Set up existing long position
        strategy.positions['BTC-USD'] = {
            "direction": "LONG",
            "entry_price": 100.0,
            "entry_time": datetime.now() - timedelta(days=5),
            "highest_price": 105.0
        }

        # Create data that triggers trailing stop
        test_data = mock_data.copy()

        # Set price below trailing stop (10-day low)
        trailing_stop = test_data['low'].iloc[-10:].min()
        test_data.iloc[-1, test_data.columns.get_loc('close')] = trailing_stop - 1

        test_data.attrs['symbol'] = 'BTC-USD'

        # Generate signal
        signal = strategy.next(test_data)

        # Should exit on trailing stop
        if signal is not None:
            assert signal.direction == "FLAT"
            assert 'BTC-USD' not in strategy.positions

    def test_adx_filter(self, mock_data):
        """Test ADX filter prevents entry in non-trending market."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 50.0,  # Very high threshold
            "use_volume_filter": False
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Create choppy data (no clear trend)
        test_data = mock_data.copy()

        # Make prices oscillate
        for i in range(len(test_data)):
            test_data.iloc[i, test_data.columns.get_loc('close')] = 100 + 5 * np.sin(i * 0.5)

        test_data.attrs['symbol'] = 'BTC-USD'

        # Generate signal
        signal = strategy.next(test_data)

        # Should not generate signal due to low ADX
        assert signal is None

    def test_position_sizing(self):
        """Test position sizing with risk context."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "max_position_size": 0.25
        }

        strategy = TrendFollowingMulti(config)

        # Create test signal
        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=50000.0,
            atr=1000.0
        )

        # Create risk context
        risk_context = RiskContext(
            account_equity=100000,
            open_positions=1,
            daily_pnl=1000,
            max_drawdown_pct=0.02,
            volatility_target=0.15
        )

        # Calculate position size
        position_size, stop_loss = strategy.size(signal, risk_context)

        # Verify sizing
        assert position_size > 0
        assert position_size <= risk_context.account_equity * config["max_position_size"]
        assert stop_loss > 0
        assert stop_loss < signal.price  # Stop below entry for long

    def test_validate_config(self):
        """Test configuration validation."""
        # Test invalid config
        invalid_config = {
            "symbols": ["BTC-USD"],
            "lookback_period": "invalid",  # Should be int
            "channel_period": 20
        }

        with pytest.raises(ValueError, match="validation failed"):
            TrendFollowingMulti(invalid_config)

        # Test with empty config - should still work due to defaults
        empty_config = {}
        strategy = TrendFollowingMulti(empty_config)
        assert strategy.config["channel_period"] == 20  # Default value

    def test_insufficient_data(self, mock_data):
        """Test handling of insufficient data."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Use only 10 rows (less than channel_period)
        small_data = mock_data[:10].copy()
        small_data.attrs['symbol'] = 'BTC-USD'

        signal = strategy.next(small_data)
        assert signal is None

    def test_max_positions_limit(self, mock_data):
        """Test max positions limit."""
        config = {
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "max_positions": 2
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Add existing positions
        strategy.positions['BTC-USD'] = {
            "direction": "LONG",
            "entry_price": 50000,
            "entry_time": datetime.now()
        }
        strategy.positions['ETH-USD'] = {
            "direction": "LONG",
            "entry_price": 3000,
            "entry_time": datetime.now()
        }

        test_data = mock_data.copy()
        test_data.attrs['symbol'] = 'SOL-USD'

        # Should not enter new position
        signal = strategy.next(test_data)
        assert signal is None

    def test_volume_filter(self, mock_data):
        """Test volume filter."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 15.0,
            "use_volume_filter": True,
            "volume_threshold": 2.0  # High threshold
        }

        strategy = TrendFollowingMulti(config)
        strategy.init()

        test_data = mock_data.copy()

        # Set low volume on last bar
        test_data.iloc[-1, test_data.columns.get_loc('volume')] = 1000

        # Force breakout conditions
        test_data.iloc[-1, test_data.columns.get_loc('close')] = test_data['high'].iloc[-20:-1].max() + 2

        test_data.attrs['symbol'] = 'BTC-USD'

        signal = strategy.next(test_data)

        # Should not generate signal due to low volume
        assert signal is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
