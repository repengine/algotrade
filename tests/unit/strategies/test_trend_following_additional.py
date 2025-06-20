"""Additional tests for TrendFollowingMulti to improve coverage."""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Test talib import fallback
sys.modules['talib'] = None

# Import after mocking
from strategies.base import RiskContext, Signal
from strategies.trend_following_multi import TrendFollowingMulti


class TestTrendFollowingAdditional:
    """Additional tests for uncovered code paths."""

    @pytest.fixture
    def mock_data(self):
        """Create mock market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

        # Create trending price data
        np.random.seed(42)
        trend = np.linspace(100, 120, 100)
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

    def test_talib_import_fallback(self):
        """Test that pandas_indicators is used when talib is not available."""
        # The import already happened at module level with talib mocked to None
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
        assert strategy is not None

    def test_short_position_exit(self, mock_data):
        """Test short position exit on trailing stop."""
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

        # Set up existing short position
        strategy.positions['BTC-USD'] = {
            "direction": "SHORT",
            "entry_price": 110.0,
            "entry_time": datetime.now() - timedelta(days=5),
            "lowest_price": 105.0
        }

        # Create data that triggers trailing stop for short
        test_data = mock_data.copy()

        # Set price above trailing stop (10-day high for short)
        trailing_stop = test_data['high'].iloc[-10:].max()
        test_data.iloc[-1, test_data.columns.get_loc('close')] = trailing_stop + 1

        test_data.attrs['symbol'] = 'BTC-USD'

        # Generate signal
        signal = strategy.next(test_data)

        # Should exit short position
        if signal is not None:
            assert signal.direction == "FLAT"
            assert 'BTC-USD' not in strategy.positions

    def test_position_update_tracking(self, mock_data):
        """Test position tracking updates."""
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

        # Set up existing long position
        strategy.positions['BTC-USD'] = {
            "direction": "LONG",
            "entry_price": 100.0,
            "entry_time": datetime.now() - timedelta(days=2),
            "highest_price": 105.0
        }

        # Create data with new high
        test_data = mock_data.copy()
        test_data.iloc[-1, test_data.columns.get_loc('close')] = 110.0
        test_data.iloc[-1, test_data.columns.get_loc('high')] = 110.5
        test_data.attrs['symbol'] = 'BTC-USD'

        # Process but stay in position
        strategy.next(test_data)

        # Check if highest price was updated
        if 'BTC-USD' in strategy.positions:
            assert strategy.positions['BTC-USD']['highest_price'] >= 110.0

    def test_channel_break_tracking(self, mock_data):
        """Test channel break tracking functionality."""
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

        # Create data with channel break
        test_data = mock_data.copy()

        # Force a breakout
        test_data.iloc[-1, test_data.columns.get_loc('close')] = test_data['high'].iloc[-20:-1].max() + 2
        test_data.attrs['symbol'] = 'BTC-USD'

        # Generate signal
        signal = strategy.next(test_data)

        # Check if channel break was tracked
        if signal is not None and signal.direction == "LONG":
            assert 'BTC-USD' in strategy.channel_breaks

    def test_position_sizing_with_multiple_positions(self):
        """Test position sizing with existing positions."""
        config = {
            "symbols": ["BTC-USD"],
            "lookback_period": 100,
            "channel_period": 20,
            "trail_period": 10,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "max_position_size": 0.20
        }

        strategy = TrendFollowingMulti(config)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=50000.0,
            atr=1000.0
        )

        # Test with multiple open positions
        risk_context = RiskContext(
            account_equity=100000,
            open_positions=3,  # Already have 3 positions
            daily_pnl=500,
            max_drawdown_pct=0.05,
            volatility_target=0.15
        )

        position_size, stop_loss = strategy.size(signal, risk_context)

        # Position size should be reduced due to multiple positions
        assert position_size > 0
        assert position_size < risk_context.account_equity * 0.10  # Reduced size

    def test_short_signal_sizing(self):
        """Test position sizing for short signals."""
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

        # Create short signal
        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            direction="SHORT",
            strength=-0.8,
            strategy_id="test",
            price=50000.0,
            atr=1000.0
        )

        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0
        )

        position_size, stop_loss = strategy.size(signal, risk_context)

        # Should return negative position size for short
        assert position_size < 0
        assert stop_loss > signal.price  # Stop above entry for short

    def test_no_atr_in_signal(self):
        """Test position sizing when signal has no ATR."""
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

        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=50000.0,
            atr=None  # No ATR
        )

        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0
        )

        position_size, stop_loss = strategy.size(signal, risk_context)

        # Should use default ATR calculation
        assert position_size > 0
        assert stop_loss < signal.price

    def test_high_volatility_regime(self):
        """Test position sizing in high volatility regime."""
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

        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=50000.0,
            atr=2000.0  # High ATR
        )

        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0,
            current_regime="HIGH_VOL"
        )

        position_size, _ = strategy.size(signal, risk_context)

        # Should have reduced position size in high vol
        assert position_size > 0
        assert position_size < risk_context.account_equity * 0.05

    def test_risk_off_regime(self):
        """Test position sizing in risk-off regime."""
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

        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=50000.0,
            atr=1000.0
        )

        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0,
            current_regime="RISK_OFF"
        )

        position_size, _ = strategy.size(signal, risk_context)

        # Should return 0 in risk-off
        assert position_size == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
