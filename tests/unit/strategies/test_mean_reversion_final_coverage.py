"""Final coverage tests for MeanReversionEquity strategy."""

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Test talib import fallback
sys.modules['talib'] = None

from strategies.base import RiskContext, Signal
from strategies.mean_reversion_equity import MeanReversionEquity


class TestMeanReversionFinalCoverage:
    """Final tests to achieve 100% coverage for MeanReversionEquity."""

    @pytest.fixture
    def mock_data(self):
        """Create mock market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

        # Create realistic price movement
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'open': close_prices * np.random.uniform(0.98, 1.02, 100),
            'high': close_prices * np.random.uniform(1.01, 1.03, 100),
            'low': close_prices * np.random.uniform(0.97, 0.99, 100),
            'close': close_prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)

        return data

    def test_insufficient_data(self, mock_data):
        """Test handling of insufficient data."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "atr_period": 14
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        # Use only 10 rows of data (less than atr_period)
        small_data = mock_data[:10].copy()
        small_data.attrs['symbol'] = 'AAPL'

        signal = strategy.next(small_data)
        assert signal is None  # Should return None due to insufficient data

    def test_no_position_no_entry_conditions(self, mock_data):
        """Test when no positions and no entry conditions met."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 10.0,  # Very low threshold
            "rsi_overbought": 90.0,  # Very high threshold
            "atr_period": 14,
            "atr_band_mult": 0.1,  # Very tight bands
            "volume_filter": True  # Enable volume filter
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        test_data = mock_data.copy()
        test_data.attrs['symbol'] = 'AAPL'

        # Ensure no entry conditions are met
        # By using normal market data, RSI should be around 50
        signal = strategy.next(test_data)
        assert signal is None  # No signal when conditions not met

    def test_entry_with_volume_filter_fail(self, mock_data):
        """Test entry conditions met but volume filter fails."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 2,  # Short period
            "rsi_oversold": 50.0,  # Easy to trigger
            "rsi_overbought": 90.0,
            "atr_period": 14,
            "atr_band_mult": 10.0,  # Wide bands
            "volume_filter": True,  # Enable volume filter
            "max_positions": 5
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        test_data = mock_data.copy()

        # Force low volume on last bar
        test_data.iloc[-1, test_data.columns.get_loc('volume')] = 100  # Very low volume
        test_data.attrs['symbol'] = 'AAPL'

        strategy.next(test_data)
        # May return None due to volume filter

    def test_position_size_with_signal_without_atr(self):
        """Test position sizing when signal has no ATR."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "stop_loss_atr": 2.0
        }

        strategy = MeanReversionEquity(config)

        # Create signal without ATR
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=None  # No ATR
        )

        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0
        )

        position_size, stop_loss = strategy.size(signal, risk_context)

        # Should use default ATR
        assert position_size > 0
        assert stop_loss == signal.price - (config["stop_loss_atr"] * signal.price * 0.02)

    def test_short_position_sizing(self):
        """Test position sizing for short positions."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "stop_loss_atr": 2.0
        }

        strategy = MeanReversionEquity(config)

        # Create short signal
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="SHORT",
            strength=-0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0
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

    def test_position_already_at_max(self, mock_data):
        """Test when already at max positions."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "max_positions": 2  # Only allow 2 positions
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        # Add 2 existing positions
        strategy.positions['AAPL'] = {
            "entry_price": 100,
            "entry_atr": 2.0,
            "entry_time": datetime.now()
        }
        strategy.positions['GOOGL'] = {
            "entry_price": 2800,
            "entry_atr": 50.0,
            "entry_time": datetime.now()
        }

        test_data = mock_data.copy()
        test_data.attrs['symbol'] = 'MSFT'

        signal = strategy.next(test_data)
        assert signal is None  # Should not enter new position

    def test_drawdown_reduction(self):
        """Test position size reduction during drawdown."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }

        strategy = MeanReversionEquity(config)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0
        )

        # Risk context with significant drawdown
        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=-5000,
            max_drawdown_pct=0.15  # 15% drawdown
        )

        position_size, _ = strategy.size(signal, risk_context)

        # Compare with no drawdown
        risk_context_normal = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0
        )

        position_size_normal, _ = strategy.size(signal, risk_context_normal)

        # Position should be smaller during drawdown
        assert abs(position_size) < abs(position_size_normal)

    def test_risk_off_regime(self):
        """Test position sizing in risk-off regime."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }

        strategy = MeanReversionEquity(config)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0
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

    def test_multiple_open_positions_scaling(self):
        """Test position scaling with multiple open positions."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "max_positions": 5
        }

        strategy = MeanReversionEquity(config)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0
        )

        # Test with different numbers of open positions
        position_sizes = []

        for open_positions in [0, 2, 4]:
            risk_context = RiskContext(
                account_equity=100000,
                open_positions=open_positions,
                daily_pnl=0,
                max_drawdown_pct=0.0
            )

            position_size, _ = strategy.size(signal, risk_context)
            position_sizes.append(abs(position_size))

        # Position size should decrease with more open positions
        assert position_sizes[0] > position_sizes[1] > position_sizes[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
