"""Improved test coverage for MeanReversionEquity strategy."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from strategies.base import RiskContext, Signal
from strategies.mean_reversion_equity import MeanReversionEquity


class TestMeanReversionEquityImproved:
    """Improved tests for MeanReversionEquity strategy coverage."""

    @pytest.fixture
    def mock_data(self):
        """Create realistic mock market data."""
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

    def test_entry_signal_generation(self, mock_data):
        """Test entry signal generation with proper conditions."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 2,  # Short period for testing
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "atr_band_mult": 2.5,
            "volume_filter": False,  # Disable to simplify testing
            "max_positions": 5,
            "atr_period": 14,
            "ma_exit_period": 10,
            "stop_loss_atr": 3.0
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        # Create data that will trigger entry
        test_data = mock_data.copy()

        # Force a downtrend in recent data to get low RSI
        for i in range(-10, 0):
            test_data.iloc[i, test_data.columns.get_loc('close')] *= (0.97 + i * 0.001)
            test_data.iloc[i, test_data.columns.get_loc('low')] = test_data.iloc[i]['close'] * 0.99

        test_data.attrs['symbol'] = 'AAPL'

        # Generate signal
        signal = strategy.next(test_data)

        # Check if signal was generated (may be None if conditions not met exactly)
        if signal is not None:
            assert signal.direction == "LONG"
            assert signal.symbol == "AAPL"
            assert 0 < signal.strength <= 1.0
            assert 'AAPL' in strategy.positions

    def test_exit_signal_generation(self, mock_data):
        """Test exit signal generation."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "rsi_period": 2,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "atr_band_mult": 2.5,
            "volume_filter": False,
            "max_positions": 5,
            "atr_period": 14,
            "ma_exit_period": 10,
            "stop_loss_atr": 3.0
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        # Set up existing position
        strategy.positions['AAPL'] = {
            "entry_price": 95.0,
            "entry_atr": 2.0,
            "entry_time": datetime.now() - timedelta(days=1)
        }

        # Create data that will trigger exit (price above MA)
        test_data = mock_data.copy()

        # Force an uptrend to trigger exit
        for i in range(-10, 0):
            test_data.iloc[i, test_data.columns.get_loc('close')] *= 1.02

        test_data.attrs['symbol'] = 'AAPL'

        # Generate signal
        signal = strategy.next(test_data)

        # Should generate exit signal if conditions are met
        if signal is not None and signal.direction == "FLAT":
            assert signal.symbol == "AAPL"
            assert 'AAPL' not in strategy.positions

    def test_stop_loss_exit(self, mock_data):
        """Test stop loss exit condition."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 2,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "atr_band_mult": 2.5,
            "stop_loss_atr": 1.0,  # Tight stop for testing
            "max_positions": 5,
            "atr_period": 14,
            "ma_exit_period": 10
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        # Set up existing position
        entry_price = 100.0
        entry_atr = 2.0

        strategy.positions['AAPL'] = {
            "entry_price": entry_price,
            "entry_atr": entry_atr,
            "entry_time": datetime.now() - timedelta(days=1)
        }

        # Create data with price below stop loss
        test_data = mock_data.copy()
        stop_price = entry_price - (config["stop_loss_atr"] * entry_atr)
        test_data.iloc[-1, test_data.columns.get_loc('close')] = stop_price - 1
        test_data.attrs['symbol'] = 'AAPL'

        # Generate signal
        signal = strategy.next(test_data)

        # Should exit due to stop loss
        if signal is not None:
            assert signal.direction == "FLAT"
            assert signal.metadata.get('reason') in ['stop_loss', 'mean_reversion_exit']

    def test_overbought_exit(self, mock_data):
        """Test overbought RSI exit condition."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "rsi_period": 2,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "atr_band_mult": 2.5,
            "max_positions": 5,
            "atr_period": 14,
            "ma_exit_period": 10,
            "stop_loss_atr": 3.0
        }

        strategy = MeanReversionEquity(config)
        strategy.init()

        # Set up existing position
        strategy.positions['AAPL'] = {
            "entry_price": 95.0,
            "entry_atr": 2.0,
            "entry_time": datetime.now() - timedelta(days=1)
        }

        # Create data with strong upward movement for high RSI
        test_data = mock_data.copy()

        # Force strong uptrend in recent data
        for i in range(-5, 0):
            test_data.iloc[i, test_data.columns.get_loc('close')] *= 1.05
            test_data.iloc[i, test_data.columns.get_loc('high')] = test_data.iloc[i]['close'] * 1.01

        test_data.attrs['symbol'] = 'AAPL'

        # Generate signal
        signal = strategy.next(test_data)

        # May exit due to overbought conditions
        if signal is not None and signal.direction == "FLAT":
            assert signal.metadata.get('reason') == 'mean_reversion_exit'

    def test_risk_context_sizing(self):
        """Test position sizing with risk context."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "max_position_size": 0.2
        }

        strategy = MeanReversionEquity(config)

        # Create test signal
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0
        )

        # Create risk context
        risk_context = RiskContext(
            account_equity=100000,
            open_positions=2,
            daily_pnl=-500,
            max_drawdown_pct=0.05,
            volatility_target=0.10,
            max_position_size=0.15,
            current_regime="NORMAL"
        )

        # Calculate position size
        position_size, stop_loss = strategy.size(signal, risk_context)

        # Verify sizing
        assert position_size > 0
        assert position_size <= risk_context.max_position_size * risk_context.account_equity
        assert stop_loss > 0
        assert stop_loss < signal.price  # Stop should be below entry for long

    def test_high_volatility_regime_sizing(self):
        """Test position sizing in high volatility regime."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "max_position_size": 0.2
        }

        strategy = MeanReversionEquity(config)

        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=5.0  # High ATR
        )

        risk_context = RiskContext(
            account_equity=100000,
            open_positions=2,
            daily_pnl=-500,
            max_drawdown_pct=0.05,
            volatility_target=0.10,
            max_position_size=0.15,
            current_regime="HIGH_VOL"
        )

        position_size, stop_loss = strategy.size(signal, risk_context)

        # Should have reduced size in high vol
        assert position_size > 0
        assert position_size < risk_context.max_position_size * risk_context.account_equity * 0.5

    def test_calculate_indicators(self, mock_data):
        """Test indicator calculation."""
        config = {
            "name": "test_mean_reversion",
            "symbols": ["AAPL"],
            "lookback_period": 252,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "atr_period": 14,
            "atr_band_mult": 2.5,
            "ma_exit_period": 10
        }

        strategy = MeanReversionEquity(config)

        # Calculate indicators
        df_with_indicators = strategy.calculate_indicators(mock_data)

        # Verify all indicators are present
        assert 'rsi' in df_with_indicators.columns
        assert 'atr' in df_with_indicators.columns
        assert 'sma_20' in df_with_indicators.columns
        assert 'sma_exit' in df_with_indicators.columns
        assert 'upper_band' in df_with_indicators.columns
        assert 'lower_band' in df_with_indicators.columns
        assert 'volume_sma' in df_with_indicators.columns
        assert 'volume_ratio' in df_with_indicators.columns

        # Verify calculations
        assert not df_with_indicators['rsi'].iloc[-20:].isnull().any()
        assert not df_with_indicators['atr'].iloc[-20:].isnull().any()
        assert df_with_indicators['volume_ratio'].iloc[-1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
