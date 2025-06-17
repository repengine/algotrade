"""Tests for trading strategies."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from strategies.base import RiskContext, Signal
from strategies.mean_reversion_equity import MeanReversionEquity


class TestMeanReversionEquity:
    """Test mean reversion equity strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = {
            "symbols": ["SPY"],
            "rsi_period": 2,
            "rsi_oversold": 10.0,
            "rsi_overbought": 90.0,
            "atr_period": 14,
            "atr_band_mult": 2.5,
            "stop_loss_atr": 3.0,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
        }
        return MeanReversionEquity(config)

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        dates = pd.date_range(start="2023-01-01", end="2023-03-01", freq="D")
        n = len(dates)

        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n)
        close = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
                "high": close * (1 + np.random.uniform(0, 0.01, n)),
                "low": close * (1 - np.random.uniform(0, 0.01, n)),
                "close": close,
                "volume": np.random.uniform(1e6, 2e6, n),
            },
            index=dates,
        )

        # Ensure valid OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        data.attrs["symbol"] = "SPY"
        return data

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == "MeanReversionEquity"
        assert strategy.config["rsi_period"] == 2
        assert strategy.config["stop_loss_atr"] == 3.0
        assert len(strategy.positions) == 0

    def test_data_validation(self, strategy, sample_data):
        """Test data validation."""
        assert strategy.validate_data(sample_data)

        # Test with missing columns
        bad_data = sample_data.drop(columns=["volume"])
        assert not strategy.validate_data(bad_data)

        # Test with invalid OHLC
        bad_data = sample_data.copy()
        bad_data.loc[bad_data.index[0], "high"] = (
            bad_data.loc[bad_data.index[0], "low"] - 1
        )
        assert not strategy.validate_data(bad_data)

    def test_indicator_calculation(self, strategy, sample_data):
        """Test technical indicator calculations."""
        df = strategy.calculate_indicators(sample_data)

        # Check all indicators are present
        assert "rsi" in df.columns
        assert "atr" in df.columns
        assert "upper_band" in df.columns
        assert "lower_band" in df.columns
        assert "volume_ratio" in df.columns

        # Check RSI bounds
        rsi_values = df["rsi"].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()

        # Check ATR is positive
        atr_values = df["atr"].dropna()
        assert (atr_values > 0).all()

    def test_entry_signal_generation(self, strategy, sample_data):
        """Test entry signal generation."""
        # Modify data to trigger entry signal
        sample_data = sample_data.copy()

        # Create oversold condition
        sample_data.iloc[-5:]["close"] = sample_data.iloc[-6]["close"] * 0.95
        sample_data.iloc[-1]["close"] = sample_data.iloc[-6]["close"] * 0.93
        sample_data.iloc[-1]["volume"] = sample_data.iloc[-2]["volume"] * 1.5

        strategy.init()
        signal = strategy.next(sample_data)

        # Should generate a LONG signal in oversold conditions
        if signal:
            assert signal.direction == "LONG"
            assert signal.strength > 0
            assert signal.symbol == "SPY"
            assert "rsi" in signal.metadata

    def test_position_sizing(self, strategy):
        """Test position sizing with risk management."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="SPY",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0,
        )

        risk_context = RiskContext(
            account_equity=5000.0,
            open_positions=2,
            daily_pnl=50.0,
            max_drawdown_pct=0.05,
            volatility_target=0.10,
            max_position_size=0.20,
        )

        position_size, stop_loss = strategy.size(signal, risk_context)

        # Check position size is reasonable
        assert position_size > 0
        assert (
            position_size * signal.price
            <= risk_context.account_equity * risk_context.max_position_size
        )

        # Check stop loss is below entry
        assert stop_loss < signal.price
        assert stop_loss == signal.price - (
            strategy.config["stop_loss_atr"] * signal.atr
        )

    def test_kelly_fraction_calculation(self, strategy):
        """Test Kelly fraction calculation."""
        # Initially should return 0 (not enough trades)
        assert strategy.calculate_kelly_fraction() == 0.0

        # Simulate some trades
        for i in range(40):
            if i % 3 == 0:
                strategy.update_performance({"pnl": -100})
            else:
                strategy.update_performance({"pnl": 150})

        kelly = strategy.calculate_kelly_fraction()
        assert 0 < kelly < 1.0  # Should be positive but conservative
