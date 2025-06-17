"""Tests for pandas indicators."""

import numpy as np
import pandas as pd
import pytest

from scripts.pandas_indicators import PandasIndicators, create_talib_compatible_module


class TestPandasIndicators:
    """Test pandas indicator implementations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        data = pd.DataFrame(
            {
                "open": close + np.random.randn(100) * 0.2,
                "high": close + np.abs(np.random.randn(100) * 0.3),
                "low": close - np.abs(np.random.randn(100) * 0.3),
                "close": close,
                "volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    def test_sma(self, sample_data):
        """Test Simple Moving Average."""
        result = PandasIndicators.SMA(sample_data["close"], 10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()

        # First 9 values should use expanding window
        assert result.iloc[9] == sample_data["close"].iloc[:10].mean()

    def test_ema(self, sample_data):
        """Test Exponential Moving Average."""
        result = PandasIndicators.EMA(sample_data["close"], 10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()

    def test_rsi(self, sample_data):
        """Test Relative Strength Index."""
        result = PandasIndicators.RSI(sample_data["close"], 14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()

        # RSI should be between 0 and 100
        assert result.min() >= 0
        assert result.max() <= 100

        # Test with period 2 (as used in strategies)
        result_2 = PandasIndicators.RSI(sample_data["close"], 2)
        assert result_2.min() >= 0
        assert result_2.max() <= 100

    def test_macd(self, sample_data):
        """Test MACD."""
        macd, signal, hist = PandasIndicators.MACD(sample_data["close"])

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_data)

        # MACD histogram should be the difference
        expected_hist = macd - signal
        pd.testing.assert_series_equal(hist, expected_hist, check_names=False)

    def test_bbands(self, sample_data):
        """Test Bollinger Bands."""
        upper, middle, lower = PandasIndicators.BBANDS(sample_data["close"])

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # Upper should be above middle, lower below (excluding NaN values)
        mask = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[mask] >= middle[mask]).all()
        assert (middle[mask] >= lower[mask]).all()

    def test_atr(self, sample_data):
        """Test Average True Range."""
        result = PandasIndicators.ATR(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert (result >= 0).all()  # ATR should be non-negative

    def test_adx(self, sample_data):
        """Test Average Directional Index."""
        result = PandasIndicators.ADX(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_stoch(self, sample_data):
        """Test Stochastic Oscillator."""
        slowk, slowd = PandasIndicators.STOCH(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert isinstance(slowk, pd.Series)
        assert isinstance(slowd, pd.Series)
        assert len(slowk) == len(sample_data)

        # Stochastic should be between 0 and 100
        assert slowk.min() >= 0
        assert slowk.max() <= 100
        assert slowd.min() >= 0
        assert slowd.max() <= 100

    def test_roc(self, sample_data):
        """Test Rate of Change."""
        result = PandasIndicators.ROC(sample_data["close"], 10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

        # Manual calculation for verification
        expected = (
            (sample_data["close"] - sample_data["close"].shift(10))
            / sample_data["close"].shift(10)
            * 100
        )
        pd.testing.assert_series_equal(
            result[10:], expected[10:], check_names=False, rtol=1e-5
        )

    def test_mom(self, sample_data):
        """Test Momentum."""
        result = PandasIndicators.MOM(sample_data["close"], 10)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

        # Manual calculation
        expected = sample_data["close"] - sample_data["close"].shift(10)
        pd.testing.assert_series_equal(result[10:], expected[10:], check_names=False)

    def test_ppo(self, sample_data):
        """Test Percentage Price Oscillator."""
        result = PandasIndicators.PPO(sample_data["close"])

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_talib_compatibility(self, sample_data):
        """Test TA-Lib compatible module."""
        talib = create_talib_compatible_module()

        # Test that we can call indicators through the module
        rsi = talib.RSI(sample_data["close"], timeperiod=14)
        assert isinstance(rsi, pd.Series)

        macd, signal, hist = talib.MACD(sample_data["close"])
        assert isinstance(macd, pd.Series)

        # Test with DataFrame input
        rsi_df = talib.RSI(sample_data[["close"]], timeperiod=14)
        assert isinstance(rsi_df, pd.Series)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        pd.DataFrame()
        empty_series = pd.Series(dtype=float)

        result = PandasIndicators.SMA(empty_series, 10)
        assert len(result) == 0

        # Single value
        single = pd.Series([100.0])
        result = PandasIndicators.RSI(single, 14)
        assert len(result) == 1
        assert result.iloc[0] == 50  # Default neutral value

        # All same values (no price change)
        constant = pd.Series([100.0] * 50)
        rsi = PandasIndicators.RSI(constant, 14)
        # When no price change, RSI should be neutral or very extreme
        # Our implementation clips to 0.01-99.99 range
        assert rsi.iloc[-1] == 50.0 or rsi.iloc[-1] <= 1.0 or rsi.iloc[-1] >= 99.0
