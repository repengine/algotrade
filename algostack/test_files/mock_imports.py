"""Mock imports for missing dependencies during testing."""

import sys
from unittest.mock import MagicMock

import numpy as np

# Try to import real scipy first, mock only if not available
try:
    import scipy
    import scipy.optimize
    import scipy.stats

    print("✅ Using real scipy")
except ImportError:
    # Mock scipy
    scipy_mock = MagicMock()
    scipy_mock.optimize.minimize = MagicMock()
    scipy_mock.stats.norm = MagicMock()
    sys.modules["scipy"] = scipy_mock
    sys.modules["scipy.optimize"] = scipy_mock.optimize
    sys.modules["scipy.stats"] = scipy_mock.stats
    print("⚠️ Using mocked scipy")

# Try to import real talib first, mock only if not available
try:
    import talib

    print("✅ Using real talib")
except ImportError:
    # Mock talib with realistic functions
    talib_mock = MagicMock()

    def mock_rsi(data, timeperiod=14):
        """Mock RSI that returns correct length with NaN for initial values."""
        length = len(data)
        result = np.full(length, np.nan)
        result[timeperiod:] = 50.0  # Fill with 50 after warmup period
        return result

    def mock_atr(high, low, close, timeperiod=14):
        """Mock ATR that returns correct length with NaN for initial values."""
        length = len(high)
        result = np.full(length, np.nan)
        result[timeperiod:] = 1.0  # Fill with 1.0 after warmup period
        return result

    def mock_bbands(data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        """Mock Bollinger Bands that returns correct length with NaN for initial values."""
        length = len(data)
        upper = np.full(length, np.nan)
        middle = np.full(length, np.nan)
        lower = np.full(length, np.nan)
        upper[timeperiod:] = 101.0
        middle[timeperiod:] = 100.0
        lower[timeperiod:] = 99.0
        return upper, middle, lower

    talib_mock.RSI = mock_rsi
    talib_mock.ATR = mock_atr
    talib_mock.BBANDS = mock_bbands
    sys.modules["talib"] = talib_mock
    print("⚠️ Using mocked talib")

# Try to import real backtrader first, mock only if not available
try:
    import backtrader
    import backtrader.analyzers

    print("✅ Using real backtrader")
except ImportError:
    # Mock backtrader
    backtrader_mock = MagicMock()
    backtrader_mock.analyzers = MagicMock()
    backtrader_mock.analyzers.SharpeRatio = MagicMock()
    backtrader_mock.analyzers.DrawDown = MagicMock()
    backtrader_mock.analyzers.Returns = MagicMock()
    backtrader_mock.analyzers.TradeAnalyzer = MagicMock()
    sys.modules["backtrader"] = backtrader_mock
    sys.modules["backtrader.analyzers"] = backtrader_mock.analyzers
    print("⚠️ Using mocked backtrader")

# Mock pyarrow
pyarrow_mock = MagicMock()
pyarrow_mock.__name__ = "pyarrow"
pyarrow_mock.__version__ = "1.0.0"
pyarrow_mock.parquet = MagicMock()
pyarrow_mock.parquet.__name__ = "pyarrow.parquet"
sys.modules["pyarrow"] = pyarrow_mock
sys.modules["pyarrow.parquet"] = pyarrow_mock.parquet

# Mock yfinance
yfinance_mock = MagicMock()
sys.modules["yfinance"] = yfinance_mock

# Mock statsmodels
statsmodels_mock = MagicMock()
statsmodels_mock.tsa = MagicMock()
statsmodels_mock.tsa.stattools = MagicMock()
statsmodels_mock.tsa.stattools.adfuller = MagicMock(
    return_value=(0.1, 0.5, 10, 100, {"1%": -3.5}, 0.1)
)
statsmodels_mock.tsa.stattools.coint = MagicMock(return_value=(0.1, 0.5, None))
sys.modules["statsmodels"] = statsmodels_mock
sys.modules["statsmodels.tsa"] = statsmodels_mock.tsa
sys.modules["statsmodels.tsa.stattools"] = statsmodels_mock.tsa.stattools

# Mock sklearn
sklearn_mock = MagicMock()
sklearn_mock.linear_model = MagicMock()
sklearn_mock.linear_model.LinearRegression = MagicMock()
sys.modules["sklearn"] = sklearn_mock
sys.modules["sklearn.linear_model"] = sklearn_mock.linear_model
