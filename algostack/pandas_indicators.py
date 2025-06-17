"""Pandas-based technical indicators as a fallback for TA-Lib."""

import numpy as np
import pandas as pd


def SMA(series: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=timeperiod).mean()


def EMA(series: pd.Series, timeperiod: int = 30) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=timeperiod, adjust=False).mean()


def RSI(series: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge cases
    rsi = rsi.fillna(50)  # Neutral RSI for NaN values
    rsi[loss == 0] = 100  # When all gains
    
    return rsi


def BBANDS(series: pd.Series, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2, matype: int = 0):
    """Bollinger Bands."""
    middle = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    
    return upper, middle, lower


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    
    return atr


def ADX(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Average Directional Index."""
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # When both are positive, only keep the larger
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[mask & (plus_dm < minus_dm)] = 0
    minus_dm[mask & (plus_dm >= minus_dm)] = 0
    
    # Calculate TR
    tr = ATR(high, low, close, 1)
    
    # Calculate DI+, DI-
    plus_di = 100 * (plus_dm.rolling(timeperiod).mean() / tr.rolling(timeperiod).mean())
    minus_di = 100 * (minus_dm.rolling(timeperiod).mean() / tr.rolling(timeperiod).mean())
    
    # Calculate DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(timeperiod).mean()
    
    return adx


def MACD(series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
    """MACD - Moving Average Convergence/Divergence."""
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd - signal
    
    return macd, signal, histogram


def STOCH(high: pd.Series, low: pd.Series, close: pd.Series, 
          fastk_period: int = 5, slowk_period: int = 3, 
          slowk_matype: int = 0, slowd_period: int = 3, 
          slowd_matype: int = 0):
    """Stochastic Oscillator."""
    # Calculate %K
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()
    
    fast_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate slow %K (smoothed %K)
    slow_k = fast_k.rolling(window=slowk_period).mean()
    
    # Calculate slow %D (smoothed slow %K)
    slow_d = slow_k.rolling(window=slowd_period).mean()
    
    return slow_k, slow_d


def MFI(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    price_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(price_diff > 0, 0)
    negative_flow = raw_money_flow.where(price_diff < 0, 0)
    
    # Calculate money flow ratio
    positive_mf = positive_flow.rolling(window=timeperiod).sum()
    negative_mf = negative_flow.rolling(window=timeperiod).sum()
    
    mf_ratio = positive_mf / negative_mf.replace(0, 1)  # Avoid division by zero
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + mf_ratio))
    
    return mfi


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    # Determine if price went up, down, or unchanged
    price_diff = close.diff()
    
    # Calculate OBV
    obv = volume.where(price_diff > 0, -volume).where(price_diff != 0, 0).cumsum()
    
    return obv


def create_talib_compatible_module():
    """Create a module-like object with TA-Lib compatible functions."""
    class TALibCompat:
        SMA = staticmethod(SMA)
        EMA = staticmethod(EMA)
        RSI = staticmethod(RSI)
        BBANDS = staticmethod(BBANDS)
        ATR = staticmethod(ATR)
        ADX = staticmethod(ADX)
        MACD = staticmethod(MACD)
        STOCH = staticmethod(STOCH)
        MFI = staticmethod(MFI)
        OBV = staticmethod(OBV)
    
    return TALibCompat()