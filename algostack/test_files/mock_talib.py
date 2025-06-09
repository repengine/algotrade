"""Mock TA-Lib module to allow dashboard to run without TA-Lib installed"""

import numpy as np
import pandas as pd


# Mock TA-Lib functions used in strategies
def RSI(close, timeperiod=14):
    """Simple RSI calculation without TA-Lib"""
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def ATR(high, low, close, timeperiod=14):
    """Simple ATR calculation without TA-Lib"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    return atr.values


def SMA(close, timeperiod=20):
    """Simple moving average"""
    return pd.Series(close).rolling(window=timeperiod).mean().values


def EMA(close, timeperiod=20):
    """Exponential moving average"""
    return pd.Series(close).ewm(span=timeperiod, adjust=False).mean().values


def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD calculation"""
    close = pd.Series(close)
    exp1 = close.ewm(span=fastperiod, adjust=False).mean()
    exp2 = close.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    return macd.values, signal.values, hist.values


def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands calculation"""
    close = pd.Series(close)
    sma = close.rolling(window=timeperiod).mean()
    std = close.rolling(window=timeperiod).std()
    upper = sma + (std * nbdevup)
    lower = sma - (std * nbdevdn)
    return upper.values, sma.values, lower.values


def ADX(high, low, close, timeperiod=14):
    """Simple ADX calculation"""
    # Simplified version - just return random values for now
    return np.full(len(close), 25.0)


def STOCH(
    high,
    low,
    close,
    fastk_period=5,
    slowk_period=3,
    slowk_matype=0,
    slowd_period=3,
    slowd_matype=0,
):
    """Stochastic calculation"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()

    fastk = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    slowk = fastk.rolling(window=slowk_period).mean()
    slowd = slowk.rolling(window=slowd_period).mean()

    return slowk.values, slowd.values
