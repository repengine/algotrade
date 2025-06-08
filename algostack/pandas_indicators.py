"""
Pure pandas implementation of technical indicators.

This module provides all technical indicators used by AlgoStack strategies
implemented using only pandas and numpy, eliminating the need for TA-Lib.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional


class PandasIndicators:
    """Pure pandas implementation of technical indicators."""
    
    @staticmethod
    def SMA(series: Union[pd.Series, pd.DataFrame], timeperiod: int) -> pd.Series:
        """Simple Moving Average."""
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series.rolling(window=timeperiod, min_periods=1).mean()
    
    @staticmethod
    def EMA(series: Union[pd.Series, pd.DataFrame], timeperiod: int) -> pd.Series:
        """Exponential Moving Average."""
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series.ewm(span=timeperiod, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def RSI(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing method."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing method (exponential moving average with alpha = 1/timeperiod)
        # This is equivalent to EMA with span = 2*timeperiod - 1
        alpha = 1.0 / timeperiod
        avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=timeperiod).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=timeperiod).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill initial NaN values with 50 (neutral RSI)
        rsi = rsi.fillna(50)
        
        # Ensure RSI is between 0 and 100 (no hard clipping to 0 or 100)
        rsi = rsi.clip(lower=0.01, upper=99.99)
        
        return rsi
    
    @staticmethod
    def MACD(close: Union[pd.Series, pd.DataFrame], 
             fastperiod: int = 12, 
             slowperiod: int = 26, 
             signalperiod: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        ema_fast = close.ewm(span=fastperiod, adjust=False, min_periods=1).mean()
        ema_slow = close.ewm(span=slowperiod, adjust=False, min_periods=1).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False, min_periods=1).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def BBANDS(close: Union[pd.Series, pd.DataFrame], 
               timeperiod: int = 20, 
               nbdevup: float = 2.0, 
               nbdevdn: float = 2.0, 
               matype: int = 0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        middle = close.rolling(window=timeperiod, min_periods=1).mean()
        std = close.rolling(window=timeperiod, min_periods=1).std()
        
        upper = middle + (nbdevup * std)
        lower = middle - (nbdevdn * std)
        
        return upper, middle, lower
    
    @staticmethod
    def ATR(high: Union[pd.Series, pd.DataFrame], 
            low: Union[pd.Series, pd.DataFrame], 
            close: Union[pd.Series, pd.DataFrame], 
            timeperiod: int = 14) -> pd.Series:
        """Average True Range."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is SMA of TR
        atr = tr.rolling(window=timeperiod, min_periods=1).mean()
        
        return atr
    
    @staticmethod
    def STOCH(high: Union[pd.Series, pd.DataFrame], 
              low: Union[pd.Series, pd.DataFrame], 
              close: Union[pd.Series, pd.DataFrame], 
              fastk_period: int = 14, 
              slowk_period: int = 3, 
              slowk_matype: int = 0,
              slowd_period: int = 3,
              slowd_matype: int = 0) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Calculate Fast %K
        lowest_low = low.rolling(window=fastk_period, min_periods=1).min()
        highest_high = high.rolling(window=fastk_period, min_periods=1).max()
        
        # Handle division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, 1e-10)
        
        fastk = 100 * ((close - lowest_low) / denom)
        
        # Calculate Slow %K (SMA of Fast %K)
        slowk = fastk.rolling(window=slowk_period, min_periods=1).mean()
        
        # Calculate Slow %D (SMA of Slow %K)
        slowd = slowk.rolling(window=slowd_period, min_periods=1).mean()
        
        return slowk, slowd
    
    @staticmethod
    def ADX(high: Union[pd.Series, pd.DataFrame], 
            low: Union[pd.Series, pd.DataFrame], 
            close: Union[pd.Series, pd.DataFrame], 
            timeperiod: int = 14) -> pd.Series:
        """Average Directional Index."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Calculate directional movements
        up_move = high.diff()
        down_move = -low.diff()
        
        # Positive directional movement
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        # Negative directional movement
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the indicators
        atr = tr.rolling(window=timeperiod, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(window=timeperiod, min_periods=1).mean() / atr.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=timeperiod, min_periods=1).mean() / atr.replace(0, 1e-10))
        
        # Calculate DX
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, 1e-10)
        dx = 100 * abs(plus_di - minus_di) / di_sum
        
        # ADX is smoothed DX
        adx = dx.rolling(window=timeperiod, min_periods=1).mean()
        
        return adx
    
    @staticmethod
    def PLUS_DI(high: Union[pd.Series, pd.DataFrame], 
                low: Union[pd.Series, pd.DataFrame], 
                close: Union[pd.Series, pd.DataFrame], 
                timeperiod: int = 14) -> pd.Series:
        """Plus Directional Indicator."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Calculate directional movements
        up_move = high.diff()
        down_move = -low.diff()
        
        # Positive directional movement
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the indicators
        atr = tr.rolling(window=timeperiod, min_periods=1).mean()
        plus_di = 100 * (plus_dm.rolling(window=timeperiod, min_periods=1).mean() / atr.replace(0, 1e-10))
        
        return plus_di
    
    @staticmethod
    def MINUS_DI(high: Union[pd.Series, pd.DataFrame], 
                 low: Union[pd.Series, pd.DataFrame], 
                 close: Union[pd.Series, pd.DataFrame], 
                 timeperiod: int = 14) -> pd.Series:
        """Minus Directional Indicator."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Calculate directional movements
        up_move = high.diff()
        down_move = -low.diff()
        
        # Negative directional movement
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the indicators
        atr = tr.rolling(window=timeperiod, min_periods=1).mean()
        minus_di = 100 * (minus_dm.rolling(window=timeperiod, min_periods=1).mean() / atr.replace(0, 1e-10))
        
        return minus_di
    
    @staticmethod
    def CCI(high: Union[pd.Series, pd.DataFrame], 
            low: Union[pd.Series, pd.DataFrame], 
            close: Union[pd.Series, pd.DataFrame], 
            timeperiod: int = 14) -> pd.Series:
        """Commodity Channel Index."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # Moving average of TP
        ma = tp.rolling(window=timeperiod, min_periods=1).mean()
        
        # Mean deviation
        md = tp.rolling(window=timeperiod, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        # CCI
        cci = (tp - ma) / (0.015 * md.replace(0, 1e-10))
        
        return cci
    
    @staticmethod
    def MFI(high: Union[pd.Series, pd.DataFrame], 
            low: Union[pd.Series, pd.DataFrame], 
            close: Union[pd.Series, pd.DataFrame], 
            volume: Union[pd.Series, pd.DataFrame],
            timeperiod: int = 14) -> pd.Series:
        """Money Flow Index."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # Money Flow
        mf = tp * volume
        
        # Positive and Negative Money Flow
        tp_diff = tp.diff()
        pmf = mf.where(tp_diff > 0, 0)
        nmf = mf.where(tp_diff < 0, 0)
        
        # Money Flow Ratio
        pmf_sum = pmf.rolling(window=timeperiod, min_periods=1).sum()
        nmf_sum = nmf.rolling(window=timeperiod, min_periods=1).sum()
        
        mfr = pmf_sum / nmf_sum.replace(0, 1e-10)
        
        # MFI
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    @staticmethod
    def WILLR(high: Union[pd.Series, pd.DataFrame], 
              low: Union[pd.Series, pd.DataFrame], 
              close: Union[pd.Series, pd.DataFrame], 
              timeperiod: int = 14) -> pd.Series:
        """Williams %R."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Highest high and lowest low
        hh = high.rolling(window=timeperiod, min_periods=1).max()
        ll = low.rolling(window=timeperiod, min_periods=1).min()
        
        # Williams %R
        willr = -100 * (hh - close) / (hh - ll).replace(0, 1e-10)
        
        return willr
    
    @staticmethod
    def OBV(close: Union[pd.Series, pd.DataFrame], 
            volume: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """On Balance Volume."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        
        # Calculate direction
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # OBV
        obv = (direction * volume).cumsum()
        
        return obv
    
    @staticmethod
    def SAR(high: Union[pd.Series, pd.DataFrame], 
            low: Union[pd.Series, pd.DataFrame], 
            acceleration: float = 0.02, 
            maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR."""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        
        # Initialize
        sar = pd.Series(index=high.index, dtype=float)
        ep = pd.Series(index=high.index, dtype=float)
        af = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        
        # First value
        sar.iloc[0] = low.iloc[0]
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = acceleration
        trend.iloc[0] = 1
        
        for i in range(1, len(high)):
            # Update SAR
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                # Check for reversal
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                # Check for reversal
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return sar
    
    @staticmethod
    def TRIX(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 30) -> pd.Series:
        """TRIX - Triple Exponential Average."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Triple EMA
        ema1 = close.ewm(span=timeperiod, adjust=False, min_periods=1).mean()
        ema2 = ema1.ewm(span=timeperiod, adjust=False, min_periods=1).mean()
        ema3 = ema2.ewm(span=timeperiod, adjust=False, min_periods=1).mean()
        
        # Rate of change
        trix = 10000 * ema3.pct_change()
        
        return trix
    
    @staticmethod
    def ROC(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 10) -> pd.Series:
        """Rate of Change."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # ROC = (close - close[n]) / close[n] * 100
        roc = ((close - close.shift(timeperiod)) / close.shift(timeperiod)) * 100
        
        return roc
    
    @staticmethod
    def ROCP(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 10) -> pd.Series:
        """Rate of Change Percentage (same as ROC but as decimal)."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # ROCP = (close - close[n]) / close[n]
        rocp = (close - close.shift(timeperiod)) / close.shift(timeperiod)
        
        return rocp
    
    @staticmethod
    def MOM(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 10) -> pd.Series:
        """Momentum."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # MOM = close - close[n]
        mom = close - close.shift(timeperiod)
        
        return mom
    
    @staticmethod
    def PPO(close: Union[pd.Series, pd.DataFrame], 
            fastperiod: int = 12, 
            slowperiod: int = 26, 
            matype: int = 0) -> pd.Series:
        """Percentage Price Oscillator."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Calculate EMAs
        ema_fast = close.ewm(span=fastperiod, adjust=False, min_periods=1).mean()
        ema_slow = close.ewm(span=slowperiod, adjust=False, min_periods=1).mean()
        
        # PPO = (fast - slow) / slow * 100
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        
        return ppo
    
    @staticmethod
    def STDDEV(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 5, nbdev: float = 1) -> pd.Series:
        """Standard Deviation."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Standard deviation
        stddev = close.rolling(window=timeperiod, min_periods=1).std() * nbdev
        
        return stddev
    
    @staticmethod
    def VAR(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 5, nbdev: float = 1) -> pd.Series:
        """Variance."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Variance
        var = close.rolling(window=timeperiod, min_periods=1).var() * nbdev
        
        return var
    
    @staticmethod
    def LINEARREG(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 14) -> pd.Series:
        """Linear Regression."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        def linreg(x):
            if len(x) < 2:
                return x.iloc[-1] if len(x) > 0 else np.nan
            y = np.arange(len(x))
            A = np.vstack([y, np.ones(len(y))]).T
            m, c = np.linalg.lstsq(A, x.values, rcond=None)[0]
            return m * (len(x) - 1) + c
        
        return close.rolling(window=timeperiod, min_periods=1).apply(linreg)
    
    @staticmethod
    def LINEARREG_SLOPE(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 14) -> pd.Series:
        """Linear Regression Slope."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        def slope(x):
            if len(x) < 2:
                return 0
            y = np.arange(len(x))
            A = np.vstack([y, np.ones(len(y))]).T
            m, c = np.linalg.lstsq(A, x.values, rcond=None)[0]
            return m
        
        return close.rolling(window=timeperiod, min_periods=2).apply(slope)
    
    @staticmethod
    def LINEARREG_ANGLE(close: Union[pd.Series, pd.DataFrame], timeperiod: int = 14) -> pd.Series:
        """Linear Regression Angle."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        # Get slope and convert to angle in degrees
        slope = PandasIndicators.LINEARREG_SLOPE(close, timeperiod)
        angle = np.arctan(slope) * (180 / np.pi)
        
        return angle


# Create a compatibility layer that matches TA-Lib's interface
def create_talib_compatible_module():
    """Create a module that mimics TA-Lib's interface using pandas indicators."""
    import types
    
    # Create module
    talib = types.ModuleType('talib')
    
    # Create instance of PandasIndicators
    indicators = PandasIndicators()
    
    # Map all methods to module
    for attr_name in dir(indicators):
        if not attr_name.startswith('_'):
            setattr(talib, attr_name, getattr(indicators, attr_name))
    
    # Add any additional indicators that might be needed
    # These are simple implementations for compatibility
    
    def HT_TRENDLINE(close):
        """Hilbert Transform - Instantaneous Trendline (simplified)."""
        # Just return SMA as approximation
        return PandasIndicators.SMA(close, 4)
    
    def KAMA(close, timeperiod=30):
        """Kaufman Adaptive Moving Average (simplified)."""
        # Use EMA as approximation
        return PandasIndicators.EMA(close, timeperiod)
    
    def T3(close, timeperiod=5, vfactor=0.7):
        """Triple Exponential Moving Average T3 (simplified)."""
        # Use triple EMA
        ema1 = PandasIndicators.EMA(close, timeperiod)
        ema2 = PandasIndicators.EMA(ema1, timeperiod)
        ema3 = PandasIndicators.EMA(ema2, timeperiod)
        return ema3
    
    def DEMA(close, timeperiod=30):
        """Double Exponential Moving Average."""
        ema1 = PandasIndicators.EMA(close, timeperiod)
        ema2 = PandasIndicators.EMA(ema1, timeperiod)
        return 2 * ema1 - ema2
    
    def TEMA(close, timeperiod=30):
        """Triple Exponential Moving Average."""
        ema1 = PandasIndicators.EMA(close, timeperiod)
        ema2 = PandasIndicators.EMA(ema1, timeperiod)
        ema3 = PandasIndicators.EMA(ema2, timeperiod)
        return 3 * ema1 - 3 * ema2 + ema3
    
    def WMA(close, timeperiod=30):
        """Weighted Moving Average."""
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        weights = np.arange(1, timeperiod + 1)
        return close.rolling(timeperiod).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    # Add these additional functions to the module
    talib.HT_TRENDLINE = HT_TRENDLINE
    talib.KAMA = KAMA
    talib.T3 = T3
    talib.DEMA = DEMA
    talib.TEMA = TEMA
    talib.WMA = WMA
    
    return talib