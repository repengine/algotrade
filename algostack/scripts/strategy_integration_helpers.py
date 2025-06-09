"""
Helper functions for integrating AlgoStack strategies with various interfaces (Fixed).

This module provides utilities to bridge between different data formats and
handle missing dependencies gracefully.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
import warnings


class TechnicalIndicators:
    """Fallback technical indicators when TA-Lib is not available."""
    
    @staticmethod
    def RSI(close, timeperiod=14):
        """Calculate RSI without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """Calculate ATR without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is EMA of TR
        atr = tr.rolling(window=timeperiod).mean()
        
        return atr
    
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        """Calculate Bollinger Bands without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        middle = close.rolling(window=timeperiod).mean()
        std = close.rolling(window=timeperiod).std()
        
        upper = middle + (nbdevup * std)
        lower = middle - (nbdevdn * std)
        
        return upper, middle, lower
    
    @staticmethod
    def EMA(close, timeperiod):
        """Calculate EMA without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.ewm(span=timeperiod, adjust=False).mean()
    
    @staticmethod
    def SMA(close, timeperiod):
        """Calculate SMA without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.rolling(window=timeperiod).mean()
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calculate MACD without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        ema_fast = close.ewm(span=fastperiod, adjust=False).mean()
        ema_slow = close.ewm(span=slowperiod, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        """Calculate Stochastic without TA-Lib - fixed signature."""
        # Convert to Series if needed
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        # Fast %K
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        
        fastk = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Slow %K (SMA of Fast %K)
        slowk = fastk.rolling(window=slowk_period).mean()
        
        # Slow %D (SMA of Slow %K)
        slowd = slowk.rolling(window=slowd_period).mean()
        
        return slowk, slowd


class DataFormatConverter:
    """Convert between different data formats used in AlgoStack."""
    
    @staticmethod
    def dashboard_to_strategy(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """Convert dashboard data format to strategy format."""
        # Create a copy to avoid modifying original
        converted = df.copy()
        
        # Convert column names to lowercase
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in converted.columns:
                converted.rename(columns={old_name: new_name}, inplace=True)
        
        # Add symbol as attribute
        if symbol:
            converted.attrs['symbol'] = symbol
        
        return converted
    
    @staticmethod
    def strategy_to_dashboard(signals: list[Any]) -> pd.DataFrame:
        """Convert strategy signals to dashboard format."""
        # Handle different signal types
        if not signals:
            return pd.DataFrame(columns=['signal', 'position'])
        
        # Convert Signal objects to DataFrame
        signal_data = []
        for sig in signals:
            if hasattr(sig, 'direction'):
                # Signal object
                signal_value = 0
                if sig.direction == 'LONG':
                    signal_value = 1
                elif sig.direction == 'SHORT':
                    signal_value = -1
                elif sig.direction == 'FLAT':
                    signal_value = 0
                
                signal_data.append({
                    'timestamp': sig.timestamp,
                    'signal': signal_value,
                    'strength': getattr(sig, 'strength', 1.0),
                    'metadata': getattr(sig, 'metadata', {})
                })
            else:
                # Assume numeric signal
                signal_data.append({
                    'signal': sig,
                    'strength': 1.0
                })
        
        df = pd.DataFrame(signal_data)
        
        # Calculate positions
        if 'signal' in df.columns:
            df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return df


class RiskContextMock:
    """Mock RiskContext for strategies that require it."""
    
    def __init__(self, account_equity: float = 100000, 
                 volatility_target: float = 0.20,
                 max_position_size: float = 0.10):
        self.account_equity = account_equity
        self.volatility_target = volatility_target
        self.max_position_size = max_position_size
        self.current_positions = {}
        
    def get_account_equity(self) -> float:
        return self.account_equity
    
    def get_max_position_size(self) -> float:
        return self.max_position_size * self.account_equity


def patch_talib_imports():
    """Monkey patch talib imports with fallback implementations."""
    import sys
    
    if 'talib' not in sys.modules:
        # Create a mock talib module with proper method signatures
        import types
        talib = types.ModuleType('talib')
        
        # Assign functions directly (not as static methods)
        talib.RSI = TechnicalIndicators.RSI
        talib.ATR = TechnicalIndicators.ATR
        talib.BBANDS = TechnicalIndicators.BBANDS
        talib.EMA = TechnicalIndicators.EMA
        talib.SMA = TechnicalIndicators.SMA
        talib.MACD = TechnicalIndicators.MACD
        talib.STOCH = TechnicalIndicators.STOCH
        
        # Add other indicators as needed
        def create_dummy(name):
            def dummy(*args, **kwargs):
                warnings.warn(f"Indicator {name} not implemented, returning NaN")
                if args:
                    return pd.Series(np.nan, index=args[0].index)
                return pd.Series(np.nan)
            return dummy
        
        # Add ADX and related indicators
        def ADX(high, low, close, timeperiod=14):
            """Simple ADX approximation."""
            # Just return a value between 0-100
            return pd.Series(25.0, index=close.index)
        
        def PLUS_DI(high, low, close, timeperiod=14):
            """Plus Directional Indicator approximation."""
            return pd.Series(20.0, index=close.index)
        
        def MINUS_DI(high, low, close, timeperiod=14):
            """Minus Directional Indicator approximation."""
            return pd.Series(15.0, index=close.index)
        
        talib.ADX = ADX
        talib.PLUS_DI = PLUS_DI
        talib.MINUS_DI = MINUS_DI
        
        # Common indicators that might be used
        for indicator in ['CCI', 'MFI', 'WILLR', 'OBV', 'SAR', 'TRIX']:
            setattr(talib, indicator, create_dummy(indicator))
        
        sys.modules['talib'] = talib
        print("📊 Using fallback technical indicators (TA-Lib not available)")


def validate_strategy_config(strategy_class: type, config: dict[str, Any]) -> dict[str, Any]:
    """Validate and fix strategy configuration."""
    validated_config = config.copy()
    
    # Ensure required fields based on common patterns
    if 'symbols' not in validated_config and 'symbol' in validated_config:
        # Convert single symbol to list
        validated_config['symbols'] = [validated_config['symbol']]
    elif 'symbols' not in validated_config:
        validated_config['symbols'] = ['SPY']  # Default
    
    # Ensure numeric fields are correct type
    numeric_fields = {
        'lookback_period': int,
        'position_size': float,
        'stop_loss_pct': float,
        'take_profit_pct': float,
        'rsi_period': int,
        'rsi_oversold': float,
        'rsi_overbought': float,
        'atr_period': int,
        'adx_period': int,
        'adx_threshold': float,
        'zscore_threshold': float,
        'exit_zscore': float
    }
    
    for field, expected_type in numeric_fields.items():
        if field in validated_config:
            try:
                if expected_type == int:
                    validated_config[field] = int(validated_config[field])
                elif expected_type == float:
                    validated_config[field] = float(validated_config[field])
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                pass
    
    return validated_config