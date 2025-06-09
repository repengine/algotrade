#!/usr/bin/env python3
"""Intraday Opening Range Breakout (ORB) strategy."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any
from datetime import datetime, time, timedelta

try:
    import talib
except ImportError:
    # Use pandas implementation if talib is not available
    from pandas_indicators import create_talib_compatible_module
    talib = create_talib_compatible_module()

from strategies.base import BaseStrategy, Signal, RiskContext
from utils.validators.strategy_validators import validate_intraday_orb_config


class IntradayORB(BaseStrategy):
    """Opening Range Breakout strategy for intraday trading.
    
    Entry: Break above/below first 30-minute range with volume confirmation
    Exit: End of day or stop loss at opposite range boundary
    Trade: Once per day per symbol
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        default_config = {
            'name': 'IntradayORB',
            'symbols': ['SPY', 'QQQ', 'IWM'],
            'lookback_period': 20,  # Days for ATR calculation
            'opening_minutes': 30,   # Minutes to establish range
            'volume_threshold': 1.5, # Volume must be 50% above average
            'atr_filter': 0.5,      # Range must be > 0.5 ATR
            'breakout_buffer': 0.1, # 10% buffer above/below range
            'stop_buffer': 0.05,    # 5% stop buffer
            'trade_start_time': time(10, 0),  # Start looking for trades at 10:00
            'trade_end_time': time(15, 30),   # Stop trading at 15:30
            'exit_time': time(15, 55),        # Exit all positions
            'max_trades_per_day': 1,          # Per symbol
        }
        
        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)
        
        self.opening_ranges = {}  # Store daily opening ranges
        self.daily_trades = {}    # Track trades per day
        self.positions = {}       # Current positions
    
    def init(self) -> None:
        """Initialize strategy components."""
        self.opening_ranges.clear()
        self.daily_trades.clear()
        self.positions.clear()
    
    def calculate_opening_range(
        self, 
        data: pd.DataFrame,
        current_date: datetime.date
    ) -> Optional[dict[str, float]]:
        """Calculate the opening range for the day."""
        # Get market open time (9:30 AM)
        market_open = datetime.combine(current_date, time(9, 30))
        range_end = market_open + timedelta(minutes=self.config['opening_minutes'])
        
        # Filter data for opening range period
        mask = (data.index >= market_open) & (data.index <= range_end)
        opening_data = data[mask]
        
        if opening_data.empty:
            return None
        
        # Calculate range
        range_high = opening_data['high'].max()
        range_low = opening_data['low'].min()
        range_size = range_high - range_low
        
        # Volume during opening range
        range_volume = opening_data['volume'].sum()
        avg_volume = opening_data['volume'].mean()
        
        return {
            'high': range_high,
            'low': range_low,
            'size': range_size,
            'midpoint': (range_high + range_low) / 2,
            'volume': range_volume,
            'avg_volume': avg_volume,
            'established_time': range_end
        }
    
    def next_intraday(
        self, 
        data: pd.DataFrame,
        current_time: datetime
    ) -> Optional[Signal]:
        """Process intraday data and generate signals.
        
        Args:
            data: Intraday OHLCV data (e.g., 5-minute bars)
            current_time: Current timestamp
        """
        if not self.validate_data(data):
            return None
        
        symbol = data.attrs.get('symbol', 'UNKNOWN')
        current_date = current_time.date()
        current_time_only = current_time.time()
        
        # Skip if outside trading hours
        if (current_time_only < self.config['trade_start_time'] or
            current_time_only > self.config['exit_time']):
            return None
        
        # Force exit at end of day
        if current_time_only >= self.config['exit_time'] and symbol in self.positions:
            return Signal(
                timestamp=current_time,
                symbol=symbol,
                direction='FLAT',
                strength=0.0,
                strategy_id=self.name,
                price=data.iloc[-1]['close'],
                metadata={
                    'reason': 'end_of_day_exit',
                    'time': str(current_time_only)
                }
            )
        
        # Calculate opening range if not already done
        date_key = f"{symbol}_{current_date}"
        if date_key not in self.opening_ranges:
            or_data = self.calculate_opening_range(data, current_date)
            if or_data:
                self.opening_ranges[date_key] = or_data
            else:
                return None
        
        opening_range = self.opening_ranges[date_key]
        
        # Skip if we've already traded this symbol today
        if self.daily_trades.get(date_key, 0) >= self.config['max_trades_per_day']:
            return None
        
        # Skip if still in opening range period
        if current_time < opening_range['established_time']:
            return None
        
        # Get current price and volume
        latest = data.iloc[-1]
        current_price = latest['close']
        
        # Calculate recent volume (last 30 minutes)
        recent_mask = data.index > (current_time - timedelta(minutes=30))
        recent_volume = data[recent_mask]['volume'].mean()
        
        # Calculate ATR for filtering
        atr = talib.ATR(
            data['high'].values,
            data['low'].values, 
            data['close'].values,
            timeperiod=14
        )[-1] if len(data) >= 14 else opening_range['size']
        
        # Check if range is significant (ATR filter)
        if opening_range['size'] < self.config['atr_filter'] * atr:
            return None
        
        # Volume confirmation
        volume_confirmed = recent_volume > (opening_range['avg_volume'] * 
                                          self.config['volume_threshold'])
        
        # Calculate breakout levels with buffer
        breakout_high = opening_range['high'] * (1 + self.config['breakout_buffer'])
        breakout_low = opening_range['low'] * (1 - self.config['breakout_buffer'])
        
        # Check for breakout signals
        if symbol not in self.positions:
            # Long breakout
            if (current_price > breakout_high and 
                volume_confirmed and
                current_time_only <= self.config['trade_end_time']):
                
                # Calculate signal strength based on breakout magnitude
                breakout_strength = min(1.0, 
                    (current_price - opening_range['high']) / opening_range['size']) if opening_range['size'] > 0 else 0.5
                
                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction='LONG',
                    strength=breakout_strength,
                    strategy_id=self.name,
                    price=current_price,
                    atr=atr,
                    metadata={
                        'reason': 'orb_breakout_long',
                        'range_high': opening_range['high'],
                        'range_low': opening_range['low'],
                        'range_size': opening_range['size'],
                        'breakout_level': breakout_high,
                        'volume_ratio': recent_volume / opening_range['avg_volume'] if opening_range['avg_volume'] > 0 else 1.0,
                        'time': str(current_time_only)
                    }
                )
                
                # Track position and daily trades
                self.positions[symbol] = {
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'stop_loss': opening_range['low'] * (1 - self.config['stop_buffer'])
                }
                self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
                
                return signal
            
            # Short breakout
            elif (current_price < breakout_low and 
                  volume_confirmed and
                  current_time_only <= self.config['trade_end_time']):
                
                # Calculate signal strength
                breakout_strength = min(1.0,
                    (opening_range['low'] - current_price) / opening_range['size']) if opening_range['size'] > 0 else 0.5
                
                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction='SHORT',
                    strength=-breakout_strength,
                    strategy_id=self.name,
                    price=current_price,
                    atr=atr,
                    metadata={
                        'reason': 'orb_breakout_short',
                        'range_high': opening_range['high'],
                        'range_low': opening_range['low'],
                        'range_size': opening_range['size'],
                        'breakout_level': breakout_low,
                        'volume_ratio': recent_volume / opening_range['avg_volume'] if opening_range['avg_volume'] > 0 else 1.0,
                        'time': str(current_time_only)
                    }
                )
                
                # Track position and daily trades
                self.positions[symbol] = {
                    'direction': 'SHORT',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'stop_loss': opening_range['high'] * (1 + self.config['stop_buffer'])
                }
                self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
                
                return signal
        
        # Check for stop loss if in position
        else:
            pos = self.positions[symbol]
            
            if pos['direction'] == 'LONG' and current_price <= pos['stop_loss']:
                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction='FLAT',
                    strength=0.0,
                    strategy_id=self.name,
                    price=current_price,
                    metadata={
                        'reason': 'stop_loss',
                        'entry_price': pos['entry_price'],
                        'stop_price': pos['stop_loss'],
                        'pnl_pct': ((current_price - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0.0,
                        'time': str(current_time_only)
                    }
                )
                del self.positions[symbol]
                return signal
                
            elif pos['direction'] == 'SHORT' and current_price >= pos['stop_loss']:
                signal = Signal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction='FLAT',
                    strength=0.0,
                    strategy_id=self.name,
                    price=current_price,
                    metadata={
                        'reason': 'stop_loss',
                        'entry_price': pos['entry_price'],
                        'stop_price': pos['stop_loss'],
                        'pnl_pct': ((pos['entry_price'] - current_price) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0.0,
                        'time': str(current_time_only)
                    }
                )
                del self.positions[symbol]
                return signal
        
        return None
    
    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Standard next method for compatibility.
        
        For intraday strategies, use next_intraday instead.
        """
        # This could convert daily data to a signal if needed
        # But ORB requires intraday data
        return None
    
    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """Calculate position size for ORB trades."""
        if signal.direction == 'FLAT':
            return 0.0, 0.0
        
        # Get range information from metadata
        range_size = signal.metadata.get('range_size', 0.01 * signal.price)
        
        # Risk per trade based on range size
        # Smaller range = larger position (but capped)
        range_risk = range_size / signal.price if signal.price > 0 else 0.01
        
        # Base allocation (more conservative for intraday)
        base_allocation = risk_context.account_equity * 0.10  # 10% per trade
        
        # Adjust for range-based risk
        # Inverse relationship: smaller range = larger size
        range_multiplier = min(2.0, 0.02 / range_risk) if range_risk > 0 else 1.0  # Cap at 2x
        
        position_value = base_allocation * range_multiplier * abs(signal.strength)
        
        # Apply maximum position constraint
        max_position_value = risk_context.account_equity * 0.15  # 15% max for intraday
        position_value = min(position_value, max_position_value)
        
        # Calculate shares
        position_size = position_value / signal.price if signal.price > 0 else 0
        
        # Stop loss is already defined in the strategy logic
        stop_loss = signal.metadata.get('stop_price', 0.0)
        
        return position_size, stop_loss
    
    def reset_daily_counters(self) -> None:
        """Reset daily trade counters. Call this at start of each day."""
        self.daily_trades.clear()
        self.opening_ranges.clear()
        # Keep positions as they might be from previous day
        # (though ORB should close all by end of day)
    
    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate intraday ORB strategy configuration."""
        return validate_intraday_orb_config(config)