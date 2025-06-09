#!/usr/bin/env python3
"""Overnight drift strategy exploiting structural market biases."""

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
from utils.validators.strategy_validators import validate_overnight_drift_config


class OvernightDrift(BaseStrategy):
    """Overnight holding strategy exploiting market drift patterns.
    
    Entry: Buy at close on specific days/conditions
    Exit: Sell at next open
    Filters: Skip high-risk days (FOMC, earnings, etc.)
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        default_config = {
            'name': 'OvernightDrift',
            'symbols': ['SPY', 'QQQ'],  # Works best with index ETFs
            'lookback_period': 252,
            'hold_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],  # Skip Friday
            'vix_threshold': 30,         # Skip if VIX > threshold
            'volume_threshold': 0.8,     # Skip if volume < 80% of average
            'trend_filter': True,        # Only trade in uptrend
            'sma_period': 50,           # Trend filter period
            'momentum_period': 20,       # Short-term momentum
            'min_atr': 0.005,           # Minimum volatility (0.5%)
            'max_atr': 0.03,            # Maximum volatility (3%)
            'earnings_blackout_days': 2, # Days before/after earnings
            'fomc_blackout_days': 1,     # Days around FOMC
            'max_positions': 2,          # Maximum overnight positions
        }
        
        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)
        
        self.positions = {}
        self.event_calendar = {}  # Store upcoming events
        
    def init(self) -> None:
        """Initialize strategy components."""
        self.positions.clear()
        self.event_calendar.clear()
        
    def load_event_calendar(self, events: dict[str, list[datetime]]) -> None:
        """Load economic events calendar.
        
        Args:
            events: Dict with keys like 'fomc', 'earnings_SPY', etc.
        """
        self.event_calendar = events
    
    def is_blackout_period(self, symbol: str, date: datetime) -> bool:
        """Check if current date is in blackout period."""
        # Check FOMC blackout
        fomc_dates = self.event_calendar.get('fomc', [])
        for fomc_date in fomc_dates:
            days_diff = abs((date - fomc_date).days)
            if days_diff <= self.config['fomc_blackout_days']:
                return True
        
        # Check earnings blackout
        earnings_key = f'earnings_{symbol}'
        earnings_dates = self.event_calendar.get(earnings_key, [])
        for earnings_date in earnings_dates:
            days_diff = abs((date - earnings_date).days)
            if days_diff <= self.config['earnings_blackout_days']:
                return True
        
        # Check for major holidays (market closed next day)
        # This would need a proper holiday calendar
        
        return False
    
    def calculate_overnight_edge(self, data: pd.DataFrame) -> float:
        """Calculate expected overnight return based on patterns."""
        if len(data) < self.config['lookback_period']:
            return 0.0
        
        # Calculate historical overnight returns
        overnight_returns = []
        for i in range(1, len(data)):
            # Overnight return = Open[i] / Close[i-1] - 1
            overnight_ret = (data['open'].iloc[i] / data['close'].iloc[i-1]) - 1 if data['close'].iloc[i-1] > 0 else 0.0
            overnight_returns.append(overnight_ret)
        
        overnight_returns = pd.Series(overnight_returns)
        
        # Calculate metrics
        avg_overnight = overnight_returns.mean()
        win_rate = (overnight_returns > 0).mean()
        
        # Recent momentum
        recent_momentum = data['close'].pct_change(self.config['momentum_period']).iloc[-1]
        
        # Calculate edge score
        edge = avg_overnight * 252  # Annualized
        
        # Adjust for win rate
        if win_rate > 0.52:  # Slight positive bias
            edge *= 1.2
        elif win_rate < 0.48:  # Negative bias
            edge *= 0.8
            
        # Adjust for momentum
        if self.config['trend_filter'] and recent_momentum > 0:
            edge *= 1.1
            
        return edge
    
    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process daily data and generate overnight holding signals."""
        if not self.validate_data(data):
            return None
            
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        if len(df) < self.config['sma_period']:
            return None
            
        latest = df.iloc[-1]
        symbol = data.attrs.get('symbol', 'UNKNOWN')
        current_date = df.index[-1]
        day_of_week = current_date.strftime('%A')
        
        # Skip if not a holding day
        if day_of_week not in self.config['hold_days']:
            # Exit if holding over weekend
            if symbol in self.positions and day_of_week == 'Friday':
                return Signal(
                    timestamp=current_date,
                    symbol=symbol,
                    direction='FLAT',
                    strength=0.0,
                    strategy_id=self.name,
                    price=latest['close'],
                    metadata={
                        'reason': 'weekend_exit',
                        'day': day_of_week
                    }
                )
            return None
        
        # Check for position exit (sell at open)
        # In practice, this would be triggered at market open
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Simulate selling at open (next day)
            exit_price = latest['open']  # This would be next day's open
            pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0.0
            
            signal = Signal(
                timestamp=current_date,
                symbol=symbol,
                direction='FLAT', 
                strength=0.0,
                strategy_id=self.name,
                price=exit_price,
                metadata={
                    'reason': 'overnight_exit',
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'held_overnight': True
                }
            )
            del self.positions[symbol]
            return signal
        
        # Check entry conditions
        if len(self.positions) >= self.config['max_positions']:
            return None
            
        # Skip if in blackout period
        if self.is_blackout_period(symbol, current_date):
            return None
        
        # Volatility filter
        if latest['atr'] < self.config['min_atr'] or latest['atr'] > self.config['max_atr']:
            return None
            
        # Volume filter
        if latest['volume_ratio'] < self.config['volume_threshold']:
            return None
        
        # Trend filter
        if self.config['trend_filter']:
            if latest['close'] < latest['sma']:
                return None
                
        # VIX filter (would need VIX data)
        vix = latest.get('vix', 20)  # Default VIX
        if vix > self.config['vix_threshold']:
            return None
        
        # Calculate overnight edge
        edge = self.calculate_overnight_edge(df)
        
        # Only trade if positive edge
        if edge > 0.02:  # 2% annualized edge threshold
            strength = min(1.0, edge * 10)  # Scale edge to strength
            
            signal = Signal(
                timestamp=current_date,
                symbol=symbol,
                direction='LONG',
                strength=strength,
                strategy_id=self.name,
                price=latest['close'],
                atr=latest['atr'],
                metadata={
                    'reason': 'overnight_entry',
                    'day': day_of_week,
                    'edge': edge,
                    'momentum': latest.get('momentum', 0),
                    'volume_ratio': latest['volume_ratio'],
                    'atr': latest['atr'],
                    'entry_time': 'close'
                }
            )
            
            self.positions[symbol] = {
                'entry_price': latest['close'],
                'entry_date': current_date,
                'expected_edge': edge
            }
            
            return signal
            
        return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for overnight strategy."""
        df = data.copy()
        
        # Trend indicator
        df['sma'] = talib.SMA(df['close'], timeperiod=self.config['sma_period'])
        
        # ATR for volatility
        df['atr'] = talib.ATR(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=self.config['atr_period']
        )
        df['atr'] = np.where(df['close'] > 0, df['atr'] / df['close'], 0.01)  # Normalize as percentage
        
        # Volume ratio
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = np.where(df['volume_sma'] > 0, df['volume'] / df['volume_sma'], 1.0)
        
        # Momentum
        df['momentum'] = df['close'].pct_change(self.config['momentum_period'])
        
        # RSI for mean reversion aspect
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Day of week (for pattern analysis)
        df['day_of_week'] = df.index.dayofweek
        
        return df
    
    def size(self, signal: Signal, risk_context: RiskContext) -> tuple[float, float]:
        """Calculate position size for overnight trades."""
        if signal.direction == 'FLAT':
            return 0.0, 0.0
            
        # Conservative sizing for overnight risk
        # Base allocation
        base_allocation = risk_context.account_equity * 0.05  # 5% per overnight trade
        
        # Adjust for expected edge
        edge = signal.metadata.get('edge', 0.02)
        edge_multiplier = min(2.0, 1 + (edge * 10))  # Scale with edge
        
        # Adjust for volatility (inverse relationship)
        atr = signal.metadata.get('atr', 0.01)
        vol_multiplier = min(1.5, 0.01 / atr) if atr > 0 else 1.0  # Lower vol = larger size
        
        # Final position value
        position_value = base_allocation * edge_multiplier * vol_multiplier * signal.strength
        
        # Apply maximum constraint
        max_position_value = risk_context.account_equity * 0.10  # 10% max overnight
        position_value = min(position_value, max_position_value)
        
        # Calculate shares
        position_size = position_value / signal.price if signal.price > 0 else 0
        
        # No stop loss for overnight (exit at open)
        stop_loss = 0.0
        
        return position_size, stop_loss
    
    def backtest_metrics(self, trades: pd.DataFrame) -> dict:
        """Calculate overnight-specific metrics."""
        if trades.empty:
            return {}
            
        metrics = {
            'total_overnight_trades': len(trades),
            'avg_overnight_return': trades['pnl_pct'].mean(),
            'overnight_win_rate': (trades['pnl'] > 0).mean(),
            'best_day': None,
            'worst_day': None
        }
        
        # Analyze by day of week
        if 'metadata' in trades.columns:
            day_returns = {}
            for _, trade in trades.iterrows():
                day = trade['metadata'].get('day', 'Unknown')
                if day not in day_returns:
                    day_returns[day] = []
                day_returns[day].append(trade['pnl_pct'])
            
            # Find best/worst days
            avg_by_day = {day: np.mean(returns) for day, returns in day_returns.items()}
            if avg_by_day:
                metrics['best_day'] = max(avg_by_day, key=avg_by_day.get)
                metrics['worst_day'] = min(avg_by_day, key=avg_by_day.get)
                metrics['returns_by_day'] = avg_by_day
        
        return metrics
    
    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate overnight drift strategy configuration."""
        return validate_overnight_drift_config(config)