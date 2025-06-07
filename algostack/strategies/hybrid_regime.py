#!/usr/bin/env python3
"""Hybrid regime strategy that switches between mean reversion and trend following."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import talib

from strategies.base import BaseStrategy, Signal, RiskContext
from utils.validators.strategy_validators import validate_hybrid_regime_config
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti


class HybridRegime(BaseStrategy):
    """Regime-based strategy that dynamically allocates between strategies.
    
    Low volatility regime: Mean reversion
    High volatility/trending regime: Trend following
    Uses ADX, Bollinger Band width, and volatility metrics for regime detection
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        default_config = {
            'name': 'HybridRegime',
            'symbols': ['SPY', 'QQQ', 'IWM', 'DIA'],
            'lookback_period': 252,
            'regime_window': 20,           # Days for regime calculation
            'adx_threshold': 25,           # ADX > 25 = trending
            'bb_width_threshold': 0.15,    # BB width < 15% = low vol
            'vol_percentile_low': 30,      # Low vol regime
            'vol_percentile_high': 70,     # High vol regime
            'regime_change_buffer': 3,     # Days to confirm regime change
            'allocation_mr': 0.6,          # Allocation to MR in low vol
            'allocation_tf': 0.6,          # Allocation to TF in trending
            'blend_zone_width': 5,         # Blend strategies near thresholds
        }
        
        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)
        
        # Initialize sub-strategies
        mr_config = {
            'symbols': self.config['symbols'],
            'rsi_period': 2,
            'rsi_oversold': 10,
            'rsi_overbought': 90,
            'atr_period': 14,
            'lookback_period': self.config['lookback_period']
        }
        
        tf_config = {
            'symbols': self.config['symbols'],
            'channel_period': 20,
            'trail_period': 10,
            'atr_period': 14,
            'adx_period': 14,
            'lookback_period': self.config['lookback_period']
        }
        
        self.mean_reversion = MeanReversionEquity(mr_config)
        self.trend_following = TrendFollowingMulti(tf_config)
        
        self.current_regime = {}  # Track regime per symbol
        self.regime_history = {}  # Track regime changes
        self.positions = {}       # Track positions by strategy
        
    def init(self) -> None:
        """Initialize strategy components."""
        self.mean_reversion.init()
        self.trend_following.init()
        self.current_regime.clear()
        self.regime_history.clear()
        self.positions.clear()
    
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime using multiple indicators."""
        df = data.copy()
        
        # Calculate indicators
        # ADX for trend strength
        df['adx'] = talib.ADX(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=self.config['adx_period']
        )
        
        # Bollinger Bands for volatility regime
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=self.config['regime_window'],
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        # BB width as percentage
        df['bb_width'] = np.where(df['bb_middle'] > 0, (df['bb_upper'] - df['bb_lower']) / df['bb_middle'], 0.0)
        
        # Historical volatility
        df['returns'] = df['close'].pct_change()
        df['hvol'] = df['returns'].rolling(self.config['regime_window']).std() * np.sqrt(252)
        
        # Volatility percentile
        vol_percentile = df['hvol'].iloc[-1]
        vol_history = df['hvol'].dropna()
        if len(vol_history) > self.config['regime_window']:
            vol_rank = (vol_history < vol_percentile).sum() / len(vol_history) * 100 if len(vol_history) > 0 else 50
        else:
            vol_rank = 50
        
        # Trend indicators
        df['sma_fast'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_slow'] = talib.SMA(df['close'], timeperiod=30)
        trend_strength = (df['sma_fast'].iloc[-1] - df['sma_slow'].iloc[-1]) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0.0
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Determine regime
        regime_scores = {
            'mean_reversion': 0,
            'trend_following': 0,
            'neutral': 0
        }
        
        # Low volatility favors mean reversion
        if latest['bb_width'] < self.config['bb_width_threshold']:
            regime_scores['mean_reversion'] += 2
        
        if vol_rank < self.config['vol_percentile_low']:
            regime_scores['mean_reversion'] += 1
            
        # High ADX favors trend following
        if latest['adx'] > self.config['adx_threshold']:
            regime_scores['trend_following'] += 2
            
        # Strong trend favors trend following
        if abs(trend_strength) > 0.02:  # 2% difference
            regime_scores['trend_following'] += 1
            
        # High volatility can favor trend (breakouts)
        if vol_rank > self.config['vol_percentile_high']:
            regime_scores['trend_following'] += 1
            
        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        
        # Calculate confidence (0-1)
        total_score = sum(regime_scores.values())
        confidence = regime_scores[primary_regime] / total_score if total_score > 0 else 0
        
        return {
            'regime': primary_regime,
            'confidence': confidence,
            'scores': regime_scores,
            'indicators': {
                'adx': latest['adx'],
                'bb_width': latest['bb_width'],
                'vol_rank': vol_rank,
                'trend_strength': trend_strength
            }
        }
    
    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process data and route to appropriate sub-strategy."""
        if not self.validate_data(data):
            return None
            
        symbol = data.attrs.get('symbol', 'UNKNOWN')
        current_time = data.index[-1]
        
        # Detect current regime
        regime_info = self.detect_regime(data)
        current_regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        # Track regime changes
        if symbol not in self.current_regime:
            self.current_regime[symbol] = current_regime
            self.regime_history[symbol] = [(current_time, current_regime)]
        else:
            # Check for regime change
            if current_regime != self.current_regime[symbol]:
                # Require confirmation over buffer period
                recent_history = self.regime_history[symbol][-self.config['regime_change_buffer']:]
                if len(recent_history) >= self.config['regime_change_buffer'] - 1:
                    # Confirmed regime change
                    self.current_regime[symbol] = current_regime
                    self.regime_history[symbol].append((current_time, current_regime))
                    
                    # Exit positions from inactive strategy
                    if symbol in self.positions:
                        old_strategy = self.positions[symbol]['strategy']
                        if (old_strategy == 'mean_reversion' and current_regime == 'trend_following') or \
                           (old_strategy == 'trend_following' and current_regime == 'mean_reversion'):
                            # Generate exit signal
                            return Signal(
                                timestamp=current_time,
                                symbol=symbol,
                                direction='FLAT',
                                strength=0.0,
                                strategy_id=self.name,
                                price=data.iloc[-1]['close'],
                                metadata={
                                    'reason': 'regime_change',
                                    'old_regime': old_strategy,
                                    'new_regime': current_regime
                                }
                            )
        
        # Route to appropriate strategy
        signal = None
        
        if current_regime == 'mean_reversion':
            # Get signal from mean reversion strategy
            mr_signal = self.mean_reversion.next(data)
            if mr_signal:
                # Adjust signal strength based on regime confidence
                mr_signal.strength *= confidence * self.config['allocation_mr']
                mr_signal.metadata['regime'] = 'mean_reversion'
                mr_signal.metadata['regime_confidence'] = confidence
                signal = mr_signal
                
                # Track position
                if mr_signal.direction != 'FLAT':
                    self.positions[symbol] = {
                        'strategy': 'mean_reversion',
                        'entry_time': current_time
                    }
                elif symbol in self.positions:
                    del self.positions[symbol]
                    
        elif current_regime == 'trend_following':
            # Get signal from trend following strategy
            tf_signal = self.trend_following.next(data)
            if tf_signal:
                # Adjust signal strength based on regime confidence
                tf_signal.strength *= confidence * self.config['allocation_tf']
                tf_signal.metadata['regime'] = 'trend_following'
                tf_signal.metadata['regime_confidence'] = confidence
                signal = tf_signal
                
                # Track position
                if tf_signal.direction != 'FLAT':
                    self.positions[symbol] = {
                        'strategy': 'trend_following',
                        'entry_time': current_time
                    }
                elif symbol in self.positions:
                    del self.positions[symbol]
        
        # In neutral regime or low confidence, blend strategies
        elif current_regime == 'neutral' or confidence < 0.5:
            # Get signals from both strategies
            mr_signal = self.mean_reversion.next(data)
            tf_signal = self.trend_following.next(data)
            
            # Blend signals if both present
            if mr_signal and tf_signal:
                # Average the signals with regime-based weighting
                total_score = sum(regime_info['scores'].values())
                mr_weight = regime_info['scores']['mean_reversion'] / total_score if total_score > 0 else 0.5
                tf_weight = regime_info['scores']['trend_following'] / total_score if total_score > 0 else 0.5
                
                if mr_signal.direction == tf_signal.direction:
                    # Strategies agree - stronger signal
                    blended_strength = (mr_signal.strength * mr_weight + 
                                      tf_signal.strength * tf_weight)
                    signal = Signal(
                        timestamp=current_time,
                        symbol=symbol,
                        direction=mr_signal.direction,
                        strength=blended_strength,
                        strategy_id=self.name,
                        price=mr_signal.price,
                        atr=mr_signal.atr,
                        metadata={
                            'regime': 'blended',
                            'mr_weight': mr_weight,
                            'tf_weight': tf_weight,
                            'agreement': True
                        }
                    )
                else:
                    # Strategies disagree - skip or take stronger signal
                    if abs(mr_signal.strength * mr_weight) > abs(tf_signal.strength * tf_weight):
                        signal = mr_signal
                        signal.strength *= mr_weight
                    else:
                        signal = tf_signal
                        signal.strength *= tf_weight
                        
            elif mr_signal:
                signal = mr_signal
                signal.strength *= 0.5  # Reduce strength in neutral regime
            elif tf_signal:
                signal = tf_signal
                signal.strength *= 0.5
        
        # Add regime information to signal metadata
        if signal:
            signal.metadata.update({
                'hybrid_regime': current_regime,
                'regime_indicators': regime_info['indicators']
            })
            
        return signal
    
    def size(self, signal: Signal, risk_context: RiskContext) -> Tuple[float, float]:
        """Calculate position size based on active sub-strategy."""
        if signal.direction == 'FLAT':
            return 0.0, 0.0
            
        # Determine which strategy generated the signal
        regime = signal.metadata.get('regime', 'mean_reversion')
        
        if regime == 'mean_reversion' or regime == 'blended':
            return self.mean_reversion.size(signal, risk_context)
        else:
            return self.trend_following.size(signal, risk_context)
    
    def backtest_metrics(self, trades: pd.DataFrame) -> dict:
        """Calculate hybrid strategy specific metrics."""
        if trades.empty:
            return {}
            
        metrics = {
            'total_trades': len(trades),
            'regime_changes': 0,
            'mr_trades': 0,
            'tf_trades': 0,
            'blended_trades': 0
        }
        
        # Analyze by regime
        for _, trade in trades.iterrows():
            if 'metadata' in trade and isinstance(trade['metadata'], dict):
                regime = trade['metadata'].get('regime', 'unknown')
                if regime == 'mean_reversion':
                    metrics['mr_trades'] += 1
                elif regime == 'trend_following':
                    metrics['tf_trades'] += 1
                elif regime == 'blended':
                    metrics['blended_trades'] += 1
                    
        # Calculate regime effectiveness
        if metrics['mr_trades'] > 0:
            mr_trades_df = trades[trades.apply(
                lambda x: x.get('metadata', {}).get('regime') == 'mean_reversion', axis=1
            )]
            metrics['mr_win_rate'] = (mr_trades_df['pnl'] > 0).mean()
            metrics['mr_avg_pnl'] = mr_trades_df['pnl'].mean()
            
        if metrics['tf_trades'] > 0:
            tf_trades_df = trades[trades.apply(
                lambda x: x.get('metadata', {}).get('regime') == 'trend_following', axis=1
            )]
            metrics['tf_win_rate'] = (tf_trades_df['pnl'] > 0).mean()
            metrics['tf_avg_pnl'] = tf_trades_df['pnl'].mean()
        
        return metrics
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hybrid regime strategy configuration."""
        return validate_hybrid_regime_config(config)