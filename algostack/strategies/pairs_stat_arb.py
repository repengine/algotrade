#!/usr/bin/env python3
"""Pairs statistical arbitrage strategy using cointegration and z-score signals."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import talib
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression

from strategies.base import BaseStrategy, Signal, RiskContext
from utils.validators.strategy_validators import validate_pairs_trading_config


class PairsStatArb(BaseStrategy):
    """Statistical arbitrage strategy for cointegrated pairs.
    
    Entry: Z-score exceeds ±2 standard deviations
    Exit: Z-score returns to 0 (mean reversion)
    Pairs: Automatically selected based on cointegration tests
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        default_config = {
            'name': 'PairsStatArb',
            'symbols': [],  # Will be populated with pairs
            'lookback_period': 90,  # Days for cointegration test
            'zscore_window': 30,    # Rolling window for z-score
            'entry_threshold': 2.0,  # Z-score threshold
            'exit_threshold': 0.2,   # Close to mean
            'stop_threshold': 3.5,   # Stop loss z-score
            'min_half_life': 5,     # Minimum half-life in days
            'max_half_life': 30,    # Maximum half-life in days
            'adf_pvalue': 0.05,     # ADF test p-value threshold
            'max_pairs': 5,         # Maximum concurrent pairs
            'recalibrate_days': 30, # Recalibrate beta every N days
        }
        
        # Merge with provided config
        full_config = {**default_config, **config}
        super().__init__(full_config)
        
        self.pairs = {}  # Dict of {(symbol1, symbol2): pair_data}
        self.positions = {}  # Track open positions
        self.last_calibration = {}  # Track calibration dates
    
    def init(self) -> None:
        """Initialize strategy components."""
        self.pairs.clear()
        self.positions.clear()
        self.last_calibration.clear()
    
    def find_cointegrated_pairs(
        self, 
        symbols: List[str], 
        price_data: Dict[str, pd.DataFrame]
    ) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs from a list of symbols."""
        pairs = []
        
        # Test all combinations
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if sym1 not in price_data or sym2 not in price_data:
                    continue
                    
                # Get price series
                prices1 = price_data[sym1]['close']
                prices2 = price_data[sym2]['close']
                
                # Align series
                aligned = pd.DataFrame({
                    'p1': prices1,
                    'p2': prices2
                }).dropna()
                
                if len(aligned) < self.config['lookback_period']:
                    continue
                
                # Test for cointegration
                try:
                    _, pvalue, _ = coint(aligned['p1'], aligned['p2'])
                    
                    if pvalue < self.config['adf_pvalue']:
                        # Calculate half-life
                        spread = self._calculate_spread(aligned['p1'], aligned['p2'])
                        half_life = self._calculate_half_life(spread)
                        
                        # Check half-life constraints
                        if (self.config['min_half_life'] <= half_life <= 
                            self.config['max_half_life']):
                            pairs.append((sym1, sym2, pvalue))
                            
                except Exception as e:
                    continue
        
        # Sort by p-value (lower is better)
        pairs.sort(key=lambda x: x[2])
        
        return pairs[:self.config['max_pairs']]
    
    def _calculate_spread(
        self, 
        prices1: pd.Series, 
        prices2: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """Calculate spread using rolling OLS regression."""
        if window is None:
            window = self.config['lookback_period']
            
        # Use rolling regression for dynamic hedge ratio
        spreads = []
        betas = []
        
        for i in range(window, len(prices1)):
            # Get window data
            p1_window = prices1.iloc[i-window:i].values.reshape(-1, 1)
            p2_window = prices2.iloc[i-window:i].values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(p1_window, p2_window)
            beta = model.coef_[0]
            
            # Calculate spread
            spread = prices2.iloc[i] - beta * prices1.iloc[i]
            spreads.append(spread)
            betas.append(beta)
        
        spread_series = pd.Series(spreads, index=prices1.index[window:])
        self.betas = pd.Series(betas, index=prices1.index[window:])
        
        return spread_series
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life using OLS."""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align series
        spread_lag = spread_lag[spread_diff.index]
        
        # OLS regression: spread_diff = lambda * spread_lag + c
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        lambda_coef = model.coef_[0]
        
        # Half-life = -log(2) / lambda
        if lambda_coef < 0:
            half_life = -np.log(2) / lambda_coef if lambda_coef != 0 else float('inf')
        else:
            half_life = float('inf')
            
        return half_life
    
    def _calculate_zscore(
        self, 
        spread: pd.Series, 
        window: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling z-score of spread."""
        if window is None:
            window = self.config['zscore_window']
            
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std()
        
        zscore = np.where(spread_std > 0, (spread - spread_mean) / spread_std, 0.0)
        
        return zscore
    
    def next(self, data: pd.DataFrame) -> Optional[Signal]:
        """Process new data and generate trading signal.
        
        Note: This strategy requires multiple symbols, so data should contain
        price information for all symbols in the universe.
        """
        # For pairs trading, we need a different approach
        # This would typically be called with a dict of DataFrames
        # For now, return None as this requires portfolio-level implementation
        return None
    
    def next_pairs(
        self, 
        price_data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """Process multiple symbols and generate pairs trading signals."""
        signals = []
        current_time = datetime.now()
        
        # Recalibrate pairs periodically
        if (not self.pairs or 
            not self.last_calibration or
            (current_time - self.last_calibration.get('date', datetime.min)).days 
            >= self.config['recalibrate_days']):
            
            symbols = list(price_data.keys())
            cointegrated_pairs = self.find_cointegrated_pairs(symbols, price_data)
            
            # Update pairs
            self.pairs.clear()
            for sym1, sym2, pvalue in cointegrated_pairs:
                self.pairs[(sym1, sym2)] = {
                    'pvalue': pvalue,
                    'calibration_date': current_time
                }
            
            self.last_calibration['date'] = current_time
        
        # Check each pair for signals
        for (sym1, sym2), pair_info in self.pairs.items():
            if sym1 not in price_data or sym2 not in price_data:
                continue
                
            # Get aligned price data
            prices1 = price_data[sym1]['close']
            prices2 = price_data[sym2]['close']
            
            aligned = pd.DataFrame({
                'p1': prices1,
                'p2': prices2
            }).dropna()
            
            if len(aligned) < self.config['lookback_period']:
                continue
            
            # Calculate spread and z-score
            spread = self._calculate_spread(aligned['p1'], aligned['p2'])
            zscore = self._calculate_zscore(spread)
            
            if zscore.empty:
                continue
                
            latest_zscore = zscore.iloc[-1]
            latest_beta = self.betas.iloc[-1] if hasattr(self, 'betas') else 1.0
            
            pair_key = f"{sym1}_{sym2}"
            
            # Check for exit signals first
            if pair_key in self.positions:
                pos = self.positions[pair_key]
                exit_signal = False
                
                # Exit conditions
                if pos['direction'] == 'LONG_SPREAD':
                    # Exit if z-score returns to mean or exceeds stop
                    if (abs(latest_zscore) < self.config['exit_threshold'] or
                        latest_zscore < -self.config['stop_threshold']):
                        exit_signal = True
                        
                elif pos['direction'] == 'SHORT_SPREAD':
                    # Exit if z-score returns to mean or exceeds stop
                    if (abs(latest_zscore) < self.config['exit_threshold'] or
                        latest_zscore > self.config['stop_threshold']):
                        exit_signal = True
                
                if exit_signal:
                    # Generate exit signals for both legs
                    # Exit leg 1
                    signals.append(Signal(
                        timestamp=current_time,
                        symbol=sym1,
                        direction='FLAT',
                        strength=0.0,
                        strategy_id=self.name,
                        price=aligned['p1'].iloc[-1],
                        metadata={
                            'pair': pair_key,
                            'leg': 1,
                            'zscore': latest_zscore,
                            'reason': 'pairs_exit'
                        }
                    ))
                    
                    # Exit leg 2
                    signals.append(Signal(
                        timestamp=current_time,
                        symbol=sym2,
                        direction='FLAT',
                        strength=0.0,
                        strategy_id=self.name,
                        price=aligned['p2'].iloc[-1],
                        metadata={
                            'pair': pair_key,
                            'leg': 2,
                            'zscore': latest_zscore,
                            'reason': 'pairs_exit'
                        }
                    ))
                    
                    del self.positions[pair_key]
            
            # Check for entry signals
            elif len(self.positions) < self.config['max_pairs']:
                # Long spread: z-score < -threshold (spread undervalued)
                if latest_zscore < -self.config['entry_threshold']:
                    # Long sym2, short beta*sym1
                    strength = min(1.0, abs(latest_zscore) / 3.0)
                    
                    # Signal for leg 1 (short)
                    signals.append(Signal(
                        timestamp=current_time,
                        symbol=sym1,
                        direction='SHORT',
                        strength=-strength * latest_beta,
                        strategy_id=self.name,
                        price=aligned['p1'].iloc[-1],
                        metadata={
                            'pair': pair_key,
                            'leg': 1,
                            'beta': latest_beta,
                            'zscore': latest_zscore,
                            'spread': spread.iloc[-1],
                            'reason': 'pairs_entry_long_spread'
                        }
                    ))
                    
                    # Signal for leg 2 (long)
                    signals.append(Signal(
                        timestamp=current_time,
                        symbol=sym2,
                        direction='LONG',
                        strength=strength,
                        strategy_id=self.name,
                        price=aligned['p2'].iloc[-1],
                        metadata={
                            'pair': pair_key,
                            'leg': 2,
                            'beta': latest_beta,
                            'zscore': latest_zscore,
                            'spread': spread.iloc[-1],
                            'reason': 'pairs_entry_long_spread'
                        }
                    ))
                    
                    self.positions[pair_key] = {
                        'direction': 'LONG_SPREAD',
                        'entry_zscore': latest_zscore,
                        'entry_spread': spread.iloc[-1],
                        'beta': latest_beta,
                        'entry_time': current_time
                    }
                
                # Short spread: z-score > threshold (spread overvalued)
                elif latest_zscore > self.config['entry_threshold']:
                    # Short sym2, long beta*sym1
                    strength = min(1.0, abs(latest_zscore) / 3.0)
                    
                    # Signal for leg 1 (long)
                    signals.append(Signal(
                        timestamp=current_time,
                        symbol=sym1,
                        direction='LONG',
                        strength=strength * latest_beta,
                        strategy_id=self.name,
                        price=aligned['p1'].iloc[-1],
                        metadata={
                            'pair': pair_key,
                            'leg': 1,
                            'beta': latest_beta,
                            'zscore': latest_zscore,
                            'spread': spread.iloc[-1],
                            'reason': 'pairs_entry_short_spread'
                        }
                    ))
                    
                    # Signal for leg 2 (short)
                    signals.append(Signal(
                        timestamp=current_time,
                        symbol=sym2,
                        direction='SHORT',
                        strength=-strength,
                        strategy_id=self.name,
                        price=aligned['p2'].iloc[-1],
                        metadata={
                            'pair': pair_key,
                            'leg': 2,
                            'beta': latest_beta,
                            'zscore': latest_zscore,
                            'spread': spread.iloc[-1],
                            'reason': 'pairs_entry_short_spread'
                        }
                    ))
                    
                    self.positions[pair_key] = {
                        'direction': 'SHORT_SPREAD',
                        'entry_zscore': latest_zscore,
                        'entry_spread': spread.iloc[-1],
                        'beta': latest_beta,
                        'entry_time': current_time
                    }
        
        return signals
    
    def size(self, signal: Signal, risk_context: RiskContext) -> Tuple[float, float]:
        """Calculate position size for pairs trading legs."""
        if signal.direction == 'FLAT':
            return 0.0, 0.0
        
        # Extract pair information
        pair_key = signal.metadata.get('pair', '')
        leg = signal.metadata.get('leg', 1)
        beta = signal.metadata.get('beta', 1.0)
        
        # Base allocation per pair (split capital among max pairs)
        pair_allocation = risk_context.account_equity / (self.config['max_pairs'] * 2) if self.config['max_pairs'] > 0 else risk_context.account_equity * 0.1
        
        # Adjust for beta (hedge ratio)
        if leg == 1:
            position_value = pair_allocation * abs(beta)
        else:
            position_value = pair_allocation
        
        # Apply signal strength
        position_value *= abs(signal.strength)
        
        # Calculate shares
        position_size = position_value / signal.price if signal.price > 0 else 0
        
        # No traditional stop loss for pairs trading
        # Risk is managed through z-score thresholds
        stop_loss = 0.0
        
        return position_size, stop_loss
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pairs trading strategy configuration."""
        return validate_pairs_trading_config(config)