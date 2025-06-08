#!/usr/bin/env python3
"""Fix OvernightDrift strategy config and test signal generation."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
from dashboard_pandas import AlphaVantageDataManager
from strategies.overnight_drift import OvernightDrift
from strategy_defaults import get_strategy_defaults
from strategy_integration_helpers import DataFormatConverter

def test_overnight_drift_fix():
    """Test OvernightDrift with proper config."""
    
    data_manager = AlphaVantageDataManager()
    converter = DataFormatConverter()
    
    # Get data
    print("Testing with Yahoo Finance data...")
    yf_data = data_manager.fetch_data("SPY", "1y", "1d", "yfinance")
    yf_strategy_data = converter.dashboard_to_strategy(yf_data, symbol='SPY')
    
    print(f"Data shape: {yf_data.shape}")
    print(f"Data date range: {yf_data.index[0]} to {yf_data.index[-1]}")
    
    # Get proper defaults
    defaults = get_strategy_defaults('OvernightDrift')
    print(f"\nDefault config: {defaults}")
    
    # Create full config with missing parameters
    config = {
        'symbol': 'SPY',
        'symbols': ['SPY'],
        'lookback_period': 60,
        'position_size': 0.95,
        'atr_period': 14,  # Add missing parameter
        'volume_filter': False,  # Disable for testing
        'trend_filter': False,   # Disable for testing
        **defaults  # Include all defaults
    }
    
    print(f"\nFull config: {config}")
    
    # Initialize strategy
    strategy = OvernightDrift(config)
    strategy.init()
    
    # Test signal generation over last 100 days
    signals_generated = []
    positions_held = []
    
    for i in range(100, len(yf_strategy_data)):
        test_data = yf_strategy_data.iloc[:i+1].copy()
        test_data.attrs['symbol'] = 'SPY'
        
        signal = strategy.next(test_data)
        if signal:
            signals_generated.append({
                'date': test_data.index[-1],
                'direction': signal.direction,
                'day_of_week': test_data.index[-1].strftime('%A'),
                'metadata': signal.metadata
            })
            
            # Track positions
            if signal.direction == 'LONG':
                positions_held.append(test_data.index[-1])
            elif signal.direction == 'FLAT' and positions_held:
                positions_held.pop()
    
    print(f"\nSignals generated: {len(signals_generated)}")
    if signals_generated:
        print("\nFirst 5 signals:")
        for sig in signals_generated[:5]:
            print(f"  {sig['date'].strftime('%Y-%m-%d')} ({sig['day_of_week']}): {sig['direction']} - {sig['metadata'].get('reason', 'entry')}")
    
    # Now test with Alpha Vantage
    print("\n" + "="*60)
    print("Testing with Alpha Vantage data...")
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
    av_strategy_data = converter.dashboard_to_strategy(av_data, symbol='SPY')
    
    print(f"Data shape: {av_data.shape}")
    
    # Test with last 10 bars to see what's happening
    test_data = av_strategy_data.iloc[-10:].copy()
    test_data.attrs['symbol'] = 'SPY'
    
    # Check day of week
    print("\nLast 10 days in Alpha Vantage data:")
    for idx in test_data.index[-10:]:
        print(f"  {idx.strftime('%Y-%m-%d %A')}")
    
    # Initialize new strategy instance
    strategy2 = OvernightDrift(config)
    strategy2.init()
    
    # Try to get a signal
    signal = strategy2.next(test_data)
    print(f"\nSignal with 10 bars: {signal}")
    
    # Check with more data
    test_data_100 = av_strategy_data.iloc[-100:].copy()
    test_data_100.attrs['symbol'] = 'SPY'
    signal2 = strategy2.next(test_data_100)
    print(f"Signal with 100 bars: {signal2}")

def check_strategy_defaults():
    """Check and fix strategy defaults."""
    print("\nChecking strategy defaults...")
    
    # Check each strategy's defaults
    strategies = ['OvernightDrift', 'MeanReversionEquity', 'HybridRegime', 'IntradayORB', 'TrendFollowingMulti']
    
    for strat in strategies:
        defaults = get_strategy_defaults(strat)
        print(f"\n{strat}:")
        if 'atr_period' in defaults:
            print(f"  ✓ Has atr_period: {defaults['atr_period']}")
        else:
            print(f"  ✗ Missing atr_period")

if __name__ == "__main__":
    check_strategy_defaults()
    print("\n" + "="*60)
    test_overnight_drift_fix()