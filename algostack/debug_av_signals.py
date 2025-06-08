#!/usr/bin/env python3
"""Debug Alpha Vantage signal generation issues."""

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

def debug_av_vs_yf():
    """Compare Alpha Vantage and Yahoo Finance data."""
    
    data_manager = AlphaVantageDataManager()
    
    # Get both data sources
    print("Fetching Alpha Vantage data...")
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
    
    print("\nFetching Yahoo Finance data...")
    yf_data = data_manager.fetch_data("SPY", "1y", "1d", "yfinance")
    
    print(f"\nAlpha Vantage shape: {av_data.shape}")
    print(f"Yahoo Finance shape: {yf_data.shape}")
    
    print(f"\nAlpha Vantage columns: {list(av_data.columns)}")
    print(f"Yahoo Finance columns: {list(yf_data.columns)}")
    
    print(f"\nAlpha Vantage sample:")
    print(av_data.tail(3))
    
    print(f"\nYahoo Finance sample:")
    print(yf_data.tail(3))
    
    # Check volume data specifically
    print(f"\nAlpha Vantage volume stats:")
    print(f"  Mean: {av_data['volume'].mean():.2f}")
    print(f"  Min: {av_data['volume'].min()}")
    print(f"  Max: {av_data['volume'].max()}")
    print(f"  Zero count: {(av_data['volume'] == 0).sum()}")
    
    print(f"\nYahoo Finance volume stats:")
    yf_vol_col = 'Volume' if 'Volume' in yf_data.columns else 'volume'
    print(f"  Mean: {yf_data[yf_vol_col].mean():.2f}")
    print(f"  Min: {yf_data[yf_vol_col].min()}")
    print(f"  Max: {yf_data[yf_vol_col].max()}")
    print(f"  Zero count: {(yf_data[yf_vol_col] == 0).sum()}")
    
    # Test OvernightDrift strategy with both
    print("\n" + "="*60)
    print("Testing OvernightDrift Strategy")
    print("="*60)
    
    # Convert to strategy format
    from strategy_integration_helpers import DataFormatConverter
    converter = DataFormatConverter()
    
    av_strategy_data = converter.dashboard_to_strategy(av_data, symbol='SPY')
    yf_strategy_data = converter.dashboard_to_strategy(yf_data, symbol='SPY')
    
    # Initialize strategy
    config = {
        'symbol': 'SPY',
        'symbols': ['SPY'],
        'lookback_period': 60,
        'position_size': 0.95,
        'volume_filter': False  # Disable volume filter for testing
    }
    
    # Test with Alpha Vantage data
    print("\nTesting with Alpha Vantage data:")
    strategy = OvernightDrift(config)
    strategy.init()
    
    # Try to get a signal from the last 100 bars
    test_data = av_strategy_data.iloc[-100:].copy()
    test_data.attrs['symbol'] = 'SPY'
    
    signal = strategy.next(test_data)
    print(f"Signal: {signal}")
    
    # Check strategy internals
    if hasattr(strategy, 'overnight_returns'):
        print(f"Overnight returns calculated: {len(strategy.overnight_returns)}")
        print(f"Sample overnight returns: {strategy.overnight_returns.tail()}")
    
    # Test with Yahoo Finance data
    print("\nTesting with Yahoo Finance data:")
    strategy2 = OvernightDrift(config)
    strategy2.init()
    
    test_data2 = yf_strategy_data.iloc[-100:].copy()
    test_data2.attrs['symbol'] = 'SPY'
    
    signal2 = strategy2.next(test_data2)
    print(f"Signal: {signal2}")

if __name__ == "__main__":
    debug_av_vs_yf()