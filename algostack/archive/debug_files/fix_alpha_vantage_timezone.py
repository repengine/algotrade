#!/usr/bin/env python3
"""Fix Alpha Vantage timezone issue for OvernightDrift strategy."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
import pytz
from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager
from strategies.overnight_drift import OvernightDrift
from strategy_defaults import merge_with_defaults

def test_timezone_fix():
    """Test if adding timezone fixes the issue."""
    
    data_manager = AlphaVantageDataManager()
    strategy_manager = PandasStrategyManager()
    
    # Get Alpha Vantage data
    av_data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
    print(f"Original AV data timezone: {av_data.index.tz}")
    print(f"Original AV data shape: {av_data.shape}")
    
    # Add timezone to match Yahoo Finance
    av_data_tz = av_data.copy()
    av_data_tz.index = av_data_tz.index.tz_localize('America/New_York')
    print(f"\nFixed AV data timezone: {av_data_tz.index.tz}")
    
    # Test backtest with timezone-aware data
    strategy_name = "OvernightDrift"
    strategy_class = strategy_manager.strategies[strategy_name]
    
    user_params = {
        'symbol': 'SPY',
        'position_size': 0.95,
        'lookback_period': 60,
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False
    }
    
    print("\nRunning backtest with timezone-aware Alpha Vantage data...")
    results = strategy_manager.run_backtest(
        strategy_class,
        strategy_name,
        user_params,
        av_data_tz,
        initial_capital=100000
    )
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Total return: {results.get('total_return', 0):.2f}%")
        print(f"Number of trades: {results.get('num_trades', 0)}")
        
        signals_df = results.get('signals', pd.DataFrame())
        if not signals_df.empty:
            print(f"Signals generated: {len(signals_df)}")
        else:
            print("Still no signals!")
            
            # Debug further
            print("\nDebugging OvernightDrift logic...")
            
            # Initialize strategy directly
            from strategy_integration_helpers import DataFormatConverter
            converter = DataFormatConverter()
            
            strategy_data = converter.dashboard_to_strategy(av_data_tz, symbol='SPY')
            
            full_config = merge_with_defaults(strategy_name, user_params)
            strategy = OvernightDrift(full_config)
            strategy.init()
            
            # Test with specific data window
            test_data = strategy_data.iloc[-100:].copy()
            test_data.attrs['symbol'] = 'SPY'
            
            # Check overnight returns calculation
            print("\nChecking overnight returns calculation...")
            edge = strategy.calculate_overnight_edge(test_data)
            print(f"Overnight edge: {edge}")
            
            # Check what day of week the last few days are
            print("\nLast 5 days:")
            for idx in test_data.index[-5:]:
                day_of_week = idx.strftime('%A')
                print(f"  {idx.strftime('%Y-%m-%d')} - {day_of_week}")
                
                # Check if it's a holding day
                if day_of_week in strategy.config.get('hold_days', ['Monday', 'Tuesday', 'Wednesday', 'Thursday']):
                    print(f"    ✓ Is a holding day")
                else:
                    print(f"    ✗ Not a holding day")

def check_overnight_drift_logic():
    """Check the actual logic of OvernightDrift strategy."""
    
    print("\n" + "="*60)
    print("Checking OvernightDrift strategy logic")
    print("="*60)
    
    # Check the strategy's calculate_overnight_edge method
    from strategies.overnight_drift import OvernightDrift
    
    # Create dummy data
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'open': 100 + pd.Series(range(100)) * 0.1,
        'high': 101 + pd.Series(range(100)) * 0.1,
        'low': 99 + pd.Series(range(100)) * 0.1,
        'close': 100.5 + pd.Series(range(100)) * 0.1,
        'volume': 1000000
    }, index=dates)
    
    dummy_data.attrs['symbol'] = 'SPY'
    
    config = {
        'symbol': 'SPY',
        'symbols': ['SPY'],
        'lookback_period': 60,
        'momentum_period': 10,
        'trend_filter': False
    }
    
    strategy = OvernightDrift(config)
    edge = strategy.calculate_overnight_edge(dummy_data)
    print(f"Dummy data overnight edge: {edge}")

if __name__ == "__main__":
    test_timezone_fix()
    check_overnight_drift_logic()