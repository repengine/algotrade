#!/usr/bin/env python3
"""Detailed debugging of strategy signal generation with Alpha Vantage data."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager, PandasStrategyManager
from strategies.mean_reversion_equity import MeanReversionEquity
from strategy_integration_helpers import DataFormatConverter
import pandas as pd

def debug_mean_reversion_signals():
    """Debug MeanReversionEquity strategy in detail."""
    
    os.environ['ALPHA_VANTAGE_API_KEY'] = '991AR2LC298IGMX7'
    
    data_manager = AlphaVantageDataManager()
    converter = DataFormatConverter()
    
    # Get Alpha Vantage data
    print("Fetching Alpha Vantage data...")
    av_data = data_manager.fetch_data("SPY", "3mo", "1d", "alpha_vantage")
    
    print(f"\nData shape: {av_data.shape}")
    print(f"Data columns: {list(av_data.columns)}")
    
    # Convert data
    strategy_data = converter.dashboard_to_strategy(av_data, symbol='SPY')
    
    # Initialize strategy with debug-friendly parameters
    config = {
        'symbol': 'SPY',
        'symbols': ['SPY'],
        'lookback_period': 20,
        'rsi_period': 2,  # Short period as used by the strategy
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'position_size': 0.95,
        'volume_filter': False,  # Disable all filters
        'atr_period': 14,
        'atr_band_mult': 2.5,
        'stop_loss_atr': 3.0,
        'max_positions': 5
    }
    
    strategy = MeanReversionEquity(config)
    strategy.init()
    
    # Check positions attribute
    print(f"\nStrategy positions: {getattr(strategy, 'positions', 'No positions attribute')}")
    
    # Test with the most recent data
    test_data = strategy_data.copy()
    test_data.attrs['symbol'] = 'SPY'
    
    # Calculate indicators
    df_with_indicators = strategy.calculate_indicators(test_data)
    
    print("\nIndicator values (last 10 rows):")
    print(df_with_indicators[['close', 'rsi', 'atr', 'sma_20', 'lower_band', 'volume_ratio']].tail(10))
    
    # Check for oversold conditions in the entire dataset
    oversold_days = df_with_indicators[df_with_indicators['rsi'] < 30]
    print(f"\nDays with RSI < 30 (oversold): {len(oversold_days)}")
    if len(oversold_days) > 0:
        print("\nSample oversold days:")
        print(oversold_days[['close', 'rsi', 'lower_band', 'volume_ratio']].head(5))
    
    # Test signal generation at specific oversold points
    if len(oversold_days) > 0:
        print("\nTesting signal generation at oversold points...")
        
        for idx, (date, row) in enumerate(oversold_days.iterrows()):
            if idx >= 5:  # Test first 5 oversold days
                break
                
            # Get data up to this point
            date_idx = test_data.index.get_loc(date)
            historical_data = test_data.iloc[:date_idx+1].copy()
            historical_data.attrs['symbol'] = 'SPY'
            
            # Try to generate signal
            try:
                signal = strategy.next(historical_data)
                
                print(f"\nDate: {date}")
                print(f"  RSI: {row['rsi']:.2f}")
                print(f"  Close: {row['close']:.2f}")
                print(f"  Lower Band: {row['lower_band']:.2f}")
                print(f"  Volume Ratio: {row['volume_ratio']:.2f}")
                print(f"  Signal: {signal}")
                
                # Check entry conditions manually
                entry_signal = (
                    row['rsi'] < 30 and
                    row['close'] < row['lower_band']
                )
                print(f"  Manual entry check: {entry_signal}")
                
                # Check if there's an existing position
                if hasattr(strategy, 'positions') and strategy.positions:
                    print(f"  Current positions: {list(strategy.positions.keys())}")
                
            except Exception as e:
                print(f"  Error generating signal: {e}")
                import traceback
                traceback.print_exc()
    
    # Test with even more relaxed parameters
    print("\n" + "="*60)
    print("Testing with very relaxed parameters...")
    
    relaxed_config = config.copy()
    relaxed_config.update({
        'rsi_oversold': 50,  # Very high threshold
        'atr_band_mult': 0.5,  # Very tight bands
    })
    
    relaxed_strategy = MeanReversionEquity(relaxed_config)
    relaxed_strategy.init()
    
    # Recalculate indicators with relaxed parameters
    df_relaxed = relaxed_strategy.calculate_indicators(test_data)
    
    oversold_relaxed = df_relaxed[df_relaxed['rsi'] < 50]
    print(f"\nDays with RSI < 50: {len(oversold_relaxed)}")
    
    below_band_relaxed = df_relaxed[df_relaxed['close'] < df_relaxed['lower_band']]
    print(f"Days with close < lower_band: {len(below_band_relaxed)}")
    
    # Find days meeting both conditions
    both_conditions = df_relaxed[(df_relaxed['rsi'] < 50) & (df_relaxed['close'] < df_relaxed['lower_band'])]
    print(f"Days meeting both conditions: {len(both_conditions)}")
    
    if len(both_conditions) > 0:
        print("\nSample days meeting both conditions:")
        print(both_conditions[['close', 'rsi', 'lower_band', 'volume_ratio']].head())

if __name__ == "__main__":
    debug_mean_reversion_signals()