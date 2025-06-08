#!/usr/bin/env python3
"""Test OvernightDrift fix with Alpha Vantage data."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager

def test_overnight_drift_fix():
    """Test OvernightDrift with fixed parameters."""
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    # Test with both data sources
    for data_source in ["alpha_vantage", "yfinance"]:
        print(f"\n{'='*60}")
        print(f"Testing OvernightDrift with {data_source}")
        print(f"{'='*60}")
        
        # Fetch data
        data = data_manager.fetch_data("SPY", "1y", "1d", data_source)
        
        # Test with permissive parameters
        user_params = {
            'symbol': 'SPY',
            'position_size': 0.95,
            'lookback_period': 60,
            'volume_filter': False,      # Disable volume filter
            'trend_filter': False,       # Disable trend filter
            'volatility_filter': False,  # Disable volatility filter
            # Add more overrides
            'min_edge': 0.0001,         # Very low edge threshold
            'vix_threshold': 100        # Very high VIX threshold
        }
        
        # Run backtest
        results = strategy_manager.run_backtest(
            strategy_manager.strategies['OvernightDrift'],
            'OvernightDrift',
            user_params,
            data,
            initial_capital=100000
        )
        
        # Print results
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Total return: {results.get('total_return', 0):.2f}%")
            print(f"Number of trades: {results.get('num_trades', 0)}")
            
            signals_df = results.get('signals', pd.DataFrame())
            if not signals_df.empty:
                print(f"Signals generated: {len(signals_df)}")
                
                # Show some signals
                print("\nFirst 5 signals:")
                for idx, row in signals_df.head().iterrows():
                    print(f"  {row['timestamp']}: {row['direction']}")
            else:
                print("No signals generated!")
                
                # Debug the actual edge calculation
                print("\nDebugging edge calculation...")
                
                from strategy_integration_helpers import DataFormatConverter
                from strategies.overnight_drift import OvernightDrift
                from strategy_defaults import merge_with_defaults
                
                converter = DataFormatConverter()
                strategy_data = converter.dashboard_to_strategy(data, symbol='SPY')
                
                # Get full config
                full_config = merge_with_defaults('OvernightDrift', user_params)
                
                # Override the hardcoded values
                full_config['volume_threshold'] = 0.0
                full_config['min_atr'] = 0.0
                full_config['max_atr'] = 1.0
                full_config['trend_filter'] = False
                
                strategy = OvernightDrift(full_config)
                strategy.init()
                
                # Test with a chunk of data
                test_data = strategy_data.iloc[-100:].copy()
                test_data.attrs['symbol'] = 'SPY'
                
                # Calculate indicators
                df_with_indicators = strategy.calculate_indicators(test_data)
                
                # Check the last row's values
                latest = df_with_indicators.iloc[-1]
                print(f"\nLatest data point:")
                print(f"  Date: {df_with_indicators.index[-1]}")
                print(f"  Day: {df_with_indicators.index[-1].strftime('%A')}")
                print(f"  Close: {latest['close']:.2f}")
                print(f"  ATR: {latest['atr']:.4f}")
                print(f"  Volume ratio: {latest['volume_ratio']:.2f}")
                print(f"  SMA: {latest['sma']:.2f}")
                print(f"  Above SMA: {latest['close'] > latest['sma']}")
                
                # Calculate overnight edge
                edge = strategy.calculate_overnight_edge(test_data)
                print(f"\nOvernight edge: {edge:.4f}")
                print(f"Edge > 0.02: {edge > 0.02}")

if __name__ == "__main__":
    test_overnight_drift_fix()