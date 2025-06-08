#!/usr/bin/env python3
"""Test strategy signal generation with proper configuration."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager
from strategy_defaults import get_strategy_defaults

def test_all_strategies():
    """Test all strategies with proper configuration."""
    
    # Get data
    data_manager = AlphaVantageDataManager()
    
    # Use Yahoo Finance for consistent daily data
    print("Fetching daily data from Yahoo Finance...")
    data = data_manager.fetch_data("SPY", "1y", "1d", "yfinance")
    
    print(f"Data shape: {data.shape}")
    print(f"Data date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize strategy manager
    manager = PandasStrategyManager()
    
    # Test each strategy
    for strategy_name, strategy_class in manager.strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing {strategy_name}")
        print(f"{'='*60}")
        
        # Get appropriate config for each strategy
        if strategy_name == "IntradayORB":
            # IntradayORB needs intraday data
            print("  Fetching intraday data for IntradayORB...")
            intraday_data = data_manager.fetch_data("SPY", "1mo", "15m", "alpha_vantage")
            if intraday_data.empty:
                print("  No intraday data available, skipping IntradayORB")
                continue
            test_data = intraday_data
        else:
            test_data = data
        
        # Adjust parameters based on data length
        data_length = len(test_data)
        
        # Get defaults and adjust lookback if needed
        defaults = get_strategy_defaults(strategy_name)
        user_params = {
            'symbol': 'SPY',
            'position_size': 0.95
        }
        
        # Adjust lookback period if it's too large
        default_lookback = defaults.get('lookback_period', 50)
        if isinstance(default_lookback, dict):
            default_lookback = max(default_lookback.values())
        
        if default_lookback >= data_length * 0.8:
            # Use at most 20% of data for lookback
            user_params['lookback_period'] = int(data_length * 0.2)
            print(f"  Adjusted lookback from {default_lookback} to {user_params['lookback_period']}")
        
        # Run backtest
        results = manager.run_backtest(
            strategy_class,
            strategy_name,
            user_params,
            test_data,
            initial_capital=100000
        )
        
        # Report results
        if 'error' in results:
            print(f"  ❌ Error: {results['error']}")
        else:
            print(f"  ✅ Success!")
            print(f"     Total return: {results.get('total_return', 0):.2f}%")
            print(f"     Number of trades: {results.get('num_trades', 0)}")
            print(f"     Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            
            # Check signals
            signals_df = results.get('signals', pd.DataFrame())
            if not signals_df.empty:
                print(f"     Signals generated: {len(signals_df)}")
                # Show signal distribution
                if 'direction' in signals_df.columns:
                    signal_counts = signals_df['direction'].value_counts()
                    print(f"     Signal breakdown: {signal_counts.to_dict()}")
            else:
                print(f"     ⚠️  No signals generated")

if __name__ == "__main__":
    test_all_strategies()