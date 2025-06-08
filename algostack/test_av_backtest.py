#!/usr/bin/env python3
"""Test Alpha Vantage integration with pandas dashboard."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager

def test_av_backtest():
    """Test backtest with Alpha Vantage data."""
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    if not data_manager.av_fetcher:
        print("Alpha Vantage not available, using Yahoo Finance")
        data = data_manager.fetch_data("SPY", "1mo", "1d", "yfinance")
    else:
        print("Testing Alpha Vantage data fetch...")
        # Test intraday data
        data = data_manager.fetch_data("SPY", "1mo", "5m", "alpha_vantage")
        
        if data.empty:
            print("No intraday data, trying daily...")
            data = data_manager.fetch_data("SPY", "1mo", "1d", "alpha_vantage")
    
    print(f"\nData shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    if data.empty:
        print("No data fetched!")
        return
    
    # Test with Intraday ORB strategy for intraday data
    if '5m' in str(data.index[1] - data.index[0]):
        strategy_name = "IntradayORB"
    else:
        strategy_name = "MeanReversionEquity"
    
    print(f"\nTesting {strategy_name} strategy")
    strategy_class = strategy_manager.strategies[strategy_name]
    
    user_params = {
        'symbol': 'SPY',
        'lookback_period': 20,
        'position_size': 0.95
    }
    
    results = strategy_manager.run_backtest(
        strategy_class,
        strategy_name,
        user_params,
        data,
        initial_capital=100000
    )
    
    if 'error' in results:
        print(f"\nBacktest error: {results['error']}")
    else:
        print(f"\nBacktest completed!")
        print(f"Total return: {results.get('total_return', 0):.2f}%")
        print(f"Number of trades: {results.get('num_trades', 0)}")

if __name__ == "__main__":
    test_av_backtest()