#!/usr/bin/env python3
"""Test backtesting with fixed Alpha Vantage data."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager

def test_av_backtest():
    """Test backtesting with fixed Alpha Vantage data."""
    
    # Set API key
    os.environ['ALPHA_VANTAGE_API_KEY'] = '991AR2LC298IGMX7'
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    # Test different strategies with Alpha Vantage data
    strategies_to_test = ['MeanReversionEquity', 'HybridRegime', 'OvernightDrift']
    
    for strategy_name in strategies_to_test:
        if strategy_name not in strategy_manager.strategies:
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing {strategy_name} with Alpha Vantage data")
        print(f"{'='*60}")
        
        # Fetch data
        data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
        
        if data.empty:
            print("Failed to fetch data")
            continue
            
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {list(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Run backtest
        user_params = {
            'symbol': 'SPY',
            'position_size': 0.95,
            'lookback_period': 50 if strategy_name == 'HybridRegime' else 20
        }
        
        # Special parameters for OvernightDrift
        if strategy_name == 'OvernightDrift':
            user_params.update({
                'volume_filter': False,
                'trend_filter': False,
                'volatility_filter': False,
                'min_edge': -10  # Allow negative edge for testing
            })
        
        results = strategy_manager.run_backtest(
            strategy_manager.strategies[strategy_name],
            strategy_name,
            user_params,
            data,
            initial_capital=100000
        )
        
        # Display results
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Backtest successful!")
            print(f"   Total return: {results.get('total_return', 0):.2f}%")
            print(f"   Number of trades: {results.get('num_trades', 0)}")
            print(f"   Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            
            signals_df = results.get('signals', pd.DataFrame())
            if not signals_df.empty:
                print(f"   Signals generated: {len(signals_df)}")
                if 'direction' in signals_df.columns:
                    signal_counts = signals_df['direction'].value_counts()
                    print(f"   Signal breakdown: {signal_counts.to_dict()}")
            else:
                print("   ⚠️  No signals generated")
    
    # Test intraday data
    print(f"\n{'='*60}")
    print("Testing IntradayORB with Alpha Vantage intraday data")
    print(f"{'='*60}")
    
    intraday_data = data_manager.fetch_data("SPY", "1mo", "5m", "alpha_vantage")
    
    if not intraday_data.empty:
        print(f"Intraday data shape: {intraday_data.shape}")
        print(f"Date range: {intraday_data.index[0]} to {intraday_data.index[-1]}")
        
        results = strategy_manager.run_backtest(
            strategy_manager.strategies.get('IntradayORB'),
            'IntradayORB',
            {'symbol': 'SPY', 'position_size': 0.95},
            intraday_data,
            initial_capital=100000
        )
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Intraday backtest successful!")
            print(f"   Total return: {results.get('total_return', 0):.2f}%")
            print(f"   Number of trades: {results.get('num_trades', 0)}")

if __name__ == "__main__":
    import pandas as pd
    test_av_backtest()