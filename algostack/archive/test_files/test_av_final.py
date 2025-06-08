#!/usr/bin/env python3
"""Final test of Alpha Vantage integration with all fixes."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

from dashboard_pandas import PandasStrategyManager, AlphaVantageDataManager

def test_av_final():
    """Test strategies with Alpha Vantage data after all fixes."""
    
    # Set API key
    os.environ['ALPHA_VANTAGE_API_KEY'] = '991AR2LC298IGMX7'
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    # Test MeanReversionEquity with relaxed parameters
    print("Testing MeanReversionEquity with Alpha Vantage data")
    print("="*60)
    
    # Fetch data
    data = data_manager.fetch_data("SPY", "1y", "1d", "alpha_vantage")
    
    if data.empty:
        print("Failed to fetch data")
        return
        
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Run backtest with filters disabled
    user_params = {
        'symbol': 'SPY',
        'position_size': 0.95,
        'volume_filter': False,  # Disable volume filter
        'rsi_oversold': 40,  # More relaxed threshold
        'atr_band_mult': 1.5,  # Tighter bands
    }
    
    results = strategy_manager.run_backtest(
        strategy_manager.strategies['MeanReversionEquity'],
        'MeanReversionEquity',
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
        print(f"   Max drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"   Win rate: {results.get('win_rate', 0):.1f}%")
        
        # Show some trades if any
        if 'trades' in results and not results['trades'].empty:
            print(f"\nFirst 5 trades:")
            print(results['trades'][['entry_time', 'exit_time', 'direction', 'pnl', 'return']].head())
        
        # Check signal count
        signals_df = results.get('signals', pd.DataFrame())
        if not signals_df.empty:
            print(f"\nTotal signals generated: {len(signals_df)}")
        else:
            print("\n⚠️  No signals generated")
    
    # Test with Yahoo Finance for comparison
    print("\n" + "="*60)
    print("Testing same strategy with Yahoo Finance data for comparison")
    print("="*60)
    
    yf_data = data_manager.fetch_data("SPY", "1y", "1d", "yfinance")
    
    if not yf_data.empty:
        yf_results = strategy_manager.run_backtest(
            strategy_manager.strategies['MeanReversionEquity'],
            'MeanReversionEquity',
            user_params,
            yf_data,
            initial_capital=100000
        )
        
        if 'error' not in yf_results:
            print(f"Yahoo Finance results:")
            print(f"   Total return: {yf_results.get('total_return', 0):.2f}%")
            print(f"   Number of trades: {yf_results.get('num_trades', 0)}")
            print(f"   Sharpe ratio: {yf_results.get('sharpe_ratio', 0):.2f}")

if __name__ == "__main__":
    import pandas as pd
    test_av_final()