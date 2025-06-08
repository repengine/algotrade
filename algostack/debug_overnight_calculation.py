#!/usr/bin/env python3
"""Debug the overnight return calculation for Alpha Vantage data."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

import pandas as pd
import numpy as np
from dashboard_pandas import AlphaVantageDataManager
from strategy_integration_helpers import DataFormatConverter

def analyze_overnight_returns():
    """Analyze overnight returns for both data sources."""
    
    data_manager = AlphaVantageDataManager()
    converter = DataFormatConverter()
    
    for data_source in ["alpha_vantage", "yfinance"]:
        print(f"\n{'='*60}")
        print(f"Analyzing {data_source} overnight returns")
        print(f"{'='*60}")
        
        # Get data
        data = data_manager.fetch_data("SPY", "3mo", "1d", data_source)
        strategy_data = converter.dashboard_to_strategy(data, symbol='SPY')
        
        # Calculate overnight returns manually
        overnight_returns = []
        for i in range(1, len(strategy_data)):
            # Overnight return = Open[i] / Close[i-1] - 1
            if strategy_data['close'].iloc[i-1] > 0:
                overnight_ret = (strategy_data['open'].iloc[i] / strategy_data['close'].iloc[i-1]) - 1
                overnight_returns.append({
                    'date': strategy_data.index[i],
                    'prev_close': strategy_data['close'].iloc[i-1],
                    'open': strategy_data['open'].iloc[i],
                    'overnight_return': overnight_ret,
                    'day_of_week': strategy_data.index[i].strftime('%A')
                })
        
        overnight_df = pd.DataFrame(overnight_returns)
        
        print(f"\nTotal overnight periods: {len(overnight_df)}")
        print(f"Average overnight return: {overnight_df['overnight_return'].mean():.6f}")
        print(f"Annualized overnight return: {overnight_df['overnight_return'].mean() * 252:.4f}")
        
        # By day of week
        print("\nOvernight returns by day of week:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            day_returns = overnight_df[overnight_df['day_of_week'] == day]['overnight_return']
            if len(day_returns) > 0:
                print(f"  {day}: {day_returns.mean():.6f} (n={len(day_returns)})")
        
        # Check for outliers
        print(f"\nReturn statistics:")
        print(f"  Min: {overnight_df['overnight_return'].min():.6f}")
        print(f"  Max: {overnight_df['overnight_return'].max():.6f}")
        print(f"  Std: {overnight_df['overnight_return'].std():.6f}")
        
        # Show some extreme returns
        print("\nTop 5 positive overnight returns:")
        top_returns = overnight_df.nlargest(5, 'overnight_return')
        for _, row in top_returns.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')} ({row['day_of_week']}): {row['overnight_return']:.4f}")
        
        print("\nBottom 5 negative overnight returns:")
        bottom_returns = overnight_df.nsmallest(5, 'overnight_return')
        for _, row in bottom_returns.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')} ({row['day_of_week']}): {row['overnight_return']:.4f}")

def check_data_alignment():
    """Check if open/close data is properly aligned."""
    
    print("\n" + "="*60)
    print("Checking data alignment")
    print("="*60)
    
    data_manager = AlphaVantageDataManager()
    
    # Get a small sample
    av_data = data_manager.fetch_data("SPY", "1mo", "1d", "alpha_vantage")
    yf_data = data_manager.fetch_data("SPY", "1mo", "1d", "yfinance")
    
    print("\nAlpha Vantage last 5 days:")
    print(av_data[['open', 'close']].tail())
    
    print("\nYahoo Finance last 5 days:")
    close_col = 'Close' if 'Close' in yf_data.columns else 'close'
    open_col = 'Open' if 'Open' in yf_data.columns else 'open'
    print(yf_data[[open_col, close_col]].tail())

if __name__ == "__main__":
    analyze_overnight_returns()
    check_data_alignment()