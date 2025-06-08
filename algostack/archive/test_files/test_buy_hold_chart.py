#!/usr/bin/env python3
"""Test buy-and-hold comparison chart."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()

from dashboard_pandas import AlphaVantageDataManager, PandasStrategyManager, create_performance_chart
import pandas as pd
import plotly.io as pio

def test_buy_hold_chart():
    """Test the buy-and-hold comparison feature."""
    
    os.environ['ALPHA_VANTAGE_API_KEY'] = '991AR2LC298IGMX7'
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    # Fetch data
    print("Fetching data...")
    data = data_manager.fetch_data("SPY", "6mo", "1d", "yfinance")
    
    if data.empty:
        print("Failed to fetch data")
        return
        
    print(f"Data shape: {data.shape}")
    
    # Run a simple backtest
    print("\nRunning backtest...")
    config = {
        'symbol': 'SPY',
        'position_size': 0.95,
        'volume_filter': False,
        'rsi_oversold': 40,
        'atr_band_mult': 1.5,
    }
    
    results = strategy_manager.run_backtest(
        strategy_manager.strategies['MeanReversionEquity'],
        'MeanReversionEquity',
        config,
        data,
        initial_capital=100000
    )
    
    if 'error' in results:
        print(f"Backtest error: {results['error']}")
        return
    
    print(f"Backtest results:")
    print(f"  Strategy return: {results.get('total_return', 0):.2f}%")
    print(f"  Number of trades: {results.get('num_trades', 0)}")
    
    # Create chart
    print("\nCreating performance chart...")
    fig = create_performance_chart(results, data, 100000)
    
    # Save chart to HTML
    output_file = "buy_hold_comparison.html"
    pio.write_html(fig, output_file)
    print(f"Chart saved to {output_file}")
    
    # Display some statistics
    equity_curve = results['equity_curve']
    close_col = 'Close' if 'Close' in data.columns else 'close'
    
    # Calculate buy-and-hold return
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    
    # Handle timezone
    if hasattr(data.index, 'tz') and data.index.tz is not None and hasattr(start_date, 'tz') and start_date.tz is None:
        start_date = start_date.tz_localize(data.index.tz)
        end_date = end_date.tz_localize(data.index.tz)
    
    mask = (data.index >= start_date) & (data.index <= end_date)
    backtest_data = data[mask]
    
    if len(backtest_data) > 0:
        initial_price = backtest_data[close_col].iloc[0]
        final_price = backtest_data[close_col].iloc[-1]
        buy_hold_return = (final_price / initial_price - 1) * 100
        
        print(f"\nBuy & Hold Statistics:")
        print(f"  Initial price: ${initial_price:.2f}")
        print(f"  Final price: ${final_price:.2f}")
        print(f"  Buy & Hold return: {buy_hold_return:.2f}%")
        print(f"  Strategy outperformance: {results.get('total_return', 0) - buy_hold_return:.2f}%")

if __name__ == "__main__":
    test_buy_hold_chart()