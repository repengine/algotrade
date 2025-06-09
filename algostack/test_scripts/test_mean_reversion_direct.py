#!/usr/bin/env python3
"""Direct test of mean reversion intraday strategy."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import the strategy
from strategies.mean_reversion_intraday import MeanReversionIntraday
from dashboard_pandas import PandasStrategyManager, DataFormatConverter
from strategy_integration_helpers import RiskContextMock

def main():
    print("Testing Mean Reversion Intraday Strategy")
    print("=" * 60)
    
    # Initialize strategy manager
    strategy_manager = PandasStrategyManager()
    converter = DataFormatConverter()
    
    # Check if strategy is loaded
    if 'MeanReversionIntraday' in strategy_manager.strategies:
        print("✓ MeanReversionIntraday strategy found!")
    else:
        print("✗ MeanReversionIntraday strategy NOT found!")
        print("Available strategies:", list(strategy_manager.strategies.keys()))
        return
    
    # Test configuration with optimal parameters
    config = {
        'symbol': 'SPY',
        'lookback_period': 15,
        'zscore_threshold': 1.5,
        'exit_zscore': 0.0,
        'rsi_period': 3,
        'rsi_oversold': 25.0,
        'stop_loss_atr': 2.5,
        'position_size': 0.95
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load data
    print("\nLoading 5-minute data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    spy = yf.Ticker('SPY')
    data = spy.history(start=start_date, end=end_date, interval='5m')
    
    if data.empty:
        print("✗ No data loaded!")
        return
        
    print(f"✓ Loaded {len(data)} bars")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    
    # Run quick backtest
    print("\nRunning backtest...")
    strategy_class = strategy_manager.strategies['MeanReversionIntraday']
    
    try:
        results = strategy_manager.run_backtest(
            strategy_class=strategy_class,
            strategy_name='MeanReversionIntraday',
            user_params=config,
            data=data,
            initial_capital=10000
        )
        
        print("\n✓ Backtest completed!")
        print(f"  Total Return: {results.get('total_return', 0):.2f}%")
        print(f"  Number of Trades: {results.get('num_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1f}%")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        
        # Check if this matches our expected performance
        if results.get('total_return', 0) > 2:
            print("\n✓ Strategy performing as expected!")
        else:
            print("\n⚠ Strategy performance lower than expected")
            print("  This could be due to different market conditions")
            
    except Exception as e:
        print(f"\n✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()