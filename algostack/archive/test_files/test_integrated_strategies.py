#!/usr/bin/env python3
"""
Test script to verify integrated strategies work properly
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Import mock system
from test_files import mock_imports

# Import our integration components
from dashboard_integrated_strategies import StrategyLoader, StrategyBridge


def test_strategy_discovery():
    """Test that we can discover strategies."""
    print("Testing Strategy Discovery")
    print("=" * 50)
    
    loader = StrategyLoader()
    strategies = loader.discover_strategies()
    
    print(f"Found {len(strategies)} strategies:")
    for name, cls in strategies.items():
        print(f"  - {name}: {cls.__name__}")
    
    return strategies


def test_strategy_parameters(strategies):
    """Test parameter extraction."""
    print("\nTesting Parameter Extraction")
    print("=" * 50)
    
    loader = StrategyLoader()
    
    for name, cls in strategies.items():
        print(f"\n{name} parameters:")
        params = loader.get_strategy_parameters(cls)
        for param, value in params.items():
            print(f"  - {param}: {value} ({type(value).__name__})")


def test_strategy_execution(strategy_name, strategy_class):
    """Test a single strategy execution."""
    print(f"\nTesting {strategy_name} Execution")
    print("=" * 50)
    
    # Fetch test data
    symbol = 'SPY'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print(f"Fetching {symbol} data...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        print("ERROR: No data fetched")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Get default parameters
    loader = StrategyLoader()
    params = loader.get_strategy_parameters(strategy_class)
    params['symbol'] = symbol
    
    # Create bridge and run backtest
    bridge = StrategyBridge(strategy_class, params)
    
    print("\nRunning backtest...")
    signals = bridge.run_backtest(data)
    
    # Analyze results
    total_signals = (signals['signal'] != 0).sum()
    buy_signals = (signals['signal'] == 1).sum()
    sell_signals = (signals['signal'] == -1).sum()
    
    print(f"\nSignal Summary:")
    print(f"  Total signals: {total_signals}")
    print(f"  Buy signals: {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    
    # Calculate simple returns
    price_returns = data['Close'].pct_change()
    strategy_returns = signals['position'].shift(1) * price_returns
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    # Buy and hold
    buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    
    print(f"\nPerformance:")
    print(f"  Strategy return: {total_return:.2f}%")
    print(f"  Buy & hold return: {buy_hold_return:.2f}%")
    print(f"  Excess return: {total_return - buy_hold_return:.2f}%")
    
    # Show some signal examples
    if total_signals > 0:
        print(f"\nFirst few signals:")
        signal_dates = signals[signals['signal'] != 0].head(5)
        for date, row in signal_dates.iterrows():
            signal_type = 'BUY' if row['signal'] == 1 else 'SELL'
            price = data.loc[date, 'Close']
            print(f"  {date.date()}: {signal_type} at ${price:.2f}")


def main():
    """Run all tests."""
    print("🚀 Testing AlgoStack Strategy Integration")
    print("=" * 70)
    
    # Test discovery
    strategies = test_strategy_discovery()
    
    if not strategies:
        print("\nERROR: No strategies found!")
        return
    
    # Test parameters
    test_strategy_parameters(strategies)
    
    # Test execution for each strategy
    print("\n" + "=" * 70)
    print("TESTING STRATEGY EXECUTION")
    print("=" * 70)
    
    for name, cls in strategies.items():
        try:
            test_strategy_execution(name, cls)
        except Exception as e:
            print(f"\nERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Integration tests complete!")


if __name__ == "__main__":
    main()