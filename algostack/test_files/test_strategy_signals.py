#!/usr/bin/env python3
"""
Test script to debug strategy signal generation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def test_signals():
    # Fetch some data
    print("Fetching SPY data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    
    # Test 1: RSI Strategy
    print("\n" + "="*50)
    print("Testing RSI Strategy")
    print("="*50)
    
    period = 14
    oversold = 30
    overbought = 70
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    print(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
    print(f"RSI current: {rsi.iloc[-1]:.2f}")
    
    # Check RSI values
    oversold_days = (rsi < oversold).sum()
    overbought_days = (rsi > overbought).sum()
    print(f"Days with RSI < {oversold}: {oversold_days}")
    print(f"Days with RSI > {overbought}: {overbought_days}")
    
    # Generate signals with crossovers
    buy_signal = (rsi < oversold) & (rsi.shift(1) >= oversold)
    sell_signal = (rsi > overbought) & (rsi.shift(1) <= overbought)
    
    print(f"Buy signals (RSI crosses below {oversold}): {buy_signal.sum()}")
    print(f"Sell signals (RSI crosses above {overbought}): {sell_signal.sum()}")
    
    # Show some examples
    if buy_signal.sum() > 0:
        print("\nFirst buy signal:")
        buy_dates = data[buy_signal].index
        print(f"Date: {buy_dates[0]}, Price: ${data.loc[buy_dates[0], 'Close']:.2f}, RSI: {rsi.loc[buy_dates[0]]:.2f}")
    
    # Test 2: MA Crossover Strategy
    print("\n" + "="*50)
    print("Testing MA Crossover Strategy")
    print("="*50)
    
    fast_period = 10
    slow_period = 30
    
    fast_ma = data['Close'].rolling(window=fast_period).mean()
    slow_ma = data['Close'].rolling(window=slow_period).mean()
    
    print(f"Fast MA current: ${fast_ma.iloc[-1]:.2f}")
    print(f"Slow MA current: ${slow_ma.iloc[-1]:.2f}")
    
    # Detect crossovers
    ma_diff = fast_ma - slow_ma
    ma_diff_prev = ma_diff.shift(1)
    
    bullish_cross = (ma_diff > 0) & (ma_diff_prev <= 0)
    bearish_cross = (ma_diff < 0) & (ma_diff_prev >= 0)
    
    print(f"Bullish crossovers: {bullish_cross.sum()}")
    print(f"Bearish crossovers: {bearish_cross.sum()}")
    
    # Show some examples
    if bullish_cross.sum() > 0:
        print("\nFirst bullish crossover:")
        bull_dates = data[bullish_cross].index
        print(f"Date: {bull_dates[0]}, Price: ${data.loc[bull_dates[0], 'Close']:.2f}")
        print(f"Fast MA: ${fast_ma.loc[bull_dates[0]]:.2f}, Slow MA: ${slow_ma.loc[bull_dates[0]]:.2f}")
    
    # Test 3: Position tracking
    print("\n" + "="*50)
    print("Testing Position Tracking")
    print("="*50)
    
    # Create a simple signal series
    signals = pd.Series(0, index=data.index)
    
    # Add some MA crossover signals
    signals[bullish_cross] = 1
    signals[bearish_cross] = -1
    
    # Forward fill positions
    positions = signals.replace(0, np.nan).ffill().fillna(0)
    
    # Count position changes
    position_changes = (positions != positions.shift(1)).sum()
    long_periods = (positions == 1).sum()
    short_periods = (positions == -1).sum()
    neutral_periods = (positions == 0).sum()
    
    print(f"Position changes: {position_changes}")
    print(f"Days long: {long_periods}")
    print(f"Days short: {short_periods}")
    print(f"Days neutral: {neutral_periods}")
    
    # Calculate simple returns
    returns = data['Close'].pct_change()
    strategy_returns = positions.shift(1) * returns
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    final_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    print(f"\nFinal strategy return: {final_return:.2f}%")
    
    # Buy and hold comparison
    buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    print(f"Buy and hold return: {buy_hold_return:.2f}%")
    
    # Show last few position changes
    print("\nLast few position changes:")
    changes = positions[positions != positions.shift(1)]
    if len(changes) > 0:
        for date, pos in changes.tail(5).items():
            price = data.loc[date, 'Close']
            print(f"{date.date()}: Position = {pos:.0f}, Price = ${price:.2f}")

if __name__ == "__main__":
    test_signals()