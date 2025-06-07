#!/usr/bin/env python3
"""
Simple test to verify strategy returns calculation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def simple_ma_strategy():
    """Test a simple MA crossover strategy"""
    
    # Fetch data
    print("Fetching SPY data for testing...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    print(f"Data fetched: {len(data)} days")
    
    # Create signals dataframe
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    signals['position'] = 0.0
    
    # Calculate MAs
    fast_ma = data['Close'].rolling(window=10).mean()
    slow_ma = data['Close'].rolling(window=30).mean()
    
    # Detect crossovers
    ma_diff = fast_ma - slow_ma
    ma_diff_prev = ma_diff.shift(1)
    
    bullish_cross = (ma_diff > 0) & (ma_diff_prev <= 0)
    bearish_cross = (ma_diff < 0) & (ma_diff_prev >= 0)
    
    print(f"\nCrossovers detected:")
    print(f"Bullish: {bullish_cross.sum()}")
    print(f"Bearish: {bearish_cross.sum()}")
    
    # Set signals
    signals.loc[bullish_cross, 'signal'] = 1
    signals.loc[bearish_cross, 'signal'] = -1
    
    # Forward fill positions
    signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Calculate returns
    signals['returns'] = data['Close'].pct_change().fillna(0)
    signals['strategy_returns'] = signals['position'].shift(1).fillna(0) * signals['returns']
    
    # Calculate cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    signals['buy_hold_returns'] = (1 + signals['returns']).cumprod()
    
    # Make sure first value is 1.0
    if len(signals) > 0:
        signals.iloc[0, signals.columns.get_loc('cumulative_returns')] = 1.0
        signals.iloc[0, signals.columns.get_loc('buy_hold_returns')] = 1.0
    
    # Calculate final returns
    strategy_final = (signals['cumulative_returns'].iloc[-1] - 1) * 100
    buy_hold_final = (signals['buy_hold_returns'].iloc[-1] - 1) * 100
    
    print(f"\nResults:")
    print(f"Strategy return: {strategy_final:.2f}%")
    print(f"Buy & Hold return: {buy_hold_final:.2f}%")
    
    # Show some trades
    trades = signals[signals['signal'] != 0]
    print(f"\nTrades executed: {len(trades)}")
    print("\nFirst 5 trades:")
    print(trades[['price', 'signal', 'position']].head())
    
    # Check if cumulative returns are changing
    print(f"\nCumulative returns over time:")
    sample_dates = [0, 50, 100, 150, 200, -1]
    for i in sample_dates:
        if 0 <= i < len(signals) or i == -1:
            date = signals.index[i]
            cum_ret = signals['cumulative_returns'].iloc[i]
            print(f"{date.date()}: {(cum_ret - 1) * 100:.2f}%")
    
    return signals

if __name__ == "__main__":
    signals = simple_ma_strategy()