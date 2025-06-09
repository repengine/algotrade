#!/usr/bin/env python3
"""Debug the futures momentum strategy to see why no trades are generated."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from strategies.futures_momentum import FuturesMomentum

# Fetch data
print("Fetching SPY data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

spy = yf.Ticker('SPY')
data = spy.history(start=start_date, end=end_date, interval='5m')
data.columns = data.columns.str.lower()

print(f"Loaded {len(data)} bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Create strategy
config = {
    'symbols': ['SPY'],
    'lookback_period': 20,
    'breakout_threshold': 0.5,  # 0.5% breakout
    'rsi_period': 14,
    'rsi_threshold': 60,
    'atr_period': 14,
    'stop_loss_atr': 2.0,
    'profit_target_atr': 3.0,
    'volume_multiplier': 1.2,
    'position_size': 0.95
}

strategy = FuturesMomentum(config)
strategy.init()

# Test for signals
print("\nLooking for trading opportunities...")
signals_found = 0

for i in range(50, min(200, len(data))):  # Check first 200 bars after warmup
    window = data.iloc[:i+1].copy()
    window.attrs['symbol'] = 'SPY'
    
    # Get current values
    current_price = window['close'].iloc[-1]
    high_20 = window['high'].iloc[-20:].max()
    breakout_level = high_20 * (1 + config['breakout_threshold'] / 100)
    
    # Calculate RSI manually
    delta = window['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # Volume
    volume_avg = window['volume'].rolling(20).mean().iloc[-1]
    current_volume = window['volume'].iloc[-1]
    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 0
    
    # Check signal
    signal = strategy.next(window)
    
    if i % 100 == 0 or signal:
        print(f"\nBar {i} ({window.index[-1]}):")
        print(f"  Price: ${current_price:.2f}")
        print(f"  20-bar High: ${high_20:.2f}")
        print(f"  Breakout Level: ${breakout_level:.2f}")
        print(f"  Above Breakout: {current_price > breakout_level}")
        print(f"  RSI: {current_rsi:.1f} (threshold: {config['rsi_threshold']})")
        print(f"  Volume Ratio: {volume_ratio:.2f} (threshold: {config['volume_multiplier']})")
        
    if signal:
        signals_found += 1
        print(f"  *** SIGNAL: {signal.direction} - {signal.reason}")

print(f"\nTotal signals found: {signals_found}")

# Check why no signals
if signals_found == 0:
    print("\nDiagnosing why no signals...")
    
    # Check price breakouts
    breakouts = 0
    for i in range(20, len(data)):
        high_20 = data['high'].iloc[i-20:i].max()
        breakout_level = high_20 * 1.005  # 0.5% breakout
        if data['close'].iloc[i] > breakout_level:
            breakouts += 1
    
    print(f"Price breakouts found: {breakouts} out of {len(data)-20} bars ({breakouts/(len(data)-20)*100:.1f}%)")
    
    # Lower thresholds to see what would work
    print("\nTesting with lower thresholds...")
    for threshold in [0.3, 0.2, 0.1, 0.05]:
        breakouts = 0
        for i in range(20, len(data)):
            high_20 = data['high'].iloc[i-20:i].max()
            breakout_level = high_20 * (1 + threshold/100)
            if data['close'].iloc[i] > breakout_level:
                breakouts += 1
        print(f"  {threshold}% threshold: {breakouts} breakouts ({breakouts/(len(data)-20)*100:.1f}%)")