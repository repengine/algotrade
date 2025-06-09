#!/usr/bin/env python3
"""Analyze recent market conditions to understand why no breakouts."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

spy = yf.Ticker('SPY')
data = spy.history(start=start_date, end=end_date, interval='5m')
data.columns = data.columns.str.lower()

print(f"Analyzing {len(data)} 5-minute bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Calculate various metrics
data['returns'] = data['close'].pct_change()
data['high_20'] = data['high'].rolling(20).max()
data['low_20'] = data['low'].rolling(20).min()
data['range_pct'] = (data['high_20'] - data['low_20']) / data['low_20'] * 100

# Volatility
data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(78)  # Annualized

# Price position within range
data['position_in_range'] = (data['close'] - data['low_20']) / (data['high_20'] - data['low_20'])

# Volume analysis
data['volume_ma'] = data['volume'].rolling(20).mean()
data['volume_ratio'] = data['volume'] / data['volume_ma']

print("\n" + "="*60)
print("MARKET ANALYSIS")
print("="*60)

# Overall statistics
print(f"\nPrice Statistics:")
print(f"  Start Price: ${data['close'].iloc[0]:.2f}")
print(f"  End Price: ${data['close'].iloc[-1]:.2f}")
print(f"  Total Return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.2f}%")
print(f"  Average Daily Range: {data['range_pct'].mean():.2f}%")
print(f"  Max Daily Range: {data['range_pct'].max():.2f}%")

print(f"\nVolatility:")
print(f"  Average Volatility: {data['volatility'].mean() * 100:.1f}%")
print(f"  Current Volatility: {data['volatility'].iloc[-1] * 100:.1f}%")

# Breakout analysis
breakouts_up = 0
breakouts_down = 0
mean_reversions = 0

for i in range(20, len(data)):
    # Check for breakouts
    if data['close'].iloc[i] > data['high_20'].iloc[i-1] * 1.001:  # 0.1% breakout
        breakouts_up += 1
    elif data['close'].iloc[i] < data['low_20'].iloc[i-1] * 0.999:  # 0.1% breakdown
        breakouts_down += 1
    
    # Check for mean reversion opportunities
    if data['position_in_range'].iloc[i] < 0.2:  # Bottom 20% of range
        mean_reversions += 1

print(f"\nMarket Behavior (last 30 days):")
print(f"  Upward Breakouts: {breakouts_up} ({breakouts_up/len(data)*100:.1f}%)")
print(f"  Downward Breakouts: {breakouts_down} ({breakouts_down/len(data)*100:.1f}%)")
print(f"  Mean Reversion Setups: {mean_reversions} ({mean_reversions/len(data)*100:.1f}%)")

# Suggest better strategy
print("\n" + "="*60)
print("RECOMMENDED STRATEGY APPROACH")
print("="*60)

if mean_reversions > breakouts_up + breakouts_down:
    print("\n✅ Mean Reversion is more suitable for current market conditions")
    print("   - Market is range-bound with frequent pullbacks")
    print("   - Buy when price touches lower Bollinger Band")
    print("   - Sell when price returns to moving average")
elif data['volatility'].mean() > 0.15:
    print("\n✅ Volatility-based strategies are recommended")
    print("   - High volatility creates opportunities")
    print("   - Trade volatility expansions and contractions")
    print("   - Use tighter stops and quicker profits")
else:
    print("\n✅ Scalping small moves is recommended")
    print("   - Low volatility requires capturing small moves")
    print("   - Trade micro patterns and order flow")
    print("   - Focus on high-volume periods")

# Daily pattern analysis
data['hour'] = data.index.hour
hourly_returns = data.groupby('hour')['returns'].agg(['mean', 'std', 'count'])
hourly_returns['abs_mean'] = np.abs(hourly_returns['mean'])

print(f"\n📊 Best Trading Hours (by average movement):")
best_hours = hourly_returns.nlargest(5, 'abs_mean')
for hour, row in best_hours.iterrows():
    print(f"   {hour}:00 - Avg Move: {row['abs_mean']*100:.3f}%, Volatility: {row['std']*100:.3f}%")