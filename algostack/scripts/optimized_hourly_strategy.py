#!/usr/bin/env python3
"""
Optimized hourly strategy with parameters tuned for intraday trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json

# Hourly-optimized parameters
configs_to_test = [
    {
        "name": "Tight Mean Reversion",
        "lookback_hours": 48,  # 48 hours = ~1 week of trading
        "zscore_threshold": 1.5,  # Tighter threshold for hourly
        "exit_zscore": 0.0,
        "rsi_hours": 14,  # 14 hours
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0
    },
    {
        "name": "Very Tight Mean Reversion", 
        "lookback_hours": 24,  # 24 hours = ~3 trading days
        "zscore_threshold": 1.0,  # Very tight
        "exit_zscore": -0.25,
        "rsi_hours": 8,  # 8 hours = 1 trading day
        "rsi_oversold": 35.0,
        "rsi_overbought": 65.0
    },
    {
        "name": "Adaptive Mean Reversion",
        "lookback_hours": 36,
        "zscore_threshold": 1.25,
        "exit_zscore": 0.0,
        "rsi_hours": 10,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0
    }
]

# Download hourly data
print("Downloading hourly SPY data...")
spy = yf.Ticker('SPY')
hourly_data = spy.history(start='2025-01-01', end='2025-06-08', interval='1h')
hourly_data.columns = hourly_data.columns.str.lower()

print(f"Loaded {len(hourly_data)} hourly bars")
print(f"Date range: {hourly_data.index[0]} to {hourly_data.index[-1]}")

# Function to calculate RSI
def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Test each configuration
best_config = None
best_return = -np.inf

for config in configs_to_test:
    print(f"\nTesting {config['name']}...")
    
    # Calculate indicators
    prices = hourly_data['close']
    returns = prices.pct_change()
    
    # Z-score
    rolling_mean = prices.rolling(window=config['lookback_hours']).mean()
    rolling_std = prices.rolling(window=config['lookback_hours']).std()
    zscore = (prices - rolling_mean) / rolling_std
    
    # RSI
    rsi = calculate_rsi(prices, config['rsi_hours'])
    
    # Trading simulation
    position = 0
    cash = 10000
    shares = 0
    trades = []
    equity_curve = []
    
    for i in range(config['lookback_hours'], len(hourly_data)):
        current_price = prices.iloc[i]
        current_z = zscore.iloc[i]
        current_rsi = rsi.iloc[i]
        current_time = hourly_data.index[i]
        
        # Skip if indicators are NaN
        if pd.isna(current_z) or pd.isna(current_rsi):
            equity = cash + (shares * current_price)
            equity_curve.append(equity)
            continue
        
        # Trading logic
        if position == 0:
            # Entry: oversold conditions
            if current_z < -config['zscore_threshold'] and current_rsi < config['rsi_oversold']:
                shares = int(cash * 0.95 / current_price)
                if shares > 0:
                    cash -= shares * current_price
                    position = 1
                    trades.append({
                        'time': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'zscore': current_z,
                        'rsi': current_rsi
                    })
        
        elif position == 1:
            # Exit: mean reversion or stop loss
            if current_z > config['exit_zscore']:
                cash += shares * current_price
                trades.append({
                    'time': current_time,
                    'action': 'SELL',
                    'price': current_price,
                    'zscore': current_z,
                    'rsi': current_rsi
                })
                position = 0
                shares = 0
            # Stop loss at -3 std dev
            elif current_z < -3.0:
                cash += shares * current_price
                trades.append({
                    'time': current_time,
                    'action': 'STOP_LOSS',
                    'price': current_price,
                    'zscore': current_z,
                    'rsi': current_rsi
                })
                position = 0
                shares = 0
        
        # Track equity
        equity = cash + (shares * current_price)
        equity_curve.append(equity)
    
    # Close final position
    if position == 1:
        final_price = prices.iloc[-1]
        cash += shares * final_price
        trades.append({
            'time': hourly_data.index[-1],
            'action': 'FINAL_SELL',
            'price': final_price
        })
    
    # Calculate metrics
    final_value = cash
    total_return = (final_value - 10000) / 10000 * 100
    num_trades = len([t for t in trades if t['action'] == 'BUY'])
    
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Number of Trades: {num_trades}")
    
    if total_return > best_return and num_trades > 5:
        best_return = total_return
        best_config = config.copy()
        best_config['trades'] = trades
        best_config['equity_curve'] = equity_curve
        best_config['total_return'] = total_return
        best_config['num_trades'] = num_trades

# Show best results
if best_config:
    print("\n" + "="*80)
    print("🏆 BEST HOURLY CONFIGURATION")
    print("="*80)
    print(f"Strategy: {best_config['name']}")
    print(f"Total Return: {best_config['total_return']:.2f}%")
    print(f"Number of Trades: {best_config['num_trades']}")
    
    # Calculate additional metrics
    buy_trades = [t for t in best_config['trades'] if t['action'] == 'BUY']
    sell_trades = [t for t in best_config['trades'] if t['action'] in ['SELL', 'STOP_LOSS']]
    
    if len(buy_trades) > 0 and len(sell_trades) > 0:
        wins = 0
        total_pnl = 0
        trade_returns = []
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            trade_return = (sell_price - buy_price) / buy_price
            trade_returns.append(trade_return)
            if trade_return > 0:
                wins += 1
        
        win_rate = wins / len(trade_returns) * 100
        avg_win = np.mean([r for r in trade_returns if r > 0]) * 100 if wins > 0 else 0
        avg_loss = np.mean([r for r in trade_returns if r < 0]) * 100 if wins < len(trade_returns) else 0
        
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        
        # Sharpe ratio of trades
        if len(trade_returns) > 1:
            trade_sharpe = np.sqrt(252 * 6.5) * np.mean(trade_returns) / np.std(trade_returns)
            print(f"Trade Sharpe Ratio: {trade_sharpe:.2f}")
        
        # Annualized return
        days = (hourly_data.index[-1] - hourly_data.index[0]).days
        annual_return = (1 + best_config['total_return']/100) ** (365/days) - 1
        print(f"Annualized Return: {annual_return*100:.1f}%")
        
        print("\nConfiguration:")
        for key, value in best_config.items():
            if key not in ['trades', 'equity_curve', 'total_return', 'num_trades']:
                print(f"  {key}: {value}")
        
        print("\nRecent Trades:")
        for trade in best_config['trades'][-10:]:
            print(f"  {trade['action']} at {trade['time']}: ${trade['price']:.2f} "
                  f"(z={trade.get('zscore', 0):.2f}, RSI={trade.get('rsi', 0):.1f})")
        
        # Save winning configuration
        winning_config = {
            "strategy": "MeanReversionHourly",
            "timeframe": "1h",
            "name": best_config['name'],
            "parameters": {
                "lookback_hours": best_config['lookback_hours'],
                "zscore_threshold": best_config['zscore_threshold'],
                "exit_zscore": best_config['exit_zscore'],
                "rsi_hours": best_config['rsi_hours'],
                "rsi_oversold": best_config['rsi_oversold'],
                "rsi_overbought": best_config['rsi_overbought']
            },
            "performance": {
                "total_return": best_config['total_return'],
                "num_trades": best_config['num_trades'],
                "win_rate": win_rate,
                "annualized_return": annual_return * 100
            },
            "tested_period": {
                "start": str(hourly_data.index[0]),
                "end": str(hourly_data.index[-1]),
                "days": days
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"WINNING_HOURLY_CONFIG_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(winning_config, f, indent=2)
        
        print(f"\n✅ Winning hourly configuration saved to: {filename}")
        print("="*80)
else:
    print("\nNo profitable configuration found with sufficient trades.")