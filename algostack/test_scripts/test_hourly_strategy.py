#!/usr/bin/env python3
"""
Test the winning daily strategy on hourly timeframe.
"""

import json
from datetime import datetime

import yfinance as yf

# Test our winning daily config on hourly data
config = {
    "symbols": ["SPY"],
    "lookback_period": 20,  # Will use 20*8 = 160 hourly bars
    "zscore_threshold": 2.0,
    "exit_zscore": 0.5,
    "rsi_period": 2,  # 2*8 = 16 hourly bars
    "rsi_oversold": 20.0,
    "rsi_overbought": 80.0,
}

# Download hourly data
print("Downloading hourly SPY data...")
spy = yf.Ticker("SPY")
hourly_data = spy.history(start="2025-04-01", end="2025-06-08", interval="1h")
hourly_data.columns = hourly_data.columns.str.lower()

print(f"Loaded {len(hourly_data)} hourly bars")
print(f"Date range: {hourly_data.index[0]} to {hourly_data.index[-1]}")

# Simple backtest
lookback_hours = config["lookback_period"] * 8  # Convert days to hours
rsi_hours = config["rsi_period"] * 8

position = 0
cash = 10000
shares = 0
trades = []

for i in range(lookback_hours, len(hourly_data)):
    # Calculate z-score
    window = hourly_data["close"].iloc[i - lookback_hours : i]
    mean = window.mean()
    std = window.std()

    if std > 0:
        zscore = (hourly_data["close"].iloc[i] - mean) / std
    else:
        zscore = 0

    # Calculate RSI
    rsi_window = hourly_data["close"].iloc[i - rsi_hours : i + 1]
    gains = rsi_window.diff().clip(lower=0)
    losses = -rsi_window.diff().clip(upper=0)

    avg_gain = gains.rolling(window=rsi_hours).mean().iloc[-1]
    avg_loss = losses.rolling(window=rsi_hours).mean().iloc[-1]

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 100 if avg_gain > 0 else 50

    current_price = hourly_data["close"].iloc[i]
    current_time = hourly_data.index[i]

    # Trading logic
    if (
        position == 0
        and zscore < -config["zscore_threshold"]
        and rsi < config["rsi_oversold"]
    ):
        # Buy signal
        shares = int(cash * 0.95 / current_price)
        if shares > 0:
            cash -= shares * current_price
            position = 1
            trades.append(
                {
                    "time": current_time,
                    "action": "BUY",
                    "price": current_price,
                    "zscore": zscore,
                    "rsi": rsi,
                }
            )
            print(
                f"BUY at {current_time}: ${current_price:.2f} (z={zscore:.2f}, RSI={rsi:.1f})"
            )

    elif position == 1 and zscore > -config["exit_zscore"]:
        # Sell signal
        cash += shares * current_price
        trades.append(
            {
                "time": current_time,
                "action": "SELL",
                "price": current_price,
                "zscore": zscore,
                "rsi": rsi,
            }
        )
        print(
            f"SELL at {current_time}: ${current_price:.2f} (z={zscore:.2f}, RSI={rsi:.1f})"
        )
        position = 0
        shares = 0

# Close final position
if position == 1:
    final_price = hourly_data["close"].iloc[-1]
    cash += shares * final_price
    trades.append(
        {"time": hourly_data.index[-1], "action": "SELL", "price": final_price}
    )

# Calculate results
final_value = cash
total_return = (final_value - 10000) / 10000 * 100
num_trades = len([t for t in trades if t["action"] == "BUY"])

print("\n" + "=" * 80)
print("HOURLY TRADING RESULTS")
print("=" * 80)
print("Initial Capital: $10,000")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Number of Round-Trip Trades: {num_trades}")

if num_trades > 0:
    # Calculate trade statistics
    buy_prices = [t["price"] for t in trades if t["action"] == "BUY"]
    sell_prices = [t["price"] for t in trades if t["action"] == "SELL"]

    wins = 0
    total_pnl = 0
    for i in range(min(len(buy_prices), len(sell_prices))):
        pnl = (sell_prices[i] - buy_prices[i]) / buy_prices[i]
        total_pnl += pnl
        if pnl > 0:
            wins += 1

    win_rate = wins / min(len(buy_prices), len(sell_prices)) * 100
    avg_pnl = total_pnl / min(len(buy_prices), len(sell_prices)) * 100

    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Trade P&L: {avg_pnl:.2f}%")

    # Annualize return
    days = (hourly_data.index[-1] - hourly_data.index[0]).days
    annual_return = (1 + total_return / 100) ** (365 / days) - 1
    print(f"Annualized Return: {annual_return*100:.1f}%")

    # Show recent trades
    print("\nRecent Trades:")
    for trade in trades[-10:]:
        action = trade["action"]
        time = trade["time"]
        price = trade["price"]
        z = trade.get("zscore", 0)
        rsi = trade.get("rsi", 0)
        print(f"  {action} at {time}: ${price:.2f} (z={z:.2f}, RSI={rsi:.1f})")

print("=" * 80)

# Save results
results = {
    "timeframe": "1h",
    "config": config,
    "performance": {
        "total_return": total_return,
        "num_trades": num_trades,
        "win_rate": win_rate if num_trades > 0 else 0,
        "annualized_return": annual_return * 100 if num_trades > 0 else 0,
    },
    "trades": trades,
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"hourly_test_results_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {filename}")
