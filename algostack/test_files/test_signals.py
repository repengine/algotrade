#!/usr/bin/env python3
"""
Test script to verify strategy signal generation
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Add the algostack directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Handle TA-Lib import gracefully
try:
    import talib
except ImportError:
    import mock_talib as talib

    sys.modules["talib"] = talib


def test_rsi_signals():
    """Test RSI signal generation"""
    print("Testing RSI Signal Generation")
    print("=" * 50)

    # Fetch some data
    symbol = "SPY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)

    print(f"Fetching data for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Calculate RSI
    period = 14
    oversold = 30
    overbought = 70

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Generate signals
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data["Close"]
    signals["rsi"] = rsi
    signals["signal"] = 0

    buy_condition = rsi < oversold
    sell_condition = rsi > overbought

    signals.loc[buy_condition, "signal"] = 1
    signals.loc[sell_condition, "signal"] = -1

    # Forward fill positions
    signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(0)

    # Count signals
    buy_signals = (signals["signal"] == 1).sum()
    sell_signals = (signals["signal"] == -1).sum()
    position_changes = (signals["position"].diff() != 0).sum()

    print("\nSignal Statistics:")
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    print(f"Position changes: {position_changes}")

    # Show some examples
    print("\nFirst 5 buy signals:")
    buy_rows = signals[signals["signal"] == 1].head()
    if not buy_rows.empty:
        print(buy_rows[["price", "rsi", "signal", "position"]])
    else:
        print("No buy signals found!")

    print("\nFirst 5 sell signals:")
    sell_rows = signals[signals["signal"] == -1].head()
    if not sell_rows.empty:
        print(sell_rows[["price", "rsi", "signal", "position"]])
    else:
        print("No sell signals found!")

    # Calculate returns
    signals["returns"] = data["Close"].pct_change().fillna(0)
    signals["strategy_returns"] = (
        signals["position"].shift(1).fillna(0) * signals["returns"]
    )
    signals["cumulative_returns"] = (1 + signals["strategy_returns"]).cumprod()

    final_return = (signals["cumulative_returns"].iloc[-1] - 1) * 100
    print(f"\nFinal return: {final_return:.2f}%")

    return signals


def test_ma_crossover_signals():
    """Test moving average crossover signals"""
    print("\n\nTesting MA Crossover Signal Generation")
    print("=" * 50)

    # Fetch some data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)

    print(f"Fetching data for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    print(f"Data shape: {data.shape}")

    # Calculate moving averages
    fast_period = 20
    slow_period = 50

    fast_ma = data["Close"].rolling(window=fast_period).mean()
    slow_ma = data["Close"].rolling(window=slow_period).mean()

    # Generate signals
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data["Close"]
    signals["fast_ma"] = fast_ma
    signals["slow_ma"] = slow_ma
    signals["signal"] = 0

    # Signal when fast MA crosses above/below slow MA
    signals["ma_diff"] = fast_ma - slow_ma
    signals["ma_diff_prev"] = signals["ma_diff"].shift(1)

    # Crossover detection
    bullish_cross = (signals["ma_diff"] > 0) & (signals["ma_diff_prev"] <= 0)
    bearish_cross = (signals["ma_diff"] < 0) & (signals["ma_diff_prev"] >= 0)

    signals.loc[bullish_cross, "signal"] = 1
    signals.loc[bearish_cross, "signal"] = -1

    # Forward fill positions
    signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(0)

    # Count signals
    buy_signals = (signals["signal"] == 1).sum()
    sell_signals = (signals["signal"] == -1).sum()

    print("\nSignal Statistics:")
    print(f"Bullish crossovers: {buy_signals}")
    print(f"Bearish crossovers: {sell_signals}")

    # Show crossover points
    print("\nCrossover points:")
    crossovers = signals[signals["signal"] != 0][
        ["price", "fast_ma", "slow_ma", "signal"]
    ].head(10)
    if not crossovers.empty:
        print(crossovers)
    else:
        print("No crossovers found!")

    return signals


def test_trade_tracking():
    """Test trade entry/exit tracking"""
    print("\n\nTesting Trade Entry/Exit Tracking")
    print("=" * 50)

    # Create a simple signal pattern
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    signals = pd.DataFrame(index=dates)

    # Create some manual signals
    signals["signal"] = 0
    signals.iloc[10] = 1  # Buy
    signals.iloc[20] = -1  # Sell
    signals.iloc[30] = 1  # Buy
    signals.iloc[40] = 0  # Exit
    signals.iloc[50] = -1  # Short
    signals.iloc[60] = 0  # Exit

    # Forward fill positions
    signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(0)

    # Track entries and exits
    entry_mask = (signals["position"].shift(1).fillna(0) == 0) & (
        signals["position"] != 0
    )
    exit_mask = (signals["position"].shift(1).fillna(0) != 0) & (
        signals["position"] == 0
    )

    signals["trade_entry"] = 0
    signals["trade_exit"] = 0

    signals.loc[entry_mask, "trade_entry"] = 1
    signals.loc[exit_mask, "trade_exit"] = 1

    print("Signal pattern:")
    print(
        signals[signals["signal"] != 0][
            ["signal", "position", "trade_entry", "trade_exit"]
        ]
    )

    print(f"\nTotal entries: {entry_mask.sum()}")
    print(f"Total exits: {exit_mask.sum()}")

    return signals


if __name__ == "__main__":
    print("AlgoStack Signal Generation Test")
    print("================================\n")

    try:
        # Test RSI signals
        rsi_signals = test_rsi_signals()

        # Test MA crossover signals
        ma_signals = test_ma_crossover_signals()

        # Test trade tracking
        trade_signals = test_trade_tracking()

        print("\n\nAll tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
