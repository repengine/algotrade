#!/usr/bin/env python3
"""
Backtest the winning mean reversion strategy on 5-minute data.
Based on our earlier successful configuration.
"""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_indicators(data, lookback=20, zscore_threshold=2.0, rsi_period=2):
    """Calculate mean reversion indicators."""

    df = data.copy()

    # Z-score
    df["sma"] = df["close"].rolling(lookback).mean()
    df["std"] = df["close"].rolling(lookback).std()
    df["zscore"] = (df["close"] - df["sma"]) / df["std"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR for stops
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["atr"] = true_range.rolling(14).mean()

    return df


def run_mean_reversion_backtest(data, config):
    """Run mean reversion backtest with 5-minute data."""

    # Calculate indicators
    df = calculate_indicators(
        data,
        lookback=config["lookback_period"],
        zscore_threshold=config["zscore_threshold"],
        rsi_period=config["rsi_period"],
    )

    # Initialize
    cash = 10000
    position = 0
    shares = 0
    trades = []
    equity_curve = []

    for i in range(config["lookback_period"], len(df)):
        current_price = df["close"].iloc[i]
        current_zscore = df["zscore"].iloc[i]
        current_rsi = df["rsi"].iloc[i]
        current_time = df.index[i]
        current_atr = df["atr"].iloc[i]

        # Skip if indicators are invalid
        if pd.isna(current_zscore) or pd.isna(current_rsi):
            continue

        # Track equity
        if position > 0:
            current_equity = cash + (shares * current_price)
        else:
            current_equity = cash

        equity_curve.append(
            {
                "time": current_time,
                "equity": current_equity,
                "zscore": current_zscore,
                "rsi": current_rsi,
            }
        )

        # Check trading hours (optional)
        if hasattr(current_time, "hour"):
            hour = current_time.hour + current_time.minute / 60
            if hour < 9.5 or hour > 15.5:  # Only trade market hours
                continue

        # Trading logic
        if position == 0:
            # Entry: Oversold conditions
            if (
                current_zscore < -config["zscore_threshold"]
                and current_rsi < config["rsi_oversold"]
            ):

                # Buy
                shares = int(cash * config["position_size"] / current_price)
                if shares > 0:
                    cash -= shares * current_price
                    cash -= 0.52  # MES commission
                    position = 1
                    entry_price = current_price
                    stop_price = current_price - (current_atr * config["stop_loss_atr"])

                    trades.append(
                        {
                            "time": current_time,
                            "action": "BUY",
                            "price": current_price,
                            "shares": shares,
                            "zscore": current_zscore,
                            "rsi": current_rsi,
                        }
                    )

        else:  # position == 1
            # Exit conditions
            exit_signal = False
            exit_reason = ""

            # Take profit at mean reversion
            if current_zscore > -config["exit_zscore"]:
                exit_signal = True
                exit_reason = "mean_reversion"

            # Stop loss
            elif current_price <= stop_price:
                exit_signal = True
                exit_reason = "stop_loss"

            # Time-based exit (optional)
            elif len(trades) > 0:
                bars_held = i - df.index.get_loc(trades[-1]["time"])
                if bars_held > 100:  # Exit after 100 bars (~8 hours)
                    exit_signal = True
                    exit_reason = "time_exit"

            if exit_signal:
                # Sell
                cash += shares * current_price
                cash -= 0.52  # Commission

                pnl_pct = (current_price - entry_price) / entry_price * 100

                trades.append(
                    {
                        "time": current_time,
                        "action": "SELL",
                        "price": current_price,
                        "shares": shares,
                        "pnl_pct": pnl_pct,
                        "reason": exit_reason,
                        "zscore": current_zscore,
                    }
                )

                position = 0
                shares = 0

    # Close final position
    if position > 0:
        final_price = df["close"].iloc[-1]
        cash += shares * final_price - 0.52
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append(
            {
                "action": "SELL",
                "price": final_price,
                "pnl_pct": pnl_pct,
                "reason": "end_of_test",
            }
        )

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    total_return = (cash - 10000) / 10000 * 100

    # Trade statistics
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]

    num_trades = len(buy_trades)
    if num_trades > 0:
        winning_trades = len([t for t in sell_trades if t.get("pnl_pct", 0) > 0])
        win_rate = winning_trades / num_trades * 100

        # Win/loss analysis
        wins = [t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) > 0]
        losses = [t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # Sharpe ratio
    if len(equity_df) > 1:
        returns = equity_df["equity"].pct_change().dropna()
        # Annualize for 5-min data
        sharpe = np.sqrt(78 * 252) * returns.mean() / (returns.std() + 1e-6)
    else:
        sharpe = 0

    return {
        "total_return": total_return,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "final_value": cash,
        "trades": trades,
        "equity_curve": equity_df,
    }


def main():
    logger.info("=" * 80)
    logger.info("BACKTESTING WINNING MEAN REVERSION STRATEGY")
    logger.info("Using parameters optimized for current market conditions")
    logger.info("=" * 80)

    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    spy = yf.Ticker("SPY")
    data = spy.history(start=start_date, end=end_date, interval="5m")
    data.columns = data.columns.str.lower()

    logger.info(f"Loaded {len(data)} 5-minute bars")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Test multiple configurations
    configs = [
        {
            "name": "Conservative",
            "lookback_period": 20,  # 100 minutes
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 2,
            "rsi_oversold": 20.0,
            "rsi_overbought": 80.0,
            "stop_loss_atr": 3.0,
            "position_size": 0.95,
        },
        {
            "name": "Moderate",
            "lookback_period": 15,  # 75 minutes
            "zscore_threshold": 1.5,
            "exit_zscore": 0.0,
            "rsi_period": 3,
            "rsi_oversold": 25.0,
            "rsi_overbought": 75.0,
            "stop_loss_atr": 2.5,
            "position_size": 0.95,
        },
        {
            "name": "Aggressive",
            "lookback_period": 10,  # 50 minutes
            "zscore_threshold": 1.0,
            "exit_zscore": -0.25,
            "rsi_period": 2,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "stop_loss_atr": 2.0,
            "position_size": 0.95,
        },
    ]

    results = []

    for config in configs:
        logger.info(f"\nTesting {config['name']} configuration...")
        result = run_mean_reversion_backtest(data, config)
        result["config_name"] = config["name"]
        result["config"] = config
        results.append(result)

        logger.info(f"  Total Return: {result['total_return']:.2f}%")
        logger.info(f"  Trades: {result['num_trades']}")
        logger.info(f"  Win Rate: {result['win_rate']:.1f}%")

    # Find best
    best = max(results, key=lambda x: x["total_return"])

    print("\n" + "=" * 80)
    print("🏆 BACKTEST RESULTS - MEAN REVERSION")
    print("=" * 80)

    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"  Trades: {result['num_trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    print("\n" + "-" * 80)
    print(f"BEST CONFIGURATION: {best['config_name']}")
    print("-" * 80)
    print(f"Total Return: {best['total_return']:.2f}%")
    print(f"Final Value: ${best['final_value']:,.2f}")
    print(f"Number of Trades: {best['num_trades']}")

    if best["num_trades"] > 0:
        print(f"Win Rate: {best['win_rate']:.1f}%")
        print(f"Average Win: {best['avg_win']:.2f}%")
        print(f"Average Loss: {best['avg_loss']:.2f}%")
        print(f"Profit Factor: {best['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")

        # Annualize
        days = (data.index[-1] - data.index[0]).days
        if days > 0:
            annual_factor = 365 / days
            annual_return = (1 + best["total_return"] / 100) ** annual_factor - 1
            print(f"\n📈 Projected Annual Return: {annual_return*100:.1f}%")

        # Show recent trades
        if best["trades"]:
            print("\n📋 Recent Trades:")
            for trade in best["trades"][-10:]:
                if trade["action"] == "BUY":
                    print(
                        f"  BUY at {trade['time']}: ${trade['price']:.2f} "
                        f"(Z={trade['zscore']:.2f}, RSI={trade['rsi']:.1f})"
                    )
                else:
                    print(
                        f"  SELL at {trade.get('time', 'end')}: ${trade['price']:.2f} "
                        f"({trade['reason']}, P&L={trade.get('pnl_pct', 0):.2f}%)"
                    )

    # Save results
    output = {
        "strategy": "Mean Reversion 5-min",
        "test_period": str(data.index[0]) + " to " + str(data.index[-1]),
        "best_config": best["config"],
        "performance": {
            "total_return": best["total_return"],
            "num_trades": best["num_trades"],
            "win_rate": best["win_rate"],
            "sharpe_ratio": best["sharpe_ratio"],
        },
        "all_results": [
            {
                "name": r["config_name"],
                "return": r["total_return"],
                "trades": r["num_trades"],
                "win_rate": r["win_rate"],
            }
            for r in results
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mean_reversion_5min_backtest_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()
