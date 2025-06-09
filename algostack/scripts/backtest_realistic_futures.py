#!/usr/bin/env python3
"""
Realistic futures momentum backtest with appropriate parameters for 5-min bars.
"""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from strategies.futures_momentum import FuturesMomentum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest_with_multiple_configs(data):
    """Test multiple realistic configurations."""

    configs = [
        {
            "name": "Tight Breakout",
            "lookback_period": 20,
            "breakout_threshold": 0.1,  # 0.1% for 5-min bars
            "rsi_period": 14,
            "rsi_threshold": 55,  # Lower RSI threshold
            "atr_period": 14,
            "stop_loss_atr": 1.5,  # Tighter stop
            "profit_target_atr": 2.5,  # Better R:R
            "volume_multiplier": 1.1,  # Lower volume requirement
            "position_size": 0.95,
        },
        {
            "name": "Scalper",
            "lookback_period": 10,  # Shorter lookback
            "breakout_threshold": 0.05,  # Very tight
            "rsi_period": 7,
            "rsi_threshold": 50,  # Neutral RSI
            "atr_period": 10,
            "stop_loss_atr": 1.0,  # 1 ATR stop
            "profit_target_atr": 1.5,  # Quick profits
            "volume_multiplier": 1.0,  # No volume filter
            "position_size": 0.95,
        },
        {
            "name": "Momentum Rider",
            "lookback_period": 30,
            "breakout_threshold": 0.15,
            "rsi_period": 21,
            "rsi_threshold": 60,
            "atr_period": 14,
            "stop_loss_atr": 2.0,
            "profit_target_atr": 4.0,  # 2:1 R:R
            "volume_multiplier": 1.2,
            "position_size": 0.95,
        },
    ]

    results = []

    for config in configs:
        logger.info(f"\nTesting {config['name']} configuration...")

        # Create strategy
        strategy_config = {"symbols": ["SPY"], **config}

        strategy = FuturesMomentum(strategy_config)
        strategy.init()

        # Run backtest
        cash = 10000
        position = 0
        shares = 0
        trades = []
        equity_curve = []

        for i in range(strategy.config["lookback_period"], len(data)):
            window = data.iloc[: i + 1].copy()
            window.attrs["symbol"] = "SPY"

            signal = strategy.next(window)

            current_price = data["close"].iloc[i]
            current_time = data.index[i]

            # Track equity
            if position > 0:
                current_equity = cash + (shares * current_price)
            else:
                current_equity = cash

            equity_curve.append(current_equity)

            # Process signals
            if signal:
                if signal.direction == "LONG" and position == 0:
                    # Buy
                    shares = int(cash * 0.95 / current_price)
                    cash -= shares * current_price + 0.52  # Commission
                    position = 1
                    entry_price = current_price

                    trades.append(
                        {
                            "time": current_time,
                            "action": "BUY",
                            "price": current_price,
                            "shares": shares,
                        }
                    )

                elif signal.direction == "FLAT" and position > 0:
                    # Sell
                    cash += shares * current_price - 0.52  # Commission

                    pnl_pct = (current_price - entry_price) / entry_price * 100

                    trades.append(
                        {
                            "time": current_time,
                            "action": "SELL",
                            "price": current_price,
                            "shares": shares,
                            "pnl_pct": pnl_pct,
                            "reason": signal.reason,
                        }
                    )

                    position = 0
                    shares = 0

        # Close final position
        if position > 0:
            final_price = data["close"].iloc[-1]
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
        total_return = (cash - 10000) / 10000 * 100
        num_trades = len([t for t in trades if t["action"] == "BUY"])

        if num_trades > 0:
            winning_trades = len([t for t in trades if t.get("pnl_pct", 0) > 0])
            win_rate = winning_trades / num_trades * 100

            # Average win/loss
            wins = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0]
            losses = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0

            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = np.sqrt(78 * 252) * returns.mean() / (returns.std() + 1e-6)
        else:
            sharpe = 0

        result = {
            "config_name": config["name"],
            "total_return": total_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "final_value": cash,
            "config": config,
        }

        results.append(result)

        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Trades: {num_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")

    return results


def main():
    logger.info("=" * 80)
    logger.info("REALISTIC FUTURES MOMENTUM BACKTEST")
    logger.info("Testing multiple configurations on 5-minute data")
    logger.info("=" * 80)

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    spy = yf.Ticker("SPY")
    data = spy.history(start=start_date, end=end_date, interval="5m")
    data.columns = data.columns.str.lower()

    logger.info(f"Loaded {len(data)} 5-minute bars")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Run backtests
    results = run_backtest_with_multiple_configs(data)

    # Find best configuration
    best = max(results, key=lambda x: x["total_return"])

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"  Trades: {result['num_trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")

    print("\n" + "=" * 80)
    print(f"🏆 BEST CONFIGURATION: {best['config_name']}")
    print("=" * 80)
    print(f"Total Return: {best['total_return']:.2f}%")
    print(f"Final Value: ${best['final_value']:,.2f}")
    print(f"Number of Trades: {best['num_trades']}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Average Win: {best['avg_win']:.2f}%")
    print(f"Average Loss: {best['avg_loss']:.2f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")

    # Annualize return
    days = (data.index[-1] - data.index[0]).days
    if days > 0 and best["total_return"] > 0:
        annual_return = (1 + best["total_return"] / 100) ** (365 / days) - 1
        print(f"\n📈 Projected Annual Return: {annual_return*100:.1f}%")

    print("\nOptimal Parameters:")
    for key, value in best["config"].items():
        if key != "name":
            print(f"  {key}: {value}")

    # Save results
    output = {
        "test_period": {
            "start": str(data.index[0]),
            "end": str(data.index[-1]),
            "days": days,
            "bars": len(data),
        },
        "best_config": best,
        "all_results": results,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"futures_backtest_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()
