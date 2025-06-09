#!/usr/bin/env python3
"""
Optimized Futures Strategy - Multiple approaches for high returns
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


class OptimizedFuturesStrategy:
    """
    Multiple futures strategies optimized for high returns:
    1. Momentum Breakout - Ride strong trends
    2. Mean Reversion - Fade extremes
    3. Opening Range Breakout - Capture morning volatility
    4. Overnight Gap Trade - Exploit session transitions
    """

    def __init__(self):
        self.strategies = {
            "momentum_breakout": {
                "description": "Trade breakouts with the trend",
                "lookback": 20,
                "breakout_threshold": 0.5,  # 0.5% above 20-period high
                "rsi_confirm": 60,  # RSI > 60 for long
                "atr_multiplier": 2.0,  # Stop loss
                "profit_multiplier": 3.0,  # Risk:reward = 1:1.5
                "max_risk_per_trade": 0.02,  # 2% risk per trade
            },
            "opening_range": {
                "description": "Trade first 30-min range breakouts",
                "range_minutes": 30,
                "breakout_buffer": 2,  # 2 ticks above range
                "stop_buffer": 5,  # 5 ticks below range
                "target_multiplier": 2.5,  # 2.5x the range
                "time_limit": 120,  # Exit within 2 hours
                "only_with_volume": True,
            },
            "gap_fade": {
                "description": "Fade overnight gaps",
                "min_gap_percent": 0.3,  # 0.3% minimum gap
                "max_gap_percent": 1.0,  # 1.0% maximum gap
                "entry_retracement": 0.382,  # Enter at 38.2% retracement
                "stop_at_highs": True,
                "target_fill_percent": 0.618,  # Target 61.8% gap fill
            },
            "volatility_expansion": {
                "description": "Trade volatility breakouts",
                "bb_period": 20,
                "bb_std": 2.0,
                "volume_spike": 1.5,  # 1.5x average volume
                "min_range_expansion": 1.5,  # Range must expand 1.5x
                "quick_profit": True,  # Take profits quickly
                "scalp_ticks": 4,  # Quick 4-tick scalps
            },
        }

    def test_momentum_breakout(self, data):
        """Test momentum breakout strategy."""

        df = data.copy()
        config = self.strategies["momentum_breakout"]

        # Calculate indicators
        df["high_20"] = df["high"].rolling(config["lookback"]).max()
        df["low_20"] = df["low"].rolling(config["lookback"]).min()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR for stops
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["atr"] = true_range.rolling(14).mean()

        # Simulate trading
        trades = []
        position = 0
        capital = 10000

        for i in range(30, len(df)):
            if pd.isna(df["rsi"].iloc[i]):
                continue

            price = df["close"].iloc[i]

            if position == 0:
                # Breakout entry
                breakout_level = df["high_20"].iloc[i - 1] * (
                    1 + config["breakout_threshold"] / 100
                )

                if price > breakout_level and df["rsi"].iloc[i] > config["rsi_confirm"]:
                    # Calculate position size
                    stop_distance = df["atr"].iloc[i] * config["atr_multiplier"]
                    risk_amount = capital * config["max_risk_per_trade"]
                    contracts = int(risk_amount / (stop_distance * 5))  # MES = $5/point

                    if contracts > 0:
                        position = contracts
                        entry_price = price
                        stop_price = price - stop_distance
                        target_price = price + (
                            stop_distance * config["profit_multiplier"]
                        )

                        trades.append(
                            {
                                "entry_time": df.index[i],
                                "entry_price": entry_price,
                                "contracts": contracts,
                                "stop": stop_price,
                                "target": target_price,
                            }
                        )

            elif position > 0:
                # Exit logic
                if price <= stop_price or price >= target_price:
                    pnl = position * (price - entry_price) * 5  # MES multiplier
                    capital += pnl - (position * 0.52 * 2)  # Commission

                    trades[-1]["exit_price"] = price
                    trades[-1]["pnl"] = pnl
                    position = 0

        # Calculate returns
        total_return = (capital - 10000) / 10000 * 100
        win_rate = (
            len([t for t in trades if t.get("pnl", 0) > 0]) / len(trades) * 100
            if trades
            else 0
        )

        return {
            "strategy": "momentum_breakout",
            "total_return": total_return,
            "num_trades": len(trades),
            "win_rate": win_rate,
        }

    def test_opening_range_breakout(self, data):
        """Test opening range breakout strategy."""

        df = data.copy()
        config = self.strategies["opening_range"]

        # Add time features
        df["time"] = df.index.time
        df["is_opening_range"] = (df.index.hour == 9) & (df.index.minute < 30)
        df["is_tradeable"] = (df.index.hour >= 9) & (
            df.index.hour < 12
        )  # Trade until noon

        trades = []
        capital = 10000

        # Group by date
        for _date, day_data in df.groupby(df.index.date):
            # Get opening range
            opening_data = day_data[day_data["is_opening_range"]]
            if len(opening_data) < 6:  # Need 30 min of data
                continue

            range_high = opening_data["high"].max()
            range_low = opening_data["low"].min()
            range_size = range_high - range_low

            if range_size < 0.5:  # Skip tiny ranges
                continue

            # Trade the breakout
            position = 0
            for idx in day_data[day_data["is_tradeable"]].index:
                price = day_data.loc[idx, "close"]

                if position == 0:
                    # Long breakout
                    if price > range_high + (
                        config["breakout_buffer"] * 0.25
                    ):  # 0.25 = tick size
                        position = 1
                        entry_price = price
                        stop_price = range_low - (config["stop_buffer"] * 0.25)
                        target_price = entry_price + (
                            range_size * config["target_multiplier"]
                        )
                        entry_time = idx

                    # Short breakout
                    elif price < range_low - (config["breakout_buffer"] * 0.25):
                        position = -1
                        entry_price = price
                        stop_price = range_high + (config["stop_buffer"] * 0.25)
                        target_price = entry_price - (
                            range_size * config["target_multiplier"]
                        )
                        entry_time = idx

                elif position != 0:
                    # Check exits
                    time_elapsed = (idx - entry_time).total_seconds() / 60

                    if position == 1:  # Long position
                        if (
                            price <= stop_price
                            or price >= target_price
                            or time_elapsed > config["time_limit"]
                        ):
                            pnl = (
                                price - entry_price
                            ) * 5 - 1.04  # MES P&L and commission
                            capital += pnl
                            trades.append({"pnl": pnl, "direction": "long"})
                            position = 0

                    else:  # Short position
                        if (
                            price >= stop_price
                            or price <= target_price
                            or time_elapsed > config["time_limit"]
                        ):
                            pnl = (entry_price - price) * 5 - 1.04
                            capital += pnl
                            trades.append({"pnl": pnl, "direction": "short"})
                            position = 0

        total_return = (capital - 10000) / 10000 * 100
        win_rate = (
            len([t for t in trades if t["pnl"] > 0]) / len(trades) * 100
            if trades
            else 0
        )

        return {
            "strategy": "opening_range_breakout",
            "total_return": total_return,
            "num_trades": len(trades),
            "win_rate": win_rate,
        }

    def find_best_strategy(self, data):
        """Test all strategies and find the best one."""

        results = []

        # Test each strategy
        print("\nTesting Momentum Breakout...")
        momentum_results = self.test_momentum_breakout(data)
        results.append(momentum_results)
        print(f"  Return: {momentum_results['total_return']:.2f}%")
        print(f"  Trades: {momentum_results['num_trades']}")
        print(f"  Win Rate: {momentum_results['win_rate']:.1f}%")

        print("\nTesting Opening Range Breakout...")
        orb_results = self.test_opening_range_breakout(data)
        results.append(orb_results)
        print(f"  Return: {orb_results['total_return']:.2f}%")
        print(f"  Trades: {orb_results['num_trades']}")
        print(f"  Win Rate: {orb_results['win_rate']:.1f}%")

        # Sort by return
        results.sort(key=lambda x: x["total_return"], reverse=True)

        return results[0] if results else None


def main():
    """Find the best futures strategy."""

    print("=" * 80)
    print("OPTIMIZED FUTURES STRATEGIES")
    print("Finding the highest return approach...")
    print("=" * 80)

    # Download data
    spy = yf.Ticker("SPY")
    data = spy.history(period="2y", interval="5m")
    data.columns = data.columns.str.lower()

    print(f"Testing on {len(data)} 5-minute bars")

    # Test strategies
    optimizer = OptimizedFuturesStrategy()
    best = optimizer.find_best_strategy(data)

    if best:
        print("\n" + "=" * 80)
        print("🏆 RECOMMENDED FUTURES STRATEGY")
        print("=" * 80)

        # Project annual returns
        days = (data.index[-1] - data.index[0]).days
        annual_return = best["total_return"] * (252 / days)

        print(f"\nStrategy: {best['strategy'].upper()}")
        print(f"Backtest Return: {best['total_return']:.2f}%")
        print(f"Projected Annual Return: {annual_return:.1f}%")
        print(f"Number of Trades: {best['num_trades']}")
        print(f"Win Rate: {best['win_rate']:.1f}%")

        # Double or triple the return with multiple contracts
        print("\n💰 SCALING OPPORTUNITIES:")
        print(f"With 2 contracts: {annual_return * 2:.1f}% annual return")
        print(f"With 3 contracts: {annual_return * 3:.1f}% annual return")
        print(f"With 5 contracts: {annual_return * 5:.1f}% annual return")

        print("\n📋 IMPLEMENTATION STEPS:")
        print("1. Open futures account (no PDT rules!)")
        print("2. Start with MES (Micro E-mini) - only $1,500 margin")
        print("3. Use 1-2 contracts initially")
        print("4. Scale up as profits grow")
        print("5. Trade multiple strategies for diversification")

        print("\n⚡ ADVANTAGES OVER STOCKS:")
        print("• No PDT rules - trade unlimited times")
        print("• 23-hour trading (huge opportunity)")
        print("• Built-in leverage without interest")
        print("• Better tax treatment (60/40 rule)")
        print("• Superior liquidity and execution")

        # Save configuration
        config = {
            "recommended_strategy": best["strategy"],
            "expected_annual_return": annual_return,
            "implementation": optimizer.strategies[best["strategy"]],
            "account_requirements": {
                "minimum": "$5,000 (for MES)",
                "recommended": "$10,000-$25,000",
                "margin_per_MES": "$1,500",
                "margin_per_ES": "$15,000",
            },
            "risk_management": {
                "max_risk_per_trade": "2%",
                "max_daily_loss": "6%",
                "position_sizing": "Based on stop distance",
                "always_use_stops": True,
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BEST_FUTURES_STRATEGY_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Strategy configuration saved to: {filename}")

    print("=" * 80)


if __name__ == "__main__":
    main()
