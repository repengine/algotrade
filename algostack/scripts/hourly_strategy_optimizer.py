#!/usr/bin/env python3
"""
Hourly timeframe strategy optimizer - finds profitable configurations on 1-hour candles.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from strategies.mean_reversion_equity import MeanReversionEquity


class HourlyStrategyOptimizer:
    """Optimize strategies on hourly timeframe."""

    def __init__(self):
        self.hourly_data = None
        self.load_hourly_data()

    def load_hourly_data(self):
        """Load hourly SPY data."""
        logger.info("Downloading hourly SPY data...")
        spy = yf.Ticker("SPY")

        # Get 60 days of hourly data (yfinance limit)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        self.hourly_data = spy.history(start=start_date, end=end_date, interval="1h")

        # Convert columns to lowercase
        self.hourly_data.columns = self.hourly_data.columns.str.lower()
        self.hourly_data.attrs["symbol"] = "SPY"

        logger.info(f"Loaded {len(self.hourly_data)} hourly bars")
        logger.info(
            f"Date range: {self.hourly_data.index[0]} to {self.hourly_data.index[-1]}"
        )

    def test_hourly_strategy(
        self, strategy_class, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Test a strategy on hourly data."""

        try:
            # Initialize strategy
            strategy = strategy_class(config)
            strategy.init()

            # Need enough data for lookback
            min_lookback = (
                config.get("lookback_period", 20) * 8
            )  # Convert days to hours (8 trading hours/day)

            # Simulate trading
            position = 0
            cash = 10000
            shares = 0
            equity_curve = []
            trades = []

            for i in range(min_lookback, len(self.hourly_data)):
                window = self.hourly_data.iloc[: i + 1].copy()
                window.attrs["symbol"] = "SPY"

                # Get signal
                signal = strategy.next(window)

                price = self.hourly_data["close"].iloc[i]
                timestamp = self.hourly_data.index[i]

                # Execute trades
                if signal and signal.direction == "LONG" and position == 0:
                    # Buy
                    shares = int(cash * 0.95 / price)
                    cash -= shares * price
                    position = 1
                    trades.append(
                        {
                            "time": timestamp,
                            "action": "BUY",
                            "price": price,
                            "shares": shares,
                        }
                    )

                elif signal and signal.direction == "FLAT" and position == 1:
                    # Sell
                    cash += shares * price
                    trades.append(
                        {
                            "time": timestamp,
                            "action": "SELL",
                            "price": price,
                            "shares": shares,
                        }
                    )
                    shares = 0
                    position = 0

                # Calculate equity
                equity = cash + (shares * price)
                equity_curve.append(
                    {"time": timestamp, "equity": equity, "price": price}
                )

            # Close final position
            if position == 1:
                final_price = self.hourly_data["close"].iloc[-1]
                cash += shares * final_price
                trades.append(
                    {
                        "time": self.hourly_data.index[-1],
                        "action": "SELL",
                        "price": final_price,
                        "shares": shares,
                    }
                )

            # Calculate metrics
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index("time", inplace=True)

            # Calculate daily returns from hourly equity curve
            daily_equity = equity_df["equity"].resample("D").last()
            daily_returns = daily_equity.pct_change().dropna()

            # Calculate hourly returns
            hourly_returns = equity_df["equity"].pct_change().dropna()

            # Metrics
            total_return = (cash - 10000) / 10000 * 100

            if len(daily_returns) > 0:
                sharpe_daily = (
                    np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-6)
                )
            else:
                sharpe_daily = 0

            if len(hourly_returns) > 0:
                # Annualized hourly Sharpe (1950 trading hours per year)
                sharpe_hourly = (
                    np.sqrt(1950)
                    * hourly_returns.mean()
                    / (hourly_returns.std() + 1e-6)
                )
            else:
                sharpe_hourly = 0

            # Calculate max drawdown
            rolling_max = equity_df["equity"].expanding().max()
            drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()

            return {
                "success": True,
                "total_return": total_return,
                "sharpe_daily": sharpe_daily,
                "sharpe_hourly": sharpe_hourly,
                "max_drawdown": max_drawdown,
                "num_trades": len(trades),
                "trades_per_day": len(trades)
                / (
                    len(self.hourly_data) / (6.5 * 5)
                ),  # 6.5 hours per day, 5 days per week
                "win_rate": self.calculate_win_rate(trades),
                "avg_trade_duration": self.calculate_avg_duration(trades),
                "config": config,
            }

        except Exception as e:
            logger.debug(f"Strategy test failed: {e}")
            return {"success": False}

    def calculate_win_rate(self, trades: list[dict]) -> float:
        """Calculate win rate from trades."""
        if len(trades) < 2:
            return 0.0

        wins = 0
        for i in range(0, len(trades) - 1, 2):  # Buy-sell pairs
            if i + 1 < len(trades):
                buy_price = trades[i]["price"]
                sell_price = trades[i + 1]["price"]
                if sell_price > buy_price:
                    wins += 1

        total_pairs = len(trades) // 2
        return wins / total_pairs if total_pairs > 0 else 0.0

    def calculate_avg_duration(self, trades: list[dict]) -> float:
        """Calculate average trade duration in hours."""
        if len(trades) < 2:
            return 0.0

        durations = []
        for i in range(0, len(trades) - 1, 2):  # Buy-sell pairs
            if i + 1 < len(trades):
                buy_time = trades[i]["time"]
                sell_time = trades[i + 1]["time"]
                duration = (sell_time - buy_time).total_seconds() / 3600  # Hours
                durations.append(duration)

        return np.mean(durations) if durations else 0.0

    def optimize_hourly_strategies(self):
        """Find optimal parameters for hourly trading."""

        results = []

        # Test Mean Reversion on hourly timeframe
        logger.info("Testing Mean Reversion strategies on hourly data...")

        # Adjusted parameters for hourly timeframe
        hourly_params = {
            "symbols": [["SPY"]],
            "lookback_period": [
                5,
                10,
                15,
                20,
            ],  # Days (will be converted to hours internally)
            "zscore_threshold": [1.5, 2.0, 2.5],
            "exit_zscore": [0.0, 0.25, 0.5],
            "rsi_period": [2, 3, 5],
            "rsi_oversold": [15.0, 20.0, 25.0],
            "rsi_overbought": [75.0, 80.0, 85.0],
        }

        # Test a subset of combinations
        tested = 0
        for lookback in hourly_params["lookback_period"]:
            for zscore in hourly_params["zscore_threshold"]:
                for exit_z in hourly_params["exit_zscore"]:
                    if exit_z > zscore:
                        continue
                    for rsi_period in hourly_params["rsi_period"]:
                        for rsi_os in hourly_params["rsi_oversold"]:
                            for rsi_ob in hourly_params["rsi_overbought"]:
                                if rsi_os >= rsi_ob:
                                    continue

                                config = {
                                    "symbols": ["SPY"],
                                    "lookback_period": lookback,
                                    "zscore_threshold": zscore,
                                    "exit_zscore": exit_z,
                                    "rsi_period": rsi_period,
                                    "rsi_oversold": rsi_os,
                                    "rsi_overbought": rsi_ob,
                                }

                                result = self.test_hourly_strategy(
                                    MeanReversionEquity, config
                                )
                                tested += 1

                                if result["success"]:
                                    results.append(result)

                                    if (
                                        result["sharpe_hourly"] > 1.5
                                        and result["num_trades"] > 10
                                    ):
                                        logger.info("Good hourly strategy found!")
                                        logger.info(
                                            f"  Return: {result['total_return']:.2f}%"
                                        )
                                        logger.info(
                                            f"  Hourly Sharpe: {result['sharpe_hourly']:.2f}"
                                        )
                                        logger.info(
                                            f"  Trades/day: {result['trades_per_day']:.1f}"
                                        )
                                        logger.info(
                                            f"  Win Rate: {result['win_rate']*100:.1f}%"
                                        )

                                if tested % 50 == 0:
                                    logger.info(f"Tested {tested} configurations...")

        # Sort by Sharpe ratio
        results.sort(key=lambda x: x["sharpe_hourly"], reverse=True)

        return results


def main():
    """Find optimal hourly trading strategy."""

    logger.info("=" * 80)
    logger.info("HOURLY STRATEGY OPTIMIZER")
    logger.info("Finding profitable intraday trading strategies...")
    logger.info("=" * 80)

    optimizer = HourlyStrategyOptimizer()
    results = optimizer.optimize_hourly_strategies()

    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hourly_strategies_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results[:10], f, indent=2, default=str)  # Top 10

        logger.info(f"\nResults saved to: {filename}")

        # Show best strategy
        if results[0]["sharpe_hourly"] > 0:
            best = results[0]

            print("\n" + "=" * 80)
            print("🏆 BEST HOURLY STRATEGY")
            print("=" * 80)
            print(f"Total Return (60 days): {best['total_return']:.2f}%")
            print(f"Hourly Sharpe Ratio: {best['sharpe_hourly']:.2f}")
            print(f"Daily Sharpe Ratio: {best['sharpe_daily']:.2f}")
            print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
            print(f"Number of Trades: {best['num_trades']}")
            print(f"Trades per Day: {best['trades_per_day']:.1f}")
            print(f"Win Rate: {best['win_rate']*100:.1f}%")
            print(f"Avg Trade Duration: {best['avg_trade_duration']:.1f} hours")
            print("\nConfiguration:")
            print(json.dumps(best["config"], indent=2))
            print("=" * 80)

            # Annualize the return
            days_tested = 60
            annual_return = (1 + best["total_return"] / 100) ** (365 / days_tested) - 1
            print(f"\nProjected Annual Return: {annual_return*100:.1f}%")

            # Create production config
            prod_config = {
                "strategy": "MeanReversionEquity",
                "timeframe": "1h",
                "config": best["config"],
                "expected_performance": {
                    "hourly_sharpe": best["sharpe_hourly"],
                    "daily_sharpe": best["sharpe_daily"],
                    "trades_per_day": best["trades_per_day"],
                    "win_rate": best["win_rate"],
                    "projected_annual_return": annual_return * 100,
                },
                "notes": "Optimized for hourly trading on SPY",
            }

            prod_filename = f"HOURLY_PRODUCTION_CONFIG_{timestamp}.json"
            with open(prod_filename, "w") as f:
                json.dump(prod_config, f, indent=2)

            print(f"\n✅ Production config saved to: {prod_filename}")
    else:
        logger.warning("No successful strategies found")


if __name__ == "__main__":
    main()
