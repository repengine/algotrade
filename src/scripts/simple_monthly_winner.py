#!/usr/bin/env python3
"""
Simple monthly winner finder - tests strategies for consistent monthly profits.
"""

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# Import our strategies
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBacktester:
    """Simple backtester for quick strategy evaluation."""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital

    def run_backtest(self, strategy, data: pd.DataFrame) -> dict[str, Any]:
        """Run a simple backtest."""

        # Initialize strategy
        strategy.init()

        # Track equity and positions
        equity = self.initial_capital
        position = 0
        equity_curve = []
        trades = []

        # Process each bar
        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            if len(current_data) < 50:  # Need minimum data
                equity_curve.append(equity)
                continue

            # Get signal
            try:
                signal = strategy.next(current_data)
            except Exception:
                signal = None

            current_price = data["close"].iloc[i]

            # Execute trades
            if signal:
                if signal.direction == "LONG" and position == 0:
                    # Buy
                    shares = int(equity * 0.95 / current_price)
                    position = shares
                    equity -= shares * current_price
                    trades.append(
                        {
                            "date": data.index[i],
                            "action": "BUY",
                            "price": current_price,
                            "shares": shares,
                        }
                    )

                elif signal.direction == "FLAT" and position > 0:
                    # Sell
                    equity += position * current_price
                    trades.append(
                        {
                            "date": data.index[i],
                            "action": "SELL",
                            "price": current_price,
                            "shares": position,
                        }
                    )
                    position = 0

            # Calculate current equity
            current_equity = equity + (position * current_price)
            equity_curve.append(current_equity)

        # Close any open positions
        if position > 0:
            final_price = data["close"].iloc[-1]
            equity += position * final_price
            current_equity = equity

        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=data.index)
        returns = equity_series.pct_change().dropna()

        total_return = (
            (current_equity - self.initial_capital) / self.initial_capital * 100
        )

        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)
            max_dd = self.calculate_max_drawdown(equity_series)
        else:
            sharpe_ratio = 0
            max_dd = 0

        return {
            "equity_curve": equity_series,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "num_trades": len(trades),
            "trades": trades,
        }

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        return drawdown.min()


def test_strategy_configs():
    """Test various strategy configurations."""

    # Download SPY data
    logger.info("Downloading SPY data...")
    spy = yf.Ticker("SPY")
    data = spy.history(start="2020-10-01", end="2023-01-01")

    # Convert column names to lowercase for compatibility
    data.columns = data.columns.str.lower()
    data.index.name = "date"

    # Test period: 2021-01-01 to 2023-01-01 (24 months)
    test_data = data["2021-01-01":"2023-01-01"].copy()

    # Get monthly returns for buy-and-hold
    # Handle both 'Close' and 'close' column names
    close_col = "Close" if "Close" in test_data.columns else "close"
    monthly_prices = test_data[close_col].resample("M").last()
    benchmark_monthly_returns = monthly_prices.pct_change().dropna()

    logger.info(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    logger.info(f"Number of months: {len(benchmark_monthly_returns)}")

    # Strategy configurations to test
    results = []
    backtester = SimpleBacktester()

    # Test Mean Reversion configurations
    logger.info("\nTesting Mean Reversion strategies...")
    for lookback in [15, 20, 25]:
        for rsi_period in [2, 3, 5]:
            for rsi_oversold in [10, 15, 20, 25]:
                config = {
                    "symbols": ["SPY"],
                    "lookback_period": lookback,
                    "rsi_period": rsi_period,
                    "rsi_oversold": rsi_oversold,
                    "rsi_overbought": 85,
                    "atr_band_mult": 2.5,
                    "ma_exit_period": 10,
                    "stop_loss_atr": 3.0,
                }

                try:
                    strategy = MeanReversionEquity(config)
                    backtest_results = backtester.run_backtest(strategy, test_data)

                    # Calculate monthly performance
                    equity_monthly = (
                        backtest_results["equity_curve"].resample("M").last()
                    )
                    strategy_monthly_returns = equity_monthly.pct_change().dropna()

                    # Check monthly criteria
                    outperformance = (
                        strategy_monthly_returns - benchmark_monthly_returns
                    )
                    num_profitable = (strategy_monthly_returns > 0).sum()
                    num_outperform = (outperformance > 0).sum()

                    result = {
                        "strategy": "MeanReversionEquity",
                        "config": config,
                        "total_return": backtest_results["total_return"],
                        "sharpe_ratio": backtest_results["sharpe_ratio"],
                        "max_drawdown": backtest_results["max_drawdown"],
                        "num_trades": backtest_results["num_trades"],
                        "profitable_months": num_profitable,
                        "outperforming_months": num_outperform,
                        "total_months": len(strategy_monthly_returns),
                        "avg_monthly_return": strategy_monthly_returns.mean() * 100,
                        "avg_outperformance": outperformance.mean() * 100,
                        "perfect": num_profitable == len(strategy_monthly_returns)
                        and num_outperform == len(strategy_monthly_returns),
                    }

                    results.append(result)

                    if result["perfect"]:
                        logger.info("🎉 PERFECT STRATEGY FOUND!")
                        logger.info(f"Config: {json.dumps(config, indent=2)}")
                        logger.info(f"Total Return: {result['total_return']:.2f}%")
                        logger.info(f"Sharpe: {result['sharpe_ratio']:.2f}")

                except Exception as e:
                    logger.debug(f"Error with config: {e}")

    # Test Trend Following configurations
    logger.info("\nTesting Trend Following strategies...")
    for fast in [10, 15, 20]:
        for slow in [40, 50]:
            for atr_mult in [1.5, 2.0, 2.5]:
                config = {
                    "symbols": ["SPY"],
                    "channel_period": 20,
                    "trail_period": fast,
                    "fast_ma": fast,
                    "slow_ma": slow,
                    "atr_period": 14,
                    "atr_multiplier": atr_mult,
                    "adx_period": 14,
                    "adx_threshold": 25,
                    "lookback_period": 252,
                }

                try:
                    strategy = TrendFollowingMulti(config)
                    backtest_results = backtester.run_backtest(strategy, test_data)

                    # Calculate monthly performance
                    equity_monthly = (
                        backtest_results["equity_curve"].resample("M").last()
                    )
                    strategy_monthly_returns = equity_monthly.pct_change().dropna()

                    # Check monthly criteria
                    outperformance = (
                        strategy_monthly_returns - benchmark_monthly_returns
                    )
                    num_profitable = (strategy_monthly_returns > 0).sum()
                    num_outperform = (outperformance > 0).sum()

                    result = {
                        "strategy": "TrendFollowingMulti",
                        "config": config,
                        "total_return": backtest_results["total_return"],
                        "sharpe_ratio": backtest_results["sharpe_ratio"],
                        "max_drawdown": backtest_results["max_drawdown"],
                        "num_trades": backtest_results["num_trades"],
                        "profitable_months": num_profitable,
                        "outperforming_months": num_outperform,
                        "total_months": len(strategy_monthly_returns),
                        "avg_monthly_return": strategy_monthly_returns.mean() * 100,
                        "avg_outperformance": outperformance.mean() * 100,
                        "perfect": num_profitable == len(strategy_monthly_returns)
                        and num_outperform == len(strategy_monthly_returns),
                    }

                    results.append(result)

                    if result["perfect"]:
                        logger.info("🎉 PERFECT STRATEGY FOUND!")
                        logger.info(f"Config: {json.dumps(config, indent=2)}")
                        logger.info(f"Total Return: {result['total_return']:.2f}%")
                        logger.info(f"Sharpe: {result['sharpe_ratio']:.2f}")

                except Exception as e:
                    logger.debug(f"Error with config: {e}")

    return results


def main():
    """Find winning strategies."""

    logger.info("Starting strategy search for 24-month winners...")

    results = test_strategy_configs()

    # Sort by performance
    results.sort(
        key=lambda x: (x["outperforming_months"], x["total_return"]), reverse=True
    )

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"strategy_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find perfect strategies
    perfect_strategies = [r for r in results if r["perfect"]]

    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY SEARCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total strategies tested: {len(results)}")
    logger.info(f"Perfect strategies found: {len(perfect_strategies)}")

    if perfect_strategies:
        logger.info(
            "\n🎉 PERFECT STRATEGIES (profitable EVERY month, beat benchmark EVERY month):"
        )
        for i, strategy in enumerate(perfect_strategies[:5]):  # Show top 5
            logger.info(f"\n#{i+1}:")
            logger.info(f"Strategy: {strategy['strategy']}")
            logger.info(f"Total Return: {strategy['total_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {strategy['max_drawdown']:.2f}%")
            logger.info(f"Avg Monthly Return: {strategy['avg_monthly_return']:.2f}%")
            logger.info(f"Avg Outperformance: {strategy['avg_outperformance']:.2f}%")
            logger.info(f"Config: {json.dumps(strategy['config'], indent=2)}")

        # Save winning config
        with open(f"WINNING_CONFIG_{timestamp}.json", "w") as f:
            json.dump(
                {
                    "strategy": perfect_strategies[0]["strategy"],
                    "config": perfect_strategies[0]["config"],
                    "performance": {
                        "total_return": perfect_strategies[0]["total_return"],
                        "sharpe_ratio": perfect_strategies[0]["sharpe_ratio"],
                        "max_drawdown": perfect_strategies[0]["max_drawdown"],
                        "avg_monthly_return": perfect_strategies[0][
                            "avg_monthly_return"
                        ],
                        "avg_outperformance": perfect_strategies[0][
                            "avg_outperformance"
                        ],
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"\n✅ Winning configuration saved to WINNING_CONFIG_{timestamp}.json")
    else:
        # Show best performers
        logger.info("\nNo perfect strategies found. Best performers:")
        for i, strategy in enumerate(results[:5]):
            logger.info(f"\n#{i+1}:")
            logger.info(f"Strategy: {strategy['strategy']}")
            logger.info(
                f"Outperforming Months: {strategy['outperforming_months']}/{strategy['total_months']}"
            )
            logger.info(f"Total Return: {strategy['total_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
