#!/usr/bin/env python3
"""
Find winning strategies that are profitable every month for 24 months.
Uses existing strategy configuration format.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.yf_fetcher import YFinanceFetcher
from backtests.run_backtests import BacktestEngine
from core.data_handler import DataHandler
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.overnight_drift import OvernightDrift
from strategies.trend_following_multi import TrendFollowingMulti
from utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MonthlyWinnerFinder:
    """Find strategies that win every single month."""

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self.data_handler = DataHandler(YFinanceFetcher())
        self.backtest_engine = BacktestEngine()

    def test_strategy_monthly(
        self, strategy_class, config: dict[str, Any], start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Test a strategy and get monthly performance."""

        try:
            # Fetch data with extra buffer for indicators
            data_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=100)
            data_end = datetime.strptime(end_date, "%Y-%m-%d")

            data = self.data_handler.fetch_data(self.symbol, data_start, data_end)

            # Initialize strategy
            strategy = strategy_class(config)

            # Run backtest
            results = self.backtest_engine.run(strategy, data)

            # Get equity curve starting from actual start date
            equity_curve = results["equity_curve"]
            equity_curve = equity_curve[equity_curve.index >= start_date]

            # Calculate monthly returns
            monthly_equity = equity_curve.resample("M").last()
            monthly_returns = monthly_equity.pct_change().dropna()

            # Get benchmark monthly returns
            benchmark = data["close"][data.index >= start_date]
            benchmark_monthly = benchmark.resample("M").last()
            benchmark_returns = benchmark_monthly.pct_change().dropna()

            # Align returns
            aligned = pd.DataFrame(
                {"strategy": monthly_returns, "benchmark": benchmark_returns}
            ).dropna()

            # Calculate statistics
            outperformance = aligned["strategy"] - aligned["benchmark"]

            # Check monthly criteria
            all_profitable = (aligned["strategy"] > 0).all()
            all_beat_benchmark = (outperformance > 0).all()
            num_profitable = (aligned["strategy"] > 0).sum()
            num_outperforming = (outperformance > 0).sum()

            return {
                "config": config,
                "strategy_class": strategy_class.__name__,
                "total_return": results["total_return"],
                "sharpe_ratio": results["sharpe_ratio"],
                "max_drawdown": results["max_drawdown"],
                "num_trades": results["num_trades"],
                "num_months": len(aligned),
                "all_profitable": all_profitable,
                "all_beat_benchmark": all_beat_benchmark,
                "num_profitable_months": num_profitable,
                "num_outperforming_months": num_outperforming,
                "avg_monthly_return": aligned["strategy"].mean(),
                "avg_monthly_outperformance": outperformance.mean(),
                "min_monthly_return": aligned["strategy"].min(),
                "max_monthly_return": aligned["strategy"].max(),
                "monthly_returns": aligned["strategy"].tolist(),
                "monthly_outperformance": outperformance.tolist(),
            }

        except Exception as e:
            logger.error(f"Error testing strategy: {e}")
            return None

    def get_strategy_configs(self) -> list[tuple[Any, dict[str, Any]]]:
        """Generate strategy configurations to test."""

        configs = []

        # Mean Reversion configurations
        for lookback in [15, 20, 25, 30]:
            for rsi_period in [2, 3, 5]:
                for rsi_oversold in [10, 15, 20, 25]:
                    for atr_mult in [2.0, 2.5, 3.0]:
                        config = {
                            "symbols": [self.symbol],
                            "lookback_period": lookback,
                            "rsi_period": rsi_period,
                            "rsi_oversold": rsi_oversold,
                            "rsi_overbought": 90,
                            "atr_band_mult": atr_mult,
                            "ma_exit_period": 10,
                            "stop_loss_atr": 3.0,
                            "max_positions": 1,
                        }
                        configs.append((MeanReversionEquity, config))

        # Trend Following configurations
        for fast in [10, 15, 20]:
            for slow in [40, 50, 60]:
                for channel_period in [20, 30]:
                    for atr_mult in [1.5, 2.0, 2.5]:
                        config = {
                            "symbols": [self.symbol],
                            "channel_period": channel_period,
                            "trail_period": fast,
                            "fast_ma": fast,
                            "slow_ma": slow,
                            "atr_period": 14,
                            "atr_multiplier": atr_mult,
                            "adx_period": 14,
                            "adx_threshold": 25,
                            "lookback_period": 252,
                            "max_positions": 1,
                        }
                        configs.append((TrendFollowingMulti, config))

        # Overnight Drift configurations
        for lookback in [20, 30, 40]:
            for threshold in [0.002, 0.003, 0.004]:
                for volume_mult in [1.2, 1.5, 2.0]:
                    config = {
                        "symbols": [self.symbol],
                        "lookback_period": lookback,
                        "entry_threshold": threshold,
                        "position_size": 0.9,
                        "volume_filter": True,
                        "volume_threshold": volume_mult,
                        "regime_filter": True,
                        "max_positions": 1,
                    }
                    configs.append((OvernightDrift, config))

        return configs

    def find_perfect_strategies(
        self, start_date: str = "2021-01-01", end_date: str = "2023-01-01"
    ):
        """Find strategies that meet all criteria."""

        logger.info(f"Testing strategies from {start_date} to {end_date}")
        logger.info(
            "Requirement: Profitable EVERY month and beat buy-and-hold EVERY month"
        )

        configs = self.get_strategy_configs()
        logger.info(f"Testing {len(configs)} strategy configurations...")

        perfect_strategies = []
        good_strategies = []

        for i, (strategy_class, config) in enumerate(configs):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(configs)}")

            result = self.test_strategy_monthly(
                strategy_class, config, start_date, end_date
            )

            if result:
                # Check if perfect (all months profitable and beat benchmark)
                if result["all_profitable"] and result["all_beat_benchmark"]:
                    logger.info(
                        f"🎉 PERFECT STRATEGY FOUND! {result['strategy_class']}"
                    )
                    logger.info(f"Config: {json.dumps(config, indent=2)}")
                    logger.info(f"Total Return: {result['total_return']:.2f}%")
                    logger.info(f"Sharpe: {result['sharpe_ratio']:.2f}")
                    perfect_strategies.append(result)

                # Also track "good" strategies (>90% win rate)
                elif result["num_outperforming_months"] >= result["num_months"] * 0.9:
                    good_strategies.append(result)

        return perfect_strategies, good_strategies

    def save_results(self, perfect_strategies: list[dict], good_strategies: list[dict]):
        """Save results to file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save perfect strategies
        if perfect_strategies:
            filename = f"PERFECT_strategies_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(perfect_strategies, f, indent=2, default=str)
            logger.info(f"Perfect strategies saved to {filename}")

            # Also save a simplified config file
            config_file = f"winning_config_{timestamp}.json"
            winning_config = {
                "strategy": perfect_strategies[0]["strategy_class"],
                "config": perfect_strategies[0]["config"],
                "performance": {
                    "total_return": perfect_strategies[0]["total_return"],
                    "sharpe_ratio": perfect_strategies[0]["sharpe_ratio"],
                    "max_drawdown": perfect_strategies[0]["max_drawdown"],
                    "num_trades": perfect_strategies[0]["num_trades"],
                    "monthly_win_rate": "100%",
                },
            }
            with open(config_file, "w") as f:
                json.dump(winning_config, f, indent=2)
            logger.info(f"Winning config saved to {config_file}")

        # Save good strategies
        if good_strategies:
            filename = f"good_strategies_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(good_strategies, f, indent=2, default=str)
            logger.info(f"Good strategies saved to {filename}")


def main():
    """Find the perfect strategy."""

    finder = MonthlyWinnerFinder(symbol="SPY")

    # Find strategies for 2-year period
    perfect, good = finder.find_perfect_strategies(
        start_date="2021-01-01", end_date="2023-01-01"
    )

    # Save results
    finder.save_results(perfect, good)

    # Print summary
    print("\n" + "=" * 80)
    print("STRATEGY SEARCH COMPLETE")
    print("=" * 80)

    if perfect:
        print(f"\n🎉 Found {len(perfect)} PERFECT strategies!")
        print("\nBest Perfect Strategy:")
        best = max(perfect, key=lambda x: x["sharpe_ratio"])
        print(f"Strategy: {best['strategy_class']}")
        print(f"Total Return: {best['total_return']:.2f}%")
        print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
        print(f"Avg Monthly Return: {best['avg_monthly_return']*100:.2f}%")
        print(
            f"Avg Monthly Outperformance: {best['avg_monthly_outperformance']*100:.2f}%"
        )
        print("\nConfig:")
        print(json.dumps(best["config"], indent=2))
    else:
        print("\n⚠️  No perfect strategies found.")
        if good:
            print(f"\nFound {len(good)} good strategies (>90% win rate)")
            best = max(good, key=lambda x: x["num_outperforming_months"])
            print("\nBest Good Strategy:")
            print(f"Strategy: {best['strategy_class']}")
            print(
                f"Win Rate: {best['num_outperforming_months']}/{best['num_months']} months"
            )
            print(f"Total Return: {best['total_return']:.2f}%")
            print(f"Config: {json.dumps(best['config'], indent=2)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
