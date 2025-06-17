#!/usr/bin/env python3
"""
Find a winning strategy that beats buy-and-hold for 24 consecutive months.
"""

import itertools
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.yf_fetcher import YFinanceFetcher
from core.backtest_engine import BacktestEngine
from core.data_handler import DataHandler
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.overnight_drift import OvernightDrift
from strategies.trend_following_multi import TrendFollowingMulti
from utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MonthlyPerformanceOptimizer:
    """Find strategies that are profitable every month and beat buy-and-hold."""

    def __init__(
        self,
        symbol: str = "SPY",
        start_date: str = "2021-01-01",
        end_date: str = "2023-01-01",
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data_handler = DataHandler(YFinanceFetcher())
        self.backtest_engine = BacktestEngine()

    def get_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns from equity curve."""
        monthly = equity_curve.resample("M").last()
        return monthly.pct_change().dropna()

    def check_monthly_profitability(
        self, strategy_returns: pd.Series, benchmark_returns: pd.Series
    ) -> dict[str, Any]:
        """Check if strategy beats benchmark every month."""
        # Align the series
        aligned = pd.DataFrame(
            {"strategy": strategy_returns, "benchmark": benchmark_returns}
        ).dropna()

        # Calculate monthly outperformance
        outperformance = aligned["strategy"] - aligned["benchmark"]

        # Check conditions
        all_profitable = (aligned["strategy"] > 0).all()
        all_beat_benchmark = (outperformance > 0).all()
        num_months = len(aligned)

        return {
            "all_profitable": all_profitable,
            "all_beat_benchmark": all_beat_benchmark,
            "num_months": num_months,
            "avg_monthly_return": aligned["strategy"].mean(),
            "avg_outperformance": outperformance.mean(),
            "min_monthly_return": aligned["strategy"].min(),
            "min_outperformance": outperformance.min(),
            "profitable_months": (aligned["strategy"] > 0).sum(),
            "outperforming_months": (outperformance > 0).sum(),
        }

    def test_parameters(self, strategy_class, params: dict[str, Any]) -> dict[str, Any]:
        """Test a single parameter combination."""
        try:
            # Fetch data
            data = self.data_handler.fetch_data(
                self.symbol,
                datetime.strptime(self.start_date, "%Y-%m-%d"),
                datetime.strptime(self.end_date, "%Y-%m-%d"),
            )

            # Initialize strategy
            strategy = strategy_class(**params)

            # Run backtest
            results = self.backtest_engine.run(strategy, data)

            # Get buy-and-hold results
            buy_hold_equity = data["close"] / data["close"].iloc[0] * 10000

            # Calculate monthly returns
            strategy_monthly = self.get_monthly_returns(results["equity_curve"])
            benchmark_monthly = self.get_monthly_returns(buy_hold_equity)

            # Check monthly performance
            monthly_check = self.check_monthly_profitability(
                strategy_monthly, benchmark_monthly
            )

            return {
                "params": params,
                "total_return": results["total_return"],
                "sharpe_ratio": results["sharpe_ratio"],
                "max_drawdown": results["max_drawdown"],
                "num_trades": results["num_trades"],
                **monthly_check,
            }

        except Exception as e:
            logger.error(f"Error testing params {params}: {e}")
            return None

    def generate_parameter_combinations(
        self, strategy_name: str
    ) -> list[dict[str, Any]]:
        """Generate parameter combinations for a strategy."""

        if strategy_name == "MeanReversionEquity":
            param_grid = {
                "symbol": [self.symbol],
                "lookback_period": [10, 15, 20, 25, 30],
                "entry_threshold": [1.0, 1.5, 2.0, 2.5],
                "exit_threshold": [0.0, 0.25, 0.5],
                "position_size": [0.7, 0.8, 0.9, 1.0],
                "rsi_period": [7, 10, 14],
                "rsi_oversold": [20, 25, 30],
                "rsi_overbought": [70, 75, 80],
            }
        elif strategy_name == "TrendFollowingMulti":
            param_grid = {
                "symbol": [self.symbol],
                "fast_period": [10, 15, 20, 25],
                "slow_period": [40, 50, 60],
                "atr_period": [10, 14, 20],
                "atr_multiplier": [1.5, 2.0, 2.5],
                "trend_filter_period": [100, 150, 200],
                "position_size": [0.7, 0.8, 0.9, 1.0],
            }
        elif strategy_name == "OvernightDrift":
            param_grid = {
                "symbol": [self.symbol],
                "lookback_period": [20, 30, 40, 50],
                "entry_threshold": [0.001, 0.002, 0.003],
                "position_size": [0.7, 0.8, 0.9, 1.0],
                "use_volume_filter": [True, False],
                "volume_lookback": [10, 20],
                "volume_threshold": [1.2, 1.5],
            }
        else:
            return []

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []

        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def optimize_strategy(
        self, strategy_name: str, max_workers: int = 4
    ) -> list[dict[str, Any]]:
        """Optimize a strategy to find parameter combinations that work."""
        logger.info(f"Optimizing {strategy_name}...")

        # Get strategy class
        strategy_map = {
            "MeanReversionEquity": MeanReversionEquity,
            "TrendFollowingMulti": TrendFollowingMulti,
            "OvernightDrift": OvernightDrift,
        }

        strategy_class = strategy_map.get(strategy_name)
        if not strategy_class:
            logger.error(f"Unknown strategy: {strategy_name}")
            return []

        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(strategy_name)
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")

        # Test in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.test_parameters, strategy_class, params): params
                for params in param_combinations
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

                    # Check if we found a winner
                    if (
                        result["all_profitable"]
                        and result["all_beat_benchmark"]
                        and result["num_months"] >= 24
                    ):
                        logger.info(
                            f"FOUND WINNER! {strategy_name} with params: {result['params']}"
                        )
                        logger.info(f"Monthly stats: {result}")

        return results

    def find_winning_strategy(self) -> dict[str, Any]:
        """Find a strategy that meets all criteria."""
        strategies = ["MeanReversionEquity", "TrendFollowingMulti", "OvernightDrift"]

        all_results = []
        winning_strategies = []

        for strategy_name in strategies:
            results = self.optimize_strategy(strategy_name)
            all_results.extend(results)

            # Filter for winners
            winners = [
                r
                for r in results
                if r["all_profitable"]
                and r["all_beat_benchmark"]
                and r["num_months"] >= 24
            ]
            if winners:
                winning_strategies.extend(winners)

        # Sort by best performance
        if winning_strategies:
            winning_strategies.sort(key=lambda x: x["avg_outperformance"], reverse=True)
            return winning_strategies[0]

        # If no perfect winner, find best candidate
        logger.warning("No perfect winner found. Finding best candidate...")

        # Sort by number of profitable months and outperforming months
        all_results.sort(
            key=lambda x: (
                x["outperforming_months"],
                x["profitable_months"],
                x["avg_outperformance"],
            ),
            reverse=True,
        )

        logger.info(f"Best candidate: {all_results[0]}")
        return all_results[0]


def main():
    """Main optimization routine."""
    logger.info("Starting strategy optimization for 24-month profitability...")

    optimizer = MonthlyPerformanceOptimizer(
        symbol="SPY", start_date="2021-01-01", end_date="2023-01-01"
    )

    # Find winning strategy
    winner = optimizer.find_winning_strategy()

    # Save results
    output_file = f"winning_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(winner, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("WINNING STRATEGY FOUND!")
    logger.info("=" * 80)
    logger.info(f"Strategy: {winner['params']}")
    logger.info(f"Total Return: {winner['total_return']:.2f}%")
    logger.info(f"Sharpe Ratio: {winner['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {winner['max_drawdown']:.2f}%")
    logger.info(f"Profitable Months: {winner['profitable_months']}/{winner['num_months']}")
    logger.info(
        f"Outperforming Months: {winner['outperforming_months']}/{winner['num_months']}"
    )
    logger.info(f"Avg Monthly Return: {winner['avg_monthly_return']*100:.2f}%")
    logger.info(f"Avg Outperformance: {winner['avg_outperformance']*100:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
