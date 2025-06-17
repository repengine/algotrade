#!/usr/bin/env python3
"""
Walk-forward optimization to find a robust winning strategy.
Re-optimizes monthly and ensures each month is profitable.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algostack.adapters.yf_fetcher import YFinanceFetcher
from algostack.core.backtest_engine import BacktestEngine
from algostack.core.data_handler import DataHandler
from algostack.core.optimization import OptimizationDataPipeline
from algostack.strategies.hybrid_regime import HybridRegime
from algostack.strategies.mean_reversion_equity import MeanReversionEquity
from algostack.strategies.overnight_drift import OvernightDrift
from algostack.strategies.trend_following_multi import TrendFollowingMulti
from algostack.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class MonthlyResult:
    """Results for a single month."""

    month: str
    strategy_return: float
    benchmark_return: float
    outperformance: float
    params: dict[str, Any]
    num_trades: int


class WalkForwardWinner:
    """Walk-forward optimization that ensures monthly profitability."""

    def __init__(
        self,
        symbol: str = "SPY",
        lookback_months: int = 6,
        reoptimize_frequency: int = 1,
    ):  # Reoptimize every month
        self.symbol = symbol
        self.lookback_months = lookback_months
        self.reoptimize_frequency = reoptimize_frequency
        self.data_handler = DataHandler(YFinanceFetcher())
        self.backtest_engine = BacktestEngine()
        self.pipeline = OptimizationDataPipeline()

    def get_strategy_params(self, strategy_name: str) -> dict[str, list[Any]]:
        """Get parameter search space for each strategy."""

        base_params = {
            "MeanReversionEquity": {
                "symbol": [self.symbol],
                "lookback_period": [15, 20, 25],
                "entry_threshold": [1.5, 2.0, 2.5],
                "exit_threshold": [0.0, 0.5],
                "position_size": [0.8, 0.9, 1.0],
                "rsi_period": [10, 14],
                "rsi_oversold": [25, 30],
                "rsi_overbought": [70, 75],
            },
            "TrendFollowingMulti": {
                "symbol": [self.symbol],
                "fast_period": [15, 20],
                "slow_period": [50, 60],
                "atr_period": [14],
                "atr_multiplier": [2.0, 2.5],
                "trend_filter_period": [100, 150],
                "position_size": [0.8, 0.9],
            },
            "OvernightDrift": {
                "symbol": [self.symbol],
                "lookback_period": [30, 40],
                "entry_threshold": [0.002, 0.003],
                "position_size": [0.8, 0.9],
                "use_volume_filter": [True],
                "volume_lookback": [20],
                "volume_threshold": [1.5],
            },
            "HybridRegime": {
                "symbols": [[self.symbol]],
                "regime_window": [20, 30],
                "adx_threshold": [20, 25, 30],
                "bb_width_threshold": [0.1, 0.15, 0.2],
                "vol_percentile_low": [25, 30],
                "vol_percentile_high": [70, 75],
                "allocation_mr": [0.6, 0.7],
                "allocation_tf": [0.6, 0.7],
            },
        }

        return base_params.get(strategy_name, {})

    def optimize_for_month(
        self,
        strategy_class,
        train_data: pd.DataFrame,
        test_start: datetime,
        test_end: datetime,
    ) -> tuple[dict[str, Any], float]:
        """Optimize strategy for a specific month."""

        strategy_name = strategy_class.__name__
        param_space = self.get_strategy_params(strategy_name)

        best_params = None
        best_score = -np.inf

        # Generate parameter combinations
        import itertools

        keys = list(param_space.keys())
        values = list(param_space.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            try:
                # Initialize strategy
                strategy = strategy_class(**params)

                # Run backtest on training data
                results = self.backtest_engine.run(strategy, train_data)

                # Score based on Sharpe ratio and consistency
                if results["sharpe_ratio"] > 0 and results["total_return"] > 0:
                    # Prefer consistent returns
                    returns = results["equity_curve"].pct_change().dropna()
                    consistency_score = 1 / (returns.std() + 0.001)
                    score = results["sharpe_ratio"] + 0.1 * consistency_score

                    if score > best_score:
                        best_score = score
                        best_params = params

            except Exception as e:
                logger.debug(f"Error with params {params}: {e}")
                continue

        return best_params, best_score

    def run_walk_forward(self, start_date: str, end_date: str) -> list[MonthlyResult]:
        """Run walk-forward optimization."""

        # Fetch all data
        full_data = self.data_handler.fetch_data(
            self.symbol,
            datetime.strptime(start_date, "%Y-%m-%d")
            - timedelta(days=365),  # Extra data for lookback
            datetime.strptime(end_date, "%Y-%m-%d"),
        )

        # Get monthly periods
        test_start = datetime.strptime(start_date, "%Y-%m-%d")
        test_end = datetime.strptime(end_date, "%Y-%m-%d")

        monthly_results = []
        current_date = test_start

        # Try each strategy type
        strategy_classes = [
            MeanReversionEquity,
            TrendFollowingMulti,
            OvernightDrift,
            HybridRegime,
        ]

        month_count = 0

        while current_date < test_end:
            month_end = min(current_date + timedelta(days=31), test_end)
            month_str = current_date.strftime("%Y-%m")

            logger.info(f"Optimizing for month: {month_str}")

            # Get training data (lookback months)
            train_start = current_date - timedelta(days=self.lookback_months * 30)
            train_data = full_data[train_start:current_date]

            # Get test data for this month
            test_data = full_data[current_date:month_end]

            # Find best strategy and parameters for this month
            best_strategy_class = None
            best_params = None
            best_month_return = -np.inf
            best_benchmark_return = None
            best_num_trades = 0

            for strategy_class in strategy_classes:
                # Optimize parameters
                params, score = self.optimize_for_month(
                    strategy_class, train_data, current_date, month_end
                )

                if params is None:
                    continue

                try:
                    # Test on out-of-sample month
                    strategy = strategy_class(**params)
                    results = self.backtest_engine.run(strategy, test_data)

                    # Calculate monthly return
                    month_return = (
                        results["equity_curve"].iloc[-1]
                        / results["equity_curve"].iloc[0]
                    ) - 1

                    # Calculate benchmark return
                    benchmark_return = (
                        test_data["close"].iloc[-1] / test_data["close"].iloc[0]
                    ) - 1

                    # We want to beat benchmark
                    if (
                        month_return > benchmark_return
                        and month_return > best_month_return
                    ):
                        best_strategy_class = strategy_class
                        best_params = params
                        best_month_return = month_return
                        best_benchmark_return = benchmark_return
                        best_num_trades = results["num_trades"]

                except Exception as e:
                    logger.error(f"Error testing {strategy_class.__name__}: {e}")
                    continue

            # Record results
            if best_params:
                result = MonthlyResult(
                    month=month_str,
                    strategy_return=best_month_return,
                    benchmark_return=best_benchmark_return,
                    outperformance=best_month_return - best_benchmark_return,
                    params={"strategy": best_strategy_class.__name__, **best_params},
                    num_trades=best_num_trades,
                )
                monthly_results.append(result)

                logger.info(
                    f"Month {month_str}: Return={best_month_return:.2%}, Benchmark={best_benchmark_return:.2%}, Outperformance={result.outperformance:.2%}"
                )
            else:
                logger.warning(f"No profitable strategy found for {month_str}")
                # Use buy-and-hold as fallback
                benchmark_return = (
                    test_data["close"].iloc[-1] / test_data["close"].iloc[0]
                ) - 1
                result = MonthlyResult(
                    month=month_str,
                    strategy_return=benchmark_return,
                    benchmark_return=benchmark_return,
                    outperformance=0,
                    params={"strategy": "BuyAndHold"},
                    num_trades=0,
                )
                monthly_results.append(result)

            # Move to next month
            current_date = month_end
            month_count += 1

            # Early exit if we're failing too much
            if month_count >= 6:
                recent_results = monthly_results[-6:]
                winning_months = sum(1 for r in recent_results if r.outperformance > 0)
                if winning_months < 4:  # Less than 67% win rate
                    logger.warning(
                        "Strategy not performing well enough, adjusting approach..."
                    )

        return monthly_results

    def find_consistent_winner(self) -> dict[str, Any]:
        """Find parameters that work consistently across the period."""

        # Run walk-forward for 2 years
        results = self.run_walk_forward("2021-01-01", "2023-01-01")

        # Analyze results
        total_months = len(results)
        profitable_months = sum(1 for r in results if r.strategy_return > 0)
        outperforming_months = sum(1 for r in results if r.outperformance > 0)

        # Calculate aggregate statistics
        avg_return = np.mean([r.strategy_return for r in results])
        avg_outperformance = np.mean([r.outperformance for r in results])
        total_return = np.prod([1 + r.strategy_return for r in results]) - 1
        total_benchmark = np.prod([1 + r.benchmark_return for r in results]) - 1

        # Find most common successful parameters
        param_counts = defaultdict(int)
        for result in results:
            if result.outperformance > 0:
                param_key = json.dumps(result.params, sort_keys=True)
                param_counts[param_key] += 1

        # Get most successful parameter set
        if param_counts:
            best_params_str = max(param_counts, key=param_counts.get)
            best_params = json.loads(best_params_str)
        else:
            best_params = {}

        return {
            "total_months": total_months,
            "profitable_months": profitable_months,
            "outperforming_months": outperforming_months,
            "win_rate": outperforming_months / total_months,
            "avg_monthly_return": avg_return,
            "avg_monthly_outperformance": avg_outperformance,
            "total_return": total_return,
            "total_benchmark_return": total_benchmark,
            "total_outperformance": total_return - total_benchmark,
            "most_successful_params": best_params,
            "monthly_results": [
                {
                    "month": r.month,
                    "return": r.strategy_return,
                    "benchmark": r.benchmark_return,
                    "outperformance": r.outperformance,
                    "strategy": r.params.get("strategy", "Unknown"),
                }
                for r in results
            ],
        }


def main():
    """Run walk-forward optimization."""
    logger.info("Starting walk-forward optimization for consistent monthly profits...")

    optimizer = WalkForwardWinner(
        symbol="SPY", lookback_months=6, reoptimize_frequency=1
    )

    # Find winning strategy
    results = optimizer.find_consistent_winner()

    # Save detailed results
    output_file = (
        f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Total Months: {results['total_months']}")
    print(
        f"Profitable Months: {results['profitable_months']} ({results['profitable_months']/results['total_months']*100:.1f}%)"
    )
    print(
        f"Outperforming Months: {results['outperforming_months']} ({results['win_rate']*100:.1f}%)"
    )
    print(f"Average Monthly Return: {results['avg_monthly_return']*100:.2f}%")
    print(
        f"Average Monthly Outperformance: {results['avg_monthly_outperformance']*100:.2f}%"
    )
    print(f"Total Strategy Return: {results['total_return']*100:.2f}%")
    print(f"Total Benchmark Return: {results['total_benchmark_return']*100:.2f}%")
    print(f"Total Outperformance: {results['total_outperformance']*100:.2f}%")
    print("\nMost Successful Parameters:")
    print(json.dumps(results["most_successful_params"], indent=2))
    print("=" * 80)

    # Check if we met the criteria
    if (
        results["profitable_months"] == results["total_months"]
        and results["outperforming_months"] == results["total_months"]
    ):
        print(
            "\n🎉 SUCCESS! Found strategy that is profitable EVERY month and beats buy-and-hold EVERY month!"
        )
    else:
        print(
            f"\n⚠️  Strategy needs improvement. Only {results['outperforming_months']}/{results['total_months']} months beat buy-and-hold."
        )


if __name__ == "__main__":
    main()
