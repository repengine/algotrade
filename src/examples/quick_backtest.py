#!/usr/bin/env python3
"""
Quick Backtest Example

This script demonstrates how to quickly backtest a single strategy.
"""

import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtests.run_backtests import BacktestEngine
from strategies.mean_reversion_equity import MeanReversionEquityStrategy
from strategies.trend_following_multi import TrendFollowingMultiStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backtest_mean_reversion():
    """Run a simple mean reversion backtest."""
    logger.info("Running Mean Reversion Strategy Backtest")
    logger.info("=" * 50)

    # Create strategy
    strategy = MeanReversionEquityStrategy(
        {
            "symbol": "SPY",
            "lookback": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss": 0.03,
            "position_size": 0.95,  # Use 95% of capital
        }
    )

    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    metrics = engine.run_backtest(
        strategy=strategy,
        symbols=["SPY"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        commission=0.001,  # 0.1% commission
        slippage=0.0005,  # 0.05% slippage
    )

    # Print results
    engine.print_summary()

    return metrics


def backtest_trend_following():
    """Run a trend following backtest."""
    logger.info("\n\nRunning Trend Following Strategy Backtest")
    logger.info("=" * 50)

    # Create strategy
    strategy = TrendFollowingMultiStrategy(
        {
            "symbols": ["QQQ", "IWM"],
            "fast_ma": 10,
            "slow_ma": 30,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "max_positions": 2,
            "position_size": 0.4,  # 40% per position
        }
    )

    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    metrics = engine.run_backtest(
        strategy=strategy,
        symbols=["QQQ", "IWM"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        commission=0.001,
        slippage=0.0005,
    )

    # Print results
    engine.print_summary()

    return metrics


def compare_strategies():
    """Compare multiple strategies."""
    logger.info("\n\nStrategy Comparison")
    logger.info("=" * 50)

    # Run both strategies
    mr_metrics = backtest_mean_reversion()
    tf_metrics = backtest_trend_following()

    # Compare key metrics
    logger.info(f"\n{'Metric':<20} {'Mean Reversion':>15} {'Trend Following':>15}")
    logger.info("-" * 50)
    logger.info(
        f"{'Total Return':<20} {mr_metrics['total_return']:>14.1f}% {tf_metrics['total_return']:>14.1f}%"
    )
    logger.info(
        f"{'Sharpe Ratio':<20} {mr_metrics['sharpe_ratio']:>15.2f} {tf_metrics['sharpe_ratio']:>15.2f}"
    )
    logger.info(
        f"{'Max Drawdown':<20} {mr_metrics['max_drawdown']:>14.1f}% {tf_metrics['max_drawdown']:>14.1f}%"
    )
    logger.info(
        f"{'Win Rate':<20} {mr_metrics['win_rate']:>14.1%} {tf_metrics['win_rate']:>14.1%}"
    )
    logger.info(
        f"{'Total Trades':<20} {mr_metrics['total_trades']:>15} {tf_metrics['total_trades']:>15}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick backtest example")
    parser.add_argument(
        "--strategy",
        choices=["mean_reversion", "trend_following", "compare"],
        default="compare",
        help="Strategy to backtest",
    )

    args = parser.parse_args()

    if args.strategy == "mean_reversion":
        backtest_mean_reversion()
    elif args.strategy == "trend_following":
        backtest_trend_following()
    else:
        compare_strategies()

    logger.info("\nBacktest complete!")
