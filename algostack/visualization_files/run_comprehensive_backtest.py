#!/usr/bin/env python3
"""
Comprehensive Backtesting Script for AlgoStack

This script runs backtests for all available strategies using both
Yahoo Finance and Alpha Vantage data sources.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtests.run_backtests import (
    BacktestEngine,
    run_walk_forward_optimization,
)
from strategies.hybrid_regime import HybridRegimeStrategy
from strategies.intraday_orb import IntradayORBStrategy
from strategies.mean_reversion_equity import MeanReversionEquityStrategy
from strategies.overnight_drift import OvernightDriftStrategy
from strategies.pairs_stat_arb import PairsStatArbStrategy
from strategies.trend_following_multi import TrendFollowingMultiStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Strategy configurations for backtesting
STRATEGY_CONFIGS = {
    "mean_reversion_spy": {
        "class": MeanReversionEquityStrategy,
        "config": {
            "symbol": "SPY",
            "lookback": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss": 0.03,
            "position_size": 0.95,  # Use 95% of capital
        },
        "symbols": ["SPY"],
        "description": "Mean reversion on S&P 500 ETF",
    },
    "trend_following_multi": {
        "class": TrendFollowingMultiStrategy,
        "config": {
            "symbols": ["QQQ", "IWM", "DIA"],
            "fast_ma": 10,
            "slow_ma": 30,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "max_positions": 2,
            "position_size": 0.3,  # 30% per position
        },
        "symbols": ["QQQ", "IWM", "DIA"],
        "description": "Trend following on major ETFs",
    },
    "overnight_drift": {
        "class": OvernightDriftStrategy,
        "config": {
            "symbol": "SPY",
            "holding_period": 1,
            "entry_time": "15:50",
            "exit_time": "09:35",
            "min_volume": 1000000,
            "position_size": 0.95,
        },
        "symbols": ["SPY"],
        "description": "Overnight holding strategy",
        "requires_intraday": True,
    },
    "opening_range_breakout": {
        "class": IntradayORBStrategy,
        "config": {
            "symbol": "QQQ",
            "orb_minutes": 30,
            "profit_target": 0.02,  # 2%
            "stop_loss": 0.01,  # 1%
            "position_size": 0.5,
        },
        "symbols": ["QQQ"],
        "description": "Opening range breakout",
        "requires_intraday": True,
    },
    "pairs_trading": {
        "class": PairsStatArbStrategy,
        "config": {
            "pair": ["XLF", "KRE"],  # Financials vs Regional Banks
            "lookback": 60,
            "entry_zscore": 2.0,
            "exit_zscore": 0.5,
            "stop_loss": 0.05,
            "position_size": 0.4,  # 40% per leg
        },
        "symbols": ["XLF", "KRE"],
        "description": "Statistical arbitrage pairs trading",
    },
    "hybrid_regime": {
        "class": HybridRegimeStrategy,
        "config": {
            "symbol": "SPY",
            "trend_lookback": 50,
            "vol_lookback": 20,
            "vol_threshold": 0.02,
            "trend_fast_ma": 10,
            "trend_slow_ma": 30,
            "mr_lookback": 20,
            "mr_entry_threshold": 2.0,
            "position_size": 0.8,
        },
        "symbols": ["SPY"],
        "description": "Regime-adaptive strategy",
    },
}


class ComprehensiveBacktester:
    """Run comprehensive backtests for all strategies."""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        self.load_api_keys()

    def load_api_keys(self):
        """Load API keys from config."""
        config_path = Path(__file__).parent / "config" / "secrets.yaml"
        if config_path.exists():
            with open(config_path) as f:
                secrets = yaml.safe_load(f)
                self.av_api_key = (
                    secrets.get("data_providers", {})
                    .get("alphavantage", {})
                    .get("api_key")
                )
        else:
            self.av_api_key = None

    def run_all_backtests(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        data_source: str = "yfinance",
    ):
        """Run backtests for all strategies."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BACKTEST REPORT")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Data Source: {data_source}")
        print("=" * 80 + "\n")

        # Set up data source
        if data_source == "alphavantage" and self.av_api_key:
            # Configure Alpha Vantage with premium key
            os.environ["ALPHA_VANTAGE_API_KEY"] = self.av_api_key

        summary_results = []

        for strategy_name, strategy_info in STRATEGY_CONFIGS.items():
            try:
                # Skip intraday strategies for now (need special data handling)
                if strategy_info.get("requires_intraday", False):
                    logger.info(f"Skipping {strategy_name} - requires intraday data")
                    continue

                print(f"\nTesting: {strategy_name}")
                print(f"Description: {strategy_info['description']}")
                print("-" * 60)

                # Create strategy instance
                strategy_class = strategy_info["class"]
                config = strategy_info["config"]
                strategy = strategy_class(config)

                # Run backtest
                engine = BacktestEngine(initial_capital=self.initial_capital)
                metrics = engine.run_backtest(
                    strategy=strategy,
                    symbols=strategy_info["symbols"],
                    start_date=start_date,
                    end_date=end_date,
                    commission=0.001,  # 0.1% commission
                    slippage=0.0005,  # 0.05% slippage
                    data_provider=data_source,
                )

                # Store results
                self.results[strategy_name] = {
                    "metrics": metrics,
                    "config": config,
                    "description": strategy_info["description"],
                }

                # Print summary
                engine.print_summary()

                # Add to summary
                summary_results.append(
                    {
                        "strategy": strategy_name,
                        "total_return": metrics["total_return"],
                        "annual_return": metrics["annual_return"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "max_drawdown": metrics["max_drawdown"],
                        "win_rate": metrics["win_rate"],
                        "profit_factor": metrics["profit_factor"],
                        "total_trades": metrics["total_trades"],
                    }
                )

            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}")

        # Print summary comparison
        self._print_summary_comparison(summary_results)

        # Save results
        self._save_results()

    def run_walk_forward_analysis(
        self,
        strategy_name: str,
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        window_size: int = 252,  # 1 year
        step_size: int = 63,  # 3 months
    ):
        """Run walk-forward analysis for a specific strategy."""
        if strategy_name not in STRATEGY_CONFIGS:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        print(f"\nWalk-Forward Analysis: {strategy_name}")
        print("=" * 60)

        strategy_info = STRATEGY_CONFIGS[strategy_name]

        # Run walk-forward optimization
        results_df = run_walk_forward_optimization(
            strategy_class=strategy_info["class"],
            config=strategy_info["config"],
            symbols=strategy_info["symbols"],
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            step_size=step_size,
        )

        # Print results
        print("\nWalk-Forward Results:")
        print(
            results_df[
                [
                    "window_start",
                    "window_end",
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                ]
            ]
        )

        # Calculate statistics
        print(f"\nAverage Annual Return: {results_df['annual_return'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}")
        print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%")
        print(
            f"Consistency (Positive periods): {(results_df['total_return'] > 0).mean():.1%}"
        )

        return results_df

    def _print_summary_comparison(self, results: list[Dict]):
        """Print summary comparison of all strategies."""
        if not results:
            return

        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 80)

        # Convert to DataFrame for easy sorting
        df = pd.DataFrame(results)

        # Sort by Sharpe ratio
        df = df.sort_values("sharpe_ratio", ascending=False)

        # Print table
        print(
            f"\n{'Strategy':<25} {'Total Return':>12} {'Annual Return':>13} {'Sharpe':>8} {'Max DD':>8} {'Win Rate':>9} {'Trades':>8}"
        )
        print("-" * 95)

        for _, row in df.iterrows():
            print(
                f"{row['strategy']:<25} "
                f"{row['total_return']:>11.1f}% "
                f"{row['annual_return']:>12.1f}% "
                f"{row['sharpe_ratio']:>8.2f} "
                f"{row['max_drawdown']:>7.1f}% "
                f"{row['win_rate']:>8.1%} "
                f"{row['total_trades']:>8}"
            )

        print("\nBest Performing Strategies:")
        print(
            f"1. Highest Return: {df.iloc[df['total_return'].argmax()]['strategy']} ({df['total_return'].max():.1f}%)"
        )
        print(
            f"2. Best Sharpe: {df.iloc[0]['strategy']} ({df.iloc[0]['sharpe_ratio']:.2f})"
        )
        print(
            f"3. Lowest Drawdown: {df.iloc[df['max_drawdown'].argmin()]['strategy']} ({df['max_drawdown'].min():.1f}%)"
        )

    def _save_results(self):
        """Save backtest results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "backtests" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"comprehensive_backtest_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def main():
    global STRATEGY_CONFIGS
    """Run comprehensive backtests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive backtests for AlgoStack"
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument(
        "--source",
        default="yfinance",
        choices=["yfinance", "alphavantage"],
        help="Data source to use",
    )
    parser.add_argument("--strategy", help="Run specific strategy only")
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run walk-forward analysis"
    )

    args = parser.parse_args()

    # Create backtester
    backtester = ComprehensiveBacktester(initial_capital=args.capital)

    if args.walk_forward and args.strategy:
        # Run walk-forward analysis for specific strategy
        backtester.run_walk_forward_analysis(
            strategy_name=args.strategy, start_date=args.start, end_date=args.end
        )
    elif args.strategy:
        # Run single strategy
        if args.strategy not in STRATEGY_CONFIGS:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available strategies: {', '.join(STRATEGY_CONFIGS.keys())}")
            return

        # Create custom config with single strategy
        single_config = {args.strategy: STRATEGY_CONFIGS[args.strategy]}

        # Temporarily replace global config
        original_configs = STRATEGY_CONFIGS
        STRATEGY_CONFIGS = single_config

        backtester.run_all_backtests(
            start_date=args.start, end_date=args.end, data_source=args.source
        )

        # Restore original config
        STRATEGY_CONFIGS = original_configs
    else:
        # Run all strategies
        backtester.run_all_backtests(
            start_date=args.start, end_date=args.end, data_source=args.source
        )

    print("\nBacktesting complete!")


if __name__ == "__main__":
    main()
