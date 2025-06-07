#!/usr/bin/env python3
"""
Quick Backtest Example

This script demonstrates how to quickly backtest a single strategy.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algostack.backtests.run_backtests import BacktestEngine
from algostack.strategies.mean_reversion_equity import MeanReversionEquityStrategy
from algostack.strategies.trend_following_multi import TrendFollowingMultiStrategy


def backtest_mean_reversion():
    """Run a simple mean reversion backtest."""
    print("Running Mean Reversion Strategy Backtest")
    print("="*50)
    
    # Create strategy
    strategy = MeanReversionEquityStrategy({
        'symbol': 'SPY',
        'lookback': 20,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 0.03,
        'position_size': 0.95,  # Use 95% of capital
    })
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    metrics = engine.run_backtest(
        strategy=strategy,
        symbols=['SPY'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        commission=0.001,  # 0.1% commission
        slippage=0.0005   # 0.05% slippage
    )
    
    # Print results
    engine.print_summary()
    
    return metrics


def backtest_trend_following():
    """Run a trend following backtest."""
    print("\n\nRunning Trend Following Strategy Backtest")
    print("="*50)
    
    # Create strategy
    strategy = TrendFollowingMultiStrategy({
        'symbols': ['QQQ', 'IWM'],
        'fast_ma': 10,
        'slow_ma': 30,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'max_positions': 2,
        'position_size': 0.4,  # 40% per position
    })
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    metrics = engine.run_backtest(
        strategy=strategy,
        symbols=['QQQ', 'IWM'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        commission=0.001,
        slippage=0.0005
    )
    
    # Print results
    engine.print_summary()
    
    return metrics


def compare_strategies():
    """Compare multiple strategies."""
    print("\n\nStrategy Comparison")
    print("="*50)
    
    # Run both strategies
    mr_metrics = backtest_mean_reversion()
    tf_metrics = backtest_trend_following()
    
    # Compare key metrics
    print(f"\n{'Metric':<20} {'Mean Reversion':>15} {'Trend Following':>15}")
    print("-"*50)
    print(f"{'Total Return':<20} {mr_metrics['total_return']:>14.1f}% {tf_metrics['total_return']:>14.1f}%")
    print(f"{'Sharpe Ratio':<20} {mr_metrics['sharpe_ratio']:>15.2f} {tf_metrics['sharpe_ratio']:>15.2f}")
    print(f"{'Max Drawdown':<20} {mr_metrics['max_drawdown']:>14.1f}% {tf_metrics['max_drawdown']:>14.1f}%")
    print(f"{'Win Rate':<20} {mr_metrics['win_rate']:>14.1%} {tf_metrics['win_rate']:>14.1%}")
    print(f"{'Total Trades':<20} {mr_metrics['total_trades']:>15} {tf_metrics['total_trades']:>15}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick backtest example")
    parser.add_argument("--strategy", choices=["mean_reversion", "trend_following", "compare"],
                        default="compare", help="Strategy to backtest")
    
    args = parser.parse_args()
    
    if args.strategy == "mean_reversion":
        backtest_mean_reversion()
    elif args.strategy == "trend_following":
        backtest_trend_following()
    else:
        compare_strategies()
        
    print("\nBacktest complete!")