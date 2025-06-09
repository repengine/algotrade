#!/usr/bin/env python3
"""
Test a single strategy configuration to debug issues.
"""

import logging
import sys
import traceback

import yfinance as yf

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, ".")

from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti


def test_mean_reversion():
    """Test a simple mean reversion strategy."""

    # Download data
    logger.info("Downloading SPY data...")
    spy = yf.Ticker("SPY")
    data = spy.history(start="2020-01-01", end="2023-01-01")

    # Convert to lowercase columns
    data.columns = data.columns.str.lower()

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data columns: {data.columns.tolist()}")
    logger.info(f"First few rows:\n{data.head()}")

    # Create strategy config
    config = {
        "symbols": ["SPY"],
        "lookback_period": 20,
        "rsi_period": 2,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
        "atr_band_mult": 2.5,
        "ma_exit_period": 10,
        "stop_loss_atr": 3.0,
    }

    try:
        # Initialize strategy
        logger.info("Initializing strategy...")
        strategy = MeanReversionEquity(config)
        strategy.init()

        # Test generating signals
        logger.info("Testing signal generation...")

        signals = []
        for i in range(50, len(data)):
            window_data = data.iloc[: i + 1].copy()
            window_data.attrs["symbol"] = "SPY"

            try:
                signal = strategy.next(window_data)
                if signal:
                    signals.append(
                        {
                            "date": data.index[i],
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                        }
                    )
                    logger.info(f"Signal on {data.index[i]}: {signal.direction}")
            except Exception as e:
                logger.error(f"Error at index {i}: {e}")
                traceback.print_exc()
                break

        logger.info(f"\nTotal signals generated: {len(signals)}")

        # Calculate simple returns
        if signals:
            logger.info("\nSimulating trades...")
            position = 0
            equity = 10000
            trades = []

            for i, signal in enumerate(signals):
                idx = data.index.get_loc(signal["date"])
                price = data["close"].iloc[idx]

                if signal["direction"] == "LONG" and position == 0:
                    # Buy
                    shares = int(equity * 0.95 / price)
                    position = shares
                    equity -= shares * price
                    trades.append(("BUY", signal["date"], price, shares))

                elif signal["direction"] == "FLAT" and position > 0:
                    # Sell
                    equity += position * price
                    trades.append(("SELL", signal["date"], price, position))
                    position = 0

            # Close final position
            if position > 0:
                final_price = data["close"].iloc[-1]
                equity += position * final_price
                trades.append(("SELL", data.index[-1], final_price, position))

            final_value = equity
            total_return = (final_value - 10000) / 10000 * 100

            logger.info("\nBacktest Results:")
            logger.info("Initial Capital: $10,000")
            logger.info(f"Final Value: ${final_value:,.2f}")
            logger.info(f"Total Return: {total_return:.2f}%")
            logger.info(f"Number of Trades: {len(trades)}")

            # Show first few trades
            logger.info("\nFirst 5 trades:")
            for trade in trades[:5]:
                logger.info(
                    f"{trade[0]} on {trade[1]} at ${trade[2]:.2f} - {trade[3]} shares"
                )

    except Exception as e:
        logger.error(f"Strategy initialization error: {e}")
        traceback.print_exc()


def test_trend_following():
    """Test trend following strategy."""

    logger.info("\n\nTesting Trend Following Strategy...")

    # Download data
    spy = yf.Ticker("SPY")
    data = spy.history(start="2020-01-01", end="2023-01-01")
    data.columns = data.columns.str.lower()

    config = {
        "symbols": ["SPY"],
        "channel_period": 20,
        "trail_period": 10,
        "fast_ma": 20,
        "slow_ma": 50,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "adx_period": 14,
        "adx_threshold": 25,
        "lookback_period": 252,
    }

    try:
        strategy = TrendFollowingMulti(config)
        strategy.init()

        # Test a single signal
        test_data = data.iloc[:100].copy()
        test_data.attrs["symbol"] = "SPY"

        signal = strategy.next(test_data)
        logger.info(f"Test signal: {signal}")

    except Exception as e:
        logger.error(f"Trend following error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_mean_reversion()
    test_trend_following()
