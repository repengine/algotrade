#!/usr/bin/env python3
"""Test data connections and API keys."""

import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algostack.core.data_handler import DataHandler


def test_yfinance():
    """Test Yahoo Finance connection."""
    print("Testing Yahoo Finance...")
    try:
        handler = DataHandler(["yfinance"])
        df = handler.get_historical(
            "SPY", datetime.now() - timedelta(days=30), datetime.now(), interval="1d"
        )
        print(f"✓ Yahoo Finance: Retrieved {len(df)} days of SPY data")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        return True
    except Exception as e:
        print(f"✗ Yahoo Finance error: {e}")
        return False


def test_alphavantage():
    """Test Alpha Vantage connection."""
    print("\nTesting Alpha Vantage (Premium)...")
    try:
        handler = DataHandler(["alphavantage"], premium_av=True)

        if "alphavantage" not in handler.adapters:
            print("✗ Alpha Vantage: No API key found")
            return False

        df = handler.get_historical(
            "SPY", datetime.now() - timedelta(days=30), datetime.now(), interval="1d"
        )
        print(f"✓ Alpha Vantage: Retrieved {len(df)} days of SPY data")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        print("  Premium features enabled with realtime data")
        return True
    except Exception as e:
        print(f"✗ Alpha Vantage error: {e}")
        return False


def main():
    """Run all tests."""
    print("AlgoStack Data Connection Test")
    print("=" * 40)

    yf_ok = test_yfinance()
    av_ok = test_alphavantage()

    print("\n" + "=" * 40)
    print("Summary:")
    print(f"  Yahoo Finance: {'✓ OK' if yf_ok else '✗ Failed'}")
    print(f"  Alpha Vantage: {'✓ OK' if av_ok else '✗ Failed'}")

    if yf_ok or av_ok:
        print("\nYou can run backtests with:")
        if yf_ok:
            print("  python run_comprehensive_backtest.py --source yfinance")
        if av_ok:
            print("  python run_comprehensive_backtest.py --source alphavantage")
    else:
        print("\nNo data sources available!")


if __name__ == "__main__":
    main()
