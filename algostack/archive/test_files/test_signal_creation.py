#!/usr/bin/env python3
"""Test Signal object creation directly."""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

from strategies.base import Signal


def test_signal_creation():
    """Test if Signal objects can be created properly."""

    print("Testing Signal object creation...")

    try:
        # Create a test signal
        signal = Signal(
            timestamp=datetime.now(),
            symbol="SPY",
            direction="LONG",
            strength=0.5,
            strategy_id="TestStrategy",
            price=599.0,
            atr=0.01,
            metadata={"reason": "test", "edge": 0.1},
        )

        print(f"✓ Signal created successfully: {signal}")
        print(f"  Direction: {signal.direction}")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Strength: {signal.strength}")

    except Exception as e:
        print(f"✗ Error creating signal: {e}")
        import traceback

        traceback.print_exc()


def test_with_bad_data():
    """Test Signal with invalid data."""
    print("\nTesting Signal with invalid strength...")

    try:
        # This should fail validation
        signal = Signal(
            timestamp=datetime.now(),
            symbol="SPY",
            direction="LONG",
            strength=-0.5,  # Invalid for LONG
            strategy_id="TestStrategy",
            price=599.0,
        )
        print(f"Signal created (shouldn't happen): {signal}")
    except Exception as e:
        print(f"✓ Correctly rejected invalid signal: {e}")


if __name__ == "__main__":
    test_signal_creation()
    test_with_bad_data()
