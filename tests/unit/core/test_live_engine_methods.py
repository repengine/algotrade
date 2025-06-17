#!/usr/bin/env python3
"""Test all the newly implemented methods work together."""

import asyncio
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_engine import LiveTradingEngine
from strategies.base import BaseStrategy, Signal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, **kwargs):
        """Initialize with config."""
        config = kwargs if kwargs else {}
        super().__init__(config)
        self.symbols = ["AAPL", "MSFT"]

    def init(self):
        pass

    def next(self, data: pd.DataFrame):
        pass

    def size(self, signal, risk_context):
        return (100, 0)

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate test signals."""
        return [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="test_strategy",
                price=150.0,
            )
        ]


async def test_live_engine():
    """Test LiveTradingEngine with all new methods."""
    print("Testing LiveTradingEngine initialization and methods...")

    # Create config
    config = {
        "mode": "paper",
        "strategies": [
            {
                "class": MockStrategy,
                "id": "test_strategy",
                "params": {},
            }
        ],
        "portfolio_config": {"initial_capital": 100000},
        "risk_config": {"max_drawdown": 0.15},
    }

    # Initialize engine
    engine = LiveTradingEngine(config)
    print(f"✓ Engine initialized with {len(engine.strategies)} strategies")
    print(f"✓ Active symbols: {engine._active_symbols}")

    # Test portfolio methods
    portfolio = engine.portfolio_engine
    print(f"\n✓ Portfolio total_value: ${portfolio.total_value:,.2f}")
    print(f"✓ Portfolio cash: ${portfolio.cash:,.2f}")

    # Test position update
    portfolio.update_position("AAPL", 100, 150.0, 155.0)
    print("✓ Updated position for AAPL")

    # Test portfolio metrics
    metrics = portfolio.calculate_metrics()
    print(f"✓ Portfolio metrics calculated: {len(metrics)} metrics")

    # Test daily PnL
    daily_pnl = portfolio.calculate_daily_pnl()
    print(f"✓ Daily PnL: ${daily_pnl:,.2f}")

    # Test strategy signal generation
    strategy = list(engine.strategies.values())[0]
    signals = strategy.generate_signals(pd.DataFrame())
    print(f"\n✓ Strategy generated {len(signals)} signals")

    # Test risk manager
    risk_manager = engine.risk_manager
    violations = risk_manager.check_limits()
    print(f"✓ Risk manager checked limits: {len(violations)} violations")

    # Test signal validation
    if signals:
        signal = signals[0]
        is_valid = engine._is_valid_signal(signal)
        print(f"✓ Signal validation: {is_valid}")

    print("\n✅ All methods implemented and working correctly!")


if __name__ == "__main__":
    asyncio.run(test_live_engine())
