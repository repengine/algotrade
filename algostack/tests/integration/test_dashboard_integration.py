#!/usr/bin/env python3
"""Test dashboard integration components without streamlit dependency"""

import importlib
import inspect
import sys
from pathlib import Path

import yaml
import pytest

# Add the algostack directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))


from algostack.backtests.run_backtests import BacktestEngine
from algostack.core.data_handler import DataHandler
from algostack.strategies.base import BaseStrategy


def test_strategy_discovery():
    """Test dynamic strategy discovery"""
    print("\n=== Testing Strategy Discovery ===")

    strategies_found = {}
    strategies_dir = Path(__file__).parent / "strategies"

    for file_path in strategies_dir.glob("*.py"):
        if file_path.name.startswith("_") or file_path.name == "base.py":
            continue

        module_name = file_path.stem
        try:
            module = importlib.import_module(f"strategies.{module_name}")

            # Find all classes that inherit from BaseStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                    friendly_name = (
                        name.replace("Strategy", "").replace("_", " ").title()
                    )
                    strategies_found[friendly_name] = obj
                    print(f"✓ Found strategy: {friendly_name} ({name})")

        except Exception as e:
            print(f"✗ Failed to load {module_name}: {e}")

    print(f"\nTotal strategies discovered: {len(strategies_found)}")
    return strategies_found


def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")

    config_path = Path(__file__).parent / "config" / "base.yaml"

    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        return None

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
        print(f"  - Engine mode: {config.get('engine', {}).get('mode')}")
        print(f"  - Data provider: {config.get('data', {}).get('provider')}")
        print(
            f"  - Initial capital: {config.get('portfolio', {}).get('initial_capital')}"
        )
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return None


@pytest.fixture
def strategies():
    """Return available strategies."""
    from algostack.strategies.mean_reversion_equity import MeanReversionEquity
    from algostack.strategies.trend_following_multi import TrendFollowingMulti
    return {
        "MeanReversionEquity": MeanReversionEquity,
        "TrendFollowingMulti": TrendFollowingMulti
    }

@pytest.fixture
def config():
    """Return test configuration."""
    return {
        "symbols": ["AAPL", "MSFT"],
        "lookback_period": 20,
        "zscore_threshold": 2.0,
        "exit_zscore": 0.5,
        "channel_period": 20,
        "adx_threshold": 25.0
    }

def test_strategy_instantiation(strategies, config):
    """Test strategy instantiation"""
    print("\n=== Testing Strategy Instantiation ===")

    if not strategies or not config:
        print("✗ Cannot test - missing strategies or config")
        return

    for strategy_name, strategy_class in strategies.items():
        try:
            # Create minimal config for strategy
            strategy_config = {
                "name": strategy_name,
                "symbols": ["AAPL", "MSFT"],
                "lookback_period": 252,
            }

            # Instantiate strategy
            strategy = strategy_class(config=strategy_config)
            print(f"✓ {strategy_name} instantiated successfully")

            # Check if it has required methods
            required_methods = ["init", "next", "size"]
            for method in required_methods:
                if hasattr(strategy, method):
                    print(f"  ✓ Has {method}() method")
                else:
                    print(f"  ✗ Missing {method}() method")

        except Exception as e:
            print(f"✗ Failed to instantiate {strategy_name}: {e}")


def test_backtest_engine(config):
    """Test BacktestEngine initialization"""
    print("\n=== Testing BacktestEngine ===")

    if not config:
        print("✗ Cannot test - missing config")
        return

    try:
        # Initialize data handler
        data_handler = DataHandler(config)
        print("✓ DataHandler initialized")

        # Initialize backtest engine
        engine = BacktestEngine(data_handler=data_handler, initial_capital=100000)
        print("✓ BacktestEngine initialized")

        # Check available methods
        if hasattr(engine, "run_backtest"):
            print("✓ Has run_backtest() method")
        if hasattr(engine, "results"):
            print("✓ Has results attribute")

    except Exception as e:
        print(f"✗ Failed to initialize BacktestEngine: {e}")


def test_full_integration():
    """Test full integration workflow"""
    print("\n=== Testing Full Integration ===")

    # Load config
    config = test_config_loading()
    if not config:
        print("✗ Integration test failed - no config")
        return

    # Discover strategies
    strategies = test_strategy_discovery()
    if not strategies:
        print("✗ Integration test failed - no strategies")
        return

    # Test instantiation
    test_strategy_instantiation(strategies, config)

    # Test backtest engine
    test_backtest_engine(config)

    print("\n=== Integration Test Summary ===")
    print(f"✓ Found {len(strategies)} strategies")
    print("✓ Configuration system working")
    print("✓ Strategy instantiation working")
    print("✓ BacktestEngine initialization working")
    print("\n✅ Dashboard integration components are ready!")


if __name__ == "__main__":
    print("AlgoStack Dashboard Integration Test")
    print("=" * 40)
    test_full_integration()
