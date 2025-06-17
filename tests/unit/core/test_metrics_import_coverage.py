"""Test to cover the import fallback lines in metrics.py"""

import importlib
import sys
from unittest.mock import patch


def test_backtest_metrics_import_fallback():
    """Test the ImportError fallback for BacktestMetrics."""
    # Save original modules
    original_modules = {}
    modules_to_save = ['core.metrics', 'algostack.core.backtest_metrics', 'algostack.core']
    for mod in modules_to_save:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]

    try:
        # Remove cached imports
        for key in list(sys.modules.keys()):
            if 'core' in key:
                del sys.modules[key]

        # Create a mock that raises ImportError for backtest_metrics
        def mock_import(name, *args, **kwargs):
            if name == 'core.backtest_metrics' or name.endswith('.backtest_metrics'):
                raise ImportError("No module named 'backtest_metrics'")
            # Use the real import for everything else
            return importlib.__import__(name, *args, **kwargs)

        # Patch the import statement
        with patch('builtins.__import__', side_effect=mock_import):
            # Import metrics module - this should trigger the ImportError branch
            from core import metrics

            # Verify that BacktestMetrics is aliased to MetricsCollector
            assert metrics.BacktestMetrics is metrics.MetricsCollector

            # Verify the classes work as expected
            collector = metrics.BacktestMetrics(initial_capital=100000)
            assert collector.initial_capital == 100000

    finally:
        # Restore original modules
        for mod, original in original_modules.items():
            sys.modules[mod] = original


def test_backtest_metrics_import_success():
    """Test successful import of BacktestMetrics."""
    # Remove any cached imports
    modules_to_remove = [key for key in sys.modules if 'core.metrics' in key]
    for module in modules_to_remove:
        del sys.modules[module]

    # Normal import should work
    import core.metrics as metrics

    # BacktestMetrics should be available
    assert hasattr(metrics, 'BacktestMetrics')
    assert hasattr(metrics, 'MetricsCollector')
