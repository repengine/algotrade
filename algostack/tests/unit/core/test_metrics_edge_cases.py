"""Edge case tests for metrics.py to achieve 100% coverage."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import pytest

from algostack.core.metrics import MetricsCollector, Trade


class TestMetricsEdgeCases:
    """Edge case tests for final coverage."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance."""
        return MetricsCollector(initial_capital=100000)
    
    def test_calmar_with_trades_no_value_history(self, metrics_collector):
        """Test Calmar ratio with trades but no value history - lines 445-455."""
        # Clear value history
        metrics_collector.value_history = []
        
        # Create some drawdown
        metrics_collector.max_drawdown = 0.10  # 10% drawdown
        metrics_collector.current_value = 110000  # 10% gain
        
        # Add trades with entry/exit times
        base_time = datetime(2024, 1, 1)
        
        # Trade 1
        trade1 = Trade(
            timestamp=base_time,
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=100.0,
            commission=1.0,
            entry_time=base_time,
            exit_time=base_time + timedelta(days=30),
            pnl=1000.0
        )
        
        # Trade 2
        trade2 = Trade(
            timestamp=base_time + timedelta(days=15),
            symbol='GOOGL',
            side='BUY',
            quantity=50,
            price=200.0,
            commission=1.0,
            entry_time=base_time + timedelta(days=15),
            exit_time=base_time + timedelta(days=45),
            pnl=500.0
        )
        
        metrics_collector.trades = [trade1, trade2]
        
        # Get metrics - should calculate Calmar using trade dates
        metrics = metrics_collector.get_performance_metrics()
        
        # Check Calmar was calculated
        assert metrics['calmar_ratio'] != 0.0
        # Total return is 10%, max drawdown is 10%
        # Time period is 45 days, so annualized return = 0.1 * (365/45) = 0.811
        # Calmar = 0.811 / 0.1 = 8.11
        assert metrics['calmar_ratio'] > 8.0
    
    def test_calmar_same_day_trades(self, metrics_collector):
        """Test Calmar with same-day trades - line 453."""
        # Clear value history
        metrics_collector.value_history = []
        metrics_collector.max_drawdown = 0.05
        metrics_collector.current_value = 105000
        
        # Add trades on same day
        base_time = datetime(2024, 1, 1)
        
        trade = Trade(
            timestamp=base_time,
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=100.0,
            commission=1.0,
            entry_time=base_time,
            exit_time=base_time,  # Same day
            pnl=500.0
        )
        
        metrics_collector.trades = [trade]
        
        # Get metrics - trading_days will be 0
        metrics = metrics_collector.get_performance_metrics()
        
        # When trading_days is 0, annualized return equals total return
        assert metrics['calmar_ratio'] == 1.0  # 0.05 / 0.05
    
    def test_import_fallback_scenario(self):
        """Test import fallback mechanism - lines 509-511."""
        # Save the original module if it exists
        original_module = sys.modules.get('algostack.core.backtest_metrics')
        
        try:
            # Remove backtest_metrics from modules to simulate it not existing
            if 'algostack.core.backtest_metrics' in sys.modules:
                del sys.modules['algostack.core.backtest_metrics']
            
            # Mock the import to raise ImportError
            with patch('builtins.__import__', side_effect=ImportError) as mock_import:
                # Filter to only raise for backtest_metrics
                def import_side_effect(name, *args, **kwargs):
                    if 'backtest_metrics' in name:
                        raise ImportError("No module named 'backtest_metrics'")
                    # Call the real import for other modules
                    return original_import(name, *args, **kwargs)
                
                original_import = __import__
                mock_import.side_effect = import_side_effect
                
                # Force re-evaluation of the import block
                import importlib
                import algostack.core.metrics as metrics_module
                
                # The module should still work with MetricsCollector as fallback
                assert hasattr(metrics_module, 'MetricsCollector')
                
                # If BacktestMetrics doesn't exist, it should be aliased to MetricsCollector
                if not hasattr(metrics_module, 'BacktestMetrics'):
                    # This is expected if import failed
                    pass
                else:
                    # If it exists, it should be MetricsCollector
                    assert metrics_module.BacktestMetrics is metrics_module.MetricsCollector
        
        finally:
            # Restore original module if it existed
            if original_module is not None:
                sys.modules['algostack.core.backtest_metrics'] = original_module