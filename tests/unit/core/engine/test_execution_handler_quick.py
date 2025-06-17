"""
Quick test for ExecutionHandler to check which methods need coverage.
"""

from unittest.mock import Mock

import pytest
from core.engine.execution_handler import (
    ExecutionAlgorithm,
    ExecutionHandler,
    ExecutionParams,
)


def test_execution_handler_basic():
    """Basic test to check coverage."""
    # Test initialization
    handler = ExecutionHandler()
    assert handler.executor is None
    assert handler.order_manager is None

    # Test with executor
    executor = Mock()
    handler = ExecutionHandler(executor=executor)
    assert handler.executor == executor

    # Test with order manager
    manager = Mock()
    handler = ExecutionHandler(order_manager=manager)
    assert handler.order_manager == manager

    # Test ExecutionParams
    params = ExecutionParams()
    assert params.algorithm == ExecutionAlgorithm.MARKET

    # Test ExecutionAlgorithm
    assert ExecutionAlgorithm.MARKET.value == "market"
    assert ExecutionAlgorithm.TWAP.value == "twap"

    # Test slippage calculation legacy
    handler = ExecutionHandler()
    slippage = handler.calculate_slippage(100.0, 101.0, 'BUY')
    assert slippage == 0.01

    slippage = handler.calculate_slippage(100.0, 99.0, 'SELL')
    assert slippage == 0.01

    # Test dict API
    slippage = handler.calculate_slippage(
        {'expected_price': 100, 'side': 'BUY'},
        {'avg_fill_price': 101, 'filled_quantity': 10}
    )
    assert slippage['amount'] == -1.0

    # Test get_execution_stats not found
    stats = handler.get_execution_stats("not_found")
    assert stats is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
