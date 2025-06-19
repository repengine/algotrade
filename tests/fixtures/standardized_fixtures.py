"""
Standardized test fixtures for AlgoStack.

This module provides comprehensive, reusable fixtures following the plan in
docs/planning/test-fix-implementation-plan.md
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

# Make pytest_asyncio optional
try:
    import pytest_asyncio
    HAS_PYTEST_ASYNCIO = True
except ImportError:
    pytest_asyncio = None
    HAS_PYTEST_ASYNCIO = False

from core.engine.order_manager import Order, OrderSide, OrderStatus, OrderType
from core.metrics import Trade

# ===============================
# Async Fixtures
# ===============================

# Only define async fixtures if pytest_asyncio is available
if HAS_PYTEST_ASYNCIO:
    # Don't override event_loop - let pytest-asyncio handle it
    # The asyncio_mode = auto in pytest.ini will create event loops as needed


    @pytest_asyncio.fixture
    async def async_order_manager():
        """
        Async order manager fixture for testing.

        Example:
            async def test_order_creation(async_order_manager):
                order = await async_order_manager.create_order(
                    symbol="AAPL",
                    order_type=OrderType.LIMIT,
                    side=OrderSide.BUY,
                    quantity=100,
                    price=150.00
                )
                assert order.status == OrderStatus.PENDING
        """
        from core.engine.order_manager import OrderManager

        # Create mock exchange connector
        exchange = AsyncMock()
        exchange.submit_order = AsyncMock(return_value=True)
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.get_order_status = AsyncMock(return_value=OrderStatus.FILLED)

        manager = OrderManager(exchange_connector=exchange)
        return manager


    @pytest_asyncio.fixture
    async def async_portfolio_engine():
        """
        Async portfolio engine fixture.

        Example:
            async def test_position_update(async_portfolio_engine):
                position = await async_portfolio_engine.update_position(
                    'AAPL', 100, 150.00
                )
                assert position['quantity'] == 100
        """
        from core.portfolio import PortfolioEngine

        portfolio = PortfolioEngine({'initial_capital': 100000})

        # Mock async methods
        portfolio.update_position = AsyncMock(
            return_value={'quantity': 100, 'avg_price': 150.00}
        )
        portfolio.calculate_metrics = AsyncMock(
            return_value={'total_value': 100000, 'positions_value': 15000}
        )

        return portfolio


# ===============================
# Trade Fixtures
# ===============================

@pytest.fixture
def sample_trades() -> List[Trade]:
    """
    Create a list of sample trades with various outcomes.

    Returns:
        List[Trade]: Mixed winning and losing trades

    Example:
        def test_metrics(sample_trades):
            metrics = BacktestMetrics()
            for trade in sample_trades:
                metrics.add_trade(trade)
            assert metrics.win_rate() > 0.5
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    trades = [
        # Winning trades
        Trade(
            timestamp=base_time,
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            commission=1.0,
            entry_time=base_time,
            exit_time=base_time + timedelta(days=5),
            entry_price=150.0,
            exit_price=155.0,
            pnl=498.0,  # (155-150)*100 - 2 commission
            pnl_percentage=3.32
        ),
        Trade(
            timestamp=base_time + timedelta(days=10),
            symbol='GOOGL',
            side='BUY',
            quantity=50,
            price=2800.0,
            commission=2.0,
            entry_time=base_time + timedelta(days=10),
            exit_time=base_time + timedelta(days=15),
            entry_price=2800.0,
            exit_price=2850.0,
            pnl=2496.0,  # (2850-2800)*50 - 4 commission
            pnl_percentage=1.78
        ),
        # Losing trade
        Trade(
            timestamp=base_time + timedelta(days=20),
            symbol='MSFT',
            side='BUY',
            quantity=75,
            price=400.0,
            commission=1.5,
            entry_time=base_time + timedelta(days=20),
            exit_time=base_time + timedelta(days=25),
            entry_price=400.0,
            exit_price=390.0,
            pnl=-753.0,  # (390-400)*75 - 3 commission
            pnl_percentage=-2.51
        ),
        # Another winning trade
        Trade(
            timestamp=base_time + timedelta(days=30),
            symbol='TSLA',
            side='SELL',  # Short trade
            quantity=30,
            price=250.0,
            commission=1.0,
            entry_time=base_time + timedelta(days=30),
            exit_time=base_time + timedelta(days=35),
            entry_price=250.0,
            exit_price=240.0,
            pnl=298.0,  # (250-240)*30 - 2 commission
            pnl_percentage=3.97
        )
    ]

    return trades


@pytest.fixture
def completed_trades(sample_trades) -> List[Trade]:
    """Trades with complete round-trip information."""
    return [t for t in sample_trades if t.pnl is not None]


# ===============================
# Order Fixtures
# ===============================

@pytest.fixture
def sample_orders() -> List[Order]:
    """
    Create sample orders for testing.

    Returns:
        List[Order]: Various order types and states

    Example:
        def test_order_manager(sample_orders):
            manager = OrderManager()
            for order in sample_orders:
                manager.orders[order.order_id] = order
            active = manager.get_active_orders()
            assert len(active) > 0
    """
    orders = [
        Order(
            order_id="ORD001",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
            status=OrderStatus.FILLED,
            filled_quantity=100.0,
            average_fill_price=150.50
        ),
        Order(
            order_id="ORD002",
            symbol="GOOGL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=50.0,
            price=2800.00,
            status=OrderStatus.PENDING
        ),
        Order(
            order_id="ORD003",
            symbol="MSFT",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=75.0,
            stop_price=380.00,
            status=OrderStatus.SUBMITTED
        ),
        Order(
            order_id="ORD004",
            symbol="TSLA",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=30.0,
            price=260.00,
            status=OrderStatus.CANCELLED
        )
    ]

    return orders


# ===============================
# Strategy Fixtures
# ===============================

@pytest.fixture
def mock_strategy():
    """
    Mock strategy for testing.

    Example:
        def test_signal_generation(mock_strategy, mock_market_data):
            signals = mock_strategy.calculate_signals(mock_market_data)
            assert 'AAPL' in signals
    """
    strategy = MagicMock()
    strategy.name = "TestStrategy"
    strategy.symbols = ['AAPL', 'GOOGL', 'MSFT']
    strategy.parameters = {
        'lookback_period': 20,
        'threshold': 2.0
    }
    strategy.positions = {'AAPL': 0, 'GOOGL': 0, 'MSFT': 0}

    # Mock signal generation
    strategy.calculate_signals = MagicMock(
        return_value={'AAPL': 0.8, 'GOOGL': -0.5, 'MSFT': 0.0}
    )
    strategy.init = MagicMock()
    strategy.next = strategy.calculate_signals

    return strategy


@pytest.fixture
def strategy_parameters() -> Dict[str, Any]:
    """
    Standard strategy parameters for testing.

    Example:
        def test_strategy_init(strategy_parameters):
            strategy = MeanReversionEquity(strategy_parameters)
            assert strategy.lookback_period == 20
    """
    return {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'lookback_period': 20,
        'zscore_threshold': 2.0,
        'exit_zscore': 0.5,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'position_size': 100,
        'use_kelly': False,
        'max_positions': 5,
        'stop_loss': 0.02,
        'take_profit': 0.05
    }


# ===============================
# Performance Fixtures
# ===============================

@pytest.fixture
def performance_metrics() -> Dict[str, float]:
    """
    Sample performance metrics for testing.

    Example:
        def test_metrics_display(performance_metrics):
            assert performance_metrics['sharpe_ratio'] > 1.0
            assert performance_metrics['max_drawdown'] < -5.0
    """
    return {
        'total_return': 25.5,
        'annual_return': 18.3,
        'sharpe_ratio': 1.45,
        'sortino_ratio': 2.1,
        'max_drawdown': -12.5,
        'calmar_ratio': 1.46,
        'win_rate': 0.65,
        'profit_factor': 2.3,
        'average_win': 850.0,
        'average_loss': 370.0,
        'expectancy': 425.0,
        'total_trades': 150,
        'recovery_factor': 3.2,
        'ulcer_index': 8.5
    }


@pytest.fixture
def equity_curve() -> pd.Series:
    """
    Sample equity curve for testing.

    Example:
        def test_drawdown_calc(equity_curve):
            drawdowns = calculate_drawdowns(equity_curve)
            assert drawdowns.min() < 0
    """
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    # Generate realistic equity curve with volatility and trend
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.015, len(dates))

    # Add some drawdown periods
    returns[50:70] = np.random.normal(-0.005, 0.02, 20)  # Drawdown period
    returns[150:160] = np.random.normal(-0.008, 0.025, 10)  # Another drawdown

    equity = 100000 * np.exp(np.cumsum(returns))

    return pd.Series(equity, index=dates, name='equity')


# ===============================
# Risk Management Fixtures
# ===============================

@pytest.fixture
def risk_limits() -> Dict[str, float]:
    """
    Standard risk limits for testing.

    Example:
        def test_risk_validation(risk_limits):
            risk_mgr = RiskManager(risk_limits)
            assert risk_mgr.max_position_size == 0.1
    """
    return {
        'max_position_size': 0.1,  # 10% of portfolio
        'max_portfolio_heat': 0.06,  # 6% total risk
        'max_correlation': 0.7,
        'max_leverage': 1.5,
        'max_drawdown': 0.15,  # 15%
        'daily_loss_limit': 0.02,  # 2%
        'position_limit': 10,
        'min_liquidity_ratio': 0.2
    }


@pytest.fixture
def mock_risk_validator():
    """
    Mock risk validator for testing.

    Example:
        async def test_order_validation(mock_risk_validator):
            is_valid, error = await mock_risk_validator.validate_order(order)
            assert is_valid
    """
    validator = AsyncMock()
    validator.validate_order = AsyncMock(return_value=(True, None))
    validator.check_position_limit = AsyncMock(return_value=True)
    validator.check_buying_power = AsyncMock(return_value=True)
    validator.check_daily_loss = AsyncMock(return_value=True)

    return validator


# ===============================
# Data Fixtures
# ===============================

@pytest.fixture
def multi_symbol_data() -> Dict[str, pd.DataFrame]:
    """
    Market data for multiple symbols.

    Example:
        def test_portfolio_calc(multi_symbol_data):
            portfolio = Portfolio()
            for symbol, data in multi_symbol_data.items():
                portfolio.update_price(symbol, data['close'].iloc[-1])
    """
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data = {}

    for i, symbol in enumerate(symbols):
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        n = len(dates)

        # Different characteristics for each symbol
        np.random.seed(42 + i)
        volatility = 0.015 + i * 0.003
        drift = 0.0003 + i * 0.0001

        returns = np.random.normal(drift, volatility, n)
        close = (100 + i * 50) * np.exp(np.cumsum(returns))

        df = pd.DataFrame(index=dates)
        df['open'] = close * (1 + np.random.uniform(-0.002, 0.002, n))
        df['high'] = np.maximum(df['open'], close) * (1 + np.abs(np.random.normal(0, 0.003, n)))
        df['low'] = np.minimum(df['open'], close) * (1 - np.abs(np.random.normal(0, 0.003, n)))
        df['close'] = close
        df['volume'] = np.random.lognormal(14 + i, 0.5, n).astype(int)

        df.attrs['symbol'] = symbol
        data[symbol] = df

    return data


# ===============================
# API Fixtures
# ===============================

@pytest.fixture
def api_client():
    """
    Test client for API testing.

    Example:
        def test_api_endpoint(api_client):
            response = api_client.get('/api/v1/positions')
            assert response.status_code == 200
    """
    from fastapi.testclient import TestClient

    from api.app import app

    return TestClient(app)


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """
    Authentication headers for API testing.

    Example:
        def test_protected_endpoint(api_client, auth_headers):
            response = api_client.get('/api/v1/orders', headers=auth_headers)
            assert response.status_code == 200
    """
    return {
        'Authorization': 'Bearer test_token_123',
        'X-API-Key': 'test_api_key'
    }


# ===============================
# Error Scenarios
# ===============================

@pytest.fixture
def connection_error():
    """Simulate connection errors."""
    error = ConnectionError("Unable to connect to broker")
    error.retry_after = 5
    return error


@pytest.fixture
def insufficient_funds_error():
    """Simulate insufficient funds error."""
    from core.exceptions import InsufficientCapitalError

    return InsufficientCapitalError(
        "Required: $15,000, Available: $10,000"
    )


@pytest.fixture
def rate_limit_error():
    """Simulate API rate limit error."""
    error = Exception("Rate limit exceeded")
    error.retry_after = 60
    error.reset_time = datetime.now() + timedelta(minutes=1)
    return error
