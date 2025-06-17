# AlgoStack Test Scaffolding Reference Guide

## Table of Contents
1. [Core Test Design Principles](#core-test-design-principles)
2. [Test Structure & Organization](#test-structure--organization)
3. [Essential Test Patterns](#essential-test-patterns)
4. [Mock Patterns & Examples](#mock-patterns--examples)
5. [Fixture Patterns](#fixture-patterns)
6. [Common Test Scenarios](#common-test-scenarios)
7. [Code Snippets & Templates](#code-snippets--templates)
8. [Best Practices Summary](#best-practices-summary)

## Core Test Design Principles

### FIRST Principles
Every test should be:
- **Fast**: Runs in milliseconds
- **Independent**: No shared state between tests
- **Repeatable**: Same result every time
- **Self-Validating**: Clear pass/fail
- **Thorough**: Cover edge cases

### AAA Pattern (Arrange-Act-Assert)
```python
def test_stop_loss_triggered():
    """
    Stop loss order is triggered when price drops below threshold.
    
    This test verifies that a stop loss order correctly triggers
    when the market price falls below the specified stop price.
    """
    # Arrange - Set up test data and state
    portfolio = Portfolio(capital=10000)
    position = Position(
        symbol="AAPL",
        quantity=100,
        entry_price=150.0,
        stop_loss=145.0
    )
    portfolio.add_position(position)
    market_data = MarketData(symbol="AAPL", price=144.0)  # Below stop
    
    # Act - Execute the behavior being tested
    orders = portfolio.check_stop_losses(market_data)
    
    # Assert - Verify the outcome
    assert len(orders) == 1
    assert orders[0].type == OrderType.MARKET
    assert orders[0].quantity == -100  # Sell order
    assert orders[0].reason == "Stop loss triggered at 145.0"
```

### Test Naming Convention
```python
# Pattern: test_[unit]_[scenario]_[expected_result]

def test_portfolio_add_position_increases_exposure():
    """When adding a position, total exposure increases."""

def test_risk_manager_correlation_check_flags_high_correlation():
    """When positions are highly correlated, risk check fails."""

def test_order_executor_market_order_fills_with_slippage():
    """When executing market order, fill includes slippage."""
```

## Test Structure & Organization

### Directory Structure
```bash
tests/
├── unit/                    # Fast, isolated tests (75%)
│   ├── core/               # Core module tests
│   │   ├── test_portfolio.py
│   │   ├── test_risk.py
│   │   └── test_executor.py
│   ├── strategies/         # Strategy tests
│   │   ├── test_base_strategy.py
│   │   └── test_mean_reversion.py
│   └── utils/              # Utility tests
├── integration/            # Component interaction tests (20%)
│   ├── test_data_pipeline.py
│   ├── test_trading_flow.py
│   └── test_backtest_full.py
├── e2e/                    # End-to-end tests (5%)
│   └── test_live_trading_simulation.py
├── benchmarks/             # Performance tests
│   └── test_backtest_performance.py
├── fixtures/               # Shared test data
│   ├── market_data.py
│   ├── portfolios.py
│   └── strategies.py
└── conftest.py            # Pytest configuration
```

### Pytest Configuration
```python
# pyproject.toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
python_classes = "Test*"

# Markers for test categories
markers = [
    "unit: Fast unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Tests that take > 1 second",
    "requires_market_data: Tests that need market data",
]

# Coverage settings
addopts = """
    --cov=algostack
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=95
    -v
"""
```

## Essential Test Patterns

### Pattern 1: Parameterized Tests
```python
import pytest

@pytest.mark.parametrize("price,quantity,expected", [
    (100, 10, 1000),      # Normal case
    (0.01, 1, 0.01),      # Penny stock
    (0, 10, 0),           # Zero price
    (100, 0, 0),          # Zero quantity
    (-100, 10, -1000),    # Short position
])
def test_position_value_calculation(price, quantity, expected):
    """Test position value across all scenarios."""
    position = Position('TEST', quantity, price)
    assert position.value == pytest.approx(expected)

# With descriptive IDs
@pytest.mark.parametrize("order_type,expected_fee", [
    ("MARKET", 0.001),
    ("LIMIT", 0.0008),
    ("STOP", 0.001),
], ids=["market_order", "limit_order", "stop_order"])
def test_commission_calculation_by_order_type(order_type, expected_fee):
    """Commission varies based on order type."""
    commission = calculate_commission(order_type, 10000)
    assert commission == pytest.approx(expected_fee * 10000)
```

### Pattern 2: Fixture Composition
```python
@pytest.fixture
def base_portfolio():
    """Base portfolio fixture."""
    return Portfolio(capital=100000)

@pytest.fixture  
def portfolio_with_positions(base_portfolio):
    """Portfolio with some positions."""
    base_portfolio.add_position('AAPL', 100, 150.0)
    base_portfolio.add_position('GOOGL', 50, 2500.0)
    return base_portfolio

@pytest.fixture
def portfolio_with_history(portfolio_with_positions):
    """Portfolio with transaction history."""
    portfolio_with_positions.close_position('AAPL', 155.0)
    portfolio_with_positions.add_position('MSFT', 200, 300.0)
    return portfolio_with_positions

# Parameterized fixture
@pytest.fixture(params=['bull', 'bear', 'sideways'])
def market_condition(request):
    """Different market conditions."""
    conditions = {
        'bull': {'trend': 'up', 'volatility': 'low'},
        'bear': {'trend': 'down', 'volatility': 'high'},
        'sideways': {'trend': 'flat', 'volatility': 'medium'}
    }
    return conditions[request.param]
```

### Pattern 3: Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    price=st.floats(min_value=0.01, max_value=10000, allow_nan=False),
    quantity=st.integers(min_value=1, max_value=1000000)
)
def test_position_value_properties(price, quantity):
    """
    Test position value calculation properties.
    
    Properties:
    1. Value is always positive for long positions
    2. Value equals price * quantity
    3. Value changes linearly with quantity
    """
    position = Position('TEST', quantity, price)
    
    # Property 1: Positive value
    assert position.value >= 0
    
    # Property 2: Calculation correctness
    assert position.value == pytest.approx(price * quantity)
    
    # Property 3: Linear scaling
    double_position = Position('TEST', quantity * 2, price)
    assert double_position.value == pytest.approx(position.value * 2)
```

## Mock Patterns & Examples

### Basic Mocking
```python
from unittest.mock import Mock, patch, MagicMock

def test_mock_external_api():
    """Mock external API calls."""
    # Create mock data fetcher
    mock_fetcher = Mock()
    mock_fetcher.fetch_data.return_value = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000000, 1100000, 1200000]
    })
    
    # Use mock in test
    data_handler = DataHandler(fetcher=mock_fetcher)
    data = data_handler.get_latest_data('AAPL')
    
    # Verify mock was called correctly
    mock_fetcher.fetch_data.assert_called_once_with('AAPL')
    assert len(data) == 3
```

### Patching External Dependencies
```python
@patch('algostack.adapters.yf_fetcher.yfinance')
def test_patch_third_party_library(mock_yfinance):
    """Patch third-party library imports."""
    # Configure mock
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({
        'Close': [150, 151, 152],
        'Volume': [1000000, 1100000, 1200000]
    })
    mock_yfinance.Ticker.return_value = mock_ticker
    
    # Test code that uses yfinance
    fetcher = YahooFinanceFetcher()
    data = fetcher.fetch_data('AAPL', '2024-01-01', '2024-01-03')
    
    # Verify
    mock_yfinance.Ticker.assert_called_with('AAPL')
    assert len(data) == 3
```

### Async Mocking
```python
from unittest.mock import AsyncMock

@pytest.fixture
async def async_order_manager():
    """Async order manager fixture."""
    exchange = AsyncMock()
    exchange.submit_order = AsyncMock(return_value=True)
    exchange.get_order_status = AsyncMock(return_value=OrderStatus.FILLED)
    
    manager = OrderManager(exchange_connector=exchange)
    return manager

async def test_async_order_creation(async_order_manager):
    """Test async order creation."""
    order = await async_order_manager.create_order(
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        price=150.00
    )
    assert order.status == OrderStatus.PENDING
```

### Mock with Side Effects
```python
def test_mock_with_side_effects():
    """Mock with changing behavior over multiple calls."""
    # Mock that returns different values each call
    mock_price_feed = Mock()
    mock_price_feed.get_price.side_effect = [100, 101, 99, 102]
    
    # Test code that calls get_price multiple times
    prices = []
    for _ in range(4):
        prices.append(mock_price_feed.get_price('AAPL'))
    
    assert prices == [100, 101, 99, 102]
    assert mock_price_feed.get_price.call_count == 4
```

## Fixture Patterns

### Market Data Fixtures
```python
@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 0,  # Will be calculated
        'low': 0,   # Will be calculated
        'close': 0, # Will be calculated
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Realistic OHLC relationships
    data['high'] = data['open'] * (1 + abs(np.random.randn(100)) * 0.01)
    data['low'] = data['open'] * (1 - abs(np.random.randn(100)) * 0.01)
    data['close'] = data['low'] + (data['high'] - data['low']) * np.random.rand(100)
    
    return data

@pytest.fixture
def volatile_market_data():
    """Generate volatile market data for stress testing."""
    # High volatility (5% daily moves)
    # Implementation here...

@pytest.fixture
def market_crash_data():
    """Generate market crash scenario data."""
    # Simulates a 20% drop over 5 days
    # Implementation here...
```

### Trade Fixtures
```python
@pytest.fixture
def sample_trades() -> List[Trade]:
    """Create a list of sample trades with various outcomes."""
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
        )
    ]
    
    return trades
```

### Strategy Fixtures
```python
@pytest.fixture
def mock_strategy():
    """Mock strategy for testing."""
    strategy = MagicMock()
    strategy.name = "TestStrategy"
    strategy.symbols = ['AAPL', 'GOOGL', 'MSFT']
    strategy.parameters = {
        'lookback_period': 20,
        'threshold': 2.0
    }
    
    # Mock signal generation
    strategy.calculate_signals = MagicMock(
        return_value={'AAPL': 0.8, 'GOOGL': -0.5, 'MSFT': 0.0}
    )
    
    return strategy

@pytest.fixture
def strategy_parameters() -> Dict[str, Any]:
    """Standard strategy parameters for testing."""
    return {
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'lookback_period': 20,
        'zscore_threshold': 2.0,
        'exit_zscore': 0.5,
        'rsi_period': 14,
        'position_size': 100,
        'max_positions': 5,
        'stop_loss': 0.02,
        'take_profit': 0.05
    }
```

## Common Test Scenarios

### Portfolio Management Tests
```python
class TestPortfolioManagement:
    """Test portfolio CRUD operations."""
    
    def test_add_long_position_reduces_cash(self, portfolio):
        """Adding a long position reduces available cash."""
        initial_cash = portfolio.cash
        
        position = portfolio.add_position(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            commission=0.001  # 0.1%
        )
        
        expected_cost = 100 * 150.0 * 1.001  # Including commission
        assert portfolio.cash == pytest.approx(initial_cash - expected_cost)
        assert position.cost_basis == pytest.approx(expected_cost)
    
    @pytest.mark.parametrize("quantity,price,commission,should_fail", [
        (100, 150.0, 0.001, False),   # Normal case
        (1000, 150.0, 0.001, True),   # Insufficient capital
        (0, 150.0, 0.001, True),      # Zero quantity
        (100, 0, 0.001, True),        # Zero price
        (-100, 150.0, 0.001, False),  # Short position (allowed)
    ])
    def test_add_position_validation(
        self, portfolio, quantity, price, commission, should_fail
    ):
        """Position creation validates inputs and capital requirements."""
        if should_fail:
            with pytest.raises((InsufficientCapitalError, InvalidOrderError)):
                portfolio.add_position('TEST', quantity, price, commission)
        else:
            position = portfolio.add_position('TEST', quantity, price, commission)
            assert position.symbol == 'TEST'
            assert position.quantity == quantity
```

### PnL Calculation Tests
```python
class TestPnLCalculations:
    """Test profit/loss calculation accuracy."""
    
    def test_unrealized_pnl_calculation(self, portfolio_with_positions):
        """Unrealized PnL reflects current market prices."""
        current_prices = {
            'AAPL': 160.0,   # +$10/share
            'GOOGL': 2550.0, # -$50/share
            'MSFT': 300.0    # Flat
        }
        
        pnl = portfolio_with_positions.calculate_unrealized_pnl(current_prices)
        
        expected_pnl = (
            100 * (160 - 150) +      # AAPL: +$1000
            50 * (2550 - 2600) +     # GOOGL: -$2500
            200 * (300 - 300)        # MSFT: $0
        )
        
        assert pnl == pytest.approx(expected_pnl)
        assert pnl == -1500  # Net loss
```

### Risk Management Tests
```python
class TestRiskManagement:
    """Test risk calculations and limits."""
    
    def test_position_concentration_limit(self):
        """Portfolio enforces position concentration limits."""
        portfolio = Portfolio(
            capital=100000,
            max_position_size=0.2  # 20% max
        )
        
        # This should succeed (20% of capital)
        portfolio.add_position('AAPL', 100, 200.0)
        
        # This should fail (would be 40% of capital)
        with pytest.raises(InvalidOrderError) as exc_info:
            portfolio.add_position('GOOGL', 20, 2000.0)
        
        assert "exceeds maximum position size" in str(exc_info.value)
```

### Strategy Tests
```python
class TestMeanReversionStrategy:
    """Test mean reversion strategy implementation."""
    
    def test_signal_generation_ranging_market(self, strategy, ranging_data):
        """Strategy generates signals in ranging market."""
        signals = strategy.generate_signals(ranging_data)
        
        # Should generate multiple signals in ranging market
        assert len(signals) > 0
        
        # Verify signal logic
        for signal in signals:
            idx = ranging_data.index.get_loc(signal.timestamp)
            price = ranging_data.iloc[idx]['close']
            
            # Calculate bands at signal time
            window = ranging_data.iloc[max(0, idx-20):idx]
            mean = window['close'].mean()
            std = window['close'].std()
            
            if signal.signal_type == SignalType.BUY:
                # Buy signal should be near lower band
                assert price < mean - 1.5 * std
            elif signal.signal_type == SignalType.SELL:
                # Sell signal should be near upper band  
                assert price > mean + 1.5 * std
```

### Integration Test Example
```python
@pytest.mark.integration
def test_signal_to_execution_flow(trading_system):
    """Test complete flow from signal to executed order."""
    portfolio = trading_system['portfolio']
    risk = trading_system['risk']
    executor = trading_system['executor']
    
    # Create a buy signal
    signal = Signal(
        symbol='AAPL',
        signal_type=SignalType.BUY,
        strength=0.8,
        timestamp=datetime.now(),
        metadata={'price': 150.0}
    )
    
    # Risk check
    position_size = risk.calculate_position_size(
        portfolio=portfolio,
        signal=signal,
        current_price=150.0
    )
    
    assert position_size > 0  # Risk approved
    assert position_size <= portfolio.cash / 150.0  # Within cash limits
    
    # Execute order
    order = {
        'symbol': signal.symbol,
        'quantity': position_size,
        'order_type': 'MARKET',
        'price': 150.0
    }
    
    fill = executor.execute_order(order)
    
    # Update portfolio
    portfolio.add_position(
        symbol=fill['symbol'],
        quantity=fill['quantity'],
        price=fill['price'],
        commission=fill['commission']
    )
    
    # Verify portfolio state
    assert 'AAPL' in portfolio.positions
    assert portfolio.positions['AAPL'].quantity == position_size
    assert portfolio.cash < 100000  # Cash reduced
```

## Code Snippets & Templates

### Test Module Template
```python
"""
Tests for [module name].

Test Categories:
- [Category 1]: [Description]
- [Category 2]: [Description]
- [Category 3]: [Description]
"""

# Standard library imports
from datetime import datetime, timedelta
from decimal import Decimal

# Third-party imports
import pytest
import pandas as pd
import numpy as np
from freezegun import freeze_time

# Local imports
from algostack.core.module import Class1, Class2
from algostack.core.exceptions import CustomError

# Module constants
DEFAULT_CAPITAL = 100000
TEST_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT']


class TestCategory1:
    """Tests for [category description]."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {...}
    
    def test_specific_behavior(self, setup_data):
        """
        [Component] should [behavior] when [condition].
        
        This test verifies that [detailed explanation].
        """
        # Arrange
        data = setup_data
        component = Component(data)
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected_value
        assert component.state == expected_state
```

### Performance Test Template
```python
@pytest.mark.benchmark
def test_performance_criteria():
    """Verify component meets performance criteria."""
    import time
    
    # Setup large dataset
    data = generate_large_dataset(1_000_000)
    
    # Time the operation
    start_time = time.time()
    result = process_data(data)
    execution_time = time.time() - start_time
    
    # Performance assertions
    assert execution_time < 2.0  # Must complete in 2 seconds
    
    # Calculate throughput
    items_per_second = len(data) / execution_time
    assert items_per_second > 500_000  # Process >500k items/second
```

### Error Testing Template
```python
def test_error_handling():
    """Component handles errors gracefully."""
    # Test specific exception type and message
    with pytest.raises(ValueError) as exc_info:
        Portfolio(capital=-1000)
    
    assert "capital must be positive" in str(exc_info.value)
    
    # Test multiple acceptable errors
    with pytest.raises((ValueError, TypeError)):
        process_invalid_input(None)
    
    # Test error attributes
    with pytest.raises(PortfolioError) as exc_info:
        portfolio.add_position('AAPL', 10000, 150.0)
    
    error = exc_info.value
    assert error.error_code == 'INSUFFICIENT_CAPITAL'
    assert error.required_capital == 1500000
```

## Best Practices Summary

### 1. Strong Assertions
```python
# ❌ AVOID: Weak assertions
assert result is not None
assert isinstance(result, dict)
assert len(result) > 0

# ✅ PREFER: Strong assertions
assert result == {
    'total_value': 100000,
    'positions': 3,
    'exposure': 0.75,
    'cash': 25000
}
assert result['sharpe_ratio'] == pytest.approx(1.5, rel=1e-2)
assert 0 <= result['win_rate'] <= 1
```

### 2. Test Documentation
```python
def test_complex_calculation():
    """
    Verify Sharpe ratio calculation handles edge cases correctly.
    
    The Sharpe ratio is calculated as:
        Sharpe = (E[R] - Rf) / σ
    
    Where:
        E[R] = Expected return
        Rf = Risk-free rate (assumed 0)
        σ = Standard deviation of returns
    
    Edge cases tested:
    1. Zero volatility → Infinity (capped at 999)
    2. Negative returns → Negative Sharpe
    3. Single return → Undefined (return 0)
    """
    # Test implementation...
```

### 3. Test Isolation
```python
# Each test should be completely independent
def test_independent_1():
    """Test 1 creates its own data."""
    portfolio = Portfolio(capital=100000)  # Fresh instance
    # Test logic...

def test_independent_2():
    """Test 2 is not affected by test 1."""
    portfolio = Portfolio(capital=100000)  # Another fresh instance
    # Test logic...
```

### 4. Use Markers
```python
@pytest.mark.unit
def test_fast_calculation():
    """Fast unit test."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_workflow():
    """Slow integration test."""
    pass

@pytest.mark.requires_market_data
def test_with_real_data():
    """Test requiring external data."""
    pass
```

### 5. Helpful Error Messages
```python
def test_with_helpful_errors():
    """Provide context when assertions fail."""
    positions = portfolio.get_positions()
    
    # Bad: No context on failure
    assert len(positions) == 3
    
    # Good: Provides debugging info
    assert len(positions) == 3, f"Expected 3 positions, got {len(positions)}: {positions}"
```

## Quick Commands Reference

```bash
# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=algostack --cov-report=html

# Run specific test file
pytest tests/unit/core/test_portfolio.py

# Run tests matching pattern
pytest -k "test_risk"

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto

# Run only failed tests from last run
pytest --lf

# Run tests and drop into debugger on failure
pytest --pdb

# Profile slow tests
pytest --durations=10
```

## Coverage Configuration

```ini
# .coveragerc
[run]
source = algostack
omit = 
    */tests/*
    */test_*
    */conftest.py
    */__pycache__/*
    */venv/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

## Test Quality Checklist

- [ ] Test has descriptive name following convention
- [ ] Test has docstring explaining what it tests
- [ ] Test follows AAA pattern (Arrange-Act-Assert)
- [ ] Test uses strong assertions (exact values, not just types)
- [ ] Test is independent (no shared state)
- [ ] Test covers edge cases and error conditions
- [ ] Test uses appropriate fixtures to reduce duplication
- [ ] Test is properly categorized with markers
- [ ] Test runs quickly (< 1 second for unit tests)
- [ ] Test provides helpful error messages on failure

---

*This guide consolidates all test patterns from the AlgoStack test design overhaul. For detailed examples and explanations, refer to the original documents in docs/planning/*