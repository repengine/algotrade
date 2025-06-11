# AlgoStack Test Design Overhaul: Complete Analysis and Implementation Plan

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Test Design Principles](#test-design-principles)
4. [Phased Implementation Plan](#phased-implementation-plan)
5. [Test Patterns and Templates](#test-patterns-and-templates)
6. [Best Practices Guide](#best-practices-guide)
7. [Metrics and Success Criteria](#metrics-and-success-criteria)

## Executive Summary

This document provides a comprehensive analysis of AlgoStack's test suite and a detailed plan to transform it into a professional-grade testing framework. The goal is to achieve:

- **100% code coverage** with meaningful tests
- **Zero test duplication** through proper organization
- **Fast test execution** through proper categorization
- **Easy maintenance** through consistent patterns
- **Confidence in deployment** through comprehensive validation

## Current State Analysis

### Test Structure Overview

```
Current Structure:
algostack/
├── tests/                    # Main test directory
│   ├── test_*.py            # Unit tests (inconsistent)
│   └── test_*_comprehensive.py  # Coverage-focused tests
├── test_files/              # Mixed scripts and tests
│   ├── quick_tests/         # Ad-hoc test scripts
│   └── *.py                 # Various test utilities
└── conftest.py              # Shared fixtures
```

### Key Issues Identified

#### 1. **Organizational Chaos**
```python
# Example: Multiple files testing the same module
tests/test_portfolio.py
tests/test_portfolio_engine.py
tests/test_portfolio_comprehensive.py
tests/test_portfolio_100_coverage.py
tests/test_portfolio_engine_comprehensive.py
```

**Impact**: Developers don't know which test to run or update.

#### 2. **Weak Test Design**
```python
# Current pattern - weak assertions
def test_calculate_metrics():
    """Test metrics calculation."""
    metrics = calculate_metrics(data)
    assert metrics is not None  # Weak!
    assert isinstance(metrics, dict)  # Still weak!
```

**Impact**: Tests pass but don't verify correctness.

#### 3. **No Test Categories**
```python
# All tests run together - no way to run "just unit tests"
def test_full_backtest():  # Slow integration test
    ...

def test_sharpe_calculation():  # Fast unit test
    ...
```

**Impact**: Can't run quick tests during development.

#### 4. **Limited Pytest Features**
```python
# Current: Repetitive test code
def test_order_market():
    order = Order(type="MARKET", ...)
    # test logic

def test_order_limit():
    order = Order(type="LIMIT", ...)
    # same test logic repeated
```

**Impact**: Maintenance nightmare, missed edge cases.

### Coverage Analysis

```
Module                  Current Coverage    Quality Score
─────────────────────────────────────────────────────────
core/portfolio.py              95%              C-
core/risk.py                   88%              D+
core/executor.py               92%              C
strategies/base.py             78%              D
dashboard.py                   45%              F
```

**Quality Score Factors**:
- Assertion strength
- Test isolation
- Edge case coverage
- Documentation quality

## Test Design Principles

### 1. **The Testing Pyramid**

```
         /\
        /  \       E2E Tests (5%)
       /────\      - Full system workflows
      /      \     - Production-like environment
     /────────\    
    /          \   Integration Tests (20%)
   /────────────\  - Component interactions
  /              \ - External dependencies
 /────────────────\
/                  \ Unit Tests (75%)
────────────────────── - Fast, isolated, specific
```

### 2. **FIRST Principles**

Every test should be:

```python
# Fast - Runs in milliseconds
@pytest.mark.unit
def test_calculate_position_value():
    """Calculate position value: price * quantity."""
    position = Position(price=100.0, quantity=10)
    assert position.value == 1000.0  # Instant calculation

# Independent - No shared state
def test_risk_check_isolated():
    """Each test creates its own data."""
    portfolio = Portfolio()  # Fresh instance
    risk = RiskManager(portfolio)  # No global state
    
# Repeatable - Same result every time
@freeze_time("2024-01-01")  # Fixed time
def test_daily_returns():
    """Returns calculation is deterministic."""
    # No random data, no external dependencies
    
# Self-Validating - Clear pass/fail
def test_position_limit_enforcement():
    """Position limit is enforced."""
    risk = RiskManager(max_position=0.1)
    size = risk.calculate_position_size(capital=10000, price=100)
    assert size == 10  # Exactly 10% = 10 shares
    
# Thorough - Covers edge cases
@pytest.mark.parametrize("price,quantity,expected", [
    (100, 10, 1000),      # Normal case
    (0.01, 1, 0.01),      # Penny stock
    (0, 10, 0),           # Zero price
    (100, 0, 0),          # Zero quantity
    (-100, 10, -1000),    # Short position
])
def test_position_value_calculation(price, quantity, expected):
    """Test position value across all scenarios."""
```

### 3. **AAA Pattern (Arrange-Act-Assert)**

```python
def test_stop_loss_triggered():
    """
    Stop loss order is triggered when price drops below threshold.
    
    This test verifies that a stop loss order correctly triggers
    when the market price falls below the specified stop price,
    accounting for slippage in the execution.
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

### 4. **Test Naming Conventions**

```python
# Pattern: test_[unit]_[scenario]_[expected_result]

def test_portfolio_add_position_increases_exposure():
    """When adding a position, total exposure increases."""

def test_risk_manager_correlation_check_flags_high_correlation():
    """When positions are highly correlated, risk check fails."""

def test_order_executor_market_order_fills_with_slippage():
    """When executing market order, fill includes slippage."""

# For edge cases and errors
def test_strategy_generate_signals_handles_empty_data():
    """When market data is empty, strategy returns no signals."""

def test_position_sizing_with_zero_capital_raises_error():
    """When capital is zero, position sizing raises ValueError."""
```

## Phased Implementation Plan

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish test infrastructure and patterns

#### 1.1 Create Test Structure
```bash
# New structure
tests/
├── unit/                    # Fast, isolated tests
│   ├── core/               # Core module tests
│   │   ├── test_portfolio.py
│   │   ├── test_risk.py
│   │   └── test_executor.py
│   ├── strategies/         # Strategy tests
│   │   ├── test_base_strategy.py
│   │   └── test_mean_reversion.py
│   └── utils/              # Utility tests
├── integration/            # Component interaction tests
│   ├── test_data_pipeline.py
│   ├── test_trading_flow.py
│   └── test_backtest_full.py
├── e2e/                    # End-to-end tests
│   └── test_live_trading_simulation.py
├── benchmarks/             # Performance tests
│   └── test_backtest_performance.py
├── fixtures/               # Shared test data
│   ├── market_data.py
│   ├── portfolios.py
│   └── strategies.py
└── conftest.py            # Pytest configuration
```

#### 1.2 Configure Pytest
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
    "unit: Fast unit tests (deselect with '-m \"not unit\"')",
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

# Test discovery
norecursedirs = [
    ".git",
    ".tox",
    "dist",
    "build",
    "*.egg",
]
```

#### 1.3 Create Base Fixtures
```python
# tests/fixtures/market_data.py
"""
Market data fixtures for testing.

This module provides reusable market data fixtures that ensure
consistent test data across the test suite.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """
    Generate sample OHLCV data for testing.
    
    Returns a DataFrame with 100 days of price data with:
    - Realistic price movements
    - Volume patterns
    - No missing data
    
    Example:
        def test_strategy(sample_ohlcv_data):
            strategy = MyStrategy()
            signals = strategy.generate_signals(sample_ohlcv_data)
            assert len(signals) > 0
    """
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
    """
    Generate volatile market data for stress testing.
    
    Features:
    - High volatility (5% daily moves)
    - Price gaps
    - Volume spikes
    
    Use this for testing risk management and edge cases.
    """
    # Implementation here...


@pytest.fixture
def market_crash_data():
    """
    Generate market crash scenario data.
    
    Simulates a 20% drop over 5 days with:
    - Increasing volume
    - Widening spreads
    - Gap downs
    """
    # Implementation here...
```

### Phase 2: Unit Test Overhaul (Week 3-4)
**Goal**: Rewrite unit tests with proper patterns

#### 2.1 Portfolio Tests Example
```python
# tests/unit/core/test_portfolio.py
"""
Unit tests for Portfolio class.

Tests cover:
- Position management
- PnL calculations
- Risk metrics
- State transitions
"""
import pytest
from decimal import Decimal
from datetime import datetime
from freezegun import freeze_time

from algostack.core.portfolio import Portfolio, Position
from algostack.core.exceptions import (
    InsufficientCapitalError,
    PositionNotFoundError,
    InvalidOrderError
)


class TestPortfolioConstruction:
    """Test portfolio initialization and configuration."""
    
    def test_portfolio_init_default_values(self):
        """
        Portfolio initializes with default values.
        
        Default portfolio should have:
        - Starting capital as current cash
        - Empty positions
        - Zero exposure
        """
        portfolio = Portfolio(capital=10000)
        
        assert portfolio.initial_capital == 10000
        assert portfolio.cash == 10000
        assert portfolio.positions == {}
        assert portfolio.total_value == 10000
        assert portfolio.exposure == 0
    
    def test_portfolio_init_with_positions(self):
        """
        Portfolio can be initialized with existing positions.
        
        This supports resuming from a saved state.
        """
        positions = {
            'AAPL': Position('AAPL', 100, 150.0),
            'GOOGL': Position('GOOGL', 50, 2500.0)
        }
        portfolio = Portfolio(capital=50000, positions=positions)
        
        assert len(portfolio.positions) == 2
        assert portfolio.cash == 50000  # Cash tracked separately
        assert portfolio.exposure == 140000  # 100*150 + 50*2500


class TestPositionManagement:
    """Test adding, updating, and removing positions."""
    
    @pytest.fixture
    def portfolio(self):
        """Standard portfolio for position tests."""
        return Portfolio(capital=100000)
    
    def test_add_long_position_reduces_cash(self, portfolio):
        """
        Adding a long position reduces available cash.
        
        Cash reduction = quantity * price * (1 + commission)
        """
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
        """
        Position creation validates inputs and capital requirements.
        
        Tests various edge cases for position creation.
        """
        if should_fail:
            with pytest.raises((InsufficientCapitalError, InvalidOrderError)):
                portfolio.add_position('TEST', quantity, price, commission)
        else:
            position = portfolio.add_position('TEST', quantity, price, commission)
            assert position.symbol == 'TEST'
            assert position.quantity == quantity


class TestPnLCalculations:
    """Test profit/loss calculation accuracy."""
    
    @pytest.fixture
    def portfolio_with_positions(self):
        """Portfolio with mixed positions for PnL testing."""
        portfolio = Portfolio(capital=100000)
        
        # Add winning position
        portfolio.add_position('AAPL', 100, 150.0)
        
        # Add losing position
        portfolio.add_position('GOOGL', 50, 2600.0)
        
        # Add flat position
        portfolio.add_position('MSFT', 200, 300.0)
        
        return portfolio
    
    def test_unrealized_pnl_calculation(self, portfolio_with_positions):
        """
        Unrealized PnL reflects current market prices.
        
        PnL = Σ(current_price - entry_price) * quantity
        """
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
    
    @freeze_time("2024-01-15 09:30:00")
    def test_daily_returns_calculation(self, portfolio_with_positions):
        """
        Daily returns calculated from portfolio value changes.
        
        Returns = (end_value - start_value) / start_value
        """
        # Set previous close values
        portfolio_with_positions.previous_close_value = 95000
        
        # Current values
        current_prices = {'AAPL': 160.0, 'GOOGL': 2550.0, 'MSFT': 300.0}
        portfolio_with_positions.update_market_values(current_prices)
        
        daily_return = portfolio_with_positions.calculate_daily_return()
        
        current_value = portfolio_with_positions.total_value
        expected_return = (current_value - 95000) / 95000
        
        assert daily_return == pytest.approx(expected_return, rel=1e-4)


class TestRiskMetrics:
    """Test portfolio risk calculations."""
    
    def test_position_concentration_limit(self):
        """
        Portfolio enforces position concentration limits.
        
        No single position should exceed max_position_size.
        """
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
    
    def test_correlation_check(self):
        """
        Portfolio checks correlation between positions.
        
        Highly correlated positions increase risk.
        """
        portfolio = Portfolio(capital=100000)
        portfolio.add_position('SPY', 100, 400.0)
        
        # Check correlation with a new position
        correlation = portfolio.check_correlation('QQQ', lookback=252)
        
        # SPY and QQQ are typically highly correlated
        assert correlation > 0.8
        
        # Portfolio should warn about high correlation
        warnings = portfolio.get_risk_warnings()
        assert any("correlation" in w.lower() for w in warnings)


class TestTransactionHistory:
    """Test transaction tracking and history."""
    
    def test_transaction_recording(self):
        """
        All transactions are recorded with full details.
        
        Each transaction should include:
        - Timestamp
        - Type (BUY/SELL)
        - Symbol, quantity, price
        - Commission
        - Running cash balance
        """
        portfolio = Portfolio(capital=100000)
        
        # Execute a buy transaction
        with freeze_time("2024-01-15 10:00:00") as frozen_time:
            portfolio.add_position('AAPL', 100, 150.0, commission=0.001)
            
            # Check transaction was recorded
            assert len(portfolio.transactions) == 1
            txn = portfolio.transactions[0]
            
            assert txn['timestamp'] == frozen_time()
            assert txn['type'] == 'BUY'
            assert txn['symbol'] == 'AAPL'
            assert txn['quantity'] == 100
            assert txn['price'] == 150.0
            assert txn['commission'] == pytest.approx(15.0)  # 0.1% of 15000
            assert txn['cash_balance'] == pytest.approx(84985.0)  # 100000 - 15015
```

#### 2.2 Strategy Tests Example
```python
# tests/unit/strategies/test_mean_reversion.py
"""
Unit tests for Mean Reversion Strategy.

Tests cover:
- Signal generation logic
- Parameter validation
- Edge cases
- Performance characteristics
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from algostack.strategies.mean_reversion import MeanReversionStrategy
from algostack.core.signals import Signal, SignalType


class TestMeanReversionStrategy:
    """Test mean reversion strategy implementation."""
    
    @pytest.fixture
    def strategy(self):
        """
        Standard mean reversion strategy for testing.
        
        Uses typical parameters:
        - 20-day lookback
        - 2 standard deviations
        - 0.02 minimum spread
        """
        return MeanReversionStrategy(
            lookback_period=20,
            num_std=2.0,
            min_spread=0.02
        )
    
    @pytest.fixture
    def trending_data(self):
        """Generate trending market data (not suitable for mean reversion)."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Strong uptrend
        trend = np.linspace(100, 150, 100)
        noise = np.random.randn(100) * 0.5
        
        return pd.DataFrame({
            'close': trend + noise,
            'volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
    
    @pytest.fixture  
    def ranging_data(self):
        """Generate ranging market data (ideal for mean reversion)."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Oscillating around 100
        prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.randn(100) * 0.5
        
        return pd.DataFrame({
            'close': prices + noise,
            'volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
    
    def test_signal_generation_ranging_market(self, strategy, ranging_data):
        """
        Strategy generates signals in ranging market.
        
        Should produce:
        - BUY signals near lower band
        - SELL signals near upper band
        - No signals within bands
        """
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
    
    def test_no_signals_in_strong_trend(self, strategy, trending_data):
        """
        Strategy avoids signals in trending markets.
        
        Mean reversion should not trade strong trends.
        """
        signals = strategy.generate_signals(trending_data)
        
        # Should generate few or no signals in trending market
        assert len(signals) < 5  # Arbitrary threshold
    
    @pytest.mark.parametrize("lookback,num_std,expected_signals", [
        (10, 2.0, 'more'),    # Shorter lookback = more signals
        (30, 2.0, 'fewer'),   # Longer lookback = fewer signals
        (20, 1.5, 'more'),    # Tighter bands = more signals
        (20, 2.5, 'fewer'),   # Wider bands = fewer signals
    ])
    def test_parameter_sensitivity(
        self, ranging_data, lookback, num_std, expected_signals
    ):
        """
        Strategy behavior changes with parameters.
        
        Tests that strategy responds appropriately to parameter changes.
        """
        # Base case
        base_strategy = MeanReversionStrategy(20, 2.0, 0.02)
        base_signals = base_strategy.generate_signals(ranging_data)
        base_count = len(base_signals)
        
        # Test case
        test_strategy = MeanReversionStrategy(lookback, num_std, 0.02)
        test_signals = test_strategy.generate_signals(ranging_data)
        test_count = len(test_signals)
        
        if expected_signals == 'more':
            assert test_count > base_count
        else:
            assert test_count < base_count
    
    def test_insufficient_data_handling(self, strategy):
        """
        Strategy handles insufficient data gracefully.
        
        Should return empty signals, not crash.
        """
        # Only 10 days of data (need 20 for calculation)
        short_data = pd.DataFrame({
            'close': np.random.randn(10) + 100,
            'volume': [1000000] * 10
        }, index=pd.date_range('2024-01-01', periods=10))
        
        signals = strategy.generate_signals(short_data)
        assert signals == []
    
    def test_signal_strength_calculation(self, strategy, ranging_data):
        """
        Signal strength inversely proportional to distance from mean.
        
        Stronger signals when price is further from mean.
        """
        signals = strategy.generate_signals(ranging_data)
        
        for signal in signals:
            assert 0 <= signal.strength <= 1
            
            # Verify strength calculation
            idx = ranging_data.index.get_loc(signal.timestamp)
            window = ranging_data.iloc[max(0, idx-20):idx]
            mean = window['close'].mean()
            std = window['close'].std()
            price = ranging_data.iloc[idx]['close']
            
            # Distance from mean in standard deviations
            z_score = abs(price - mean) / std
            expected_strength = min(z_score / strategy.num_std, 1.0)
            
            assert signal.strength == pytest.approx(expected_strength, rel=0.1)
```

### Phase 3: Integration Tests (Week 5-6)
**Goal**: Test component interactions

#### 3.1 Data Pipeline Integration
```python
# tests/integration/test_data_pipeline.py
"""
Integration tests for data pipeline.

Tests the flow from data fetching through strategy execution.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from algostack.adapters.yf_fetcher import YahooFinanceFetcher
from algostack.core.data_handler import DataHandler
from algostack.strategies.mean_reversion import MeanReversionStrategy
from algostack.core.signals import SignalType


class TestDataPipelineIntegration:
    """Test data flow from source to strategy."""
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data for integration testing."""
        # Create realistic market data
        import pandas as pd
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        data = pd.DataFrame({
            'open': [100 + i * 0.1 + np.random.randn() for i in range(30)],
            'high': [101 + i * 0.1 + np.random.randn() for i in range(30)],
            'low': [99 + i * 0.1 + np.random.randn() for i in range(30)],
            'close': [100 + i * 0.1 + np.random.randn() for i in range(30)],
            'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(30)]
        }, index=dates)
        
        return data
    
    @pytest.mark.integration
    def test_data_fetching_to_signal_generation(self, mock_market_data):
        """
        Test complete flow from data fetching to signal generation.
        
        Verifies:
        1. Data fetcher retrieves data correctly
        2. Data handler processes and validates data
        3. Strategy receives clean data and generates signals
        """
        # Mock the data fetcher
        with patch.object(YahooFinanceFetcher, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = mock_market_data
            
            # Set up components
            fetcher = YahooFinanceFetcher()
            data_handler = DataHandler(fetcher)
            strategy = MeanReversionStrategy(lookback_period=20)
            
            # Execute pipeline
            symbols = ['AAPL']
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 30)
            
            # Fetch and process data
            data = data_handler.get_data(symbols, start_date, end_date)
            
            # Verify data processing
            assert 'AAPL' in data
            assert len(data['AAPL']) == 30
            assert all(col in data['AAPL'].columns for col in ['open', 'high', 'low', 'close', 'volume'])
            
            # Generate signals
            signals = strategy.generate_signals(data['AAPL'])
            
            # Verify signal generation
            assert isinstance(signals, list)
            for signal in signals:
                assert signal.symbol == 'AAPL'
                assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
                assert 0 <= signal.strength <= 1
    
    @pytest.mark.integration
    def test_data_validation_pipeline(self):
        """
        Test data validation catches bad data.
        
        Ensures invalid data is caught and handled properly.
        """
        # Create data with issues
        bad_data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, 104],  # Missing value
            'volume': [1000000, 0, 1000000, -1000, 1000000]  # Zero and negative volume
        }, index=pd.date_range('2024-01-01', periods=5))
        
        data_handler = DataHandler(Mock())
        
        # Validation should catch issues
        with pytest.raises(ValueError) as exc_info:
            data_handler.validate_data(bad_data)
        
        assert "missing values" in str(exc_info.value).lower()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_symbol_pipeline(self):
        """
        Test pipeline handles multiple symbols efficiently.
        
        Verifies parallel processing and data alignment.
        """
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        
        with patch.object(YahooFinanceFetcher, 'fetch_data') as mock_fetch:
            # Return different data for each symbol
            def side_effect(symbol, start, end):
                base_price = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300, 'AMZN': 140, 'META': 320}[symbol]
                dates = pd.date_range(start, end, freq='D')
                return pd.DataFrame({
                    'close': [base_price + np.random.randn() for _ in dates],
                    'volume': [1000000] * len(dates)
                }, index=dates)
            
            mock_fetch.side_effect = side_effect
            
            # Test pipeline
            fetcher = YahooFinanceFetcher()
            data_handler = DataHandler(fetcher)
            
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 30)
            
            data = data_handler.get_data(symbols, start, end)
            
            # Verify all symbols processed
            assert len(data) == len(symbols)
            assert all(symbol in data for symbol in symbols)
            
            # Verify data alignment
            first_symbol_dates = data[symbols[0]].index
            for symbol in symbols[1:]:
                assert data[symbol].index.equals(first_symbol_dates)
```

#### 3.2 Trading Flow Integration
```python
# tests/integration/test_trading_flow.py
"""
Integration tests for complete trading flow.

Tests the interaction between:
- Strategy signals
- Risk management
- Order execution
- Portfolio updates
"""
import pytest
from datetime import datetime
from decimal import Decimal

from algostack.core.portfolio import Portfolio
from algostack.core.risk import RiskManager
from algostack.core.executor import Executor
from algostack.strategies.mean_reversion import MeanReversionStrategy
from algostack.core.signals import Signal, SignalType


class TestTradingFlowIntegration:
    """Test complete trading workflow integration."""
    
    @pytest.fixture
    def trading_system(self):
        """Complete trading system setup."""
        portfolio = Portfolio(capital=100000)
        risk_manager = RiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            max_correlation=0.7
        )
        executor = MockExecutor()  # Use mock for testing
        strategy = MeanReversionStrategy()
        
        return {
            'portfolio': portfolio,
            'risk': risk_manager,
            'executor': executor,
            'strategy': strategy
        }
    
    @pytest.mark.integration
    def test_signal_to_execution_flow(self, trading_system):
        """
        Test complete flow from signal to executed order.
        
        Flow:
        1. Strategy generates signal
        2. Risk manager validates position
        3. Executor places order
        4. Portfolio updates
        """
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
    
    @pytest.mark.integration
    def test_risk_rejection_flow(self, trading_system):
        """
        Test risk manager rejecting unsafe trades.
        
        Verifies risk checks prevent dangerous positions.
        """
        portfolio = trading_system['portfolio']
        risk = trading_system['risk']
        
        # Add existing position close to limit
        portfolio.add_position('AAPL', 600, 150.0)  # ~$90k position
        
        # Try to add more (would exceed position limit)
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            strength=1.0,
            timestamp=datetime.now()
        )
        
        position_size = risk.calculate_position_size(
            portfolio=portfolio,
            signal=signal,
            current_price=150.0
        )
        
        assert position_size == 0  # Risk manager rejects
        
        # Verify risk warnings
        warnings = risk.get_warnings()
        assert any('position limit' in w.lower() for w in warnings)
    
    @pytest.mark.integration
    def test_stop_loss_execution_flow(self, trading_system):
        """
        Test stop loss trigger and execution flow.
        
        Verifies:
        1. Stop loss monitoring
        2. Automatic order generation
        3. Position closure
        """
        portfolio = trading_system['portfolio']
        executor = trading_system['executor']
        
        # Add position with stop loss
        position = portfolio.add_position(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            stop_loss=145.0  # Stop at $145
        )
        
        # Simulate price drop
        current_prices = {'AAPL': 144.0}  # Below stop
        
        # Check stop losses
        stop_orders = portfolio.check_stop_losses(current_prices)
        
        assert len(stop_orders) == 1
        assert stop_orders[0]['symbol'] == 'AAPL'
        assert stop_orders[0]['quantity'] == -100  # Sell order
        
        # Execute stop loss
        fill = executor.execute_order(stop_orders[0])
        
        # Update portfolio
        portfolio.close_position(
            symbol='AAPL',
            price=fill['price'],
            commission=fill['commission']
        )
        
        # Verify position closed
        assert 'AAPL' not in portfolio.positions
        assert portfolio.realized_pnl < 0  # Loss recorded
```

### Phase 4: E2E and Performance Tests (Week 7-8)
**Goal**: Full system validation

#### 4.1 End-to-End Test Example
```python
# tests/e2e/test_complete_backtest.py
"""
End-to-end test for complete backtest workflow.

Tests the entire system from configuration to results.
"""
import pytest
import yaml
from pathlib import Path

from algostack.core.backtest_engine import BacktestEngine
from algostack.core.optimization import WalkForwardOptimizer


class TestCompleteBacktest:
    """Test complete backtest scenarios."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_backtest_workflow(self, tmp_path):
        """
        Test complete backtest from config to results.
        
        Simulates real usage:
        1. Load configuration
        2. Fetch historical data
        3. Run backtest
        4. Generate reports
        5. Save results
        """
        # Create test configuration
        config = {
            'backtest': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.001
            },
            'strategy': {
                'name': 'MeanReversion',
                'parameters': {
                    'lookback_period': 20,
                    'num_std': 2.0,
                    'min_spread': 0.02
                }
            },
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'risk': {
                'max_position_size': 0.1,
                'max_portfolio_risk': 0.02,
                'stop_loss': 0.05
            }
        }
        
        # Save config
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run backtest
        engine = BacktestEngine(config_path)
        results = engine.run()
        
        # Verify results structure
        assert 'metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        
        # Verify metrics
        metrics = results['metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Verify trades recorded
        assert len(results['trades']) > 0
        for trade in results['trades']:
            assert 'symbol' in trade
            assert 'entry_time' in trade
            assert 'exit_time' in trade
            assert 'pnl' in trade
        
        # Verify equity curve
        equity_curve = results['equity_curve']
        assert len(equity_curve) > 200  # Daily data for year
        assert equity_curve.iloc[0] == 100000  # Starting capital
        
        # Save results
        results_path = tmp_path / "results.json"
        engine.save_results(results_path)
        assert results_path.exists()
    
    @pytest.mark.e2e
    @pytest.mark.slow  
    def test_walk_forward_optimization(self):
        """
        Test walk-forward optimization workflow.
        
        Tests:
        1. Parameter optimization
        2. Out-of-sample testing
        3. Performance decay analysis
        """
        optimizer = WalkForwardOptimizer(
            strategy_class=MeanReversionStrategy,
            param_ranges={
                'lookback_period': (10, 30, 5),
                'num_std': (1.5, 2.5, 0.5),
                'min_spread': (0.01, 0.03, 0.01)
            },
            train_periods=252,  # 1 year
            test_periods=63,    # 3 months
            step_periods=21     # 1 month
        )
        
        # Run optimization
        results = optimizer.optimize(
            symbols=['SPY'],
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        
        # Verify optimization results
        assert 'windows' in results
        assert len(results['windows']) > 0
        
        for window in results['windows']:
            assert 'train_metrics' in window
            assert 'test_metrics' in window
            assert 'best_params' in window
            
            # Test performance should be realistic
            # (usually worse than training)
            train_sharpe = window['train_metrics']['sharpe_ratio']
            test_sharpe = window['test_metrics']['sharpe_ratio']
            
            # Not always true but generally expected
            assert test_sharpe < train_sharpe * 1.5
        
        # Check for performance decay
        test_sharpes = [w['test_metrics']['sharpe_ratio'] for w in results['windows']]
        
        # Calculate decay
        first_half_avg = np.mean(test_sharpes[:len(test_sharpes)//2])
        second_half_avg = np.mean(test_sharpes[len(test_sharpes)//2:])
        
        decay = (first_half_avg - second_half_avg) / first_half_avg
        assert decay < 0.5  # Less than 50% decay
```

#### 4.2 Performance Benchmarks
```python
# tests/benchmarks/test_backtest_performance.py
"""
Performance benchmarks for backtesting engine.

Ensures backtesting remains performant as codebase grows.
"""
import pytest
import time
import pandas as pd
import numpy as np

from algostack.core.backtest_engine import BacktestEngine
from algostack.strategies.mean_reversion import MeanReversionStrategy


class TestBacktestPerformance:
    """Benchmark backtest performance."""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing."""
        # 10 years of minute data
        dates = pd.date_range('2014-01-01', '2024-01-01', freq='1min')
        # Only market hours
        dates = dates[dates.hour.isin(range(9, 16))]
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
            'high': 0,
            'low': 0, 
            'close': 0,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        # Realistic OHLC
        data['high'] = data['open'] * (1 + abs(np.random.randn(len(dates))) * 0.001)
        data['low'] = data['open'] * (1 - abs(np.random.randn(len(dates))) * 0.001)
        data['close'] = data['low'] + (data['high'] - data['low']) * np.random.rand(len(dates))
        
        return data
    
    @pytest.mark.benchmark
    def test_backtest_speed_single_symbol(self, large_dataset):
        """
        Benchmark single symbol backtest speed.
        
        Target: Process 10 years of minute data in < 10 seconds.
        """
        engine = BacktestEngine(
            strategy=MeanReversionStrategy(),
            initial_capital=100000
        )
        
        start_time = time.time()
        results = engine.run_backtest(large_dataset)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 10.0  # Must complete in 10 seconds
        
        # Calculate throughput
        bars_per_second = len(large_dataset) / execution_time
        assert bars_per_second > 100000  # Process >100k bars/second
        
        # Log performance metrics
        print(f"Processed {len(large_dataset):,} bars in {execution_time:.2f} seconds")
        print(f"Throughput: {bars_per_second:,.0f} bars/second")
    
    @pytest.mark.benchmark
    def test_backtest_memory_usage(self, large_dataset):
        """
        Test memory efficiency during backtest.
        
        Ensures backtest doesn't have memory leaks.
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = BacktestEngine(
            strategy=MeanReversionStrategy(),
            initial_capital=100000
        )
        
        # Run backtest
        results = engine.run_backtest(large_dataset)
        
        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        
        # Clean up and check memory released
        del results
        del engine
        import gc
        gc.collect()
        
        cleaned_memory = process.memory_info().rss / 1024 / 1024
        
        # Most memory should be released
        assert cleaned_memory < baseline_memory + 100  # Within 100MB of baseline
    
    @pytest.mark.benchmark
    def test_multi_symbol_scaling(self):
        """
        Test performance scaling with multiple symbols.
        
        Performance should scale linearly with symbol count.
        """
        base_time = None
        
        for num_symbols in [1, 5, 10, 20]:
            # Generate data for each symbol
            symbols_data = {}
            for i in range(num_symbols):
                dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
                symbols_data[f'SYM{i}'] = pd.DataFrame({
                    'close': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
                    'volume': np.random.randint(100000, 1000000, len(dates))
                }, index=dates)
            
            # Time the backtest
            start_time = time.time()
            
            engine = BacktestEngine(
                strategy=MeanReversionStrategy(),
                initial_capital=100000
            )
            
            for symbol, data in symbols_data.items():
                engine.run_backtest(data)
            
            execution_time = time.time() - start_time
            
            if base_time is None:
                base_time = execution_time
            else:
                # Should scale roughly linearly
                expected_time = base_time * num_symbols
                assert execution_time < expected_time * 1.2  # Allow 20% overhead
            
            print(f"{num_symbols} symbols: {execution_time:.2f}s")
```

## Test Patterns and Templates

### Pattern 1: Parameterized Test Template
```python
"""
Template for parameterized tests.

Use when testing the same logic with different inputs.
"""
import pytest


class TestParameterizedExample:
    """Example of parameterized testing patterns."""
    
    @pytest.mark.parametrize("input_value,expected_output", [
        # Standard cases
        (10, 100),      # Normal input
        (0, 0),         # Zero input
        (-10, 100),     # Negative input
        
        # Edge cases  
        (0.1, 0.01),    # Small decimal
        (1e6, 1e12),    # Large number
        
        # Special cases
        (float('inf'), float('inf')),  # Infinity
    ], ids=[
        "normal", "zero", "negative",
        "small_decimal", "large_number", 
        "infinity"
    ])
    def test_calculation(self, input_value, expected_output):
        """
        Test calculation with various inputs.
        
        The ids parameter provides readable test names in output.
        """
        result = input_value ** 2
        assert result == expected_output
    
    @pytest.mark.parametrize("strategy_params", [
        pytest.param(
            {'lookback': 20, 'threshold': 2.0},
            id="standard_params"
        ),
        pytest.param(
            {'lookback': 10, 'threshold': 1.5},
            id="aggressive_params"
        ),
        pytest.param(
            {'lookback': 50, 'threshold': 3.0},
            id="conservative_params",
            marks=pytest.mark.slow  # Mark specific params
        ),
    ])
    def test_strategy_configurations(self, strategy_params):
        """
        Test strategy with different parameter sets.
        
        Using pytest.param allows adding metadata to test cases.
        """
        strategy = Strategy(**strategy_params)
        assert strategy.is_valid()
```

### Pattern 2: Fixture Composition Template
```python
"""
Template for composable fixtures.

Build complex test scenarios from simple fixtures.
"""
import pytest
from datetime import datetime, timedelta


class TestFixtureComposition:
    """Example of fixture composition patterns."""
    
    @pytest.fixture
    def base_portfolio(self):
        """Base portfolio fixture."""
        return Portfolio(capital=100000)
    
    @pytest.fixture  
    def portfolio_with_positions(self, base_portfolio):
        """Portfolio with some positions."""
        base_portfolio.add_position('AAPL', 100, 150.0)
        base_portfolio.add_position('GOOGL', 50, 2500.0)
        return base_portfolio
    
    @pytest.fixture
    def portfolio_with_history(self, portfolio_with_positions):
        """Portfolio with transaction history."""
        # Simulate some trades
        portfolio_with_positions.close_position('AAPL', 155.0)
        portfolio_with_positions.add_position('MSFT', 200, 300.0)
        return portfolio_with_positions
    
    @pytest.fixture(params=['bull', 'bear', 'sideways'])
    def market_condition(self, request):
        """
        Parameterized fixture for different market conditions.
        
        This creates three versions of any test using this fixture.
        """
        conditions = {
            'bull': {'trend': 'up', 'volatility': 'low'},
            'bear': {'trend': 'down', 'volatility': 'high'},
            'sideways': {'trend': 'flat', 'volatility': 'medium'}
        }
        return conditions[request.param]
    
    def test_portfolio_performance(
        self, portfolio_with_history, market_condition
    ):
        """
        Test portfolio under different market conditions.
        
        This test runs 3 times - once for each market condition.
        """
        # Apply market condition to portfolio
        returns = simulate_returns(market_condition)
        portfolio_with_history.apply_returns(returns)
        
        # Assertions based on market condition
        if market_condition['trend'] == 'up':
            assert portfolio_with_history.total_return > 0
        elif market_condition['trend'] == 'down':
            assert portfolio_with_history.max_drawdown > 0.1
```

### Pattern 3: Mock and Patch Template
```python
"""
Template for mocking external dependencies.

Use for isolating unit tests from external systems.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestMockingPatterns:
    """Example of mocking patterns."""
    
    def test_mock_external_api(self):
        """
        Mock external API calls.
        
        Isolates test from network dependencies.
        """
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
    
    @patch('algostack.adapters.yf_fetcher.yfinance')
    def test_patch_third_party_library(self, mock_yfinance):
        """
        Patch third-party library imports.
        
        Replaces yfinance with mock during test.
        """
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
    
    @patch('algostack.core.executor.datetime')
    def test_mock_datetime(self, mock_datetime):
        """
        Mock datetime for time-sensitive tests.
        
        Controls time during test execution.
        """
        # Set fixed time
        fixed_time = datetime(2024, 1, 15, 9, 30, 0)
        mock_datetime.now.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        
        # Test time-sensitive code
        executor = Executor()
        order = executor.place_order('AAPL', 100, 'MARKET')
        
        # Verify timestamp
        assert order.timestamp == fixed_time
    
    def test_mock_with_side_effects(self):
        """
        Mock with side effects for complex scenarios.
        
        Simulates changing behavior over multiple calls.
        """
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

### Pattern 4: Property-Based Testing Template
```python
"""
Template for property-based testing with Hypothesis.

Tests invariants rather than specific examples.
"""
import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant


class TestPropertyBased:
    """Example of property-based testing."""
    
    @given(
        price=st.floats(min_value=0.01, max_value=10000, allow_nan=False),
        quantity=st.integers(min_value=1, max_value=1000000)
    )
    def test_position_value_properties(self, price, quantity):
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
    
    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5),
            min_size=2,
            max_size=1000
        )
    )
    def test_sharpe_ratio_properties(self, returns):
        """
        Test Sharpe ratio calculation properties.
        
        Properties:
        1. Sharpe ratio is defined for any return series
        2. Higher returns → higher Sharpe (all else equal)
        3. Lower volatility → higher Sharpe (all else equal)
        """
        sharpe = calculate_sharpe_ratio(returns)
        
        # Property 1: Always defined (might be negative)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Property 2: Monotonic with returns
        higher_returns = [r + 0.01 for r in returns]
        higher_sharpe = calculate_sharpe_ratio(higher_returns)
        assert higher_sharpe > sharpe
    
    @given(
        portfolio_value=st.floats(min_value=1000, max_value=1000000),
        position_limit=st.floats(min_value=0.01, max_value=0.5)
    )
    def test_position_sizing_constraints(self, portfolio_value, position_limit):
        """
        Test position sizing always respects constraints.
        
        Property: Position size never exceeds limit percentage.
        """
        risk_manager = RiskManager(max_position_size=position_limit)
        
        # Generate random signal
        signal = Signal('TEST', SignalType.BUY, strength=1.0)
        
        size = risk_manager.calculate_position_size(
            portfolio_value=portfolio_value,
            signal=signal,
            price=100.0
        )
        
        # Position value should not exceed limit
        position_value = size * 100.0
        max_allowed = portfolio_value * position_limit
        
        assert position_value <= max_allowed + 0.01  # Small tolerance


class PortfolioStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for portfolio operations.
    
    Tests complex sequences of operations maintain invariants.
    """
    
    def __init__(self):
        super().__init__()
        self.portfolio = Portfolio(capital=100000)
    
    @rule(
        symbol=st.text(min_size=1, max_size=5, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        quantity=st.integers(min_value=1, max_value=1000),
        price=st.floats(min_value=1, max_value=1000)
    )
    def add_position(self, symbol, quantity, price):
        """Rule: Can add position if sufficient capital."""
        cost = quantity * price * 1.001  # With commission
        
        if cost <= self.portfolio.cash:
            self.portfolio.add_position(symbol, quantity, price)
    
    @rule()
    def close_random_position(self):
        """Rule: Can close any open position."""
        if self.portfolio.positions:
            symbol = random.choice(list(self.portfolio.positions.keys()))
            self.portfolio.close_position(symbol, price=100.0)  # Simplified
    
    @invariant()
    def cash_never_negative(self):
        """Invariant: Cash balance never goes negative."""
        assert self.portfolio.cash >= 0
    
    @invariant()
    def total_value_conserved(self):
        """Invariant: Total value approximately conserved."""
        # Account for commissions
        min_expected = self.portfolio.initial_capital * 0.95
        max_expected = self.portfolio.initial_capital * 1.05
        
        assert min_expected <= self.portfolio.total_value <= max_expected
```

## Best Practices Guide

### 1. Test Organization Best Practices

```python
"""
Example of well-organized test module.

Follow these patterns for consistency.
"""
# Standard library imports
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Third-party imports
import pytest
import pandas as pd
import numpy as np
from freezegun import freeze_time

# Local imports - grouped by module
from algostack.core.portfolio import Portfolio, Position
from algostack.core.risk import RiskManager
from algostack.core.exceptions import PortfolioError


# Module-level documentation
"""
Tests for portfolio management functionality.

Test Categories:
- Construction and initialization
- Position management (add/update/remove)
- PnL calculations
- Risk metrics
- Transaction history
- State persistence
"""


# Constants for tests
DEFAULT_CAPITAL = 100000
DEFAULT_COMMISSION = 0.001
TEST_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT']


# Shared fixtures at module level
@pytest.fixture
def empty_portfolio():
    """Empty portfolio with default capital."""
    return Portfolio(capital=DEFAULT_CAPITAL)


@pytest.fixture
def sample_positions():
    """Standard position set for testing."""
    return {
        'AAPL': Position('AAPL', 100, 150.0),
        'GOOGL': Position('GOOGL', 50, 2500.0)
    }


# Test classes organized by functionality
class TestPortfolioConstruction:
    """Tests for portfolio initialization."""
    
    def test_default_initialization(self):
        """Test portfolio with default parameters."""
        # Test implementation
    

class TestPositionManagement:
    """Tests for position CRUD operations."""
    
    def test_add_position(self, empty_portfolio):
        """Test adding new position."""
        # Test implementation


class TestPnLCalculations:
    """Tests for profit/loss calculations."""
    
    @pytest.mark.parametrize("scenario", ["profit", "loss", "breakeven"])
    def test_pnl_scenarios(self, empty_portfolio, scenario):
        """Test PnL calculation in different scenarios."""
        # Test implementation
```

### 2. Assertion Best Practices

```python
"""
Examples of strong vs weak assertions.
"""
import pytest
import numpy as np


class TestAssertionPatterns:
    """Demonstrate good assertion practices."""
    
    def test_weak_assertions_avoid(self):
        """
        ❌ AVOID: Weak assertions that don't ensure correctness.
        """
        result = calculate_something()
        
        # Too weak - only checks existence
        assert result is not None
        
        # Too weak - only checks type
        assert isinstance(result, dict)
        
        # Too weak - doesn't verify content
        assert len(result) > 0
    
    def test_strong_assertions_prefer(self):
        """
        ✅ PREFER: Strong assertions that verify behavior.
        """
        result = calculate_portfolio_metrics(portfolio)
        
        # Verify structure AND content
        assert result == {
            'total_value': 100000,
            'positions': 3,
            'exposure': 0.75,
            'cash': 25000
        }
        
        # Use approx for floating point
        assert result['sharpe_ratio'] == pytest.approx(1.5, rel=1e-2)
        
        # Check ranges
        assert 0 <= result['win_rate'] <= 1
        
        # Verify relationships
        assert result['total_value'] == result['cash'] + result['exposure_value']
    
    def test_custom_assertion_helpers(self):
        """
        ✅ PREFER: Custom assertions for complex validations.
        """
        def assert_valid_portfolio_state(portfolio):
            """Comprehensive portfolio validation."""
            assert portfolio.cash >= 0, "Cash cannot be negative"
            assert portfolio.total_value > 0, "Portfolio value must be positive"
            
            # Check position consistency
            calculated_exposure = sum(
                pos.value for pos in portfolio.positions.values()
            )
            assert calculated_exposure == pytest.approx(portfolio.exposure)
            
            # Check transaction history
            assert all(
                txn['timestamp'] <= datetime.now()
                for txn in portfolio.transactions
            )
        
        # Use custom assertion
        portfolio = create_test_portfolio()
        assert_valid_portfolio_state(portfolio)
    
    def test_error_assertions(self):
        """
        ✅ PREFER: Specific error checking.
        """
        # Check specific exception type and message
        with pytest.raises(ValueError) as exc_info:
            Portfolio(capital=-1000)
        
        assert "capital must be positive" in str(exc_info.value)
        
        # Check multiple acceptable errors
        with pytest.raises((ValueError, TypeError)):
            process_invalid_input(None)
        
        # Check error attributes
        with pytest.raises(PortfolioError) as exc_info:
            portfolio.add_position('AAPL', 10000, 150.0)
        
        error = exc_info.value
        assert error.error_code == 'INSUFFICIENT_CAPITAL'
        assert error.required_capital == 1500000
```

### 3. Test Data Management

```python
"""
Best practices for test data management.
"""
import pytest
from pathlib import Path


class TestDataManagement:
    """Examples of test data handling."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Directory containing test data files."""
        return Path(__file__).parent / "test_data"
    
    @pytest.fixture
    def market_data_csv(self, test_data_dir, tmp_path):
        """
        Generate test CSV data in temp directory.
        
        Using tmp_path ensures cleanup and isolation.
        """
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 2000000, 100)
        })
        
        filepath = tmp_path / "market_data.csv"
        data.to_csv(filepath, index=False)
        
        return filepath
    
    def test_with_test_data_file(self, test_data_dir):
        """
        Load test data from version-controlled files.
        
        Keep test data files small and focused.
        """
        # Use small, representative data files
        test_file = test_data_dir / "sample_trades.json"
        
        with open(test_file) as f:
            trades = json.load(f)
        
        # Process test data
        results = analyze_trades(trades)
        assert results['total_trades'] == len(trades)
    
    @pytest.fixture
    def database_fixture(self):
        """
        Create test database with known state.
        
        Use transactions for isolation.
        """
        # Create test database
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        
        # Create session with transaction
        connection = engine.connect()
        transaction = connection.begin()
        session = Session(bind=connection)
        
        # Add test data
        session.add_all([
            Portfolio(name="Test1", capital=100000),
            Portfolio(name="Test2", capital=200000)
        ])
        session.commit()
        
        yield session
        
        # Rollback transaction for cleanup
        transaction.rollback()
        connection.close()
```

### 4. Test Documentation

```python
"""
Best practices for test documentation.
"""


class TestDocumentation:
    """Demonstrate comprehensive test documentation."""
    
    def test_complex_calculation(self):
        """
        Verify Sharpe ratio calculation handles edge cases correctly.
        
        The Sharpe ratio is calculated as:
            Sharpe = (E[R] - Rf) / σ
        
        Where:
            E[R] = Expected return
            Rf = Risk-free rate (assumed 0 for simplicity)
            σ = Standard deviation of returns
        
        Edge cases tested:
        1. Zero volatility → Infinity (capped at 999)
        2. Negative returns → Negative Sharpe
        3. Single return → Undefined (return 0)
        
        Test data:
            Returns: [0.01, -0.02, 0.03, 0.01, -0.01]
            Expected Sharpe: 0.365 (approximately)
        
        References:
            - https://en.wikipedia.org/wiki/Sharpe_ratio
            - Internal design doc: docs/metrics.md
        """
        returns = [0.01, -0.02, 0.03, 0.01, -0.01]
        
        sharpe = calculate_sharpe_ratio(returns)
        
        # Verify calculation
        expected_return = np.mean(returns)
        expected_std = np.std(returns, ddof=1)
        expected_sharpe = expected_return / expected_std * np.sqrt(252)
        
        assert sharpe == pytest.approx(expected_sharpe, rel=1e-3)
    
    def test_with_inline_explanations(self):
        """
        Test portfolio rebalancing with detailed inline comments.
        
        This test verifies the monthly rebalancing algorithm.
        """
        # Setup: Create portfolio with initial positions
        portfolio = Portfolio(capital=100000)
        portfolio.add_position('AAPL', 300, 150.0)  # $45,000 (45%)
        portfolio.add_position('GOOGL', 20, 2500.0)  # $50,000 (50%)
        # Remaining cash: $5,000 (5%)
        
        # Define target weights
        target_weights = {
            'AAPL': 0.4,   # Reduce from 45% to 40%
            'GOOGL': 0.4,   # Reduce from 50% to 40%
            'CASH': 0.2     # Increase from 5% to 20%
        }
        
        # Execute rebalancing
        trades = portfolio.rebalance(target_weights)
        
        # Verify trades generated correctly
        assert len(trades) == 2  # Should sell both positions
        
        # Verify AAPL trade (need to sell ~$5k worth)
        aapl_trade = next(t for t in trades if t.symbol == 'AAPL')
        assert aapl_trade.quantity < 0  # Selling
        assert abs(aapl_trade.quantity * 150.0) == pytest.approx(5000, rel=0.1)
        
        # After rebalancing, verify target weights achieved
        portfolio.execute_trades(trades)
        weights = portfolio.get_weights()
        
        for symbol, target in target_weights.items():
            if symbol != 'CASH':
                assert weights[symbol] == pytest.approx(target, abs=0.01)
```

### 5. Test Naming Conventions

```python
"""
Standardized test naming conventions.
"""


class TestNamingConventions:
    """Examples of clear, descriptive test names."""
    
    # ✅ GOOD: Descriptive test names
    def test_portfolio_add_position_reduces_available_cash(self):
        """When adding a position, available cash decreases by cost."""
        pass
    
    def test_risk_manager_rejects_position_exceeding_limit(self):
        """Risk manager prevents positions larger than configured limit."""
        pass
    
    def test_stop_loss_triggers_when_price_drops_below_threshold(self):
        """Stop loss order executes when price falls below stop price."""
        pass
    
    # ❌ BAD: Vague test names
    def test_portfolio(self):  # Too general
        pass
    
    def test_error(self):  # Which error?
        pass
    
    def test_calculation(self):  # What calculation?
        pass
    
    # ✅ GOOD: Error case naming
    def test_calculate_sharpe_ratio_with_empty_returns_raises_value_error(self):
        """Sharpe ratio calculation requires at least one return."""
        pass
    
    def test_add_position_with_insufficient_capital_raises_portfolio_error(self):
        """Cannot add position when cost exceeds available capital."""
        pass
    
    # ✅ GOOD: Parameterized test naming
    @pytest.mark.parametrize("order_type,expected_fee", [
        ("MARKET", 0.001),
        ("LIMIT", 0.0008),
        ("STOP", 0.001),
    ], ids=["market_order", "limit_order", "stop_order"])
    def test_commission_calculation_by_order_type(self, order_type, expected_fee):
        """Commission varies based on order type."""
        pass
```

## Metrics and Success Criteria

### Coverage Targets

```yaml
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

# Coverage requirements by module type
# Critical modules: 100%
# Core modules: 95%+  
# Utilities: 90%+
# UI/Visualization: 80%+
```

### Performance Benchmarks

```python
# tests/benchmarks/performance_criteria.py
"""
Performance criteria for various components.
"""

PERFORMANCE_CRITERIA = {
    'backtest': {
        'single_symbol_daily_1year': 1.0,      # seconds
        'single_symbol_minute_1year': 10.0,    # seconds
        'multi_symbol_10_daily_1year': 5.0,    # seconds
        'optimization_100_params': 60.0,        # seconds
    },
    'data_loading': {
        'csv_1million_rows': 2.0,              # seconds
        'database_1million_rows': 5.0,         # seconds
        'api_1year_daily': 3.0,                # seconds
    },
    'calculations': {
        'sharpe_ratio_1000_points': 0.001,     # seconds
        'correlation_matrix_100x100': 0.1,     # seconds
        'portfolio_optimization': 1.0,          # seconds
    }
}


@pytest.mark.benchmark
def test_performance_criteria():
    """Verify all components meet performance criteria."""
    for component, criteria in PERFORMANCE_CRITERIA.items():
        for operation, max_time in criteria.items():
            actual_time = measure_operation_time(component, operation)
            assert actual_time <= max_time, \
                f"{component}.{operation} took {actual_time}s (max: {max_time}s)"
```

### Test Quality Metrics

```python
# scripts/test_quality_metrics.py
"""
Calculate test quality metrics.
"""
import ast
import os
from pathlib import Path


def analyze_test_quality(test_dir):
    """Analyze test quality metrics."""
    metrics = {
        'total_tests': 0,
        'parameterized_tests': 0,
        'tests_with_docstrings': 0,
        'tests_with_assertions': 0,
        'average_assertions_per_test': 0,
        'fixture_usage': 0,
        'mock_usage': 0,
    }
    
    for test_file in Path(test_dir).glob("**/test_*.py"):
        with open(test_file) as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                metrics['total_tests'] += 1
                
                # Check for docstring
                if ast.get_docstring(node):
                    metrics['tests_with_docstrings'] += 1
                
                # Count assertions
                assertions = sum(
                    1 for n in ast.walk(node)
                    if isinstance(n, ast.Assert)
                )
                if assertions > 0:
                    metrics['tests_with_assertions'] += 1
                
                # Check for parametrize decorator
                for decorator in node.decorator_list:
                    if 'parametrize' in ast.dump(decorator):
                        metrics['parameterized_tests'] += 1
    
    # Calculate averages
    if metrics['total_tests'] > 0:
        metrics['test_docstring_coverage'] = (
            metrics['tests_with_docstrings'] / metrics['total_tests'] * 100
        )
    
    return metrics


if __name__ == "__main__":
    metrics = analyze_test_quality("tests/")
    
    print("Test Quality Metrics")
    print("=" * 50)
    print(f"Total Tests: {metrics['total_tests']}")
    print(f"Parameterized Tests: {metrics['parameterized_tests']}")
    print(f"Tests with Docstrings: {metrics['test_docstring_coverage']:.1f}%")
    print(f"Tests with Assertions: {metrics['tests_with_assertions']}")
    
    # Quality gates
    assert metrics['test_docstring_coverage'] >= 80, "Need 80%+ test documentation"
    assert metrics['parameterized_tests'] >= metrics['total_tests'] * 0.2, "Need 20%+ parameterized tests"
```

## Implementation Timeline

### Week 1-2: Foundation
- [ ] Set up new test structure
- [ ] Configure pytest and coverage tools  
- [ ] Create base fixtures and utilities
- [ ] Establish naming conventions

### Week 3-4: Unit Tests
- [ ] Rewrite core module tests
- [ ] Implement parameterized tests
- [ ] Add property-based tests
- [ ] Achieve 95%+ coverage

### Week 5-6: Integration Tests  
- [ ] Create component integration tests
- [ ] Test data pipeline flows
- [ ] Test trading workflows
- [ ] Add performance benchmarks

### Week 7-8: E2E and Polish
- [ ] Implement E2E test scenarios
- [ ] Add remaining edge cases
- [ ] Document all tests
- [ ] Create test running guides

### Success Criteria
- 100% code coverage on critical modules
- All tests follow established patterns
- Test suite runs in < 5 minutes
- Zero flaky tests
- Comprehensive documentation

## Conclusion

This comprehensive overhaul will transform AlgoStack's test suite into a professional-grade testing framework that:

1. **Ensures correctness** through comprehensive coverage
2. **Enables refactoring** with confidence  
3. **Documents behavior** through clear tests
4. **Catches regressions** immediately
5. **Supports TDD** for new features

The investment in test quality will pay dividends in reduced bugs, faster development, and increased confidence in the system's reliability.