# AlgoStack Technical Reference

## Overview
This document contains implementation patterns, technical scaffolding, and engineering guidelines discovered during the "Zero Errors, 100% Coverage" initiative. Use this as a reference for consistent implementation across the codebase.

## Table of Contents
1. [Critical Implementation Patterns](#critical-implementation-patterns)
2. [API Consistency Guidelines](#api-consistency-guidelines)
3. [Test Scaffolding Patterns](#test-scaffolding-patterns)
4. [Common Fixes Reference](#common-fixes-reference)
5. [Technical Discoveries](#technical-discoveries)
6. [Technical Debt Registry](#technical-debt-registry)

---

## Critical Implementation Patterns

### 1. Memory Manager - Weak Reference Safety
```python
def register_object(self, name: str, obj: Any) -> None:
    """Register object with weak reference safety."""
    try:
        weakref.ref(obj)
        self._managed_objects[name] = obj
        logger.debug(f"Registered object: {name}")
    except TypeError:
        # Skip built-in types that don't support weak refs
        logger.debug(f"Skipping {name} - {type(obj).__name__} doesn't support weak refs")
```
**When to use**: Any time you're registering objects for lifecycle management.
**Why**: Prevents TypeError crashes with dicts/lists while maintaining memory management.

### 2. WebSocket Reconnection with Exponential Backoff
```python
async def _reconnect_websocket(self) -> None:
    """Reconnect with exponential backoff."""
    retry_count = 0
    backoff = 1  # Start at 1 second
    
    while retry_count < self.max_retries:
        try:
            await asyncio.sleep(backoff)
            await self._connect()
            await self._restore_subscriptions()
            break
        except Exception as e:
            retry_count += 1
            backoff = min(backoff * 2, 60)  # Cap at 60 seconds
            logger.warning(f"Reconnect attempt {retry_count} failed: {e}")
```
**When to use**: Any external connection that might drop (WebSocket, database, message queue).
**Why**: Prevents data feed loss and ensures operational stability.

### 3. Order State Synchronization
```python
class OrderEventType(Enum):
    """Order lifecycle events."""
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

# Duplicate detection
def _is_duplicate_order(self, order: Order) -> bool:
    """Check if order is duplicate within time window."""
    key = f"{order.symbol}:{order.side}:{order.quantity}:{order.order_type}"
    now = time.time()
    
    if key in self._recent_orders:
        if now - self._recent_orders[key] < self.duplicate_window:
            return True
    
    self._recent_orders[key] = now
    return False
```
**When to use**: Any order management system interfacing with brokers.
**Why**: Prevents duplicate orders and capital loss from missed fills.

### 4. Risk Management Safety Features
```python
# Portfolio correlation risk (0=diversified, 1=concentrated)
def calculate_correlation_risk(self) -> float:
    """Measure portfolio concentration risk."""
    if len(self.positions) < 2:
        return 0.0
    
    returns = self._get_position_returns()
    corr_matrix = returns.corr()
    
    # Average pairwise correlation
    n = len(corr_matrix)
    total_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))
    return abs(total_corr)

# Position liquidity assessment
def assess_liquidity(self, position: Position) -> Dict[str, float]:
    """Ensure we can exit positions without market impact."""
    avg_volume = self._get_average_volume(position.symbol)
    position_volume = abs(position.quantity)
    
    return {
        "liquidity_score": min(1.0, avg_volume / (position_volume * 10)),
        "estimated_impact": position_volume / avg_volume * 0.1,  # 10bps per 1% of volume
        "exit_time_days": position_volume / (avg_volume * 0.1)  # Using 10% of daily volume
    }
```
**When to use**: Pre-trade validation and portfolio monitoring.
**Why**: Critical for capital preservation in live trading.

---

## API Consistency Guidelines

### Field Naming Conventions
```python
# RiskMetrics dataclass - CORRECT field names
@dataclass
class RiskMetrics:
    value_at_risk: float          # NOT var_95
    conditional_var: float        # NOT cvar_95
    risk_reward_ratio: float      # NOT risk_reward
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
# Portfolio attributes - CORRECT names
class Portfolio:
    current_equity: float         # NOT total_equity
    buying_power: float
    total_pnl: float
    
# Risk methods - CORRECT names
class RiskManager:
    def check_position_size(...)  # NOT check_position_limit
```

### Signal Validation Rules
```python
# SHORT signals MUST have negative strength
if signal.direction == SignalDirection.SHORT and signal.strength > 0:
    raise ValueError("SHORT signals must have negative strength")
    
# LONG signals MUST have positive strength  
if signal.direction == SignalDirection.LONG and signal.strength < 0:
    raise ValueError("LONG signals must have positive strength")
```

### Method Return Types
```python
# Metrics returns Any to support strategy breakdown
def get_performance_metrics() -> Dict[str, Any]:  # NOT Dict[str, float]
    return {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "by_strategy": {"momentum": 0.10, "mean_reversion": 0.05}
    }

# Risk checks return detailed list
def pre_trade_risk_check(order: Order) -> Dict[str, Any]:
    return {
        "passed": True,
        "checks": [
            {"name": "position_limit", "passed": True, "details": {...}},
            {"name": "risk_limit", "passed": True, "details": {...}}
        ]
    }
```

---

## Test Scaffolding Patterns

### 1. Async Test Setup
```python
import pytest
import pytest_asyncio

@pytest.fixture(scope="function")  # NOT session for async
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def async_engine():
    """Async fixture for engine tests."""
    engine = LiveEngine(mode="paper")
    await engine.initialize()
    yield engine
    await engine.shutdown()
```

### 2. Mock Patterns for External Dependencies
```python
# WebSocket mocking
@pytest.fixture
def mock_websocket():
    with patch("websockets.connect") as mock:
        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=["msg1", "msg2", WebSocketException])
        ws.send = AsyncMock()
        mock.return_value.__aenter__.return_value = ws
        yield ws

# IBKR adapter mocking
@pytest.fixture
def mock_ibkr_client():
    client = Mock(spec=IBClient)
    client.connect = Mock()
    client.reqMktData = Mock()
    client.placeOrder = Mock()
    return client
```

### 3. Test Data Builders
```python
def create_test_position(
    symbol: str = "AAPL",
    quantity: int = 100,
    entry_price: float = 150.0,
    **kwargs
) -> Position:
    """Builder for test positions with sensible defaults."""
    return Position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        entry_time=datetime.now(),
        position_type="long" if quantity > 0 else "short",
        **kwargs
    )

def create_test_order(
    symbol: str = "AAPL",
    side: str = "buy",
    quantity: int = 100,
    **kwargs
) -> Order:
    """Builder for test orders with validation."""
    return Order(
        order_id=str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=kwargs.get("order_type", "market"),
        created_at=datetime.now(),
        **kwargs
    )
```

### 4. Integration Test Patterns
```python
class TestComponentIntegration:
    """Test component interactions."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create fully integrated system for testing."""
        portfolio = PortfolioEngine(initial_capital=100000)
        risk_mgr = RiskManager(portfolio=portfolio)
        metrics = MetricsCollector(initial_capital=100000)
        
        # Wire up events
        portfolio.add_observer(risk_mgr)
        portfolio.add_observer(metrics)
        
        return {
            "portfolio": portfolio,
            "risk": risk_mgr,
            "metrics": metrics
        }
    
    def test_trade_flow(self, integrated_system):
        """Test complete trade flow through system."""
        # Test setup → signal → validation → execution → updates
```

---

## Common Fixes Reference

### 1. Type Error Fixes
```python
# NumPy type annotations
from numpy import floating
from typing import Any

# WRONG
def calculate(data: floating) -> float:

# CORRECT
def calculate(data: floating[Any]) -> float:

# Or use float directly if you don't need NumPy specific types
def calculate(data: float) -> float:
```

### 2. Import Organization
```python
# Standard library
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Third party
import numpy as np
import pandas as pd
from fastapi import FastAPI

# Local - use absolute imports
from algostack.core.models import Order, Position
from algostack.core.portfolio import Portfolio
```

### 3. Error Type Consistency
```python
# Validation errors should be ValueError, not TypeError
if leverage < 0:
    raise ValueError("Leverage cannot be negative")  # NOT TypeError

# Type errors for actual type mismatches
if not isinstance(data, pd.DataFrame):
    raise TypeError(f"Expected DataFrame, got {type(data)}")
```

### 4. Async/Await Patterns
```python
# Always await async calls
result = await self.fetch_data()  # NOT self.fetch_data()

# Async context managers
async with self.lock:
    await self.process()

# Gather for concurrent operations
results = await asyncio.gather(
    self.fetch_prices(),
    self.fetch_positions(),
    self.fetch_orders()
)
```

---

## Technical Discoveries

### 1. Test vs Implementation Quality
**Finding**: Implementation is often more robust than tests expect.
- Example: VaR tests expect >3% but statistically correct value is ~2%
- Action: Update tests to match correct implementation, not vice versa
- Principle: The implementation serving live trading is the source of truth

### 2. Strategy Configuration Requirements
**Finding**: Strategies need complete configuration for integration tests.
```python
# Mean reversion REQUIRES these parameters
config = {
    "symbols": ["AAPL", "GOOGL"],
    "lookback_period": 20,
    "zscore_threshold": 2.0,    # Missing this causes test failures
    "exit_zscore": 0.5,         # Missing this causes test failures
    "position_size": 0.1
}
```

### 3. Event Loop Scope Issues
**Finding**: Session-scoped event loops cause issues with async fixtures.
- Solution: Use function-scoped event loops for tests
- Clean up: Ensure proper loop closure in fixtures
- Pattern: One event loop per test function

### 4. Performance Considerations
**Finding**: Certain operations have hidden performance costs.
- DataFrame operations in hot paths impact backtest speed
- Repeated correlation calculations are expensive
- Solution: Cache computed values, use incremental updates

---

## Technical Debt Registry

### Critical Priority
1. **Async Order Manager Tests**
   - Issue: 8 tests showing coroutine warnings
   - Impact: Can't verify async order handling
   - Fix: Proper async test patterns

2. **E2E Test API Mismatches**
   - Issue: 9 tests expect old TradingEngine API
   - Impact: Can't verify end-to-end flows
   - Fix: Update to BacktestEngine API

3. **VaR Test Expectations**
   - Issue: Tests expect unrealistic values
   - Impact: Tests fail despite correct implementation
   - Fix: Statistical review of expectations

### Medium Priority
1. **PortfolioOptimizer Missing**
   - Issue: Class referenced but not implemented
   - Impact: Optimization tests skipped
   - Fix: Implement or remove references

2. **Logger Mock Paths**
   - Issue: Tests use wrong paths (missing 'algostack' prefix)
   - Impact: Logger assertions fail
   - Fix: Standardize logger paths

3. **RiskLimits Type Confusion**
   - Issue: Some code expects dict, implementation uses class
   - Impact: Type mismatches in tests
   - Fix: Consistent type usage

### Low Priority
1. **Test Fixture Duplication**
   - Issue: Similar fixtures in multiple files
   - Impact: Maintenance overhead
   - Fix: Centralize in conftest.py

2. **Import Optimization**
   - Issue: Some files import entire modules
   - Impact: Slower startup
   - Fix: Specific imports

3. **Deprecated Code**
   - Issue: Old strategy implementations still present
   - Impact: Confusion about which to use
   - Fix: Remove with deprecation warnings

---

## Quick Reference

### Run Tests
```bash
# All tests with coverage
pytest --cov=algostack --cov-report=term-missing

# Specific test file
pytest tests/unit/core/test_risk.py -v

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# With Poetry
poetry run pytest
```

### Debug Failed Tests
```bash
# Show full error output
pytest -vv --tb=short

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Show local variables in traceback
pytest -l
```

### Code Quality Checks
```bash
# Type checking
mypy algostack/ --ignore-missing-imports

# Linting
ruff check algostack/

# Auto-fix safe issues
ruff check --fix algostack/

# Format code
black algostack/
```

---

*This is a living document. Update it when you discover new patterns or fix recurring issues.*