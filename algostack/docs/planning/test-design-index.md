# Test Design Overhaul - Quick Reference Index

## Overview
This index provides quick access to the comprehensive test design overhaul plan. The full document is available at: `docs/planning/test-design-overhaul-plan.md`

## Current Issues Summary
- **Multiple test files** for same modules (test_portfolio.py, test_portfolio_comprehensive.py, etc.)
- **Weak assertions** (only checking `is not None` or `isinstance`)
- **No test categorization** (can't run just unit tests)
- **Limited pytest features** (no parametrization, markers, or fixtures)

## Key Principles to Follow

### 1. FIRST Principles
- **Fast**: Tests run in milliseconds
- **Independent**: No shared state between tests
- **Repeatable**: Same result every time
- **Self-Validating**: Clear pass/fail
- **Thorough**: Cover edge cases

### 2. Test Structure
```
tests/
├── unit/           # 75% - Fast, isolated tests
├── integration/    # 20% - Component interaction tests
└── e2e/           # 5%  - Full system tests
```

### 3. Naming Convention
```python
def test_[component]_[action]_[expected_result]():
    """When [scenario], [component] should [behavior]."""
```

## Quick Implementation Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create new test directory structure
- [ ] Configure pytest with markers and coverage
- [ ] Set up shared fixtures in `conftest.py`
- [ ] Establish test data management

### Phase 2: Unit Tests (Weeks 3-4)
- [ ] Rewrite tests using parametrization
- [ ] Add strong assertions (exact values, not just types)
- [ ] Implement AAA pattern (Arrange-Act-Assert)
- [ ] Add docstrings to all tests

### Phase 3: Integration Tests (Weeks 5-6)
- [ ] Test data pipeline flow
- [ ] Test trading system integration
- [ ] Add component interaction tests
- [ ] Verify error propagation

### Phase 4: E2E & Performance (Weeks 7-8)
- [ ] Add full backtest workflow tests
- [ ] Implement performance benchmarks
- [ ] Add property-based tests
- [ ] Create test quality metrics

## Essential Patterns

### 1. Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (10, 100),
    (0, 0),
    (-10, 100),
])
def test_calculation(input, expected):
    assert square(input) == expected
```

### 2. Strong Assertions
```python
# ❌ Weak
assert result is not None

# ✅ Strong
assert result == {'value': 100, 'status': 'success'}
```

### 3. Fixture Composition
```python
@pytest.fixture
def portfolio():
    return Portfolio(capital=100000)

@pytest.fixture
def portfolio_with_positions(portfolio):
    portfolio.add_position('AAPL', 100, 150.0)
    return portfolio
```

### 4. Test Categories
```python
@pytest.mark.unit
def test_fast_calculation():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_workflow():
    pass
```

## Coverage Targets
- **Critical modules**: 100% (portfolio, risk, executor)
- **Core modules**: 95%+ (strategies, metrics)
- **Utilities**: 90%+ (helpers, validators)
- **UI/Dashboard**: 80%+ (visualization)

## Test Quality Gates
1. All tests have docstrings
2. Average 3+ assertions per test
3. 20%+ tests use parametrization
4. Zero flaky tests
5. Full suite runs in < 5 minutes

## Common Pitfalls to Avoid
1. **Testing implementation instead of behavior**
2. **Overmocking** - Only mock external dependencies
3. **Shared state** - Each test must be independent
4. **Magic numbers** - Use named constants
5. **Missing edge cases** - Test boundaries and errors

## Quick Commands
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
```

## Where to Find Examples
- **Portfolio Tests**: See Phase 2, Section 2.1 in full document
- **Strategy Tests**: See Phase 2, Section 2.2 in full document  
- **Integration Tests**: See Phase 3 in full document
- **Mocking Patterns**: See Pattern 3 in Test Patterns section
- **Property-Based Tests**: See Pattern 4 in Test Patterns section

## Success Metrics
- ✅ 100% coverage on critical modules
- ✅ All tests follow naming conventions
- ✅ Comprehensive test documentation
- ✅ Fast test execution (< 5 min full suite)
- ✅ No test duplication

## Next Steps
1. Review the full document for detailed examples
2. Start with Phase 1 foundation setup
3. Focus on one module at a time for conversion
4. Use the templates and patterns provided
5. Measure progress with coverage reports

---
*For complete examples and detailed explanations, refer to the full document at `docs/planning/test-design-overhaul-plan.md`*