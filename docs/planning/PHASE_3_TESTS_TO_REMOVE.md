# Phase 3: Tests to Remove - Non-Essential for Four Pillars

## Criteria for Removal
Tests that should be removed if they:
1. Test theoretical edge cases that won't occur in production
2. Focus on achieving code coverage rather than verifying trading safety
3. Test internal implementation details rather than behavior
4. Are duplicate tests with different names
5. Test mocked behavior without real value

## Files to Remove or Refactor

### 1. Edge Case Test Files
These test obscure scenarios that don't protect capital or improve trading:
- `tests/unit/core/test_metrics_edge_cases.py` - Tests like same-day Calmar ratio calculation
- `tests/unit/core/engine/test_execution_handler_edge_cases.py` - Theoretical execution scenarios

### 2. Duplicate Coverage Files
Multiple test files testing the same functionality:
- Keep ONE comprehensive test file per module, remove others:
  - `test_live_engine.py` (KEEP - primary tests)
  - `test_live_engine_comprehensive.py` (REMOVE - duplicate)
  - `test_live_engine_complete_coverage.py` (REMOVE - duplicate)
  - `test_live_engine_final_coverage.py` (REMOVE - duplicate)
  - `test_live_engine_minimal.py` (REMOVE - subset)
  - `test_live_engine_accurate_coverage.py` (REMOVE - duplicate)

### 3. Pure Coverage Tests
Tests written only to achieve 100% coverage without trading value:
- `test_metrics_import_coverage.py` - Tests import fallbacks
- `test_*_100_coverage.py` files - Coverage-focused tests

### 4. Overly Mocked Tests
Tests that mock everything and test nothing real:
- `test_live_engine_comprehensive_mocked.py` - Everything is mocked

## Tests to Keep (Serve Four Pillars)

### Pillar 1: Capital Preservation
- Risk management tests
- Position limit tests
- Order validation tests
- Circuit breaker tests

### Pillar 2: Profit Generation
- Strategy signal tests
- Order execution tests
- Optimization tests (that test real parameter selection)

### Pillar 3: Operational Stability
- Connection/reconnection tests
- Error recovery tests
- State persistence tests

### Pillar 4: Verifiable Correctness
- Backtesting accuracy tests
- Performance metric tests (real calculations)
- Order fill simulation tests

## Removal Plan

### Step 1: Remove duplicate test files
```bash
# Remove duplicate live engine tests
rm tests/unit/core/test_live_engine_comprehensive.py
rm tests/unit/core/test_live_engine_complete_coverage.py
rm tests/unit/core/test_live_engine_final_coverage.py
rm tests/unit/core/test_live_engine_minimal.py
rm tests/unit/core/test_live_engine_accurate_coverage.py
rm tests/unit/core/test_live_engine_comprehensive_mocked.py
```

### Step 2: Remove edge case files
```bash
rm tests/unit/core/test_metrics_edge_cases.py
rm tests/unit/core/engine/test_execution_handler_edge_cases.py
```

### Step 3: Remove pure coverage files
```bash
rm tests/unit/core/test_metrics_import_coverage.py
rm tests/unit/core/test_metrics_100_coverage.py
rm tests/unit/core/engine/test_execution_handler_100_coverage.py
```

### Step 4: Consolidate remaining tests
- Merge essential tests from removed files into primary test files
- Ensure all critical paths are still tested

## Expected Impact
- Reduced test suite size by ~30-40%
- Faster test execution
- Clearer focus on production-critical tests
- Easier maintenance

## Tests That MUST Remain
Even if they seem redundant, keep tests for:
1. Order submission with insufficient funds
2. Risk limit violations
3. Connection failures and recovery
4. Position tracking accuracy
5. P&L calculations
6. Stop loss and take profit execution
7. Market hours validation
8. Emergency liquidation