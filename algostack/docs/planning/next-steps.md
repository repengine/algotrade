# Next Steps for AlgoStack Test Coverage Initiative

## Current State Summary
As of Session 13 (2025-06-17):
- **Overall Tests Passing**: TBD (need to run full suite)
- **Type Errors**: 0 ✅
- **Linting Errors**: 164 remaining
- **Test Coverage**: ~50% overall (estimated)

### Recently Completed (100% Coverage)
- ✅ core/executor.py
- ✅ core/risk.py
- ✅ core/portfolio.py
- ✅ core/engine/order_manager.py
- ✅ strategies/base.py

### Near-Complete (>90% Coverage)
- ✅ core/backtest_engine.py (98%)
- ✅ core/metrics.py (99%)
- ✅ core/engine/trading_engine.py (93%)
- ✅ core/engine/execution_handler.py (96%)
- ✅ core/data_handler.py (94%) - Session 13
- ✅ strategies/mean_reversion_equity.py (78%) - Session 13

## Immediate Priority Tasks

### 1. Complete LiveTradingEngine Coverage (CRITICAL)
**Current**: 29% coverage (Session 13)
**Target**: 100% coverage
**Files to work with**:
- `tests/unit/core/test_live_engine_accurate_coverage.py`
- `tests/unit/core/test_live_engine_final_coverage.py`
- `tests/unit/core/test_live_engine_missing_coverage.py` (Session 13)

**Key challenges**:
- apscheduler dependency mocking
- Async event loops and task management
- Market data feed simulation
- State persistence testing
- Duplicate _update_market_data methods (architectural issue)

**Recommended approach**:
1. Consider refactoring to remove duplicate methods first
2. Create a comprehensive mock infrastructure for all dependencies
3. Focus on testing individual methods in isolation
4. Use the patterns from execution_handler tests for async methods
5. Consider splitting LiveTradingEngine into smaller, more testable components

### 2. Complete Data Handler Coverage ✅
**Current**: 94% coverage (Session 13 - COMPLETED)
**Remaining**: 8 lines in edge cases

**Completed in Session 13**:
- ✅ API key loading error paths
- ✅ Cache fallback mechanisms
- ✅ get_latest() method
- ✅ Stale cache updates

### 3. Strategy Module Coverage
**Priority order**:
1. mean_reversion_equity.py (78% → 100%) - Session 13 partial completion
   - ✅ Talib import fallback tested
   - ✅ Configuration validation
   - Need: Signal generation edge cases, performance metrics
2. trend_following_multi.py (11% → 100%)
3. hybrid_regime.py (0% → 100%)
4. Other strategies (all at 0%)

**Common patterns to test**:
- Signal generation logic
- Indicator calculations
- Risk parameter validation
- Edge cases (insufficient data, NaN values)
- Import fallback patterns (talib → pandas_indicators)

### 4. Adapter Coverage
**Priority order**:
1. ibkr_adapter.py (29% → 100%)
2. ibkr_executor.py (13% → 100%)
3. paper_executor.py (13% → 100%)
4. yf_fetcher.py (0% → 100%)
5. av_fetcher.py (16% → 100%)

## Technical Patterns Established

### Async Testing Pattern
```python
# Pattern 1: Mock asyncio.create_task
def create_mock_task(coro):
    task = Mock()
    task.cancel = Mock()
    task.cancelled = Mock(return_value=False)
    task._coro = coro
    return task

with patch('asyncio.create_task', side_effect=create_mock_task):
    # Test async code
```

### Dependency Mocking Pattern
```python
# Pattern 2: Mock external dependencies before import
sys.modules['apscheduler'] = Mock()
sys.modules['apscheduler.schedulers'] = Mock()
# Then import the module
```

### Coverage Measurement Pattern
```python
# Run multiple test files together for better coverage
poetry run pytest test_file1.py test_file2.py test_file3.py \
    --cov=module_name --cov-report=term-missing
```

## Known Issues and Workarounds

### 1. Async Test Hanging
**Issue**: Tests hang when asyncio.create_task is called
**Solution**: Mock create_task to return a Mock object instead of real task

### 2. External Dependencies
**Issue**: apscheduler, pandas-ta not installed in test environment
**Solution**: Mock at sys.modules level before imports

### 3. Coverage Not Detected
**Issue**: Module imported before coverage starts
**Solution**: Create standalone test runner or use multiple test files

## Recommended Session Plan

### Session 15 Focus (Next Session)
1. **Hour 1-2**: LiveTradingEngine architectural improvements
   - Identify and fix duplicate methods
   - Create better mock infrastructure
   - Focus on getting to 50%+ coverage

2. **Hour 3**: Complete mean_reversion_equity.py
   - Finish signal generation tests
   - Add performance metrics tests
   - Target 100% coverage

3. **Hour 4**: Begin trend_following_multi.py
   - Use patterns from mean_reversion tests
   - Focus on talib fallback pattern

### Session 15 Focus
1. Complete remaining strategies
2. Begin adapter coverage
3. Fix any failing tests discovered

### Session 15 Focus
1. Complete adapter coverage
2. API and Dashboard coverage
3. Final cleanup and documentation

## Success Metrics
- [x] 50% overall test coverage achieved (estimated)
- [ ] All core modules at >90% coverage (in progress)
- [ ] Zero failing tests
- [ ] All async code properly tested
- [ ] Documentation updated

## Session 13 Achievements
- ✅ DataHandler: 74% → 94% coverage
- ✅ mean_reversion_equity: 16% → 78% coverage
- ✅ Tested talib import fallback pattern
- ✅ Created 4 new comprehensive test files
- ⚠️ LiveTradingEngine remains challenging (29% coverage)

## Commands to Run First
```bash
# Check current test status
poetry run pytest --co -q | grep -E "test.*\.py" | wc -l

# Run live engine tests
poetry run pytest tests/unit/core/test_live_engine*.py -v \
    --cov=algostack.core.live_engine --cov-report=term-missing

# Check overall coverage
poetry run pytest --cov=algostack --cov-report=html
```

## Final Notes
- Maintain focus on the Four Pillars
- Every test should verify behavior that matters for real trading
- Don't just chase coverage numbers - ensure tests are meaningful
- Use established patterns to save time
- Run tests frequently to catch regressions early

---
*Last Updated: 2025-06-17*
*Prepared for: Next Claude iteration (Session 14)*