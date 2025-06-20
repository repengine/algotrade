# Phase 3 Test Failure Analysis

## Pattern Mismatches Identified

### 1. LiveTradingEngine Method Mismatches
**Issue**: Tests expect methods that don't exist
- `check_connection()` - Called in test_live_trading_simulation.py:364 but doesn't exist
- `collect_signals(market_data)` - Called with argument in test_live_trading_simulation.py:608 but method takes no arguments

**Fix**: Update test calls to match actual method signatures

### 2. Position Constructor Missing Required Fields
**Issue**: Position dataclass requires `strategy_id` but tests create without it
- Found in multiple test files
- Position class definition requires: symbol, strategy_id, direction, quantity, entry_price, entry_time, current_price

**Fix**: Add strategy_id to all Position instantiations in tests

### 3. Order Constructor Issues
**Issue**: Tests use incorrect parameter names
- Using `limit_price` instead of `price`
- Order class expects specific field names

**Fix**: Update Order creation to use correct parameter names

### 4. Mock Interface Mismatches
**Issue**: Mock objects don't match production interfaces
- DataHandler mocks missing `fetch()` method
- PortfolioEngine using `total_equity` instead of `current_equity`
- Order.side expecting Enum but receiving string

**Fix**: Update all mocks to match exact production interfaces

### 5. API Response Format Issues
**Issue**: Tests expect different response formats than implementation provides
- Tests checking for simple dict responses
- Implementation returns structured responses with data/status/timestamp

**Fix**: Update assertions to match actual API response structure

### 6. Async Pattern Issues
**Issue**: Coroutines not properly awaited, cleanup issues
- Missing await statements
- Task cleanup not handled properly
- WebSocket reconnection race conditions

**Fix**: Add proper async/await patterns and cleanup

## Priority Order for Fixes

1. **Critical Method Signatures** (Pillar 1 & 3)
   - Fix LiveTradingEngine method calls
   - Fix Position constructor calls
   - Fix Order constructor calls

2. **Mock Interfaces** (Pillar 4)
   - Update all mocks to match production
   - Ensure return types match exactly

3. **Async Patterns** (Pillar 3)
   - Add proper cleanup
   - Fix coroutine handling
   - Handle race conditions

4. **Response Format Assertions** (Pillar 4)
   - Update to match actual formats
   - Remove outdated assertions