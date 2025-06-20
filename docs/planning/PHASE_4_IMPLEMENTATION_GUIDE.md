# Phase 4: Complete Implementation Guide with AI Guardrails

## 🎯 Phase 4 Objectives
Fix 120 failing tests across 5 days by completing all remaining implementations while maintaining the Four Pillars.

## 🛡️ AI Implementation Guardrails

### Before ANY Code Changes:
1. **ALWAYS use grep/glob to verify current implementation**
2. **NEVER assume method signatures or class structures**
3. **ALWAYS run tests before and after changes**
4. **NEVER create new files unless absolutely necessary**
5. **ALWAYS check imports and dependencies exist**

### Quality Checks After Each Change:
```bash
# 1. Run specific test file
poetry run pytest tests/unit/core/test_specific.py -xvs

# 2. Run ruff checks
poetry run ruff check src/
poetry run ruff format src/ tests/

# 3. Run mypy checks (if configured)
poetry run mypy src/ --ignore-missing-imports

# 4. Check for remaining TODOs
grep -r "TODO" src/ --include="*.py" | grep -v "__pycache__"
```

### Git Workflow Per Day:
```bash
# At start of day
git checkout -b phase4/day-X-description

# After each major fix
git add -p  # Review changes carefully
git commit -m "fix: [Component] specific description of fix"

# At end of day
git push origin phase4/day-X-description
# Create PR with summary of changes
```

---

## 📅 Day 1: Core Trading Engine (48 Test Fixes)

### 🎯 Day 1 Goals
- Complete Trading Engine initialization
- Fix LiveTradingEngine lifecycle management
- Ensure foundation is solid for other components

### 📋 Day 1 Tasks

#### Task 1.1: Analyze Trading Engine TODOs
```bash
# First, examine the current implementation
poetry run grep -n "TODO" src/core/engine/trading_engine.py
poetry run grep -n "NotImplementedError" src/core/engine/trading_engine.py

# Check what tests expect
poetry run pytest tests/unit/core/engine/test_trading_engine.py -xvs --tb=short

# Read the actual implementation
cat src/core/engine/trading_engine.py | grep -A 10 -B 10 "TODO"
```

#### Task 1.2: Implement Trading Engine Core
**IMPORTANT**: Check existing manager implementations first
```bash
# Verify manager classes exist
poetry run grep -r "class.*Manager" src/core/ --include="*.py"
poetry run grep -r "PositionManager\|RiskManager\|DataManager" src/core/ --include="*.py"
```

**Expected Implementation Areas:**
1. `initialize_components()` - lines ~241
2. `process_market_data()` - lines ~251  
3. `execute_strategies()` - lines ~256
4. `validate_risks()` - lines ~261
5. `process_orders()` - lines ~266

**Before implementing each method:**
```bash
# Check method signatures in tests
poetry run grep -A 5 "def test.*process_market_data" tests/
poetry run grep -B 5 -A 5 "process_market_data(" tests/
```

#### Task 1.3: Fix LiveTradingEngine Issues
```bash
# Analyze current failures
poetry run pytest tests/unit/core/test_live_engine.py -xvs -k "test_initialization"

# Check for scheduler imports
poetry run grep -n "AsyncIOScheduler" src/core/live_engine.py
poetry run grep -n "apscheduler" pyproject.toml

# Verify strategy initialization
poetry run grep -n "_initialize_strategies" src/core/live_engine.py
```

**Key Areas to Fix:**
1. Strategy factory pattern
2. Scheduler lifecycle (start/stop)
3. Trading mode state machine
4. Error recovery in main loop

#### Task 1.4: Verify Day 1 Progress
```bash
# Run all affected tests
poetry run pytest tests/unit/core/test_live_engine*.py -v
poetry run pytest tests/unit/core/engine/test_trading_engine*.py -v

# Check test count
poetry run pytest tests/unit/core/test_live_engine*.py --collect-only | grep "test session starts" -A 5

# Quality checks
poetry run ruff check src/core/engine/ src/core/live_engine.py
poetry run ruff format src/core/engine/ src/core/live_engine.py

# Commit progress
git add -p
git commit -m "fix: [TradingEngine] implement component initialization and lifecycle"
git commit -m "fix: [LiveTradingEngine] add strategy initialization and scheduler management"
```

### ✅ Day 1 Success Criteria
- [ ] All 7 TradingEngine tests pass
- [ ] All 48 LiveTradingEngine tests pass
- [ ] No new ruff violations
- [ ] All changes committed with descriptive messages

---

## 📅 Day 2: Strategy Framework (27 Test Fixes)

### 🎯 Day 2 Goals
- Add missing abstract method to BaseStrategy
- Fix strategy configuration validation
- Resolve signal generation edge cases

### 📋 Day 2 Tasks

#### Task 2.1: Fix Base Strategy Abstract Method
```bash
# Identify the missing method
poetry run pytest tests/unit/strategies/test_base_strategy_coverage.py -xvs -k "size"

# Check how size is used in tests
poetry run grep -r "\.size(" tests/unit/strategies/ --include="*.py"

# Verify abstract class definition
poetry run grep -n "abstractmethod" src/strategies/base.py
```

**Implementation Steps:**
1. Add `@abstractmethod` decorator for `size()`
2. Define proper signature based on test usage
3. Update all concrete strategies

#### Task 2.2: Update All Concrete Strategies
```bash
# Find all strategy implementations
poetry run find src/strategies -name "*.py" -exec grep -l "BaseStrategy" {} \;

# For each strategy, check if size() exists
poetry run grep -n "def size" src/strategies/mean_reversion.py
poetry run grep -n "def size" src/strategies/trend_following.py
```

#### Task 2.3: Fix Mean Reversion Config
```bash
# Check what parameters tests expect
poetry run pytest tests/unit/strategies/test_mean_reversion_coverage.py -xvs -k "config"

# Find config validation
poetry run grep -n "validate_config\|required.*param" src/strategies/mean_reversion.py

# Check for zscore_threshold usage
poetry run grep -r "zscore_threshold" src/strategies/ tests/unit/strategies/
```

#### Task 2.4: Fix Signal Generation Edge Cases
```bash
# Run specific failing tests
poetry run pytest tests/unit/strategies/test_mean_reversion_coverage.py::TestMeanReversionStrategy::test_edge_case -xvs

# Check signal generation logic
poetry run grep -A 20 "def generate_signals" src/strategies/mean_reversion.py
```

#### Task 2.5: Verify Day 2 Progress
```bash
# Test all strategies
poetry run pytest tests/unit/strategies/ -v

# Specific test counts
poetry run pytest tests/unit/strategies/test_base_strategy*.py --collect-only | grep "collected"
poetry run pytest tests/unit/strategies/test_mean_reversion*.py --collect-only | grep "collected"
poetry run pytest tests/unit/strategies/test_trend_following*.py --collect-only | grep "collected"

# Quality checks
poetry run ruff check src/strategies/
poetry run ruff format src/strategies/

# Commit progress
git add -p
git commit -m "fix: [BaseStrategy] add abstract size() method"
git commit -m "fix: [MeanReversion] add missing config parameters and validation"
git commit -m "fix: [Strategies] handle edge cases in signal generation"
```

### ✅ Day 2 Success Criteria
- [ ] All 9 BaseStrategy tests pass
- [ ] All 12 MeanReversion tests pass
- [ ] All 6 TrendFollowing tests pass
- [ ] Total: 27 strategy tests fixed

---

## 📅 Day 3: Portfolio & Risk Management (28 Test Fixes)

### 🎯 Day 3 Goals
- Fix Portfolio Engine state management
- Complete risk validation implementation
- Fix optimization module issues

### 📋 Day 3 Tasks

#### Task 3.1: Analyze Portfolio Engine Failures
```bash
# Check current portfolio engine structure
poetry run grep -n "class PortfolioEngine" src/core/portfolio_engine.py

# Run tests to see specific failures
poetry run pytest tests/unit/core/test_portfolio_engine.py -xvs --tb=short | head -50

# Check initialization issues
poetry run grep -A 10 "__init__" src/core/portfolio_engine.py
```

#### Task 3.2: Fix State Management
```bash
# Look for state synchronization
poetry run grep -r "sync\|state\|update" src/core/portfolio_engine.py

# Check what tests expect for state
poetry run grep -r "state\|sync" tests/unit/core/test_portfolio_engine.py

# Verify position tracking
poetry run grep -n "track.*position\|update.*position" src/core/portfolio_engine.py
```

#### Task 3.3: Fix Risk Validation
```bash
# Find risk manager usage
poetry run grep -r "risk_manager\|validate.*risk" src/core/

# Check EnhancedOrderManager
poetry run grep -n "TODO" src/core/engine/enhanced_order_manager.py

# Look for volume-based checks
poetry run grep -r "volume.*risk\|check.*volume" src/core/
```

#### Task 3.4: Fix Optimization Module
```bash
# Check optimization test failures
poetry run pytest tests/unit/core/test_optimization*.py -xvs --tb=short

# Find parameter validation
poetry run grep -n "validate.*param" src/core/optimization.py

# Check results handling
poetry run grep -A 10 "OptimizationResult" src/core/optimization.py
```

#### Task 3.5: Verify Day 3 Progress
```bash
# Run portfolio tests
poetry run pytest tests/unit/core/test_portfolio*.py -v

# Run optimization tests
poetry run pytest tests/unit/core/test_optimization*.py -v

# Count fixed tests
poetry run pytest tests/unit/core/test_portfolio_engine.py -v | grep -c "PASSED"
poetry run pytest tests/unit/core/test_optimization*.py -v | grep -c "PASSED"

# Quality checks
poetry run ruff check src/core/portfolio_engine.py src/core/optimization.py
poetry run ruff format src/core/portfolio_engine.py src/core/optimization.py

# Commit progress
git add -p
git commit -m "fix: [PortfolioEngine] implement state synchronization and position tracking"
git commit -m "fix: [RiskManagement] add volume-based risk validation"
git commit -m "fix: [Optimization] fix parameter validation and results handling"
```

### ✅ Day 3 Success Criteria
- [ ] All 19 PortfolioEngine tests pass
- [ ] All 9 Optimization tests pass  
- [ ] Risk validation implemented
- [ ] Total: 28 tests fixed

---

## 📅 Day 4: Data Pipeline & Integration (10 Test Fixes)

### 🎯 Day 4 Goals
- Fix data handler issues
- Complete execution handler TODOs
- Fix integration test failures

### 📋 Day 4 Tasks

#### Task 4.1: Fix Data Handler Issues
```bash
# Check Alpha Vantage key handling
poetry run grep -n "ALPHA_VANTAGE\|api_key" src/core/data_handler.py

# Look for cache implementation
poetry run grep -n "cache\|Cache" src/core/data_handler.py

# Check parquet fallback
poetry run grep -A 10 "parquet\|to_parquet" src/core/data_handler.py

# Run specific tests
poetry run pytest tests/unit/core/test_data_handler*.py -xvs -k "cache\|api_key\|parquet"
```

#### Task 4.2: Complete Execution Handler
```bash
# Find TODOs
poetry run grep -n "TODO" src/core/engine/execution_handler.py

# Check volume profile method
poetry run grep -B 5 -A 15 "get_volume_profile" src/core/engine/execution_handler.py

# Check market volume method  
poetry run grep -B 5 -A 15 "get_market_volume" src/core/engine/execution_handler.py

# See what tests expect
poetry run grep -r "volume_profile\|market_volume" tests/unit/core/engine/
```

#### Task 4.3: Fix Integration Tests
```bash
# Check component interactions
poetry run pytest tests/integration/test_component_interactions.py -xvs --tb=short

# Find missing optimizer
poetry run grep -n "optimizer" tests/integration/test_component_interactions.py

# Check data pipeline signatures
poetry run pytest tests/integration/test_data_pipeline.py -xvs --tb=short

# Verify signal validation
poetry run grep -n "validate.*signal" tests/integration/test_integration.py
```

#### Task 4.4: Verify Day 4 Progress
```bash
# Run data handler tests
poetry run pytest tests/unit/core/test_data_handler*.py -v

# Run execution handler tests
poetry run pytest tests/unit/core/engine/test_execution_handler*.py -v

# Run integration tests
poetry run pytest tests/integration/ -v

# Quality checks
poetry run ruff check src/core/data_handler.py src/core/engine/execution_handler.py
poetry run ruff format src/core/data_handler.py src/core/engine/execution_handler.py

# Commit progress
git add -p
git commit -m "fix: [DataHandler] implement cache management and API key handling"
git commit -m "fix: [ExecutionHandler] add volume profile and market data retrieval"
git commit -m "fix: [Integration] add optimizer and fix method signatures"
```

### ✅ Day 4 Success Criteria
- [ ] All 6 DataHandler tests pass
- [ ] ExecutionHandler TODOs completed
- [ ] All 4 Integration tests pass
- [ ] Total: 10 tests fixed

---

## 📅 Day 5: E2E & API Completion (7 Test Fixes)

### 🎯 Day 5 Goals
- Fix E2E test scenarios
- Complete API implementations
- Fix remaining backtesting issues

### 📋 Day 5 Tasks

#### Task 5.1: Fix E2E Test Failures
```bash
# Check connection recovery
poetry run pytest tests/e2e/test_live_trading_simulation.py -xvs -k "connection\|recovery"

# Check rate limiting
poetry run grep -n "rate.*limit\|throttle" src/core/live_engine.py

# Check drawdown limits
poetry run pytest tests/e2e/test_complete_backtest.py -xvs -k "drawdown\|stress"

# Analyze market state transitions
poetry run grep -n "market.*state\|trading.*hours" src/core/live_engine.py
```

#### Task 5.2: Complete API Implementation
```bash
# Find all API TODOs
poetry run grep -n "TODO" src/api/app.py src/api/routers/*.py

# Check strategy status tracking
poetry run grep -B 10 -A 10 "TODO.*status" src/api/app.py

# Check parameter extraction
poetry run grep -B 10 -A 10 "TODO.*parameters" src/api/app.py

# Check signal tracking
poetry run grep -B 10 -A 10 "TODO.*signals" src/api/app.py

# Check risk calculations
poetry run grep -B 10 -A 10 "TODO.*risk" src/api/app.py
```

#### Task 5.3: Fix Backtesting Issues
```bash
# Check metrics structure
poetry run pytest tests/unit/core/test_backtesting.py -xvs --tb=short

# Find AlgoStackStrategy adapter
poetry run grep -n "AlgoStackStrategy" src/core/backtesting.py

# Verify backtrader integration
poetry run grep -n "import backtrader\|bt\." src/core/backtesting.py
```

#### Task 5.4: Final Verification
```bash
# Run ALL tests to verify Phase 4 completion
poetry run pytest tests/ -v --tb=short > phase4_final_results.txt

# Count results
echo "=== FINAL TEST RESULTS ==="
grep -c "PASSED" phase4_final_results.txt
grep -c "FAILED" phase4_final_results.txt
grep -c "ERROR" phase4_final_results.txt

# Check for remaining TODOs
echo "=== REMAINING TODOs ==="
poetry run grep -r "TODO" src/ --include="*.py" | grep -v "__pycache__" | wc -l

# Final quality check
poetry run ruff check src/
poetry run ruff format src/ tests/

# Create final commit
git add -p
git commit -m "fix: [E2E] implement connection recovery and rate limiting"
git commit -m "fix: [API] complete strategy control and risk calculations"
git commit -m "fix: [Backtesting] fix metrics structure and adapter initialization"

# Push final day
git push origin phase4/day-5-e2e-api-completion
```

### ✅ Day 5 Success Criteria
- [ ] All 5 E2E tests pass
- [ ] All API TODOs completed
- [ ] All 2 Backtesting tests pass
- [ ] Total: 7 tests fixed
- [ ] **FINAL: All 1,332 tests passing**

---

## 🚨 Emergency Procedures

### If Tests Still Fail:
```bash
# 1. Isolate the specific test
poetry run pytest path/to/test_file.py::TestClass::test_method -xvs --tb=short

# 2. Check the actual error
poetry run pytest path/to/test_file.py::TestClass::test_method -xvs --tb=long > error_details.txt

# 3. Verify implementation exists
poetry run grep -n "method_name" src/module/file.py

# 4. Check test expectations vs implementation
# Compare what test expects vs what code does
```

### If Can't Find Implementation:
```bash
# 1. Search broadly
poetry run grep -r "ClassName\|method_name" src/ --include="*.py"

# 2. Check imports
poetry run grep -r "from.*import.*ClassName" tests/ src/

# 3. Check if it should be created
# Look at test to understand if we need to create new code
# BUT REMEMBER: Prefer modifying existing code over creating new files
```

### If Circular Import:
```bash
# 1. Identify the cycle
poetry run python -c "import src.module.file" 2>&1 | grep -A 10 "circular"

# 2. Check imports
poetry run grep "^from\|^import" src/module/file.py

# 3. Move imports inside functions if needed
```

---

## 📊 Daily Progress Tracking

### Daily Checklist:
- [ ] Morning: Pull latest changes, create new branch
- [ ] Run baseline tests to know starting point
- [ ] Implement fixes with constant testing
- [ ] Run quality checks (ruff, mypy)
- [ ] Commit with descriptive messages
- [ ] Evening: Push branch, create PR
- [ ] Document any blockers or concerns

### Metrics to Track:
1. Tests fixed today: X/Y target
2. New tests broken: Should be 0
3. Code coverage change: Should increase
4. TODOs remaining: Should decrease
5. Ruff violations: Should be 0

---

## 🎯 Final Phase 4 Deliverables

1. **Zero Test Failures**: All 1,332 tests passing
2. **No TODOs**: All implementations complete
3. **Clean Code**: Ruff/mypy compliance
4. **Git History**: Clear commits for each fix
5. **Documentation**: Updated TECHNICAL_REFERENCE.md
6. **PR Ready**: Each day's work in separate PR

**Remember**: Every change must serve the Four Pillars. If a test doesn't align with Capital Preservation, Profit Generation, Operational Stability, or Verifiable Correctness, question why we're fixing it.