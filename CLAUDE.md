# Claude Development Guidelines for AlgoTrade

---We are using Poetry dependency manager in this project---

## 🎯 PRIME DIRECTIVE: Every Decision Must Serve The Trading System and Four Pillars (Listed below)

Refer to /docs/planning/test-scaffold.md before fixing or creating test files.
Refer to /docs/planning/TECHNICAL_REFERENCE.md before coding or fixing code modules.

**BEFORE EVERY ACTION - Code, Tests, Documentation, Architecture, ANYTHING - Ask:**

> "Does this move us closer to a profitable, automated trading system that can safely manage real money?"

### The Four Pillars - EVERY Decision Must Strengthen At Least One:

#### 1. **CAPITAL PRESERVATION** 
*"Will this protect money from loss?"*
- ✅ Risk limits, position sizing, stop losses
- ✅ Validation, error handling, circuit breakers
- ✅ Market disconnect handling, data validation
- ❌ Any untested code path that touches money

#### 2. **PROFIT GENERATION**
*"Will this help make money?"*
- ✅ Better entry/exit timing, reduced slippage
- ✅ Improved strategy logic, market microstructure
- ✅ Lower fees, better execution venues
- ❌ Premature optimization without measurement

#### 3. **OPERATIONAL STABILITY**
*"Will this keep the system running?"*
- ✅ Monitoring, alerting, graceful degradation
- ✅ Reconnection logic, state persistence
- ✅ Resource management, memory leaks prevention
- ❌ Complex abstractions that obscure failures

#### 4. **VERIFIABLE CORRECTNESS**
*"Can we prove this works with real money?"*
- ✅ Backtests matching live performance
- ✅ Risk metrics, drawdown calculations
- ✅ Order fill simulation, market impact modeling
- ❌ Theoretical correctness without practical validation

### 🛑 STOP SIGNALS - Abandon or Rethink If:
- It doesn't serve any of the four pillars
- It adds complexity without safety benefits
- It's "nice to have" but not trading-critical
- It's just to make tests pass without real value
- It's academic correctness over market reality

### ✅ GO SIGNALS - Proceed Immediately If:
- It prevents capital loss (Pillar 1)
- It's required for live trading (All Pillars)
- It fixes a production risk (Pillar 1 & 3)
- It directly improves P&L (Pillar 2)
- It proves the system works (Pillar 4)

**Every line of code, every test, every decision must answer:**
**"DOES THIS HELP US TRADE PROFITABLY WITH REAL MONEY?"**

If not, we're building a toy, not a trading system.

## 🔧 PRACTICAL GUIDELINES

### When Fixing Tests vs Implementation:
1. **If tests expect production-critical features** (risk checks, position limits, order validation) 
   → **FIX THE IMPLEMENTATION** to add these features
2. **If tests use outdated patterns** but implementation is more robust 
   → **UPDATE THE TESTS** to match better implementation
3. **If implementation is missing safety features** that tests check for 
   → **ADD THE SAFETY FEATURES** to implementation

### Decision Framework:
```
Is this feature needed for safe live trading?
├─ YES → Implement it properly
│   ├─ Risk management → CRITICAL, implement immediately
│   ├─ Order validation → CRITICAL, implement immediately  
│   ├─ Position tracking → CRITICAL, implement immediately
│   └─ Error handling → CRITICAL, implement immediately
└─ NO → Consider if it adds value
    ├─ Improves strategy performance → Implement if time allows
    ├─ Better code organization → Implement if simple
    └─ Academic exercise only → Skip or deprioritize
```

### Always Remember:
- **Real money will be at risk** - every line of code matters
- **Markets are unforgiving** - edge cases will happen
- **Bugs equal losses** - thorough testing saves capital
- **Production > Perfection** - working system > elegant code

POETRY DEPENDENCY MANAGER ALWAYS

## Test Design Reference
For comprehensive test design patterns and implementation guidelines, see:
- **Quick Reference**: `docs/planning/test-design-index.md` (start here)
- **Full Documentation**: `docs/planning/test-design-overhaul-plan.md` (detailed examples)

## CRITICAL: Prevent Module Bloat

**ALWAYS follow these rules when working on AlgoTrade:**

1. **SEARCH FIRST**: Before creating ANY new module or file:
   - Use Glob to find similar filenames
   - Use Grep to search for similar functionality
   - Read existing modules to understand current implementations
   - ***Use Context7 MCP server to search for up-to-date API library information.*** Remember, you were trained on old data, so you need to find new data first.

2. **EXTEND, DON'T DUPLICATE**: 
   - Prefer modifying existing code over creating new files
   - Add functions to existing modules rather than creating new ones
   - Extend existing classes instead of creating similar new ones

3. **VERIFY BEFORE CREATING**:
   - Check if functionality already exists (even partially)
   - Confirm no similar code exists in the codebase
   - Ask user for approval before creating new modules

5. **MODULE CREATION CHECKLIST**:
   - [ ] Searched for similar functionality with Grep
   - [ ] Checked existing modules with Glob
   - [ ] Read related files to understand current structure
   - [ ] Confirmed this is genuinely new functionality
   - [ ] Got explicit approval from user

**Remember**: The codebase is already feature-rich. Your job is to fix, refine, and integrate - not to recreate what already exists.