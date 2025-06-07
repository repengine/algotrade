# AlgoStack Dashboard Fix Request

## Context
I have an AlgoStack trading platform with a dashboard that isn't properly integrating with the strategies in the `strategies/` folder. The dashboard currently uses built-in simplified strategies instead of the actual strategy classes.

## Current Situation
1. The production dashboard (`dashboard.py`) works but uses hardcoded strategies in the `BuiltInStrategies` class
2. The actual strategies in `strategies/` folder (like `mean_reversion_equity.py`, `trend_following_multi.py`, etc.) aren't being used
3. Previous attempts at dynamic strategy loading resulted in strategies showing 0% returns

## What Works
- The simplified built-in strategies (MA Crossover, RSI) generate proper signals and returns (7-15%)
- The dashboard UI, charting, and metrics calculation all work correctly
- Data fetching from yfinance works properly

## What Needs Fixing
1. Make the dashboard dynamically load and use the actual strategy classes from `strategies/` folder
2. Ensure these strategies generate proper trading signals (not 0% returns)
3. The strategies should inherit from `BaseStrategy` and implement the required methods
4. Strategy parameters should be discovered from the strategy classes themselves

## File Structure
```
algostack/
├── dashboard.py              # Current working dashboard with built-in strategies
├── strategies/
│   ├── base.py              # BaseStrategy class
│   ├── mean_reversion_equity.py
│   ├── trend_following_multi.py
│   ├── intraday_orb.py
│   ├── overnight_drift.py
│   ├── pairs_stat_arb.py
│   └── hybrid_regime.py
└── dashboard_versions/       # Previous attempts stored here
```

## Previous Debug Findings
- Test scripts confirmed strategies CAN generate signals (6-7% returns)
- The issue was with how strategies were integrated into the dashboard
- Signal generation logic works when implemented directly but fails when loaded dynamically

## Goal
Create a dashboard that:
1. Dynamically discovers all strategy classes in `strategies/` folder
2. Properly instantiates and runs these strategies
3. Shows real backtest results with actual returns (not 0%)
4. Allows parameter tweaking for each strategy
5. Maintains the current UI/UX quality

Please help fix the integration between the dashboard and the actual strategy classes in the `strategies/` folder.