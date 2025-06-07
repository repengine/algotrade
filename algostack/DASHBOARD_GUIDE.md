# AlgoStack Dashboard Guide

## Overview
The AlgoStack Dashboard is a Streamlit-based web interface that provides full integration with the AlgoStack trading framework. It allows you to:
- Test strategies with different configurations
- Mix and match multiple strategies
- Run comprehensive backtests
- Analyze performance metrics
- Export results for further analysis

## Starting the Dashboard

### Method 1: Using the Start Script (Recommended)
```bash
./start_dashboard.sh
```

### Method 2: Direct Streamlit Command
```bash
streamlit run dashboard_app.py
```

### Method 3: With Custom Configuration
```bash
streamlit run dashboard_app.py --server.port 8080 --server.address 0.0.0.0
```

## Features

### 1. Dynamic Strategy Discovery
- Automatically discovers all strategies in the `strategies/` directory
- No hardcoded strategy list - fully extensible
- Shows strategy documentation and parameters

### 2. Full Configuration Support
- Loads configuration from `config/base.yaml`
- Allows runtime parameter adjustment
- Supports strategy-specific parameters
- Multi-symbol selection per strategy

### 3. Real Backtesting Engine Integration
- Uses the actual `BacktestEngine` from `backtests/run_backtests.py`
- No mock implementations - real trading logic
- Comprehensive metrics calculation
- Trade-by-trade analysis

### 4. Multi-Strategy Testing
- Test multiple strategies simultaneously
- Mix and match different strategies
- Compare performance across strategies
- Aggregate and individual results

## Dashboard Sections

### Sidebar Configuration
1. **Strategy Selection**
   - Multi-select dropdown with all available strategies
   - Expandable configuration for each strategy
   - Dynamic parameter inputs based on strategy requirements

2. **Market Selection**
   - Choose multiple symbols to test
   - Defaults loaded from configuration
   - Support for any symbols in your data source

3. **Backtest Parameters**
   - Date range selection
   - Initial capital configuration
   - Run backtest button

### Main Display Area

1. **Performance Metrics**
   - Total Return
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Total Trades
   - Profit Factor
   - Sortino Ratio
   - Calmar Ratio

2. **Equity Curve**
   - Interactive Plotly chart
   - Portfolio value over time
   - Visual performance tracking

3. **Trade Analysis**
   - Winners vs Losers breakdown
   - P&L distribution histogram
   - Recent trades table
   - Detailed trade information

4. **Strategy Performance Breakdown**
   - Comparison table of all strategies
   - Individual strategy metrics
   - Per-symbol performance details

## Available Strategies

The dashboard automatically discovers strategies including:
- **Mean Reversion Equity**: RSI-based mean reversion with ATR bands
- **Trend Following Multi**: Multi-timeframe trend following
- **Pairs Stat Arb**: Statistical arbitrage pairs trading
- **Intraday ORB**: Opening range breakout strategy
- **Overnight Drift**: Overnight holding strategy
- **Hybrid Regime**: Regime-switching hybrid strategy

## Configuration Integration

The dashboard reads from `config/base.yaml`:
```yaml
engine:
  mode: backtest
  log_level: INFO

data:
  provider: yfinance
  cache_dir: data/cache

strategies:
  mean_reversion:
    enabled: true
    params:
      rsi_period: 2
      rsi_oversold: 10
      # ... other parameters

portfolio:
  initial_capital: 100000
  currency: USD
```

## Adding New Strategies

1. Create a new strategy class in `strategies/` that inherits from `BaseStrategy`
2. Implement required methods: `init()`, `next()`, `size()`
3. The dashboard will automatically discover and include it
4. No dashboard code changes needed!

## Export Options

### 1. CSV Export
- Export all trades to CSV format
- Includes strategy name, symbol, entry/exit details
- Ready for Excel analysis

### 2. JSON Export
- Export performance metrics as JSON
- Structured data for programmatic analysis
- Includes all calculated metrics

## Debugging Features

1. **Strategy Loading**
   - Shows discovered strategies in the UI
   - Logs strategy initialization
   - Error messages for failed strategies

2. **Backtest Progress**
   - Real-time progress indicators
   - Per-symbol status updates
   - Detailed error reporting

3. **Results Validation**
   - Shows individual results per strategy-symbol
   - Aggregate metrics calculation
   - Trade-level details

## Best Practices

1. **Strategy Configuration**
   - Start with default parameters
   - Make incremental adjustments
   - Test on different time periods

2. **Multi-Strategy Testing**
   - Test strategies individually first
   - Then combine complementary strategies
   - Monitor correlation between strategies

3. **Performance Analysis**
   - Focus on risk-adjusted returns (Sharpe, Sortino)
   - Check maximum drawdown tolerance
   - Verify trade frequency matches expectations

## Troubleshooting

### Dashboard Won't Start
```bash
# Check Streamlit installation
pip install streamlit

# Check for port conflicts
lsof -i :8501

# Run with verbose logging
streamlit run dashboard_app.py --logger.level debug
```

### Strategies Not Loading
- Check strategy class inherits from `BaseStrategy`
- Verify strategy file is in `strategies/` directory
- Check logs for import errors

### Backtest Failures
- Verify data is available for selected date range
- Check strategy configuration is valid
- Review error messages in the UI
- Check `logs/` directory for detailed logs

### No Data Displayed
- Ensure `config/base.yaml` exists
- Verify data provider configuration
- Check internet connection for data downloads

## Advanced Usage

### Running from Python Script
```python
import subprocess
subprocess.run(["streamlit", "run", "dashboard_app.py"])
```

### Programmatic Access
```python
from dashboard_app import StrategyRegistry, run_backtest

# Discover strategies
registry = StrategyRegistry()
strategies = registry.list_strategies()

# Run backtest programmatically
results = run_backtest(
    strategies={"Mean Reversion Equity": {"rsi_period": 2}},
    symbols=["AAPL", "MSFT"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1)
)
```

## Future Enhancements
- Live trading integration
- Real-time performance monitoring
- Strategy optimization tools
- Advanced risk analytics
- Multi-user support