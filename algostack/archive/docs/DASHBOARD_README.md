# AlgoStack Dashboard - Setup and Usage

## Overview

The AlgoStack Dashboard is a fully integrated Streamlit-based web interface for testing and configuring trading strategies. It provides:

- **Dynamic strategy discovery** - Automatically finds all strategies in the codebase
- **Full configuration support** - Loads from config/base.yaml with runtime adjustments
- **Real backtesting** - Uses the actual BacktestEngine, no mocks or dummies
- **Multi-strategy testing** - Mix and match strategies with different configurations
- **Comprehensive metrics** - Sharpe, Sortino, Calmar ratios, drawdown analysis, and more
- **Export capabilities** - CSV and JSON export for further analysis

## Quick Start

1. **Install Dependencies** (if not already installed):
   ```bash
   cd algostack
   pip install -r requirements.txt
   ```

2. **Start the Dashboard**:
   ```bash
   ./start_dashboard.sh
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run dashboard_app.py
   ```

3. **Access the Dashboard**:
   Open your browser to `http://localhost:8501`

## Features

### Strategy Configuration
- Select one or more strategies from the dropdown
- Configure parameters for each strategy individually
- Parameters are dynamically discovered from strategy constructors

### Backtesting
- Select multiple symbols to test
- Choose date range and initial capital
- Run backtests with real data and execution logic
- View results per strategy-symbol combination

### Performance Analysis
- Overall portfolio metrics
- Individual strategy performance
- Trade-by-trade analysis
- Winners vs losers breakdown
- P&L distribution charts

### Export Options
- Export trades to CSV
- Export metrics to JSON
- Download results for external analysis

## Integration Points

The dashboard integrates with:

1. **strategies/** - All strategy classes are auto-discovered
2. **backtests/run_backtests.py** - Uses the real BacktestEngine
3. **core/data_handler.py** - Data fetching and caching
4. **config/base.yaml** - Configuration loading
5. **utils/logging.py** - Integrated logging system

## Adding New Strategies

Simply create a new strategy class in the `strategies/` directory that inherits from `BaseStrategy`. The dashboard will automatically discover it on next restart.

## Troubleshooting

1. **Module Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy yfinance backtrader streamlit plotly pyyaml
   ```

2. **Strategy Not Appearing**: Check that your strategy:
   - Inherits from `BaseStrategy`
   - Is in the `strategies/` directory
   - Has a proper class name (not starting with `_`)

3. **Backtest Failures**: 
   - Check logs in the terminal
   - Verify data is available for selected dates
   - Ensure strategy parameters are valid

## Configuration

The dashboard reads configuration from `config/base.yaml`. Key sections:

```yaml
data:
  provider: yfinance  # Data source
  cache_dir: data/cache  # Cache location

portfolio:
  initial_capital: 100000  # Default starting capital
  
strategies:
  # Strategy-specific configurations
  mean_reversion:
    enabled: true
    params:
      rsi_period: 2
      # ... other parameters
```

## Development

To modify the dashboard:

1. **dashboard_app.py** - Main application file
2. **StrategyRegistry** - Handles strategy discovery
3. **run_backtest()** - Orchestrates backtesting
4. **display_backtest_results()** - Results visualization

The dashboard is designed to be extensible. New features can be added without modifying the core trading logic.

## Performance Considerations

- Backtests run sequentially per strategy-symbol combination
- Large date ranges or many symbols may take time
- Results are cached in session state during the session
- Data is cached on disk to avoid repeated downloads

## Future Enhancements

Planned improvements include:
- Live trading integration
- Real-time performance monitoring
- Strategy optimization tools
- Portfolio allocation features
- Risk analysis dashboards

## Support

For issues or questions:
1. Check the logs in the terminal
2. Review the DASHBOARD_GUIDE.md for detailed documentation
3. Ensure all dependencies are correctly installed