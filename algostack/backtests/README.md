# AlgoStack Backtesting Guide

## Quick Start

### 1. Test Data Connections
First, verify your data sources are working:

```bash
cd /home/republic/algotrade/algostack
python test_data_connection.py
```

### 2. Run Quick Backtest
Test a single strategy quickly:

```bash
python examples/quick_backtest.py
```

### 3. Run Comprehensive Backtest
Test all strategies with detailed analysis:

```bash
# Using Yahoo Finance (free)
python run_comprehensive_backtest.py --source yfinance

# Using Alpha Vantage (premium, better data quality)
python run_comprehensive_backtest.py --source alphavantage

# Custom date range
python run_comprehensive_backtest.py --start 2021-01-01 --end 2023-12-31

# Single strategy
python run_comprehensive_backtest.py --strategy mean_reversion_spy

# Walk-forward analysis
python run_comprehensive_backtest.py --strategy trend_following_multi --walk-forward
```

## Available Strategies

1. **mean_reversion_spy** - Mean reversion on S&P 500 ETF
2. **trend_following_multi** - Trend following on major ETFs (QQQ, IWM, DIA)
3. **overnight_drift** - Overnight holding strategy (requires intraday data)
4. **opening_range_breakout** - Opening range breakout (requires intraday data)
5. **pairs_trading** - Statistical arbitrage pairs trading (XLF/KRE)
6. **hybrid_regime** - Regime-adaptive strategy

## Backtest Parameters

### Commission and Slippage
- Default commission: 0.1% per trade
- Default slippage: 0.05%
- Modify in `run_comprehensive_backtest.py` if needed

### Initial Capital
- Default: $100,000
- Change with: `--capital 50000`

### Data Sources
- **yfinance**: Free, reliable, daily data only
- **alphavantage**: Premium API with your key, supports intraday data

## Understanding Results

### Key Metrics
- **Total Return**: Overall profit/loss percentage
- **Annual Return**: Annualized return
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss (>1.5 is good)

### Result Files
Results are saved to: `backtests/results/comprehensive_backtest_YYYYMMDD_HHMMSS.json`

## Walk-Forward Analysis

Tests strategy robustness by:
1. Training on historical window (e.g., 1 year)
2. Testing on out-of-sample period (e.g., 3 months)
3. Rolling forward and repeating

Good strategies show consistent performance across all periods.

## Tips for Better Backtests

1. **Use Realistic Parameters**:
   - Include commission and slippage
   - Don't over-optimize on historical data
   
2. **Test Multiple Time Periods**:
   - Bull markets (2020-2021)
   - Bear markets (2022)
   - Sideways markets
   
3. **Check Robustness**:
   - Run walk-forward analysis
   - Test parameter sensitivity
   - Verify across different symbols

## Next Steps

After successful backtests:

1. **Paper Trading**: Test strategies with real-time data but no real money
   ```bash
   python examples/paper_trading_example.py
   ```

2. **Live Trading**: Deploy with real capital (be careful!)
   ```bash
   python examples/ibkr_live_example.py
   ```

## Troubleshooting

### No Data Error
- Check internet connection
- Verify API keys in `config/secrets.yaml`
- Try different symbols (some may be delisted)

### Import Errors
- Ensure you're in the correct directory
- Check Python path includes algostack

### Performance Issues
- Reduce date range
- Use fewer symbols
- Enable caching (automatic)

## Custom Strategies

To test your own strategy:
1. Create strategy class extending `BaseStrategy`
2. Add to `STRATEGY_CONFIGS` in `run_comprehensive_backtest.py`
3. Run backtest as normal