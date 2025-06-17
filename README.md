# AlgoStack - Multi-Strategy Algorithmic Trading Framework

A disciplined, test-first algorithmic trading system designed for small accounts (<$5k) with focus on risk management and multiple independent edges.

## Features

- **Multiple Strategies**: Mean reversion, trend following, pairs arbitrage, overnight drift, and hybrid regime strategies
- **Risk Management**: Volatility-scaled sizing, fractional Kelly criterion, portfolio-level drawdown guards
- **Data Management**: Multi-source data fetching (Yahoo Finance, Alpha Vantage) with Parquet caching
- **Backtesting**: Walk-forward analysis with realistic slippage modeling
- **Live Trading**: Support for IBKR, Robinhood, and crypto exchanges via unified interface
- **Monitoring**: Real-time performance tracking, alerts via email/Discord

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/algostack.git
cd algostack

# Install Poetry (if not already installed)
pip install poetry

# Install dependencies
poetry install

# Or using pip
pip install -r requirements.txt
```

### Configuration

1. Copy the base configuration:
```bash
cp config/base.yaml config/my_config.yaml
```

2. Set up API keys (optional):
```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
```

### Running Strategies

```bash
# Run in paper trading mode
python main.py run --config config/my_config.yaml --mode paper

# Run backtests
python main.py backtest --strategy mean_reversion --start 2020-01-01 --end 2023-12-31

# Fetch and cache data
python main.py fetch-data --symbol SPY --start 2023-01-01
```

## Strategy Overview

### Mean Reversion Equity
- **Entry**: RSI(2) < 10 AND Close < Lower Band (2.5 ATR)
- **Exit**: Close > MA(10) OR Stop Loss at 3 ATR
- **Symbols**: Liquid ETFs (SPY, QQQ, IWM, DIA)

### Trend Following Multi
- **Entry**: Donchian channel breakout (20-day)
- **Exit**: Trailing stop (10-day low/high)
- **Symbols**: Micro futures, crypto

### Pairs Statistical Arbitrage
- **Entry**: Z-score > 2 standard deviations
- **Exit**: Mean reversion to z ≈ 0
- **Pairs**: Automatically selected based on cointegration

## Architecture

```
algostack/
├── core/               # Engine, portfolio, risk management
├── strategies/         # Strategy implementations
├── adapters/          # Data and execution adapters
├── backtests/         # Backtesting framework
├── config/            # Configuration files
└── tests/             # Test suite
```

## Risk Controls

1. **Position Sizing**: Volatility-scaled with Kelly fraction
2. **Portfolio Limits**: Max 20% per position, 10% target volatility
3. **Drawdown Guard**: 15% portfolio kill switch
4. **Stop Losses**: ATR-based dynamic stops

## Development

```bash
# Run tests
pytest tests/ -v

# Code quality checks
black .
ruff .
mypy algostack/

# Run with Docker
docker build -t algostack .
docker run -v $(pwd)/config:/app/config algostack
```

## Performance Expectations

- **Target Sharpe**: > 0.9
- **Max Drawdown**: < 20%
- **Annual Volatility**: ~10%
- **Min Trade Frequency**: Daily bars for most strategies

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.