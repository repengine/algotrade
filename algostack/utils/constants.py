#!/usr/bin/env python3
"""Common constants used throughout AlgoStack."""

# Time Constants
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_TRADING_DAY = 390  # 6.5 hours for US markets
SECONDS_PER_MINUTE = 60

# Risk Constants
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence for VaR
DEFAULT_VOLATILITY_TARGET = 0.10  # 10% annualized
DEFAULT_MAX_POSITION_SIZE = 0.20  # 20% of equity
DEFAULT_MAX_DRAWDOWN = 0.15  # 15% max drawdown
DEFAULT_KELLY_FRACTION = 0.5  # Half-Kelly for safety
MIN_KELLY_FRACTION = 0.1
MAX_KELLY_FRACTION = 1.0

# Portfolio Constants
DEFAULT_INITIAL_CAPITAL = 100000.0
MIN_TRADES_FOR_KELLY = 30  # Minimum trades for Kelly calculation
DEFAULT_LOOKBACK_PERIOD = 252  # 1 year
DEFAULT_REBALANCE_FREQUENCY = 5  # Days

# Technical Indicator Constants
DEFAULT_RSI_PERIOD = 14
DEFAULT_ATR_PERIOD = 14
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_ADX_PERIOD = 14
DEFAULT_ADX_THRESHOLD = 25.0

# Trading Constants
DEFAULT_COMMISSION = 0.0
DEFAULT_SLIPPAGE = 0.0005  # 5 basis points
PENNY_STOCK_THRESHOLD = 5.0  # $5 threshold for penny stocks
MIN_VOLUME_FILTER = 100000  # Minimum daily volume

# Statistical Constants
MIN_CORRELATION = 0.70  # Minimum correlation for pairs trading
MAX_HALF_LIFE = 30  # Maximum half-life for mean reversion (days)
DEFAULT_ZSCORE_ENTRY = 2.0
DEFAULT_ZSCORE_EXIT = 0.5

# Backtesting Constants
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2023-12-31"
WALK_FORWARD_WINDOW = 252  # 1 year
WALK_FORWARD_STEP = 63  # 3 months

# Position Sizing Constants
VOLATILITY_SCALAR = 0.01  # 1% daily volatility reference
MIN_POSITION_SIZE = 0.01  # 1% minimum position
MAX_LEVERAGE = 2.0  # Maximum leverage allowed

# Performance Metrics Constants
MIN_SHARPE_RATIO = 0.5
TARGET_SHARPE_RATIO = 1.5
MIN_WIN_RATE = 0.40  # 40% minimum win rate

# Regime Detection Constants
HIGH_VOLATILITY_PERCENTILE = 75
LOW_VOLATILITY_PERCENTILE = 25
REGIME_LOOKBACK_WINDOW = 60  # Days for regime detection

# Intraday Trading Constants
MARKET_OPEN_MINUTES = 30  # Opening range period
MAX_INTRADAY_POSITION = 0.15  # 15% max for intraday
INTRADAY_STOP_LOSS = 0.01  # 1% stop loss

# Data Quality Constants
MIN_PRICE = 0.01  # Minimum valid price
MAX_PRICE_CHANGE = 0.50  # 50% max daily change (circuit breaker)
MIN_VOLUME = 0  # Allow zero volume for some instruments
