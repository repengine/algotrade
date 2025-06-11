"""
Enhanced pytest configuration and fixtures for AlgoStack test suite.

This module provides:
- Shared test fixtures following FIRST principles
- Test categorization and markers
- Mock data generators
- Test environment setup
"""
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ===============================
# Test Environment Configuration
# ===============================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Custom markers are now defined in pyproject.toml
    
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Configure pandas display options for better test output
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """
    Set up test environment once per session.
    
    Creates necessary directories and performs cleanup after tests.
    """
    # Create test directories
    test_dirs = [
        Path("data/test_cache"),
        Path("logs/test"),
        Path("backtest_results/test"),
    ]
    
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup test artifacts
    import shutil
    for dir_path in test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)


# ===============================
# Market Data Fixtures
# ===============================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.
    
    Returns a DataFrame with 100 days of realistic price data with:
    - Proper OHLC relationships
    - Realistic volume patterns
    - No missing data
    
    Example:
        def test_strategy(sample_ohlcv_data):
            strategy = MyStrategy()
            signals = strategy.generate_signals(sample_ohlcv_data)
            assert len(signals) > 0
    """
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    n = len(dates)
    
    # Generate realistic price movement with volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC with realistic relationships
    data = pd.DataFrame(index=dates)
    data['open'] = close * (1 + np.random.uniform(-0.002, 0.002, n))
    data['high'] = np.maximum(data['open'], close) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    data['low'] = np.minimum(data['open'], close) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    data['close'] = close
    data['volume'] = np.random.lognormal(14, 0.5, n).astype(int)  # Log-normal volume distribution
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def volatile_market_data() -> pd.DataFrame:
    """
    Generate volatile market data for stress testing.
    
    Features:
    - High volatility (5% daily moves)
    - Price gaps
    - Volume spikes during volatility
    
    Use this for testing risk management and edge cases.
    """
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    n = len(dates)
    
    # High volatility returns
    np.random.seed(123)
    returns = np.random.normal(0, 0.05, n)  # 5% daily volatility
    close = 100 * np.exp(np.cumsum(returns))
    
    # Add price gaps (10% chance of 2% gap)
    gaps = np.random.choice([0, 0.02, -0.02], n, p=[0.9, 0.05, 0.05])
    close = close * (1 + gaps)
    
    data = pd.DataFrame(index=dates)
    data['open'] = close * (1 + np.random.uniform(-0.01, 0.01, n))
    data['high'] = np.maximum(data['open'], close) * (1 + np.abs(np.random.normal(0, 0.02, n)))
    data['low'] = np.minimum(data['open'], close) * (1 - np.abs(np.random.normal(0, 0.02, n)))
    data['close'] = close
    
    # Volume spikes with volatility
    base_volume = np.random.lognormal(14, 0.3, n)
    volatility_multiplier = 1 + np.abs(returns) * 10  # Volume increases with volatility
    data['volume'] = (base_volume * volatility_multiplier).astype(int)
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def market_crash_data() -> pd.DataFrame:
    """
    Generate market crash scenario data.
    
    Simulates a 20% drop over 5 days with:
    - Increasing volume
    - Widening spreads
    - Gap downs
    - Recovery period
    """
    # Pre-crash period (20 days)
    pre_crash = pd.date_range(end=datetime.now() - timedelta(days=25), periods=20, freq='D')
    crash = pd.date_range(start=pre_crash[-1] + timedelta(days=1), periods=5, freq='D')
    recovery = pd.date_range(start=crash[-1] + timedelta(days=1), periods=20, freq='D')
    
    dates = pre_crash.append(crash).append(recovery)
    
    # Price action
    pre_crash_prices = 100 + np.random.normal(0, 1, len(pre_crash)).cumsum()
    crash_prices = np.linspace(pre_crash_prices[-1], pre_crash_prices[-1] * 0.8, len(crash))
    recovery_prices = crash_prices[-1] + np.random.normal(0.5, 1, len(recovery)).cumsum()
    
    close = np.concatenate([pre_crash_prices, crash_prices, recovery_prices])
    
    data = pd.DataFrame(index=dates)
    data['close'] = close
    
    # Wider spreads during crash
    normal_spread = 0.005
    crash_spread = 0.02
    spreads = np.concatenate([
        [normal_spread] * len(pre_crash),
        [crash_spread] * len(crash),
        np.linspace(crash_spread, normal_spread, len(recovery))
    ])
    
    data['open'] = close * (1 + np.random.uniform(-spreads/2, spreads/2))
    data['high'] = np.maximum(data['open'], close) * (1 + spreads/2)
    data['low'] = np.minimum(data['open'], close) * (1 - spreads/2)
    
    # Volume spikes during crash
    normal_volume = 1000000
    data['volume'] = np.concatenate([
        np.random.lognormal(np.log(normal_volume), 0.3, len(pre_crash)),
        np.linspace(normal_volume * 3, normal_volume * 5, len(crash)),
        np.linspace(normal_volume * 3, normal_volume, len(recovery))
    ]).astype(int)
    
    return data


@pytest.fixture
def multi_symbol_data() -> dict[str, pd.DataFrame]:
    """
    Generate correlated market data for multiple symbols.
    
    Returns data for SPY, QQQ, IWM, TLT with realistic correlations.
    """
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT']
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')  # 1 year
    n = len(dates)
    
    # Correlation matrix (realistic correlations)
    corr_matrix = np.array([
        [1.00, 0.95, 0.85, -0.30],  # SPY
        [0.95, 1.00, 0.80, -0.35],  # QQQ  
        [0.85, 0.80, 1.00, -0.25],  # IWM
        [-0.30, -0.35, -0.25, 1.00]  # TLT (bonds, negative correlation)
    ])
    
    # Generate correlated returns using Cholesky decomposition
    np.random.seed(42)
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.normal(0, 1, (n, len(symbols)))
    correlated = uncorrelated @ L.T
    
    # Different volatilities and base prices
    volatilities = [0.015, 0.020, 0.025, 0.010]  # SPY, QQQ, IWM, TLT
    base_prices = [400, 350, 200, 120]
    
    market_data = {}
    for i, symbol in enumerate(symbols):
        returns = correlated[:, i] * volatilities[i] + 0.0002  # Small positive drift
        prices = base_prices[i] * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = prices * (1 + np.random.uniform(-0.002, 0.002, n))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.003, n)))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.003, n)))
        data['volume'] = np.random.lognormal(14 + i, 0.5, n).astype(int)
        
        market_data[symbol] = data
    
    return market_data


# ===============================
# Portfolio and Trading Fixtures
# ===============================

@pytest.fixture
def portfolio_config() -> dict[str, Any]:
    """
    Standard portfolio configuration for testing.
    
    Provides reasonable defaults for portfolio parameters.
    """
    return {
        "initial_capital": 100000.0,
        "commission": 0.001,  # 0.1%
        "slippage": 0.0005,  # 0.05%
        "min_order_size": 1,
        "max_position_size": 0.20,  # 20% per position
        "max_portfolio_risk": 0.02,  # 2% portfolio risk
        "max_correlation": 0.70,
        "target_volatility": 0.15,  # 15% annual
        "risk_free_rate": 0.03,  # 3% annual
    }


@pytest.fixture
def empty_portfolio(portfolio_config):
    """Create an empty portfolio instance."""
    from core.portfolio import Portfolio
    return Portfolio(**portfolio_config)


@pytest.fixture
def portfolio_with_positions(empty_portfolio):
    """
    Create a portfolio with existing positions.
    
    Contains:
    - AAPL: 100 shares at $150 (long)
    - GOOGL: 50 shares at $2500 (long)
    - MSFT: -50 shares at $300 (short)
    """
    portfolio = empty_portfolio
    
    # Add positions
    portfolio.add_position('AAPL', 100, 150.0)
    portfolio.add_position('GOOGL', 50, 2500.0)
    portfolio.add_position('MSFT', -50, 300.0)  # Short position
    
    return portfolio


# ===============================
# Strategy Fixtures
# ===============================

@pytest.fixture
def strategy_params() -> dict[str, Any]:
    """Common strategy parameters for testing."""
    return {
        "lookback_period": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "stop_loss_atr": 3.0,
        "take_profit_atr": 6.0,
        "max_holding_period": 10,
        "min_volume": 1000000,
    }


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    from strategies.base import BaseStrategy, Signal
    
    class MockStrategy(BaseStrategy):
        def __init__(self, strategy_id="mock_strategy"):
            super().__init__(strategy_id=strategy_id)
            self.signal_count = 0
        
        def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
            """Generate predictable signals for testing."""
            signals = []
            
            # Generate a buy signal every 20 days
            for i in range(20, len(data), 20):
                self.signal_count += 1
                signals.append(Signal(
                    timestamp=data.index[i],
                    symbol=data.attrs.get('symbol', 'TEST'),
                    direction='LONG' if self.signal_count % 2 == 1 else 'SHORT',
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    price=data.iloc[i]['close'],
                    atr=data.iloc[i]['close'] * 0.02,  # 2% ATR
                    metadata={'signal_number': self.signal_count}
                ))
            
            return signals
    
    return MockStrategy()


# ===============================
# Risk Management Fixtures
# ===============================

@pytest.fixture
def risk_limits() -> dict[str, Any]:
    """Standard risk limits for testing."""
    return {
        "max_position_size": 0.20,  # 20% of portfolio
        "max_sector_exposure": 0.40,  # 40% per sector
        "max_correlation": 0.70,
        "max_var_95": 0.02,  # 2% VaR
        "max_drawdown": 0.15,  # 15% max drawdown
        "min_sharpe_ratio": 0.5,
        "max_leverage": 1.5,
    }


@pytest.fixture
def mock_risk_manager(risk_limits):
    """Create a mock risk manager."""
    from core.risk import RiskManager
    
    risk_manager = MagicMock(spec=RiskManager)
    risk_manager.validate_position.return_value = True
    risk_manager.calculate_position_size.return_value = 100
    risk_manager.check_risk_limits.return_value = (True, [])
    risk_manager.limits = risk_limits
    
    return risk_manager


# ===============================
# Execution Fixtures  
# ===============================

@pytest.fixture
def mock_executor():
    """Create a mock executor for testing."""
    from core.executor import Executor
    
    executor = MagicMock(spec=Executor)
    executor.submit_order.return_value = {
        'order_id': 'TEST123',
        'status': 'FILLED',
        'filled_qty': 100,
        'avg_price': 150.0,
        'commission': 0.15,
    }
    executor.get_order_status.return_value = 'FILLED'
    executor.cancel_order.return_value = True
    
    return executor


@pytest.fixture
def mock_broker_connection():
    """Mock broker connection for testing."""
    broker = MagicMock()
    broker.is_connected.return_value = True
    broker.get_positions.return_value = {}
    broker.get_account_value.return_value = 100000.0
    broker.place_order.return_value = {'order_id': 'TEST123', 'status': 'SUBMITTED'}
    
    return broker


# ===============================
# Time-based Fixtures
# ===============================

@pytest.fixture
def fixed_timestamp():
    """Fixed timestamp for reproducible tests."""
    return datetime(2024, 1, 15, 9, 30, 0)


@pytest.fixture  
def market_hours():
    """NYSE market hours for testing."""
    return {
        'open': timedelta(hours=9, minutes=30),
        'close': timedelta(hours=16, minutes=0),
        'pre_market_open': timedelta(hours=4, minutes=0),
        'after_hours_close': timedelta(hours=20, minutes=0),
    }


# ===============================
# Test Data Generators
# ===============================

def generate_random_signals(n: int = 10, symbols: list[str] = None) -> list:
    """Generate random trading signals for testing."""
    from strategies.base import Signal
    
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    signals = []
    base_time = datetime.now()
    
    for i in range(n):
        signals.append(Signal(
            timestamp=base_time + timedelta(hours=i),
            symbol=np.random.choice(symbols),
            direction=np.random.choice(['LONG', 'SHORT']),
            strength=np.random.uniform(0.5, 1.0),
            strategy_id='test_strategy',
            price=np.random.uniform(50, 500),
            atr=np.random.uniform(1, 10),
            metadata={'test': True, 'index': i}
        ))
    
    return signals


def generate_trades(n: int = 20) -> pd.DataFrame:
    """Generate sample trade history for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
    
    trades = pd.DataFrame({
        'timestamp': dates,
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], n),
        'side': np.random.choice(['BUY', 'SELL'], n),
        'quantity': np.random.randint(10, 200, n),
        'price': np.random.uniform(100, 300, n),
        'commission': np.random.uniform(0.1, 1.0, n),
        'pnl': np.random.normal(0, 100, n),
    })
    
    trades['value'] = trades['quantity'] * trades['price']
    trades['net_value'] = trades['value'] + trades['commission']
    
    return trades


# ===============================
# Assertion Helpers
# ===============================

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs):
    """Assert two DataFrames are equal with better error messages."""
    pd.testing.assert_frame_equal(df1, df2, **kwargs)


def assert_series_equal(s1: pd.Series, s2: pd.Series, **kwargs):
    """Assert two Series are equal with better error messages."""
    pd.testing.assert_series_equal(s1, s2, **kwargs)


def assert_portfolio_invariants(portfolio):
    """Assert portfolio maintains key invariants."""
    assert portfolio.cash >= 0, "Cash cannot be negative"
    assert portfolio.total_value >= 0, "Portfolio value cannot be negative"
    assert abs(portfolio.cash + portfolio.positions_value - portfolio.total_value) < 0.01
    assert all(pos.quantity != 0 for pos in portfolio.positions.values())