"""Pytest configuration and fixtures."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pytest
import pandas as pd
import numpy as np

from strategies.base import Signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': close * (1 + np.random.uniform(0, 0.01, n)),
        'low': close * (1 - np.random.uniform(0, 0.01, n)),
        'close': close,
        'volume': np.random.uniform(1e6, 2e6, n)
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def portfolio_config() -> Dict[str, Any]:
    """Standard portfolio configuration for testing."""
    return {
        'initial_capital': 10000.0,
        'target_vol': 0.10,
        'max_position_size': 0.20,
        'max_drawdown': 0.15,
        'max_correlation': 0.70,
        'use_equal_risk': True,
        'kelly_fraction': 0.5
    }


@pytest.fixture
def strategy_config() -> Dict[str, Any]:
    """Standard strategy configuration for testing."""
    return {
        'mean_reversion': {
            'enabled': True,
            'symbols': ['SPY', 'QQQ', 'IWM'],
            'params': {
                'rsi_period': 2,
                'rsi_oversold': 10,
                'rsi_overbought': 90,
                'atr_period': 14,
                'atr_band_mult': 2.5,
                'stop_loss_atr': 3.0
            }
        },
        'trend_following': {
            'enabled': True,
            'symbols': ['SPY', 'QQQ'],
            'params': {
                'channel_period': 20,
                'trail_period': 10,
                'atr_period': 14,
                'adx_period': 14,
                'adx_threshold': 25
            }
        }
    }


@pytest.fixture
def mock_market_data() -> Dict[str, pd.DataFrame]:
    """Mock market data for multiple symbols."""
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
    market_data = {}
    
    for symbol in symbols:
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        # Different volatility for each symbol
        vol = {'SPY': 0.015, 'QQQ': 0.02, 'IWM': 0.025, 'DIA': 0.012}[symbol]
        
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(0.0005, vol, n)
        close = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': close * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': close * (1 + np.random.uniform(0, 0.01, n)),
            'low': close * (1 - np.random.uniform(0, 0.01, n)),
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n),
            'returns': np.concatenate([[0], returns[:-1]])
        }, index=dates)
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        data.attrs['symbol'] = symbol
        
        market_data[symbol] = data
    
    return market_data


@pytest.fixture
def mock_signals() -> List[Signal]:
    """Generate mock trading signals."""
    from strategies.base import Signal
    
    return [
        Signal(
            timestamp=datetime.now(),
            symbol='SPY',
            direction='LONG',
            strength=0.8,
            strategy_id='mean_reversion',
            price=450.0,
            atr=5.0,
            metadata={'rsi': 8, 'volume_ratio': 1.5}
        ),
        Signal(
            timestamp=datetime.now(),
            symbol='QQQ',
            direction='SHORT',
            strength=-0.6,
            strategy_id='trend_following',
            price=380.0,
            atr=7.0,
            metadata={'adx': 30, 'channel_break': True}
        )
    ]


@pytest.fixture
def risk_config() -> Dict[str, Any]:
    """Risk management configuration."""
    return {
        'max_var_95': 0.02,
        'target_vol': 0.10,
        'max_position_size': 0.20,
        'max_drawdown': 0.15,
        'min_sharpe': 0.5,
        'risk_free_rate': 0.02,
        'use_garch': False,
        'vol_lookback': 60
    }


# Pytest plugins and settings
def pytest_configure(config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Set up test environment."""
    # Create necessary directories
    from pathlib import Path
    
    test_dirs = [
        Path("data/cache"),
        Path("logs"),
        Path("backtest_results")
    ]
    
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup can go here if needed


# Mock external dependencies
@pytest.fixture
def mock_yfinance(mocker) -> None:
    """Mock yfinance for testing."""
    mock = mocker.patch('yfinance.Ticker')
    mock.return_value.history.return_value = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 102, 103],
        'Low': [99, 100, 101],
        'Close': [100.5, 101.5, 102.5],
        'Volume': [1000000, 1100000, 1200000]
    })
    return mock


@pytest.fixture
def mock_broker_connection(mocker) -> Any:
    """Mock broker connection for testing."""
    mock = mocker.MagicMock()
    mock.is_connected.return_value = True
    mock.get_positions.return_value = {}
    mock.get_account_value.return_value = 10000.0
    return mock