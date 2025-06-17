"""
Strategy fixtures for testing.

Provides mock strategies and strategy configurations for testing.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest


class MockSignal:
    """Mock signal for testing."""
    def __init__(self, timestamp: datetime, symbol: str, direction: str,
                 strength: float, strategy_id: str, price: float,
                 atr: float = None, metadata: dict = None):
        self.timestamp = timestamp
        self.symbol = symbol
        self.direction = direction  # 'LONG' or 'SHORT'
        self.strength = strength  # -1.0 to 1.0
        self.strategy_id = strategy_id
        self.price = price
        self.atr = atr or price * 0.02  # Default 2% ATR
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Signal({self.symbol} {self.direction} @ {self.price:.2f}, strength={self.strength:.2f})"


class MockStrategy:
    """Base mock strategy for testing."""
    def __init__(self, strategy_id: str, params: dict = None):
        self.strategy_id = strategy_id
        self.params = params or {}
        self.signals_generated = 0

    def generate_signals(self, data: pd.DataFrame) -> List[MockSignal]:
        """Generate signals based on data."""
        raise NotImplementedError("Subclasses must implement generate_signals")


class AlwaysBuyStrategy(MockStrategy):
    """Strategy that always generates buy signals."""
    def __init__(self):
        super().__init__("always_buy")

    def generate_signals(self, data: pd.DataFrame) -> List[MockSignal]:
        """Generate buy signal on every bar."""
        signals = []
        symbol = data.attrs.get('symbol', 'TEST')

        for i in range(len(data)):
            self.signals_generated += 1
            signals.append(MockSignal(
                timestamp=data.index[i],
                symbol=symbol,
                direction='LONG',
                strength=1.0,
                strategy_id=self.strategy_id,
                price=data.iloc[i]['close'],
                metadata={'bar_index': i}
            ))

        return signals


class MeanReversionMockStrategy(MockStrategy):
    """Mock mean reversion strategy."""
    def __init__(self, lookback: int = 20, z_threshold: float = 2.0):
        super().__init__("mean_reversion_mock", {
            'lookback': lookback,
            'z_threshold': z_threshold
        })

    def generate_signals(self, data: pd.DataFrame) -> List[MockSignal]:
        """Generate mean reversion signals."""
        signals = []
        symbol = data.attrs.get('symbol', 'TEST')
        lookback = self.params['lookback']
        z_threshold = self.params['z_threshold']

        if len(data) < lookback:
            return signals

        # Calculate rolling statistics
        rolling_mean = data['close'].rolling(lookback).mean()
        rolling_std = data['close'].rolling(lookback).std()
        z_score = (data['close'] - rolling_mean) / rolling_std

        for i in range(lookback, len(data)):
            z = z_score.iloc[i]

            # Generate signals at extremes
            if z < -z_threshold:  # Oversold
                self.signals_generated += 1
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    symbol=symbol,
                    direction='LONG',
                    strength=min(abs(z) / 3, 1.0),  # Stronger signal for more extreme z-scores
                    strategy_id=self.strategy_id,
                    price=data.iloc[i]['close'],
                    metadata={'z_score': z, 'mean': rolling_mean.iloc[i]}
                ))
            elif z > z_threshold:  # Overbought
                self.signals_generated += 1
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    symbol=symbol,
                    direction='SHORT',
                    strength=min(abs(z) / 3, 1.0),
                    strategy_id=self.strategy_id,
                    price=data.iloc[i]['close'],
                    metadata={'z_score': z, 'mean': rolling_mean.iloc[i]}
                ))

        return signals


class TrendFollowingMockStrategy(MockStrategy):
    """Mock trend following strategy."""
    def __init__(self, fast_ma: int = 10, slow_ma: int = 30):
        super().__init__("trend_following_mock", {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma
        })

    def generate_signals(self, data: pd.DataFrame) -> List[MockSignal]:
        """Generate trend following signals based on MA crossovers."""
        signals = []
        symbol = data.attrs.get('symbol', 'TEST')
        fast_ma = self.params['fast_ma']
        slow_ma = self.params['slow_ma']

        if len(data) < slow_ma:
            return signals

        # Calculate moving averages
        fast = data['close'].rolling(fast_ma).mean()
        slow = data['close'].rolling(slow_ma).mean()

        # Find crossovers
        position = 0  # Track current position

        for i in range(slow_ma, len(data)):
            prev_fast = fast.iloc[i-1]
            prev_slow = slow.iloc[i-1]
            curr_fast = fast.iloc[i]
            curr_slow = slow.iloc[i]

            # Bullish crossover
            if prev_fast <= prev_slow and curr_fast > curr_slow and position <= 0:
                self.signals_generated += 1
                position = 1
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    symbol=symbol,
                    direction='LONG',
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    price=data.iloc[i]['close'],
                    metadata={'fast_ma': curr_fast, 'slow_ma': curr_slow}
                ))

            # Bearish crossover
            elif prev_fast >= prev_slow and curr_fast < curr_slow and position >= 0:
                self.signals_generated += 1
                position = -1
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    symbol=symbol,
                    direction='SHORT',
                    strength=0.8,
                    strategy_id=self.strategy_id,
                    price=data.iloc[i]['close'],
                    metadata={'fast_ma': curr_fast, 'slow_ma': curr_slow}
                ))

        return signals


class RandomStrategy(MockStrategy):
    """Strategy that generates random signals for stress testing."""
    def __init__(self, signal_frequency: float = 0.1, seed: int = None):
        super().__init__("random", {
            'signal_frequency': signal_frequency,
            'seed': seed
        })
        if seed:
            np.random.seed(seed)

    def generate_signals(self, data: pd.DataFrame) -> List[MockSignal]:
        """Generate random signals."""
        signals = []
        symbol = data.attrs.get('symbol', 'TEST')

        for i in range(len(data)):
            if np.random.random() < self.params['signal_frequency']:
                self.signals_generated += 1
                signals.append(MockSignal(
                    timestamp=data.index[i],
                    symbol=symbol,
                    direction=np.random.choice(['LONG', 'SHORT']),
                    strength=np.random.uniform(0.5, 1.0),
                    strategy_id=self.strategy_id,
                    price=data.iloc[i]['close'],
                    metadata={'random': True}
                ))

        return signals


# Strategy Configuration Fixtures

@pytest.fixture
def mean_reversion_config():
    """Mean reversion strategy configuration."""
    return {
        "strategy_id": "mean_reversion",
        "enabled": True,
        "symbols": ["SPY", "QQQ", "IWM"],
        "params": {
            "lookback_period": 20,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "max_holding_period": 10,
            "stop_loss_atr": 3.0,
            "take_profit_atr": 6.0,
            "min_volume": 1000000,
        }
    }


@pytest.fixture
def trend_following_config():
    """Trend following strategy configuration."""
    return {
        "strategy_id": "trend_following",
        "enabled": True,
        "symbols": ["SPY", "QQQ", "TLT", "GLD"],
        "params": {
            "fast_period": 10,
            "slow_period": 30,
            "atr_period": 14,
            "breakout_period": 20,
            "adx_threshold": 25,
            "risk_per_trade": 0.01,
            "trailing_stop_atr": 2.0,
        }
    }


@pytest.fixture
def pairs_trading_config():
    """Pairs trading strategy configuration."""
    return {
        "strategy_id": "pairs_trading",
        "enabled": True,
        "pairs": [
            ["XLF", "BAC"],  # Financials ETF vs Bank of America
            ["GLD", "GDX"],  # Gold vs Gold Miners
            ["IWM", "SPY"],  # Small caps vs Large caps
        ],
        "params": {
            "lookback_period": 60,
            "entry_z_score": 2.0,
            "exit_z_score": 0.0,
            "correlation_threshold": 0.7,
            "half_life_threshold": 30,
            "max_holding_period": 20,
        }
    }


# Strategy Instance Fixtures

@pytest.fixture
def always_buy_strategy():
    """Strategy that always buys."""
    return AlwaysBuyStrategy()


@pytest.fixture
def mean_reversion_strategy():
    """Mean reversion strategy instance."""
    return MeanReversionMockStrategy(lookback=20, z_threshold=2.0)


@pytest.fixture
def trend_following_strategy():
    """Trend following strategy instance."""
    return TrendFollowingMockStrategy(fast_ma=10, slow_ma=30)


@pytest.fixture
def random_strategy():
    """Random signal generator."""
    return RandomStrategy(signal_frequency=0.1, seed=42)


@pytest.fixture
def strategy_ensemble():
    """Collection of strategies for ensemble testing."""
    return {
        'mean_reversion': MeanReversionMockStrategy(lookback=20, z_threshold=2.0),
        'trend_following': TrendFollowingMockStrategy(fast_ma=10, slow_ma=30),
        'random': RandomStrategy(signal_frequency=0.05, seed=42),
    }


# Signal Generation Fixtures

@pytest.fixture
def sample_signals():
    """Generate sample signals for testing."""
    base_time = datetime.now()
    symbols = ['AAPL', 'GOOGL', 'MSFT']

    signals = []
    for i in range(10):
        signals.append(MockSignal(
            timestamp=base_time + timedelta(hours=i),
            symbol=symbols[i % len(symbols)],
            direction='LONG' if i % 2 == 0 else 'SHORT',
            strength=0.5 + (i % 5) * 0.1,
            strategy_id='test_strategy',
            price=100 + i * 2,
            metadata={'index': i}
        ))

    return signals


@pytest.fixture
def conflicting_signals():
    """Generate conflicting signals for testing signal resolution."""
    timestamp = datetime.now()
    symbol = 'AAPL'
    price = 150.0

    return [
        MockSignal(timestamp, symbol, 'LONG', 0.8, 'strategy_1', price),
        MockSignal(timestamp, symbol, 'SHORT', 0.6, 'strategy_2', price),
        MockSignal(timestamp, symbol, 'LONG', 0.9, 'strategy_3', price),
    ]


@pytest.fixture
def signal_history():
    """Generate historical signals for backtesting."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    signals = []

    for i, date in enumerate(dates):
        if i % 3 == 0:  # Signal every 3 days
            signals.append(MockSignal(
                timestamp=date,
                symbol='SPY',
                direction='LONG' if i % 6 == 0 else 'SHORT',
                strength=0.7,
                strategy_id='backtest_strategy',
                price=400 + i,
                metadata={'day': i}
            ))

    return signals


# Strategy Testing Helpers

@pytest.fixture
def strategy_tester():
    """Helper for testing strategies."""
    def test_strategy(strategy: MockStrategy, data: pd.DataFrame,
                     expected_signal_count: Optional[int] = None,
                     check_directions: bool = True) -> List[MockSignal]:
        """
        Test a strategy and return signals with basic validation.

        Args:
            strategy: Strategy to test
            data: Market data
            expected_signal_count: Expected number of signals (if known)
            check_directions: Whether to validate signal directions

        Returns:
            List of generated signals
        """
        signals = strategy.generate_signals(data)

        # Basic validation
        assert isinstance(signals, list), "Signals must be a list"

        for signal in signals:
            assert isinstance(signal, MockSignal), "Each signal must be a MockSignal"
            assert signal.direction in ['LONG', 'SHORT'], f"Invalid direction: {signal.direction}"
            assert -1 <= signal.strength <= 1, f"Invalid strength: {signal.strength}"
            assert signal.price > 0, f"Invalid price: {signal.price}"

        if expected_signal_count is not None:
            assert len(signals) == expected_signal_count, \
                f"Expected {expected_signal_count} signals, got {len(signals)}"

        return signals

    return test_strategy


@pytest.fixture
def signal_analyzer():
    """Helper for analyzing generated signals."""
    def analyze_signals(signals: List[MockSignal]) -> Dict[str, Any]:
        """
        Analyze a list of signals and return statistics.

        Returns:
            Dictionary with signal statistics
        """
        if not signals:
            return {
                'count': 0,
                'long_count': 0,
                'short_count': 0,
                'avg_strength': 0,
                'symbols': [],
                'strategies': [],
            }

        long_signals = [s for s in signals if s.direction == 'LONG']
        short_signals = [s for s in signals if s.direction == 'SHORT']

        return {
            'count': len(signals),
            'long_count': len(long_signals),
            'short_count': len(short_signals),
            'avg_strength': np.mean([s.strength for s in signals]),
            'min_strength': min(s.strength for s in signals),
            'max_strength': max(s.strength for s in signals),
            'symbols': list({s.symbol for s in signals}),
            'strategies': list({s.strategy_id for s in signals}),
            'time_range': (min(s.timestamp for s in signals),
                          max(s.timestamp for s in signals)),
        }

    return analyze_signals
