"""Comprehensive test suite for strategy implementations."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.base import BaseStrategy, Signal, RiskContext
from strategies.mean_reversion_equity import MeanReversionEquity as MeanReversionEquityStrategy
from strategies.trend_following_multi import TrendFollowingMulti as TrendFollowingMultiStrategy


class TestBaseStrategy:
    """Test suite for BaseStrategy abstract class."""
    
    class ConcreteStrategy(BaseStrategy):
        """Concrete implementation for testing."""
        
        def init(self):
            self.initialized = True
            
        def next(self, data):
            return Signal(
                timestamp=datetime.now(),
                symbol='TEST',
                direction='LONG',
                strength=1.0,
                strategy_id=self.name,
                price=100.0
            )
            
        def size(self, signal, risk_context):
            return (100, 95.0)  # 100 shares, $95 stop loss
        
        def calculate_signals(self, data, symbol):
            return pd.Series([1] * len(data), index=data.index)
        
        def validate_config(self, config):
            # Call parent's validation
            return super().validate_config(config)
    
    @pytest.fixture
    def strategy(self):
        """Create concrete strategy instance."""
        return self.ConcreteStrategy({'name': 'TestStrategy', 'symbols': ['TEST']})
    
    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'name': 'MyStrategy',
            'symbols': ['AAPL', 'GOOGL'],
            'lookback_period': 50,
            'enabled': False
        }
        
        strategy = self.ConcreteStrategy(config)
        
        assert strategy.name == 'MyStrategy'
        assert strategy.symbols == ['AAPL', 'GOOGL']
        assert strategy.lookback_period == 50
        assert strategy.enabled is False
        assert strategy._last_signal is None
    
    def test_performance_tracking(self, strategy):
        """Test performance statistics tracking."""
        # Update with winning trade
        strategy.update_performance({'pnl': 100})
        
        assert strategy._performance_stats['trades'] == 1
        assert strategy._performance_stats['wins'] == 1
        assert strategy._performance_stats['gross_pnl'] == 100
        
        # Update with losing trade
        strategy.update_performance({'pnl': -50})
        
        assert strategy._performance_stats['trades'] == 2
        assert strategy._performance_stats['losses'] == 1
        assert strategy._performance_stats['gross_pnl'] == 50
    
    def test_hit_rate(self, strategy):
        """Test hit rate calculation."""
        # No trades
        assert strategy.hit_rate == 0.0
        
        # Add trades
        strategy._performance_stats = {
            'trades': 10,
            'wins': 6,
            'losses': 4,
            'gross_pnl': 1000
        }
        
        assert strategy.hit_rate == 0.6
    
    def test_profit_factor(self, strategy):
        """Test profit factor calculation."""
        # No trades
        assert strategy.profit_factor == 0.0
        
        # Only wins
        strategy._performance_stats['wins'] = 5
        strategy._performance_stats['losses'] = 0
        assert strategy.profit_factor == float('inf')
        
        # Normal case
        strategy._performance_stats['wins'] = 6
        strategy._performance_stats['losses'] = 4
        assert strategy.profit_factor == 1.5
    
    def test_kelly_fraction(self, strategy):
        """Test Kelly fraction calculation."""
        # Not enough trades
        assert strategy.calculate_kelly_fraction() == 0.0
        
        # Enough trades with 60% win rate
        strategy._performance_stats = {
            'trades': 100,
            'wins': 60,
            'losses': 40
        }
        
        kelly = strategy.calculate_kelly_fraction()
        # Kelly = 2p - 1 = 2(0.6) - 1 = 0.2
        # Half Kelly = 0.1
        assert abs(kelly - 0.1) < 1e-10
        
        # Test edge case: p = 0 (all losses)
        strategy._performance_stats = {
            'trades': 50,
            'wins': 0,
            'losses': 50
        }
        assert strategy.calculate_kelly_fraction() == 0.0
        
        # Test edge case: p = 1 (all wins)
        strategy._performance_stats = {
            'trades': 50,
            'wins': 50,
            'losses': 0
        }
        assert strategy.calculate_kelly_fraction() == 0.0
    
    def test_validate_data(self, strategy):
        """Test data validation."""
        # Empty data
        assert not strategy.validate_data(pd.DataFrame())
        
        # Missing columns
        df = pd.DataFrame({'open': [100], 'close': [101]})
        assert not strategy.validate_data(df)
        
        # Valid data
        df = pd.DataFrame({
            'open': [100],
            'high': [101], 
            'low': [99],
            'close': [100.5],
            'volume': [1000000]
        })
        assert strategy.validate_data(df)
        
        # Check data quality
        df_bad = df.copy()
        df_bad.loc[0, 'high'] = 98  # High < Low
        assert not strategy.validate_data(df_bad)
        
        # Test with NaN values
        df_nan = df.copy()
        df_nan.loc[0, 'close'] = np.nan
        assert not strategy.validate_data(df_nan)
        
        # Test invalid OHLC relationships
        df_invalid = df.copy()
        df_invalid.loc[0, 'low'] = 102  # Low > High
        assert not strategy.validate_data(df_invalid)
    
    def test_validate_config(self, strategy):
        """Test configuration validation."""
        # Test with valid config
        config = {
            'name': 'TestStrategy',
            'lookback_period': 20,
            'enabled': True
        }
        validated = strategy.validate_config(config)
        assert validated == config
        
        # Test with invalid config type
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            strategy.validate_config("not a dict")
        
        # Test with invalid lookback_period (not int)
        with pytest.raises(ValueError, match="lookback_period must be a positive integer"):
            strategy.validate_config({'lookback_period': 'twenty'})
        
        # Test with invalid lookback_period (negative)
        with pytest.raises(ValueError, match="lookback_period must be a positive integer"):
            strategy.validate_config({'lookback_period': -5})
        
        # Test with invalid lookback_period (zero)
        with pytest.raises(ValueError, match="lookback_period must be a positive integer"):
            strategy.validate_config({'lookback_period': 0})
        
        # Test with invalid enabled (not bool)
        with pytest.raises(ValueError, match="enabled must be a boolean"):
            strategy.validate_config({'enabled': 'yes'})


class TestMeanReversionEquityStrategy:
    """Test suite for Mean Reversion Equity Strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Create mean-reverting price series
        price = 100
        prices = []
        for _ in range(100):
            price = price * 0.98 + 100 * 0.02 + np.random.randn() * 2
            prices.append(price)
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = {
            'symbol': 'TEST',
            'lookback': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 0.03,
            'position_size': 0.95
        }
        return MeanReversionEquityStrategy(config)
    
    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'symbol': 'SPY',
            'lookback': 30,
            'entry_threshold': 2.5,
            'exit_threshold': 0.2,
            'stop_loss': 0.02
        }
        
        strategy = MeanReversionEquityStrategy(config)
        
        assert strategy.symbol == 'SPY'
        assert strategy.lookback == 30
        assert strategy.entry_threshold == 2.5
        assert strategy.position == 0
    
    def test_calculate_signals_no_position(self, strategy, sample_data):
        """Test signal generation with no position."""
        strategy.position = 0
        
        # Create oversold condition
        oversold_data = sample_data.copy()
        oversold_data['close'] = oversold_data['close'] * 0.9  # 10% below normal
        
        signals = strategy.calculate_signals(oversold_data, 'TEST')
        
        # Should generate some buy signals
        assert (signals == 1).any()
    
    def test_calculate_signals_with_position(self, strategy, sample_data):
        """Test signal generation with existing position."""
        strategy.position = 100
        strategy.entry_price = 95.0
        
        # Price returns to mean
        mean_data = sample_data.copy()
        mean_data['close'] = 100  # At mean
        
        signals = strategy.calculate_signals(mean_data, 'TEST')
        
        # Should generate exit signals
        assert (signals == 0).any()
    
    def test_stop_loss_trigger(self, strategy, sample_data):
        """Test stop loss triggering."""
        strategy.position = 100
        strategy.entry_price = 100.0
        
        # Price drops below stop loss
        stop_data = sample_data.copy()
        stop_data['close'] = 96.0  # 4% loss (stop at 3%)
        
        signals = strategy.calculate_signals(stop_data, 'TEST')
        
        # Should exit position
        assert signals.iloc[-1] == 0
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        config = {
            'symbol': 'SPY',
            'lookback': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 0.03
        }
        
        strategy = MeanReversionEquityStrategy(config)
        validated = strategy.validate_config(config)
        assert validated == config
        
        # Invalid config (negative lookback)
        bad_config = config.copy()
        bad_config['lookback'] = -10
        
        with pytest.raises(ValueError):
            strategy.validate_config(bad_config)


class TestTrendFollowingMultiStrategy:
    """Test suite for Trend Following Multi Strategy."""
    
    @pytest.fixture
    def trending_data(self):
        """Create trending price data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create uptrending data
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 0.5
        prices = trend + noise
        
        return pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(100) * 0.5),
            'low': prices - np.abs(np.random.randn(100) * 0.5),
            'close': prices,
            'volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = {
            'symbols': ['SPY', 'QQQ', 'IWM'],
            'fast_ma': 10,
            'slow_ma': 30,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'max_positions': 2,
            'position_size': 0.3
        }
        return TrendFollowingMultiStrategy(config)
    
    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'symbols': ['AAPL', 'GOOGL'],
            'fast_ma': 20,
            'slow_ma': 50,
            'atr_period': 20,
            'atr_multiplier': 3.0,
            'max_positions': 3
        }
        
        strategy = TrendFollowingMultiStrategy(config)
        
        assert strategy.symbols == ['AAPL', 'GOOGL']
        assert strategy.fast_ma == 20
        assert strategy.slow_ma == 50
        assert strategy.positions == {}
    
    def test_calculate_signals_uptrend(self, strategy, trending_data):
        """Test signal generation in uptrend."""
        signals = strategy.calculate_signals(trending_data, 'SPY')
        
        # Should generate buy signals in uptrend
        # After moving averages are calculated
        assert (signals[strategy.slow_ma:] == 1).any()
    
    def test_calculate_signals_downtrend(self, strategy):
        """Test signal generation in downtrend."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create downtrending data
        trend = np.linspace(120, 100, 100)
        prices = trend + np.random.randn(100) * 0.5
        
        down_data = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': [1000000] * 100
        }, index=dates)
        
        signals = strategy.calculate_signals(down_data, 'SPY')
        
        # Should generate sell signals in downtrend
        assert (signals[strategy.slow_ma:] == -1).any()
    
    def test_position_limit(self, strategy, trending_data):
        """Test max positions limit."""
        # Fill up positions
        strategy.positions = {
            'SPY': 100,
            'QQQ': 100
        }
        
        # Try to add another position
        signals = strategy.calculate_signals(trending_data, 'IWM')
        
        # Should not generate new entry signals
        # when at max positions
        if len(strategy.positions) >= strategy.max_positions:
            assert (signals == 1).sum() == 0
    
    def test_stop_loss_calculation(self, strategy, trending_data):
        """Test ATR-based stop loss calculation."""
        strategy.positions['SPY'] = 100
        strategy.entry_prices['SPY'] = 110.0
        strategy.stop_losses['SPY'] = 105.0
        
        # Price hits stop loss
        stop_data = trending_data.copy()
        stop_data['close'] = 104.0
        
        signals = strategy.calculate_signals(stop_data, 'SPY')
        
        # Should exit position
        assert signals.iloc[-1] == 0
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        config = {
            'symbols': ['SPY', 'QQQ'],
            'fast_ma': 10,
            'slow_ma': 30,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'max_positions': 2
        }
        
        strategy = TrendFollowingMultiStrategy(config)
        validated = strategy.validate_config(config)
        assert validated == config
        
        # Invalid config (fast MA > slow MA)
        bad_config = config.copy()
        bad_config['fast_ma'] = 40
        
        with pytest.raises(ValueError):
            strategy.validate_config(bad_config)


class TestSignalClass:
    """Test suite for Signal data class."""
    
    def test_signal_creation(self):
        """Test creating valid signal."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='LONG',
            strength=0.8,
            strategy_id='test_strategy',
            price=150.0,
            atr=2.5,
            metadata={'reason': 'breakout'}
        )
        
        assert signal.symbol == 'AAPL'
        assert signal.direction == 'LONG'
        assert signal.strength == 0.8
    
    def test_signal_edge_case_validation(self):
        """Test edge cases where field validator info.data might be None."""
        # This tests the edge case in the validator where info.data could be None
        # We need to trigger this by creating scenarios that would cause this
        
        # Test creating a signal with zero strength for FLAT
        signal = Signal(
            timestamp=datetime.now(),
            symbol='TEST',
            direction='FLAT',
            strength=0.0,
            strategy_id='test',
            price=100.0
        )
        assert signal.strength == 0.0
    
    def test_signal_validation(self):
        """Test signal validation."""
        # Invalid direction
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',  # Should be LONG/SHORT/FLAT
                strength=0.8,
                strategy_id='test',
                price=150.0
            )
        
        # Invalid strength
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='LONG',
                strength=1.5,  # Must be between -1 and 1
                strategy_id='test',
                price=150.0
            )
    
    def test_signal_strength_validation(self):
        """Test signal strength validation."""
        # FLAT must have strength 0
        with pytest.raises(ValueError, match="FLAT signals must have strength=0"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='FLAT',
                strength=0.5,  # Must be 0 for FLAT
                strategy_id='test',
                price=150.0
            )
        
        # LONG must have positive strength
        with pytest.raises(ValueError, match="LONG signals must have positive strength"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='LONG',
                strength=-0.5,  # Must be positive for LONG
                strategy_id='test',
                price=150.0
            )
        
        # SHORT must have negative strength
        with pytest.raises(ValueError, match="SHORT signals must have negative strength"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='SHORT',
                strength=0.5,  # Must be negative for SHORT
                strategy_id='test',
                price=150.0
            )
    
    def test_valid_signal_strength_combinations(self):
        """Test valid signal/strength combinations."""
        # FLAT with 0 strength (valid)
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='FLAT',
            strength=0.0,
            strategy_id='test',
            price=150.0
        )
        assert signal.strength == 0.0
        
        # LONG with positive strength (valid)
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='LONG',
            strength=0.5,
            strategy_id='test',
            price=150.0
        )
        assert signal.strength == 0.5
        
        # SHORT with negative strength (valid) 
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='SHORT',
            strength=-0.5,
            strategy_id='test',
            price=150.0
        )
        assert signal.strength == -0.5
    
    def test_signal_validation_coverage(self):
        """Additional tests to ensure 100% coverage of Signal validation."""
        # Test FLAT with non-zero strength
        with pytest.raises(ValueError, match="FLAT signals must have strength=0"):
            Signal(
                timestamp=datetime.now(),
                symbol='TEST',
                direction='FLAT',
                strength=0.1,
                strategy_id='test',
                price=100.0
            )
        
        # Test LONG with negative strength
        with pytest.raises(ValueError, match="LONG signals must have positive strength"):
            Signal(
                timestamp=datetime.now(),
                symbol='TEST',
                direction='LONG',
                strength=-0.1,
                strategy_id='test',
                price=100.0
            )
        
        # Test SHORT with positive strength  
        with pytest.raises(ValueError, match="SHORT signals must have negative strength"):
            Signal(
                timestamp=datetime.now(),
                symbol='TEST',
                direction='SHORT',
                strength=0.1,
                strategy_id='test',
                price=100.0
            )