"""
Complete test coverage for BaseStrategy class.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from strategies.base import BaseStrategy, RiskContext, Signal


class TestSignal:
    """Test Signal pydantic model."""

    def test_signal_creation_valid(self):
        """Test creating a valid signal."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test_strategy",
            price=150.0
        )

        assert signal.symbol == "AAPL"
        assert signal.direction == "LONG"
        assert signal.strength == 0.8
        assert signal.strategy_id == "test_strategy"
        assert signal.price == 150.0

    def test_signal_optional_fields(self):
        """Test signal with optional fields."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="SHORT",
            strength=-0.5,
            strategy_id="test_strategy",
            price=150.0,
            atr=2.5,
            metadata={"reason": "MA crossover"}
        )

        assert signal.atr == 2.5
        assert signal.metadata["reason"] == "MA crossover"

    def test_signal_validation_errors(self):
        """Test signal validation errors."""
        # Invalid direction
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="INVALID",
                strength=0.5,
                strategy_id="test",
                price=150.0
            )

        # Invalid strength range
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=1.5,  # > 1.0
                strategy_id="test",
                price=150.0
            )

        # LONG with negative strength
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=-0.5,
                strategy_id="test",
                price=150.0
            )

        # SHORT with positive strength
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="SHORT",
                strength=0.5,
                strategy_id="test",
                price=150.0
            )

        # FLAT with non-zero strength
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="FLAT",
                strength=0.5,
                strategy_id="test",
                price=150.0
            )

    def test_signal_flat_direction(self):
        """Test FLAT signal creation."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="FLAT",
            strength=0.0,
            strategy_id="test",
            price=150.0
        )
        assert signal.direction == "FLAT"
        assert signal.strength == 0.0


class TestRiskContext:
    """Test RiskContext dataclass."""

    def test_risk_context_defaults(self):
        """Test default values."""
        context = RiskContext(
            account_equity=100000,
            open_positions=2,
            daily_pnl=500,
            max_drawdown_pct=0.05
        )

        assert context.volatility_target == 0.10
        assert context.max_position_size == 0.20
        assert context.current_regime == "NORMAL"

    def test_risk_context_custom(self):
        """Test custom values."""
        context = RiskContext(
            account_equity=50000,
            open_positions=5,
            daily_pnl=-1000,
            max_drawdown_pct=0.15,
            volatility_target=0.05,
            max_position_size=0.10,
            current_regime="RISK_OFF"
        )

        assert context.current_regime == "RISK_OFF"
        assert context.volatility_target == 0.05


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation for testing."""

    def init(self):
        self.initialized = True

    def next(self, data):
        return Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.5,
            strategy_id=self.name,
            price=100.0
        )

    def size(self, signal, risk_context):
        return (1000.0, 95.0)


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'name': 'TestStrategy',
            'symbols': ['AAPL', 'GOOGL'],
            'lookback_period': 30,
            'enabled': True
        }

        strategy = ConcreteStrategy(config)

        assert strategy.name == 'TestStrategy'
        assert strategy.symbols == ['AAPL', 'GOOGL']
        assert strategy.lookback_period == 30
        assert strategy.enabled is True
        assert strategy._last_signal is None
        assert strategy._performance_stats['trades'] == 0

    def test_initialization_defaults(self):
        """Test default values."""
        strategy = ConcreteStrategy({})

        assert strategy.name == 'ConcreteStrategy'
        assert strategy.symbols == []
        assert strategy.lookback_period == 252
        assert strategy.enabled is True

    def test_validate_config_valid(self):
        """Test valid configuration."""
        config = {
            'lookback_period': 20,
            'enabled': True
        }
        strategy = ConcreteStrategy(config)
        validated = strategy.validate_config(config)
        assert validated == config

    def test_validate_config_invalid(self):
        """Test invalid configuration."""
        # Not a dict
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            ConcreteStrategy("not a dict")

        # Invalid lookback_period
        with pytest.raises(ValueError, match="lookback_period must be a positive integer"):
            ConcreteStrategy({'lookback_period': -5})

        with pytest.raises(ValueError, match="lookback_period must be a positive integer"):
            ConcreteStrategy({'lookback_period': 'twenty'})

        # Invalid enabled
        with pytest.raises(ValueError, match="enabled must be a boolean"):
            ConcreteStrategy({'enabled': 'yes'})

    def test_update_performance(self):
        """Test performance tracking."""
        strategy = ConcreteStrategy({})

        # Winning trade
        strategy.update_performance({'pnl': 100})
        assert strategy._performance_stats['trades'] == 1
        assert strategy._performance_stats['wins'] == 1
        assert strategy._performance_stats['losses'] == 0
        assert strategy._performance_stats['gross_pnl'] == 100

        # Losing trade
        strategy.update_performance({'pnl': -50})
        assert strategy._performance_stats['trades'] == 2
        assert strategy._performance_stats['wins'] == 1
        assert strategy._performance_stats['losses'] == 1
        assert strategy._performance_stats['gross_pnl'] == 50

    def test_hit_rate(self):
        """Test hit rate calculation."""
        strategy = ConcreteStrategy({})

        # No trades
        assert strategy.hit_rate == 0.0

        # Add trades
        strategy.update_performance({'pnl': 100})
        strategy.update_performance({'pnl': 50})
        strategy.update_performance({'pnl': -30})

        assert strategy.hit_rate == 2/3

    def test_profit_factor(self):
        """Test profit factor calculation."""
        strategy = ConcreteStrategy({})

        # No trades
        assert strategy.profit_factor == 0.0

        # Only wins
        strategy._performance_stats['wins'] = 5
        strategy._performance_stats['losses'] = 0
        assert strategy.profit_factor == float('inf')

        # Mixed results
        strategy._performance_stats['wins'] = 6
        strategy._performance_stats['losses'] = 4
        assert strategy.profit_factor == 1.5

    def test_calculate_kelly_fraction(self):
        """Test Kelly fraction calculation."""
        strategy = ConcreteStrategy({})

        # Not enough trades
        assert strategy.calculate_kelly_fraction() == 0.0

        # Add 30 trades with 60% win rate
        strategy._performance_stats['trades'] = 30
        strategy._performance_stats['wins'] = 18

        # Kelly = 2 * 0.6 - 1 = 0.2
        # Half Kelly = 0.1
        assert pytest.approx(strategy.calculate_kelly_fraction(), 0.001) == 0.1

        # Edge cases
        strategy._performance_stats['wins'] = 0
        assert strategy.calculate_kelly_fraction() == 0.0

        strategy._performance_stats['wins'] = 30
        assert strategy.calculate_kelly_fraction() == 0.0

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        strategy = ConcreteStrategy({})

        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })

        assert strategy.validate_data(data) is True

    def test_validate_data_invalid(self):
        """Test data validation with invalid data."""
        strategy = ConcreteStrategy({})

        # Empty DataFrame
        assert strategy.validate_data(pd.DataFrame()) is False

        # Missing columns
        data = pd.DataFrame({'close': [100, 101]})
        assert strategy.validate_data(data) is False

        # NaN values
        data = pd.DataFrame({
            'open': [100, np.nan],
            'high': [105, 106],
            'low': [99, 100],
            'close': [104, 105],
            'volume': [1000, 1100]
        })
        assert strategy.validate_data(data) is False

        # Invalid OHLC relationships
        data = pd.DataFrame({
            'open': [100],
            'high': [95],  # High < Open
            'low': [99],
            'close': [104],
            'volume': [1000]
        })
        assert strategy.validate_data(data) is False

    def test_generate_signals_single_symbol(self):
        """Test generate_signals with single symbol."""
        config = {'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        data = pd.DataFrame({'close': [100, 101, 102]})

        # Base class implementation returns empty list
        signals = strategy.generate_signals(data)
        assert len(signals) == 0

    def test_generate_signals_disabled(self):
        """Test generate_signals when strategy is disabled."""
        config = {'symbols': ['AAPL'], 'enabled': False}
        strategy = ConcreteStrategy(config)

        data = pd.DataFrame({'close': [100, 101, 102]})

        signals = strategy.generate_signals(data)
        assert len(signals) == 0

    def test_generate_signals_invalid_data(self):
        """Test generate_signals with invalid data."""
        config = {'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        # Empty data
        signals = strategy.generate_signals(pd.DataFrame())
        assert len(signals) == 0

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            BaseStrategy({})


class TestBaseStrategyIntegration:
    """Integration tests for BaseStrategy."""

    def test_full_strategy_workflow(self):
        """Test complete strategy workflow."""
        class MomentumStrategy(BaseStrategy):
            def init(self):
                self.momentum_period = 10

            def next(self, data):
                if len(data) < self.momentum_period:
                    return None

                returns = data['close'].pct_change(self.momentum_period).iloc[-1]

                if returns > 0.05:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=data.iloc[-1].get('symbol', 'AAPL'),
                        direction="LONG",
                        strength=min(returns * 10, 1.0),
                        strategy_id=self.name,
                        price=data['close'].iloc[-1]
                    )
                elif returns < -0.05:
                    return Signal(
                        timestamp=datetime.now(),
                        symbol=data.iloc[-1].get('symbol', 'AAPL'),
                        direction="SHORT",
                        strength=max(returns * 10, -1.0),
                        strategy_id=self.name,
                        price=data['close'].iloc[-1]
                    )

                return None

            def size(self, signal, risk_context):
                # Kelly-based sizing
                kelly = self.calculate_kelly_fraction()
                if kelly <= 0:
                    kelly = 0.02  # Default 2% risk

                position_value = risk_context.account_equity * kelly
                shares = position_value / signal.price

                # Stop loss 2% below entry for longs
                if signal.direction == "LONG":
                    stop_loss = signal.price * 0.98
                else:
                    stop_loss = signal.price * 1.02

                return (shares, stop_loss)

        # Initialize strategy
        config = {
            'name': 'Momentum10',
            'symbols': ['AAPL'],
            'lookback_period': 15
        }
        strategy = MomentumStrategy(config)
        strategy.init()

        # Create test data with uptrend
        dates = pd.date_range('2024-01-01', periods=20)
        prices = 100 + np.arange(20) * 0.6  # 0.6% daily = 6% over 10 days
        data = pd.DataFrame({
            'open': prices - 0.5,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.random.randint(1000, 2000, 20)
        }, index=dates)

        # Generate signal
        signal = strategy.next(data)
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.strength > 0

        # Calculate position size
        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0
        )

        shares, stop_loss = strategy.size(signal, risk_context)
        assert shares > 0
        assert stop_loss < signal.price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
