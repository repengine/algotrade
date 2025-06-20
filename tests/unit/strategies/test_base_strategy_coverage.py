"""
Test coverage for BaseStrategy class.
"""

from datetime import datetime

import pandas as pd
import pytest
from strategies.base import BaseStrategy, RiskContext, Signal


class TestSignal:
    """Test Signal data class."""

    def test_signal_creation(self):
        """Test creating a signal."""
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
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
            symbol="AAPL",
            direction="SHORT",
            strength=-0.5,
            timestamp=datetime.now(),
            strategy_id="test_strategy",
            price=150.0,
            atr=2.5,  # This is an actual optional field
            metadata={"reason": "MA crossover", "stop_loss": 145.0, "take_profit": 160.0}
        )

        assert signal.atr == 2.5
        assert signal.metadata["reason"] == "MA crossover"
        # Stop loss and take profit can be stored in metadata if needed
        assert signal.metadata["stop_loss"] == 145.0
        assert signal.metadata["take_profit"] == 160.0


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'name': 'TestStrategy',
            'symbols': ['AAPL', 'GOOGL'],
            'lookback_period': 20,
            'parameters': {'threshold': 0.5}
        }

        # Create concrete implementation
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        strategy = ConcreteStrategy(config)

        assert strategy.name == 'TestStrategy'
        assert strategy.symbols == ['AAPL', 'GOOGL']
        assert strategy.lookback_period == 20
        assert strategy.get_parameters()['threshold'] == 0.5

    def test_size_method(self):
        """Test position sizing method."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                # Simple fixed size
                return (100, signal.price * 0.95)

        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        # Create test signal and risk context
        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )
        
        risk_context = RiskContext(
            account_equity=100000,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.0
        )

        size, stop_loss = strategy.size(signal, risk_context)
        assert size == 100
        assert stop_loss == 150.0 * 0.95

    def test_is_ready(self):
        """Test is_ready method."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {'name': 'Test', 'symbols': ['AAPL'], 'lookback_period': 10}
        strategy = ConcreteStrategy(config)

        # Not ready with insufficient data
        assert not strategy.is_ready(5)

        # Ready with sufficient data
        assert strategy.is_ready(15)

    def test_log_method(self):
        """Test logging method."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        # Should not raise
        strategy.log("Test message")
        strategy.log("Debug message", level="DEBUG")

    def test_get_parameters(self):
        """Test getting strategy parameters."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {
            'name': 'Test',
            'symbols': ['AAPL'],
            'parameters': {'ma_period': 20, 'threshold': 0.5}
        }
        strategy = ConcreteStrategy(config)

        params = strategy.get_parameters()
        assert params['ma_period'] == 20
        assert params['threshold'] == 0.5

    def test_update_parameters(self):
        """Test updating strategy parameters."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {
            'name': 'Test',
            'symbols': ['AAPL'],
            'parameters': {'ma_period': 20}
        }
        strategy = ConcreteStrategy(config)

        # Update parameters
        strategy.update_parameters(ma_period=30, threshold=0.7)

        params = strategy.get_parameters()
        assert params['ma_period'] == 30
        assert params['threshold'] == 0.7

    def test_validate_signal_valid(self):
        """Test signal validation with valid signal."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )

        assert strategy.validate_signal(signal) is True

    def test_validate_signal_invalid(self):
        """Test signal validation with invalid signals."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        # Test empty symbol - this is allowed by pydantic
        signal1 = Signal(
            symbol="",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )
        # Our validate_signal method should reject it
        assert strategy.validate_signal(signal1) is False

        # Invalid direction
        from pydantic import ValidationError
        try:
            signal2 = Signal(
                symbol="AAPL",
                direction="INVALID",
                strength=0.8,
                timestamp=datetime.now(),
                strategy_id="Test",
                price=150.0
            )
            assert False, "Should have raised validation error"
        except ValidationError:
            pass  # Expected

        # Invalid strength for LONG
        try:
            signal3 = Signal(
                symbol="AAPL",
                direction="LONG",
                strength=-0.5,
                timestamp=datetime.now(),
                strategy_id="Test",
                price=150.0
            )
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected

        # Invalid strength for SHORT
        try:
            signal4 = Signal(
                symbol="AAPL",
                direction="SHORT",
                strength=0.5,
                timestamp=datetime.now(),
                strategy_id="Test",
                price=150.0
            )
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected

    def test_format_signal(self):
        """Test signal formatting."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass

            def next(self, data):
                return None

            def calculate_signals(self, data):
                return []

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)

        signal = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )

        formatted = strategy.format_signal(signal)
        assert formatted == "AAPL LONG @ 0.80"


class TestRiskContext:
    """Test RiskContext data class."""

    def test_risk_context_creation(self):
        """Test creating risk context."""
        context = RiskContext(
            account_equity=100000,
            open_positions=3,
            daily_pnl=500,
            max_drawdown_pct=0.05
        )

        assert context.account_equity == 100000
        assert context.open_positions == 3
        assert context.daily_pnl == 500
        assert context.max_drawdown_pct == 0.05
        assert context.volatility_target == 0.10
        assert context.max_position_size == 0.20
        assert context.current_regime == "NORMAL"

    def test_risk_context_custom_values(self):
        """Test risk context with custom values."""
        context = RiskContext(
            account_equity=50000,
            open_positions=1,
            daily_pnl=-200,
            max_drawdown_pct=0.10,
            volatility_target=0.15,
            max_position_size=0.25,
            current_regime="HIGH_VOL"
        )

        assert context.volatility_target == 0.15
        assert context.max_position_size == 0.25
        assert context.current_regime == "HIGH_VOL"


class TestBaseStrategyIntegration:
    """Test BaseStrategy with mock data."""

    def test_strategy_workflow(self):
        """Test complete strategy workflow."""
        class TestStrategy(BaseStrategy):
            def init(self):
                self.ma_period = self.get_parameters().get('ma_period', 20)

            def next(self, data):
                if len(data) < self.ma_period:
                    return None

                # Simple MA crossover
                price = data['close'].iloc[-1]
                ma = data['close'].rolling(self.ma_period).mean().iloc[-1]

                if price > ma:
                    return Signal(
                        symbol='AAPL',
                        direction='LONG',
                        strength=0.7,
                        timestamp=datetime.now(),
                        strategy_id=self.name,
                        price=price
                    )
                else:
                    return Signal(
                        symbol='AAPL',
                        direction='SHORT',
                        strength=-0.7,
                        timestamp=datetime.now(),
                        strategy_id=self.name,
                        price=price
                    )

            def calculate_signals(self, data):
                signals = []
                for symbol in self.symbols:
                    if symbol in data:
                        signal = self.next(data[symbol])
                        if signal:
                            signals.append(signal)
                return signals

            def size(self, signal, risk_context):
                return (100, signal.price * 0.95)

        config = {
            'name': 'TestMA',
            'symbols': ['AAPL'],
            'parameters': {'ma_period': 3}
        }

        strategy = TestStrategy(config)
        strategy.init()

        # Create test data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 102, 101, 100, 99, 98, 97]
        })

        # Test signal generation
        signal = strategy.next(data)
        assert signal is not None
        assert isinstance(signal, Signal)

        # Test calculate_signals
        data_dict = {'AAPL': data}
        signals = strategy.calculate_signals(data_dict)
        assert len(signals) == 1
        assert signals[0].symbol == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])