"""
Test coverage for BaseStrategy class.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from algostack.strategies.base import BaseStrategy, Signal, RiskContext


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
            stop_loss=145.0,
            take_profit=160.0,
            metadata={"reason": "MA crossover"}
        )
        
        assert signal.stop_loss == 145.0
        assert signal.take_profit == 160.0
        assert signal.metadata["reason"] == "MA crossover"


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'name': 'TestStrategy',
            'symbols': ['AAPL', 'GOOGL'],
            'lookback': 20,
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
        
        strategy = ConcreteStrategy(config)
        
        assert strategy.name == 'TestStrategy'
        assert strategy.symbols == ['AAPL', 'GOOGL']
        assert strategy.lookback == 20
        assert strategy.parameters['threshold'] == 0.5
    
    def test_size_method(self):
        """Test position sizing method."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass
            
            def next(self, data):
                return None
            
            def calculate_signals(self, data):
                return []
            
            def size(self, symbol):
                return 100
        
        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)
        
        assert strategy.size('AAPL') == 100
    
    def test_is_ready(self):
        """Test is_ready method."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass
            
            def next(self, data):
                return None
            
            def calculate_signals(self, data):
                return []
        
        config = {'name': 'Test', 'symbols': ['AAPL'], 'lookback': 10}
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
        
        config = {
            'name': 'Test',
            'symbols': ['AAPL'],
            'parameters': {'ma_period': 20}
        }
        strategy = ConcreteStrategy(config)
        
        # Update parameters
        new_params = {'ma_period': 30, 'threshold': 0.7}
        strategy.update_parameters(new_params)
        
        assert strategy.parameters['ma_period'] == 30
        assert strategy.parameters['threshold'] == 0.7
    
    def test_validate_signal_valid(self):
        """Test signal validation with valid signal."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass
            
            def next(self, data):
                return None
            
            def calculate_signals(self, data):
                return []
        
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
        
        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)
        
        # Wrong symbol
        signal1 = Signal(
            symbol="TSLA",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )
        assert strategy.validate_signal(signal1) is False
        
        # Wrong strategy ID
        signal2 = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            timestamp=datetime.now(),
            strategy_id="Other",
            price=150.0
        )
        assert strategy.validate_signal(signal2) is False
        
        # Invalid strength for LONG
        signal3 = Signal(
            symbol="AAPL",
            direction="LONG",
            strength=-0.5,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )
        assert strategy.validate_signal(signal3) is False
        
        # Invalid strength for SHORT
        signal4 = Signal(
            symbol="AAPL",
            direction="SHORT",
            strength=0.5,
            timestamp=datetime.now(),
            strategy_id="Test",
            price=150.0
        )
        assert strategy.validate_signal(signal4) is False
    
    def test_format_signal(self):
        """Test signal formatting."""
        class ConcreteStrategy(BaseStrategy):
            def init(self):
                pass
            
            def next(self, data):
                return None
            
            def calculate_signals(self, data):
                return []
        
        config = {'name': 'Test', 'symbols': ['AAPL']}
        strategy = ConcreteStrategy(config)
        
        # Test with minimal data
        signal_data = {
            'symbol': 'AAPL',
            'direction': 'LONG',
            'strength': 0.8
        }
        
        signal = strategy.format_signal(signal_data)
        assert isinstance(signal, Signal)
        assert signal.symbol == 'AAPL'
        assert signal.direction == 'LONG'
        assert signal.strength == 0.8
        assert signal.strategy_id == 'Test'
        
        # Test with full data
        signal_data_full = {
            'symbol': 'AAPL',
            'direction': 'SHORT',
            'strength': -0.5,
            'price': 150.0,
            'stop_loss': 155.0,
            'take_profit': 140.0,
            'metadata': {'reason': 'test'}
        }
        
        signal_full = strategy.format_signal(signal_data_full)
        assert signal_full.price == 150.0
        assert signal_full.stop_loss == 155.0
        assert signal_full.take_profit == 140.0
        assert signal_full.metadata['reason'] == 'test'


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
                self.ma_period = self.parameters.get('ma_period', 20)
            
            def next(self, data):
                if len(data) < self.ma_period:
                    return None
                
                # Simple MA crossover
                price = data['close'].iloc[-1]
                ma = data['close'].rolling(self.ma_period).mean().iloc[-1]
                
                if price > ma:
                    return self.format_signal({
                        'symbol': 'AAPL',
                        'direction': 'LONG',
                        'strength': 0.7,
                        'price': price
                    })
                else:
                    return self.format_signal({
                        'symbol': 'AAPL',
                        'direction': 'SHORT',
                        'strength': -0.7,
                        'price': price
                    })
            
            def calculate_signals(self, data):
                signals = []
                for symbol in self.symbols:
                    signal = self.next(data[symbol])
                    if signal:
                        signals.append(signal)
                return signals
        
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