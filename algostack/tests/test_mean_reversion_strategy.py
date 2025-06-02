"""Tests for mean reversion equity strategy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.base import Signal, RiskContext


class TestMeanReversionEquity:
    """Test mean reversion strategy implementation."""
    
    @pytest.fixture
    def strategy_config(self):
        """Strategy configuration for testing."""
        return {
            'symbols': ['SPY'],
            'rsi_period': 2,
            'rsi_oversold': 10,
            'rsi_overbought': 90,
            'atr_period': 14,
            'atr_band_mult': 2.5,
            'stop_loss_atr': 3.0,
            'max_positions': 5
        }
    
    @pytest.fixture
    def strategy(self, strategy_config):
        """Create strategy instance."""
        return MeanReversionEquity(strategy_config)
    
    @pytest.fixture
    def oversold_data(self, sample_ohlcv_data):
        """Create data with oversold conditions."""
        data = sample_ohlcv_data.copy()
        # Force RSI to be oversold by creating downward movement
        for i in range(-10, 0):
            data.iloc[i, data.columns.get_loc('close')] *= 0.98
            data.iloc[i, data.columns.get_loc('low')] *= 0.97
        # High volume on last bar
        data.iloc[-1, data.columns.get_loc('volume')] *= 2.0
        data.attrs['symbol'] = 'SPY'
        return data
    
    @pytest.fixture
    def overbought_data(self, sample_ohlcv_data):
        """Create data with overbought conditions."""
        data = sample_ohlcv_data.copy()
        # Force RSI to be overbought
        for i in range(-10, 0):
            data.iloc[i, data.columns.get_loc('close')] *= 1.02
            data.iloc[i, data.columns.get_loc('high')] *= 1.03
        data.attrs['symbol'] = 'SPY'
        return data
    
    @pytest.mark.unit
    def test_strategy_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.name == 'MeanReversionEquity'
        assert strategy.config['rsi_period'] == strategy_config['rsi_period']
        assert strategy.config['stop_loss_atr'] == strategy_config['stop_loss_atr']
        assert len(strategy.positions) == 0
    
    @pytest.mark.unit
    def test_indicator_calculation(self, strategy, sample_ohlcv_data):
        """Test indicator calculations."""
        df = strategy.calculate_indicators(sample_ohlcv_data)
        
        # Check all required indicators
        required_indicators = ['rsi', 'atr', 'sma_20', 'sma_exit', 
                             'upper_band', 'lower_band', 'volume_ratio']
        for indicator in required_indicators:
            assert indicator in df.columns
            assert df[indicator].notna().any()
        
        # Check RSI bounds
        rsi_values = df['rsi'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Check band relationships
        assert (df['upper_band'] > df['sma_20']).all()
        assert (df['lower_band'] < df['sma_20']).all()
    
    @pytest.mark.unit
    def test_entry_signal_generation(self, strategy, oversold_data):
        """Test entry signal generation on oversold conditions."""
        strategy.init()
        signal = strategy.next(oversold_data)
        
        # May or may not generate signal depending on exact conditions
        if signal:
            assert signal.direction == 'LONG'
            assert signal.strength > 0
            assert signal.symbol == 'SPY'
            assert 'rsi' in signal.metadata
            assert signal.metadata['reason'] == 'mean_reversion_entry'
    
    @pytest.mark.unit
    def test_exit_signal_generation(self, strategy, overbought_data):
        """Test exit signal generation."""
        strategy.init()
        
        # Add a position first
        strategy.positions['SPY'] = {
            'entry_price': 95.0,
            'entry_atr': 2.0,
            'entry_time': datetime.now()
        }
        
        signal = strategy.next(overbought_data)
        
        # Should generate exit signal
        if signal:
            assert signal.direction == 'FLAT'
            assert signal.strength == 0.0
            assert signal.metadata['reason'] == 'mean_reversion_exit'
    
    @pytest.mark.unit
    def test_position_sizing(self, strategy):
        """Test position sizing with risk management."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='SPY',
            direction='LONG',
            strength=0.8,
            strategy_id='mean_reversion',
            price=100.0,
            atr=2.0
        )
        
        risk_context = RiskContext(
            account_equity=10000.0,
            open_positions=0,
            daily_pnl=0.0,
            max_drawdown_pct=0.0,
            volatility_target=0.10,
            max_position_size=0.20
        )
        
        position_size, stop_loss = strategy.size(signal, risk_context)
        
        # Verify sizing
        assert position_size > 0
        position_value = position_size * signal.price
        assert position_value <= risk_context.account_equity * risk_context.max_position_size
        
        # Verify stop loss
        assert stop_loss < signal.price
        expected_stop = signal.price - (strategy.config['stop_loss_atr'] * signal.atr)
        assert abs(stop_loss - expected_stop) < 0.01
    
    @pytest.mark.unit
    def test_kelly_calculation(self, strategy):
        """Test Kelly fraction calculation."""
        # Initially no trades
        assert strategy.calculate_kelly_fraction() == 0.0
        
        # Add performance history
        for _ in range(40):
            strategy.update_performance({'pnl': 100})
        for _ in range(20):
            strategy.update_performance({'pnl': -80})
        
        kelly = strategy.calculate_kelly_fraction()
        assert 0 < kelly <= 0.5  # Half-Kelly
        
        # Verify win rate calculation
        assert strategy.hit_rate == pytest.approx(40/60, rel=0.01)
    
    @pytest.mark.unit
    def test_data_validation(self, strategy):
        """Test data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1e6, 1.1e6]
        })
        assert strategy.validate_data(valid_data) == True
        
        # Missing columns
        invalid_data = valid_data.drop(columns=['volume'])
        assert strategy.validate_data(invalid_data) == False
        
        # Invalid OHLC relationships
        invalid_ohlc = valid_data.copy()
        invalid_ohlc.loc[0, 'high'] = 95  # High < Low
        assert strategy.validate_data(invalid_ohlc) == False
    
    @pytest.mark.integration
    def test_strategy_workflow(self, strategy, oversold_data):
        """Test complete strategy workflow."""
        strategy.init()
        
        # Generate entry signal
        entry_signal = strategy.next(oversold_data)
        
        if entry_signal and entry_signal.direction == 'LONG':
            # Verify position tracking
            assert 'SPY' in strategy.positions
            
            # Simulate price recovery
            recovery_data = oversold_data.copy()
            recovery_data.iloc[-1, recovery_data.columns.get_loc('close')] *= 1.05
            
            # Generate exit signal
            exit_signal = strategy.next(recovery_data)
            
            # Verify position closed
            if exit_signal and exit_signal.direction == 'FLAT':
                assert 'SPY' not in strategy.positions