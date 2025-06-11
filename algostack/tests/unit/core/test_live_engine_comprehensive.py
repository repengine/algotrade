"""Comprehensive test suite for live trading engine."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.live_engine import LiveTradingEngine, TradingMode


class TestLiveTradingEngine:
    """Test suite for LiveTradingEngine class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'mode': 'PAPER',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'strategies': {
                'mean_reversion': {
                    'enabled': True,
                    'params': {'lookback': 20}
                },
                'momentum': {
                    'enabled': False,
                    'params': {'period': 10}
                }
            },
            'risk': {
                'max_position_size': 0.20,
                'max_portfolio_risk': 0.06,
                'stop_loss_pct': 0.02
            },
            'execution': {
                'broker': 'paper',
                'commission': 0.001
            },
            'data': {
                'providers': ['yfinance'],
                'update_frequency': 60
            }
        }
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        return {
            'portfolio': Mock(),
            'risk_manager': Mock(),
            'data_handler': Mock(),
            'executor': Mock(),
            'strategy_loader': Mock()
        }
    
    @pytest.fixture
    def live_engine(self, config, mock_dependencies):
        """Create LiveTradingEngine instance."""
        with patch('core.live_engine.PortfolioEngine', return_value=mock_dependencies['portfolio']):
            with patch('core.live_engine.EnhancedRiskManager', return_value=mock_dependencies['risk_manager']):
                with patch('core.live_engine.DataHandler', return_value=mock_dependencies['data_handler']):
                    with patch('core.live_engine.create_executor', return_value=mock_dependencies['executor']):
                        engine = LiveTradingEngine(config)
                        engine.strategy_loader = mock_dependencies['strategy_loader']
                        return engine
    
    def test_initialization(self, config):
        """Test LiveTradingEngine initialization."""
        with patch('core.live_engine.PortfolioEngine'):
            with patch('core.live_engine.EnhancedRiskManager'):
                with patch('core.live_engine.DataHandler'):
                    with patch('core.live_engine.create_executor'):
                        engine = LiveTradingEngine(config)
                        
                        assert engine.config == config
                        assert engine.mode == TradingMode.PAPER
                        assert engine.symbols == ['AAPL', 'GOOGL', 'MSFT']
                        assert engine.running is False
                        assert engine.stats['errors'] == 0
                        assert engine.stats['signals_generated'] == 0
    
    def test_mode_validation(self):
        """Test trading mode validation."""
        # Valid modes
        for mode in ['PAPER', 'LIVE', 'BACKTEST']:
            config = {'mode': mode}
            with patch('core.live_engine.PortfolioEngine'):
                engine = LiveTradingEngine(config)
                assert engine.mode == TradingMode[mode]
        
        # Invalid mode
        config = {'mode': 'INVALID'}
        with pytest.raises(ValueError):
            LiveTradingEngine(config)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, live_engine):
        """Test starting and stopping the engine."""
        # Mock dependencies
        live_engine.executor.connect = AsyncMock(return_value=True)
        live_engine.executor.disconnect = AsyncMock()
        live_engine.data_handler.initialize = AsyncMock()
        
        # Mock the run loop to stop after one iteration
        async def mock_run():
            live_engine.running = False
        
        live_engine._run = mock_run
        
        # Start engine
        await live_engine.start()
        
        assert live_engine.executor.connect.called
        assert live_engine.data_handler.initialize.called
        
        # Stop engine
        await live_engine.stop()
        
        assert not live_engine.running
        assert live_engine.executor.disconnect.called
    
    @pytest.mark.asyncio
    async def test_update_market_data(self, live_engine):
        """Test market data updates."""
        # Mock data handler response
        market_data = {
            'AAPL': {'open': 150, 'high': 152, 'low': 149, 'close': 151, 'volume': 1000000},
            'GOOGL': {'open': 2800, 'high': 2850, 'low': 2790, 'close': 2820, 'volume': 500000}
        }
        live_engine.data_handler.get_latest = AsyncMock(return_value=market_data)
        
        # Update market data
        await live_engine._update_market_data()
        
        assert live_engine.current_prices['AAPL'] == 151
        assert live_engine.current_prices['GOOGL'] == 2820
        assert live_engine.stats['data_updates'] == 1
    
    @pytest.mark.asyncio
    async def test_process_strategies(self, live_engine):
        """Test strategy processing."""
        # Setup strategies
        mock_strategy = Mock()
        mock_strategy.enabled = True
        mock_strategy.calculate_signals = Mock(return_value=pd.Series([1]))  # Buy signal
        live_engine.strategies = {'test_strategy': mock_strategy}
        
        # Setup market data
        live_engine.market_data = {
            'AAPL': pd.DataFrame({
                'close': [150, 151, 152],
                'volume': [1000000, 1100000, 1200000]
            })
        }
        
        # Setup risk approval
        live_engine.risk_manager.pre_trade_check = Mock(
            return_value={'approved': True}
        )
        
        # Process strategies
        await live_engine._process_strategies()
        
        mock_strategy.calculate_signals.assert_called()
        assert live_engine.stats['signals_generated'] > 0
    
    @pytest.mark.asyncio
    async def test_execute_signals(self, live_engine):
        """Test signal execution."""
        # Add signal to queue
        signal = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'strategy': 'test_strategy'
        }
        await live_engine.signal_queue.put(signal)
        
        # Mock risk check and execution
        live_engine.risk_manager.pre_trade_check = Mock(
            return_value={'approved': True}
        )
        live_engine.executor.place_order = AsyncMock(
            return_value={'order_id': 'ORD123', 'status': 'FILLED'}
        )
        live_engine.portfolio_engine.process_fill = Mock()
        
        # Execute signals
        await live_engine._execute_signals()
        
        assert live_engine.signal_queue.empty()
        live_engine.executor.place_order.assert_called_once()
        assert live_engine.stats['trades_executed'] == 1
    
    @pytest.mark.asyncio
    async def test_risk_rejection(self, live_engine):
        """Test risk rejection of signals."""
        # Add high-risk signal
        signal = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10000,  # Large position
            'strategy': 'test_strategy'
        }
        await live_engine.signal_queue.put(signal)
        
        # Mock risk rejection
        live_engine.risk_manager.pre_trade_check = Mock(
            return_value={'approved': False, 'reason': 'Position too large'}
        )
        
        # Execute signals
        await live_engine._execute_signals()
        
        assert live_engine.signal_queue.empty()
        assert live_engine.stats['trades_executed'] == 0
        assert live_engine.stats['signals_rejected'] == 1
    
    @pytest.mark.asyncio
    async def test_update_positions(self, live_engine):
        """Test position updates."""
        # Setup current prices
        live_engine.current_prices = {'AAPL': 155, 'GOOGL': 2850}
        
        # Mock portfolio update
        live_engine.portfolio_engine.update_positions = Mock()
        
        # Update positions
        await live_engine._update_positions()
        
        live_engine.portfolio_engine.update_positions.assert_called_with(
            live_engine.current_prices
        )
    
    @pytest.mark.asyncio
    async def test_check_stops(self, live_engine):
        """Test stop loss checking."""
        # Setup positions with stops
        live_engine.stop_orders = {
            'AAPL': {'stop_price': 145, 'quantity': 100}
        }
        live_engine.current_prices = {'AAPL': 144}  # Below stop
        
        # Mock order execution
        live_engine.executor.place_order = AsyncMock(
            return_value={'order_id': 'STOP123', 'status': 'FILLED'}
        )
        
        # Check stops
        await live_engine._check_stops()
        
        # Verify stop triggered
        live_engine.executor.place_order.assert_called_once()
        order_call = live_engine.executor.place_order.call_args[0][0]
        assert order_call['symbol'] == 'AAPL'
        assert order_call['side'] == 'SELL'
    
    def test_add_strategy(self, live_engine):
        """Test adding a strategy."""
        mock_strategy = Mock()
        live_engine.add_strategy('new_strategy', mock_strategy)
        
        assert 'new_strategy' in live_engine.strategies
        assert live_engine.strategies['new_strategy'] == mock_strategy
    
    def test_remove_strategy(self, live_engine):
        """Test removing a strategy."""
        live_engine.strategies = {'test_strategy': Mock()}
        
        live_engine.remove_strategy('test_strategy')
        
        assert 'test_strategy' not in live_engine.strategies
    
    def test_get_status(self, live_engine):
        """Test status reporting."""
        live_engine.running = True
        live_engine.portfolio_engine.get_portfolio_value = Mock(return_value=150000)
        live_engine.portfolio_engine.positions = {'AAPL': Mock(), 'GOOGL': Mock()}
        live_engine.stats = {
            'errors': 2,
            'signals_generated': 100,
            'trades_executed': 50,
            'data_updates': 1000
        }
        
        status = live_engine.get_status()
        
        assert status['running'] is True
        assert status['mode'] == 'PAPER'
        assert status['portfolio_value'] == 150000
        assert status['position_count'] == 2
        assert status['stats'] == live_engine.stats
        assert 'strategies' in status
        assert 'timestamp' in status
    
    def test_get_performance(self, live_engine):
        """Test performance metrics retrieval."""
        mock_metrics = {
            'total_return': 12.5,
            'sharpe_ratio': 1.6,
            'max_drawdown': -8.3
        }
        live_engine.portfolio_engine.get_performance_metrics = Mock(
            return_value=mock_metrics
        )
        
        performance = live_engine.get_performance()
        
        assert performance == mock_metrics
    
    @pytest.mark.asyncio
    async def test_error_handling(self, live_engine):
        """Test error handling in main loop."""
        # Mock an error in market data update
        live_engine.data_handler.get_latest = AsyncMock(
            side_effect=Exception("Network error")
        )
        
        # Should handle error gracefully
        await live_engine._update_market_data()
        
        assert live_engine.stats['errors'] == 1
    
    @pytest.mark.asyncio
    async def test_scheduled_tasks(self, live_engine):
        """Test scheduled task execution."""
        # Mock scheduler
        mock_scheduler = Mock()
        live_engine.scheduler = mock_scheduler
        
        # Add a task
        async def test_task():
            return "task_complete"
        
        live_engine.schedule_task(test_task, interval=60, name='test_task')
        
        # Verify task scheduled
        assert mock_scheduler.add_job.called
    
    def test_emergency_stop(self, live_engine):
        """Test emergency stop functionality."""
        live_engine.running = True
        live_engine.executor.cancel_all_orders = AsyncMock()
        live_engine.executor.close_all_positions = AsyncMock()
        
        # Trigger emergency stop
        asyncio.create_task(live_engine.emergency_stop())
        
        assert not live_engine.running
        assert live_engine.emergency_shutdown is True
    
    @pytest.mark.asyncio
    async def test_save_state(self, live_engine):
        """Test state persistence."""
        live_engine.portfolio_engine.export_state = Mock(
            return_value={'positions': [], 'trades': []}
        )
        
        # Save state
        await live_engine.save_state()
        
        live_engine.portfolio_engine.export_state.assert_called_once()
        assert live_engine.last_save_time is not None
    
    @pytest.mark.asyncio
    async def test_load_state(self, live_engine):
        """Test state loading."""
        saved_state = {
            'positions': [{'symbol': 'AAPL', 'quantity': 100}],
            'stop_orders': {'AAPL': {'stop_price': 145}}
        }
        
        with patch('builtins.open', create=True):
            with patch('json.load', return_value=saved_state):
                await live_engine.load_state()
        
        assert live_engine.stop_orders == saved_state['stop_orders']