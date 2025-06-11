"""Comprehensive test suite for PortfolioEngine."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.portfolio import PortfolioEngine, Position


class TestPosition:
    """Test suite for Position class."""
    
    def test_position_initialization(self):
        """Test Position initialization."""
        pos = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            entry_time=datetime(2023, 1, 1),
            position_type='LONG'
        )
        
        assert pos.symbol == 'AAPL'
        assert pos.quantity == 100
        assert pos.entry_price == 150.0
        assert pos.position_type == 'LONG'
        assert pos.current_price == 150.0
        assert pos.realized_pnl == 0
        assert pos.unrealized_pnl == 0
    
    def test_position_update_price(self):
        """Test updating position price."""
        pos = Position('AAPL', 100, 150.0, datetime.now())
        
        # Update price
        pos.update_price(155.0)
        
        assert pos.current_price == 155.0
        assert pos.unrealized_pnl == 500.0  # (155-150) * 100
        assert pos.market_value == 15500.0
    
    def test_position_partial_close(self):
        """Test partially closing position."""
        pos = Position('AAPL', 100, 150.0, datetime.now())
        pos.update_price(160.0)
        
        # Close half
        realized = pos.reduce_position(50, 160.0)
        
        assert pos.quantity == 50
        assert realized == 500.0  # (160-150) * 50
        assert pos.realized_pnl == 500.0
    
    def test_position_to_dict(self):
        """Test position serialization."""
        pos = Position('AAPL', 100, 150.0, datetime(2023, 1, 1))
        pos.update_price(155.0)
        
        pos_dict = pos.to_dict()
        
        assert pos_dict['symbol'] == 'AAPL'
        assert pos_dict['quantity'] == 100
        assert pos_dict['entry_price'] == 150.0
        assert pos_dict['current_price'] == 155.0
        assert pos_dict['unrealized_pnl'] == 500.0
        assert pos_dict['market_value'] == 15500.0


class TestPortfolioEngine:
    """Test suite for PortfolioEngine class."""
    
    @pytest.fixture
    def portfolio(self):
        """Create portfolio instance."""
        return PortfolioEngine(
            initial_capital=100000,
            risk_config={
                'max_position_size': 0.20,
                'max_portfolio_risk': 0.06,
                'stop_loss_pct': 0.02
            }
        )
    
    @pytest.fixture
    def sample_trade(self):
        """Create sample trade."""
        return {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'side': 'BUY',
            'timestamp': datetime.now(),
            'order_id': 'TEST123'
        }
    
    def test_initialization(self):
        """Test PortfolioEngine initialization."""
        config = {
            'max_position_size': 0.25,
            'max_leverage': 1.5,
            'risk_per_trade': 0.01
        }
        
        portfolio = PortfolioEngine(
            initial_capital=50000,
            risk_config=config
        )
        
        assert portfolio.cash == 50000
        assert portfolio.initial_capital == 50000
        assert portfolio.risk_config == config
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert portfolio.total_commission == 0
    
    def test_process_fill_buy(self, portfolio, sample_trade):
        """Test processing buy order fill."""
        fill = sample_trade.copy()
        fill['commission'] = 1.0
        
        portfolio.process_fill(fill)
        
        # Check position created
        assert 'AAPL' in portfolio.positions
        pos = portfolio.positions['AAPL']
        assert pos.quantity == 100
        assert pos.entry_price == 150.0
        
        # Check cash reduced
        assert portfolio.cash == 100000 - (100 * 150.0) - 1.0
        
        # Check trade recorded
        assert len(portfolio.trades) == 1
        assert portfolio.total_commission == 1.0
    
    def test_process_fill_sell(self, portfolio):
        """Test processing sell order fill."""
        # First create a position
        portfolio.positions['AAPL'] = Position('AAPL', 100, 140.0, datetime.now())
        portfolio.cash = 86000  # 100k - 14k
        
        # Sell fill
        sell_fill = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'side': 'SELL',
            'timestamp': datetime.now(),
            'order_id': 'SELL123',
            'commission': 1.0
        }
        
        portfolio.process_fill(sell_fill)
        
        # Position should be closed
        assert 'AAPL' not in portfolio.positions
        
        # Cash should increase
        assert portfolio.cash == 86000 + (100 * 150.0) - 1.0
        
        # Check realized P&L
        assert portfolio.realized_pnl == (150.0 - 140.0) * 100 - 1.0
    
    def test_process_fill_partial(self, portfolio):
        """Test processing partial fill."""
        # Create position
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        
        # Partial sell
        partial_fill = {
            'symbol': 'AAPL',
            'quantity': 30,
            'price': 155.0,
            'side': 'SELL',
            'timestamp': datetime.now(),
            'commission': 0.5
        }
        
        portfolio.process_fill(partial_fill)
        
        # Position should be reduced
        assert portfolio.positions['AAPL'].quantity == 70
        
        # Realized P&L from partial close
        assert portfolio.realized_pnl == (155.0 - 150.0) * 30 - 0.5
    
    def test_update_positions(self, portfolio):
        """Test updating position prices."""
        # Create positions
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.positions['GOOGL'] = Position('GOOGL', 10, 2800.0, datetime.now())
        
        # Update prices
        current_prices = {
            'AAPL': 155.0,
            'GOOGL': 2750.0
        }
        
        portfolio.update_positions(current_prices)
        
        # Check updates
        assert portfolio.positions['AAPL'].current_price == 155.0
        assert portfolio.positions['AAPL'].unrealized_pnl == 500.0
        
        assert portfolio.positions['GOOGL'].current_price == 2750.0
        assert portfolio.positions['GOOGL'].unrealized_pnl == -500.0
    
    def test_get_portfolio_value(self, portfolio):
        """Test portfolio value calculation."""
        # Add positions
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.positions['GOOGL'] = Position('GOOGL', 10, 2800.0, datetime.now())
        portfolio.cash = 57000  # 100k - 15k - 28k
        
        # Update prices
        portfolio.positions['AAPL'].update_price(155.0)
        portfolio.positions['GOOGL'].update_price(2850.0)
        
        total_value = portfolio.get_portfolio_value()
        
        # Cash + positions value
        expected = 57000 + (100 * 155.0) + (10 * 2850.0)
        assert total_value == expected
    
    def test_get_positions_summary(self, portfolio):
        """Test getting positions summary."""
        # Create positions
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.positions['GOOGL'] = Position('GOOGL', -10, 2800.0, datetime.now(), 'SHORT')
        
        # Update prices
        portfolio.positions['AAPL'].update_price(155.0)
        portfolio.positions['GOOGL'].update_price(2750.0)
        
        summary = portfolio.get_positions_summary()
        
        assert len(summary) == 2
        assert summary[0]['symbol'] == 'AAPL'
        assert summary[0]['quantity'] == 100
        assert summary[0]['unrealized_pnl'] == 500.0
        
        assert summary[1]['symbol'] == 'GOOGL'
        assert summary[1]['quantity'] == -10
        assert summary[1]['position_type'] == 'SHORT'
    
    def test_get_performance_metrics(self, portfolio):
        """Test performance metrics calculation."""
        # Simulate some trading
        portfolio.cash = 95000
        portfolio.realized_pnl = 2000
        portfolio.total_commission = 50
        portfolio.trades = [1, 2, 3, 4, 5]  # Dummy trades
        
        # Add position with unrealized profit
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.positions['AAPL'].update_price(160.0)
        
        metrics = portfolio.get_performance_metrics()
        
        assert metrics['total_value'] == 95000 + 16000  # Cash + position
        assert metrics['total_return_pct'] == 11.0  # 11% return
        assert metrics['realized_pnl'] == 2000
        assert metrics['unrealized_pnl'] == 1000
        assert metrics['total_pnl'] == 3000
        assert metrics['commission_paid'] == 50
        assert metrics['net_pnl'] == 2950
        assert metrics['trade_count'] == 5
    
    def test_check_risk_limits(self, portfolio):
        """Test risk limit checking."""
        # Set up portfolio near risk limits
        portfolio.cash = 20000
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.positions['AAPL'].update_price(155.0)
        
        # Test position size limit
        new_position_value = 25000  # Would be 25% of portfolio
        total_value = portfolio.get_portfolio_value()
        
        # Check if new position exceeds limit (20%)
        position_pct = new_position_value / total_value
        exceeds_limit = position_pct > portfolio.risk_config['max_position_size']
        
        assert exceeds_limit
    
    def test_calculate_position_size(self, portfolio):
        """Test position sizing calculation."""
        signal = {
            'symbol': 'AAPL',
            'price': 150.0,
            'stop_loss': 147.0,
            'signal_strength': 0.8
        }
        
        # Calculate position size based on risk
        risk_amount = portfolio.cash * portfolio.risk_config.get('risk_per_trade', 0.01)
        stop_distance = abs(signal['price'] - signal['stop_loss'])
        shares = int(risk_amount / stop_distance)
        
        # Should not exceed max position size
        max_shares = int((portfolio.get_portfolio_value() * portfolio.risk_config['max_position_size']) / signal['price'])
        final_shares = min(shares, max_shares)
        
        assert final_shares > 0
        assert final_shares <= max_shares
    
    def test_equity_curve_update(self, portfolio):
        """Test equity curve tracking."""
        # Initial value
        portfolio.update_equity_curve()
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0]['value'] == 100000
        
        # After some trading
        portfolio.cash = 95000
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.positions['AAPL'].update_price(155.0)
        
        portfolio.update_equity_curve()
        assert len(portfolio.equity_curve) == 2
        assert portfolio.equity_curve[1]['value'] == 95000 + 15500
    
    def test_drawdown_calculation(self, portfolio):
        """Test drawdown calculation."""
        # Create equity curve with drawdown
        values = [100000, 105000, 110000, 105000, 95000, 100000, 105000]
        for i, val in enumerate(values):
            portfolio.equity_curve.append({
                'timestamp': datetime.now() + timedelta(days=i),
                'value': val
            })
        
        drawdown_info = portfolio.calculate_drawdown()
        
        # Max drawdown from 110k to 95k
        assert drawdown_info['max_drawdown_pct'] == pytest.approx(-13.64, rel=0.01)
        assert drawdown_info['max_drawdown_value'] == -15000
    
    def test_get_trade_history(self, portfolio):
        """Test trade history retrieval."""
        # Add some trades
        trades = [
            {'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 150.0, 'timestamp': datetime.now()},
            {'symbol': 'AAPL', 'side': 'SELL', 'quantity': 50, 'price': 155.0, 'timestamp': datetime.now()},
            {'symbol': 'GOOGL', 'side': 'BUY', 'quantity': 10, 'price': 2800.0, 'timestamp': datetime.now()}
        ]
        
        for trade in trades:
            portfolio.trades.append(trade)
        
        # Get all trades
        all_trades = portfolio.get_trade_history()
        assert len(all_trades) == 3
        
        # Get trades for specific symbol
        aapl_trades = portfolio.get_trade_history(symbol='AAPL')
        assert len(aapl_trades) == 2
        
        # Get trades by date range
        today = datetime.now().date()
        today_trades = portfolio.get_trade_history(start_date=today, end_date=today)
        assert len(today_trades) == 3
    
    def test_export_state(self, portfolio):
        """Test exporting portfolio state."""
        # Set up portfolio
        portfolio.cash = 85000
        portfolio.positions['AAPL'] = Position('AAPL', 100, 150.0, datetime.now())
        portfolio.realized_pnl = 1000
        portfolio.trades = [{'test': 'trade'}]
        
        state = portfolio.export_state()
        
        assert state['cash'] == 85000
        assert state['initial_capital'] == 100000
        assert 'positions' in state
        assert 'trades' in state
        assert 'performance_metrics' in state
        assert state['timestamp'] is not None