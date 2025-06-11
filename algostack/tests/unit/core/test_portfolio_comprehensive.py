"""Comprehensive test suite for portfolio management."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from core.portfolio import (
    Portfolio,
    Position,
    Trade,
    PortfolioMetrics,
    PortfolioState
)


class TestPortfolio:
    """Test suite for Portfolio class."""
    
    @pytest.fixture
    def portfolio(self):
        """Create a portfolio instance."""
        return Portfolio(
            initial_capital=100000,
            max_positions=10,
            position_size_method='equal_weight'
        )
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        return {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 200.0
        }
    
    def test_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(
            initial_capital=50000,
            max_positions=5,
            position_size_method='kelly'
        )
        
        assert portfolio.initial_capital == 50000
        assert portfolio.cash == 50000
        assert portfolio.max_positions == 5
        assert portfolio.position_size_method == 'kelly'
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0] == 50000
    
    def test_add_position(self, portfolio):
        """Test adding a new position."""
        trade = Trade(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side='BUY',
            timestamp=datetime.now(),
            commission=1.0
        )
        
        portfolio.add_position(trade)
        
        assert 'AAPL' in portfolio.positions
        assert portfolio.positions['AAPL'].quantity == 100
        assert portfolio.positions['AAPL'].avg_price == 150.0
        assert portfolio.cash == 100000 - (100 * 150.0) - 1.0
        assert len(portfolio.trades) == 1
    
    def test_add_to_existing_position(self, portfolio):
        """Test adding to an existing position."""
        # First trade
        trade1 = Trade(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side='BUY',
            timestamp=datetime.now(),
            commission=1.0
        )
        portfolio.add_position(trade1)
        
        # Second trade
        trade2 = Trade(
            symbol='AAPL',
            quantity=50,
            price=155.0,
            side='BUY',
            timestamp=datetime.now(),
            commission=1.0
        )
        portfolio.add_position(trade2)
        
        # Check position update
        position = portfolio.positions['AAPL']
        assert position.quantity == 150
        expected_avg_price = (100 * 150.0 + 50 * 155.0) / 150
        assert position.avg_price == pytest.approx(expected_avg_price)
        assert len(portfolio.trades) == 2
    
    def test_close_position_full(self, portfolio):
        """Test closing a full position."""
        # Open position
        buy_trade = Trade(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side='BUY',
            timestamp=datetime.now(),
            commission=1.0
        )
        portfolio.add_position(buy_trade)
        
        # Close position
        sell_trade = Trade(
            symbol='AAPL',
            quantity=100,
            price=160.0,
            side='SELL',
            timestamp=datetime.now(),
            commission=1.0
        )
        portfolio.close_position(sell_trade)
        
        # Check position is closed
        assert 'AAPL' not in portfolio.positions
        assert portfolio.cash == 100000 + (100 * 10.0) - 2.0  # Profit minus commissions
        assert len(portfolio.trades) == 2
        assert portfolio.realized_pnl == 1000.0 - 2.0
    
    def test_close_position_partial(self, portfolio):
        """Test partially closing a position."""
        # Open position
        buy_trade = Trade(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side='BUY',
            timestamp=datetime.now(),
            commission=1.0
        )
        portfolio.add_position(buy_trade)
        
        # Partial close
        sell_trade = Trade(
            symbol='AAPL',
            quantity=50,
            price=160.0,
            side='SELL',
            timestamp=datetime.now(),
            commission=1.0
        )
        portfolio.close_position(sell_trade)
        
        # Check position is reduced
        assert 'AAPL' in portfolio.positions
        assert portfolio.positions['AAPL'].quantity == 50
        assert len(portfolio.trades) == 2
    
    def test_update_market_values(self, portfolio, sample_prices):
        """Test updating market values."""
        # Add some positions
        portfolio.add_position(Trade('AAPL', 100, 140.0, 'BUY', datetime.now(), 1.0))
        portfolio.add_position(Trade('GOOGL', 10, 2700.0, 'BUY', datetime.now(), 1.0))
        
        # Update market values
        portfolio.update_market_values(sample_prices)
        
        # Check unrealized P&L
        aapl_pos = portfolio.positions['AAPL']
        assert aapl_pos.market_value == 100 * 150.0
        assert aapl_pos.unrealized_pnl == 100 * (150.0 - 140.0)
        
        googl_pos = portfolio.positions['GOOGL']
        assert googl_pos.market_value == 10 * 2800.0
        assert googl_pos.unrealized_pnl == 10 * (2800.0 - 2700.0)
    
    def test_get_portfolio_value(self, portfolio, sample_prices):
        """Test calculating total portfolio value."""
        # Add positions
        portfolio.add_position(Trade('AAPL', 100, 140.0, 'BUY', datetime.now(), 1.0))
        portfolio.add_position(Trade('MSFT', 50, 290.0, 'SELL', datetime.now(), 1.0))  # Short
        
        portfolio.update_market_values(sample_prices)
        
        total_value = portfolio.get_portfolio_value()
        
        # Cash + long positions - short positions
        expected_value = (
            portfolio.cash +
            100 * 150.0 +  # AAPL long
            50 * (290.0 - 300.0)  # MSFT short (negative unrealized)
        )
        assert total_value == pytest.approx(expected_value)
    
    def test_calculate_metrics(self, portfolio):
        """Test portfolio metrics calculation."""
        # Create some trading history
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        equity_values = 100000 + np.cumsum(np.random.randn(100) * 1000)
        portfolio.equity_curve = equity_values.tolist()
        
        # Add some trades
        portfolio.trades = [
            Trade('AAPL', 100, 150.0, 'BUY', dates[10], 1.0),
            Trade('AAPL', 100, 160.0, 'SELL', dates[20], 1.0),
            Trade('GOOGL', 10, 2700.0, 'BUY', dates[30], 1.0),
            Trade('GOOGL', 10, 2650.0, 'SELL', dates[40], 1.0),
        ]
        
        metrics = portfolio.calculate_metrics()
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
    
    def test_position_sizing_equal_weight(self, portfolio):
        """Test equal weight position sizing."""
        portfolio.position_size_method = 'equal_weight'
        portfolio.max_positions = 4
        
        size = portfolio.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            signal_strength=1.0
        )
        
        # Should be ~25% of capital
        expected_shares = int((100000 * 0.25) / 150.0)
        assert size == expected_shares
    
    def test_position_sizing_kelly(self, portfolio):
        """Test Kelly criterion position sizing."""
        portfolio.position_size_method = 'kelly'
        
        # Add some winning trade history
        for i in range(10):
            portfolio.trades.append(
                Trade('TEST', 100, 100.0, 'BUY', datetime.now(), 1.0, pnl=100.0)
            )
        for i in range(5):
            portfolio.trades.append(
                Trade('TEST', 100, 100.0, 'BUY', datetime.now(), 1.0, pnl=-50.0)
            )
        
        size = portfolio.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            signal_strength=1.0,
            win_rate=0.67,
            avg_win_loss_ratio=2.0
        )
        
        assert size > 0
        assert size < portfolio.cash / 150.0  # Shouldn't exceed available cash
    
    def test_position_sizing_volatility(self, portfolio):
        """Test volatility-based position sizing."""
        portfolio.position_size_method = 'volatility'
        
        size = portfolio.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            signal_strength=1.0,
            volatility=0.02,  # 2% daily vol
            risk_per_trade=0.01  # 1% risk per trade
        )
        
        # Position size = (Risk Amount) / (Price * Volatility)
        risk_amount = 100000 * 0.01
        expected_shares = int(risk_amount / (150.0 * 0.02))
        assert size == expected_shares
    
    def test_max_positions_limit(self, portfolio):
        """Test that max positions limit is enforced."""
        portfolio.max_positions = 2
        
        # Add two positions
        portfolio.add_position(Trade('AAPL', 100, 150.0, 'BUY', datetime.now(), 1.0))
        portfolio.add_position(Trade('GOOGL', 10, 2800.0, 'BUY', datetime.now(), 1.0))
        
        # Try to add third position
        can_add = portfolio.can_add_position('MSFT')
        assert not can_add
        
        # Should still be able to add to existing position
        can_add_existing = portfolio.can_add_position('AAPL')
        assert can_add_existing
    
    def test_risk_metrics(self, portfolio):
        """Test risk metric calculations."""
        # Create equity curve with drawdown
        equity_curve = [
            100000, 102000, 105000, 103000, 101000,  # Drawdown
            104000, 107000, 110000, 108000, 112000   # Recovery
        ]
        portfolio.equity_curve = equity_curve
        
        # Calculate max drawdown
        max_dd = portfolio.calculate_max_drawdown()
        assert max_dd == pytest.approx(-0.0381, rel=0.01)  # ~3.81% from 105k to 101k
        
        # Calculate Sharpe ratio
        sharpe = portfolio.calculate_sharpe_ratio()
        assert isinstance(sharpe, float)
        
        # Calculate Sortino ratio
        sortino = portfolio.calculate_sortino_ratio()
        assert isinstance(sortino, float)
    
    def test_get_open_positions(self, portfolio):
        """Test getting list of open positions."""
        # Add various positions
        portfolio.add_position(Trade('AAPL', 100, 150.0, 'BUY', datetime.now(), 1.0))
        portfolio.add_position(Trade('GOOGL', 10, 2800.0, 'BUY', datetime.now(), 1.0))
        portfolio.add_position(Trade('MSFT', 50, 300.0, 'SELL', datetime.now(), 1.0))
        
        # Get long positions
        long_positions = portfolio.get_open_positions(side='LONG')
        assert len(long_positions) == 2
        assert 'AAPL' in long_positions
        assert 'GOOGL' in long_positions
        
        # Get short positions
        short_positions = portfolio.get_open_positions(side='SHORT')
        assert len(short_positions) == 1
        assert 'MSFT' in short_positions
        
        # Get all positions
        all_positions = portfolio.get_open_positions()
        assert len(all_positions) == 3
    
    def test_transaction_costs(self, portfolio):
        """Test that transaction costs are properly accounted."""
        initial_cash = portfolio.cash
        
        # Buy with commission and slippage
        trade = Trade(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side='BUY',
            timestamp=datetime.now(),
            commission=10.0,
            slippage=5.0
        )
        portfolio.add_position(trade)
        
        # Cash should be reduced by: quantity * price + commission + slippage
        expected_cash = initial_cash - (100 * 150.0 + 10.0 + 5.0)
        assert portfolio.cash == expected_cash
    
    def test_portfolio_state_snapshot(self, portfolio, sample_prices):
        """Test creating portfolio state snapshot."""
        # Add some positions and trades
        portfolio.add_position(Trade('AAPL', 100, 140.0, 'BUY', datetime.now(), 1.0))
        portfolio.add_position(Trade('GOOGL', 10, 2700.0, 'BUY', datetime.now(), 1.0))
        portfolio.update_market_values(sample_prices)
        
        # Get snapshot
        state = portfolio.get_state_snapshot()
        
        assert isinstance(state, dict)
        assert state['cash'] == portfolio.cash
        assert state['total_value'] > 0
        assert len(state['positions']) == 2
        assert state['realized_pnl'] >= 0
        assert state['unrealized_pnl'] != 0
    
    def test_reset_portfolio(self, portfolio):
        """Test resetting portfolio to initial state."""
        # Add some activity
        portfolio.add_position(Trade('AAPL', 100, 150.0, 'BUY', datetime.now(), 1.0))
        portfolio.trades.append(Trade('GOOGL', 10, 2800.0, 'BUY', datetime.now(), 1.0))
        portfolio.equity_curve.extend([101000, 102000, 99000])
        
        # Reset
        portfolio.reset()
        
        assert portfolio.cash == portfolio.initial_capital
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0] == portfolio.initial_capital
        assert portfolio.realized_pnl == 0
    
    def test_position_allocation_constraints(self, portfolio):
        """Test position allocation constraints."""
        portfolio.max_position_size = 0.25  # Max 25% per position
        
        # Try to allocate more than 25%
        size = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            signal_strength=1.0,
            requested_allocation=0.5  # Request 50%
        )
        
        # Should be capped at 25%
        max_shares = int((100000 * 0.25) / 100.0)
        assert size <= max_shares
    
    def test_margin_requirements(self, portfolio):
        """Test margin requirement calculations for short positions."""
        # Add a short position
        portfolio.add_position(Trade('TSLA', 100, 200.0, 'SELL', datetime.now(), 1.0))
        
        # Calculate margin requirements
        margin_req = portfolio.calculate_margin_requirements()
        
        # Should have margin for short position
        assert margin_req > 0
        assert margin_req >= 100 * 200.0 * 0.3  # At least 30% margin
    
    def test_performance_attribution(self, portfolio):
        """Test performance attribution by symbol."""
        # Add trades for multiple symbols
        portfolio.trades = [
            Trade('AAPL', 100, 150.0, 'BUY', datetime(2023, 1, 1), 1.0, pnl=1000.0),
            Trade('AAPL', 100, 160.0, 'SELL', datetime(2023, 1, 10), 1.0, pnl=1000.0),
            Trade('GOOGL', 10, 2700.0, 'BUY', datetime(2023, 1, 5), 1.0, pnl=-500.0),
            Trade('GOOGL', 10, 2650.0, 'SELL', datetime(2023, 1, 15), 1.0, pnl=-500.0),
            Trade('MSFT', 50, 300.0, 'BUY', datetime(2023, 1, 8), 1.0, pnl=250.0),
        ]
        
        attribution = portfolio.get_performance_attribution()
        
        assert 'AAPL' in attribution
        assert attribution['AAPL']['total_pnl'] == 2000.0
        assert attribution['AAPL']['trade_count'] == 2
        
        assert 'GOOGL' in attribution
        assert attribution['GOOGL']['total_pnl'] == -1000.0
        assert attribution['GOOGL']['trade_count'] == 2