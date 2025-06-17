"""
Portfolio fixtures for testing.

Provides various portfolio configurations and states for comprehensive testing.
"""
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# We'll import the actual Portfolio class when it's available
# from core.portfolio import Portfolio, Position


class MockPosition:
    """Mock position for testing until real Position class is available."""
    def __init__(self, symbol: str, quantity: int, entry_price: float,
                 entry_time: datetime = None, stop_loss: float = None,
                 take_profit: float = None):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time or datetime.now()
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.cost_basis = abs(quantity * entry_price)

    @property
    def is_long(self):
        return self.quantity > 0

    @property
    def is_short(self):
        return self.quantity < 0

    def current_value(self, current_price: float) -> float:
        """Calculate current position value."""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.is_long:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * abs(self.quantity)


class MockPortfolio:
    """Mock portfolio for testing until real Portfolio class is available."""
    def __init__(self, initial_capital: float = 100000.0, **kwargs):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, MockPosition] = {}
        self.transactions = []
        self.config = kwargs

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + sum(pos.cost_basis for pos in self.positions.values())

    @property
    def positions_value(self) -> float:
        """Calculate total positions value."""
        return sum(pos.cost_basis for pos in self.positions.values())

    @property
    def exposure(self) -> float:
        """Calculate total market exposure."""
        return sum(abs(pos.cost_basis) for pos in self.positions.values())

    def add_position(self, symbol: str, quantity: int, price: float,
                    commission: float = 0.001) -> MockPosition:
        """Add a position to the portfolio."""
        cost = abs(quantity * price)
        total_cost = cost * (1 + commission)

        if total_cost > self.cash:
            raise ValueError("Insufficient capital")

        position = MockPosition(symbol, quantity, price)
        self.positions[symbol] = position
        self.cash -= total_cost

        # Record transaction
        self.transactions.append({
            'timestamp': datetime.now(),
            'type': 'BUY' if quantity > 0 else 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': cost * commission,
            'cash_balance': self.cash
        })

        return position


# Portfolio Configuration Fixtures

@pytest.fixture
def conservative_portfolio_config():
    """Conservative portfolio configuration."""
    return {
        "initial_capital": 100000.0,
        "max_position_size": 0.05,  # 5% max per position
        "max_portfolio_risk": 0.01,  # 1% portfolio risk
        "max_leverage": 1.0,  # No leverage
        "max_correlation": 0.5,  # Low correlation tolerance
        "target_volatility": 0.08,  # 8% annual vol
        "commission": 0.001,
        "slippage": 0.0005,
    }


@pytest.fixture
def aggressive_portfolio_config():
    """Aggressive portfolio configuration."""
    return {
        "initial_capital": 100000.0,
        "max_position_size": 0.25,  # 25% max per position
        "max_portfolio_risk": 0.05,  # 5% portfolio risk
        "max_leverage": 2.0,  # 2x leverage allowed
        "max_correlation": 0.8,  # Higher correlation tolerance
        "target_volatility": 0.25,  # 25% annual vol
        "commission": 0.001,
        "slippage": 0.001,
    }


@pytest.fixture
def day_trading_portfolio_config():
    """Day trading portfolio configuration."""
    return {
        "initial_capital": 25000.0,  # PDT minimum
        "max_position_size": 0.50,  # 50% max per position
        "max_portfolio_risk": 0.02,  # 2% daily risk
        "max_leverage": 4.0,  # Intraday margin
        "max_correlation": 1.0,  # No correlation limits
        "target_volatility": 0.50,  # High vol tolerance
        "commission": 0.0005,  # Lower commission for high volume
        "slippage": 0.0002,  # Tighter spreads
        "max_positions": 10,  # Multiple concurrent positions
    }


# Portfolio State Fixtures

@pytest.fixture
def empty_portfolio():
    """Empty portfolio with default configuration."""
    return MockPortfolio(initial_capital=100000.0)


@pytest.fixture
def small_portfolio():
    """Small portfolio with a few positions."""
    portfolio = MockPortfolio(initial_capital=50000.0)

    # Add some positions
    portfolio.add_position('AAPL', 100, 150.0)  # $15,000
    portfolio.add_position('MSFT', 50, 300.0)   # $15,000

    return portfolio


@pytest.fixture
def diversified_portfolio():
    """Well-diversified portfolio across sectors."""
    portfolio = MockPortfolio(initial_capital=200000.0)

    # Technology
    portfolio.add_position('AAPL', 100, 150.0)
    portfolio.add_position('MSFT', 50, 300.0)

    # Financials
    portfolio.add_position('JPM', 150, 140.0)
    portfolio.add_position('BAC', 500, 35.0)

    # Healthcare
    portfolio.add_position('JNJ', 100, 160.0)
    portfolio.add_position('PFE', 300, 40.0)

    # Consumer
    portfolio.add_position('AMZN', 50, 140.0)
    portfolio.add_position('WMT', 100, 160.0)

    return portfolio


@pytest.fixture
def leveraged_portfolio():
    """Portfolio with leverage (short positions)."""
    portfolio = MockPortfolio(initial_capital=100000.0)

    # Long positions
    portfolio.add_position('SPY', 200, 450.0)  # $90,000 long

    # Short positions
    portfolio.add_position('ARKK', -300, 50.0)  # $15,000 short
    portfolio.add_position('MEME', -1000, 10.0)  # $10,000 short

    return portfolio


@pytest.fixture
def stressed_portfolio():
    """Portfolio under stress (high drawdown, concentrated)."""
    portfolio = MockPortfolio(initial_capital=100000.0)

    # Concentrated position
    portfolio.add_position('TSLA', 100, 800.0)  # $80,000 (80% of capital)

    # Losing positions (would need to simulate losses)
    portfolio.cash = 15000  # Simulate losses

    return portfolio


@pytest.fixture
def options_portfolio():
    """Portfolio with options positions (mocked)."""
    portfolio = MockPortfolio(initial_capital=100000.0)

    # Stock positions
    portfolio.add_position('SPY', 100, 450.0)

    # Mock options positions (using negative quantities for puts)
    # Call: SPY 460C
    portfolio.add_position('SPY_460C', 10, 5.0)  # 10 contracts at $5

    # Put: SPY 440P
    portfolio.add_position('SPY_440P', -5, 3.0)  # Short 5 puts at $3

    return portfolio


# Portfolio History Fixtures

@pytest.fixture
def portfolio_with_history():
    """Portfolio with transaction history."""
    portfolio = MockPortfolio(initial_capital=100000.0)

    # Simulate trading history
    base_time = datetime.now() - timedelta(days=30)

    # Day 1: Initial purchases
    with pytest.helpers.freeze_time(base_time):
        portfolio.add_position('AAPL', 100, 140.0)
        portfolio.add_position('GOOGL', 20, 2500.0)

    # Day 10: Partial sale
    with pytest.helpers.freeze_time(base_time + timedelta(days=10)):
        if 'AAPL' in portfolio.positions:
            portfolio.positions['AAPL'].quantity = 50  # Sold 50 shares
            portfolio.cash += 50 * 145.0  # Sold at profit

    # Day 20: New position
    with pytest.helpers.freeze_time(base_time + timedelta(days=20)):
        portfolio.add_position('MSFT', 100, 320.0)

    return portfolio


@pytest.fixture
def portfolio_performance_data():
    """Historical portfolio performance data for testing metrics."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

    # Simulate portfolio value evolution
    initial_value = 100000
    returns = np.random.normal(0.0005, 0.01, len(dates))  # 12.6% annual return, 16% vol
    values = initial_value * np.exp(np.cumsum(returns))

    # Create performance dataframe
    performance = pd.DataFrame({
        'date': dates,
        'total_value': values,
        'cash': values * 0.2,  # 20% cash
        'positions_value': values * 0.8,  # 80% invested
        'daily_return': np.concatenate([[0], returns[:-1]]),
        'trades': np.random.poisson(2, len(dates)),  # Average 2 trades per day
    })

    return performance


# Test Helper Fixtures

@pytest.fixture
def position_factory():
    """Factory for creating test positions."""
    def create_position(
        symbol: str = "TEST",
        quantity: int = 100,
        entry_price: float = 100.0,
        **kwargs
    ) -> MockPosition:
        return MockPosition(symbol, quantity, entry_price, **kwargs)

    return create_position


@pytest.fixture
def portfolio_factory():
    """Factory for creating test portfolios."""
    def create_portfolio(**kwargs) -> MockPortfolio:
        return MockPortfolio(**kwargs)

    return create_portfolio


@pytest.fixture
def assert_portfolio_valid():
    """Helper to assert portfolio invariants."""
    def _assert_valid(portfolio):
        # Basic invariants
        assert portfolio.cash >= 0, "Cash cannot be negative"
        assert portfolio.total_value >= 0, "Portfolio value cannot be negative"

        # Position invariants
        for symbol, position in portfolio.positions.items():
            assert position.quantity != 0, f"Position {symbol} has zero quantity"
            assert position.entry_price > 0, f"Position {symbol} has invalid price"

        # Transaction invariants
        for txn in portfolio.transactions:
            assert txn['quantity'] != 0, "Transaction has zero quantity"
            assert txn['price'] > 0, "Transaction has invalid price"
            assert txn['commission'] >= 0, "Transaction has negative commission"

    return _assert_valid
