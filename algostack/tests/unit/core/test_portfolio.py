"""Tests for the portfolio engine."""

from datetime import datetime
from typing import Any

import pytest

from core.portfolio import PortfolioEngine, Position
from strategies.base import Signal


class TestPortfolioEngine:
    """Test the portfolio engine functionality."""

    @pytest.fixture
    def portfolio_config(self) -> dict[str, Any]:
        """Create test portfolio configuration."""
        return {
            "initial_capital": 5000.0,
            "target_vol": 0.10,
            "max_position_size": 0.20,
            "max_drawdown": 0.15,
            "use_equal_risk": True,
        }

    @pytest.fixture
    def portfolio(self, portfolio_config) -> PortfolioEngine:
        """Create portfolio engine instance."""
        return PortfolioEngine(portfolio_config)

    @pytest.fixture
    def sample_signal(self) -> Signal:
        """Create a sample trading signal."""
        return Signal(
            timestamp=datetime.now(),
            symbol="SPY",
            direction="LONG",
            strength=0.8,
            strategy_id="test_strategy",
            price=450.0,
            atr=5.0,
            metadata={"test": True},
        )

    def test_portfolio_initialization(self, portfolio, portfolio_config):
        """Test portfolio initializes correctly."""
        assert portfolio.initial_capital == portfolio_config["initial_capital"]
        assert portfolio.current_equity == portfolio_config["initial_capital"]
        assert len(portfolio.positions) == 0
        assert not portfolio.is_risk_off
        assert portfolio.current_drawdown == 0.0

    def test_position_sizing(self, portfolio, sample_signal):
        """Test position sizing calculation."""
        allocation = 0.1  # 10% allocation
        position_size, stop_loss = portfolio.size_position(sample_signal, allocation)

        # Check position size is reasonable
        assert position_size > 0
        assert (
            position_size * sample_signal.price <= portfolio.current_equity * 0.20
        )  # Max position limit

        # Check stop loss
        assert stop_loss > 0
        assert stop_loss < sample_signal.price  # Stop below entry for long

    def test_execute_signal(self, portfolio, sample_signal):
        """Test signal execution."""
        position_size = 10  # 10 shares
        stop_loss = 440.0

        position = portfolio.execute_signal(sample_signal, position_size, stop_loss)

        assert position is not None
        assert position.symbol == sample_signal.symbol
        assert position.quantity == position_size
        assert position.entry_price == sample_signal.price
        assert position.stop_loss == stop_loss
        assert sample_signal.symbol in portfolio.positions

    def test_close_position(self, portfolio, sample_signal):
        """Test closing a position."""
        # First open a position
        position_size = 10
        portfolio.execute_signal(sample_signal, position_size, 440.0)

        # Close with profit
        exit_price = 460.0
        result = portfolio.close_position(sample_signal.symbol, exit_price)

        assert result is not None
        assert result["pnl"] == (exit_price - sample_signal.price) * position_size
        assert result["pnl_pct"] > 0
        assert sample_signal.symbol not in portfolio.positions
        assert portfolio.current_equity > portfolio.initial_capital

    def test_risk_limits(self, portfolio):
        """Test risk limit checking."""
        # No violations initially
        is_compliant, violations = portfolio.check_risk_limits()
        assert is_compliant
        assert len(violations) == 0

        # Simulate large drawdown
        portfolio.current_drawdown = 0.20  # 20% drawdown
        is_compliant, violations = portfolio.check_risk_limits()
        assert not is_compliant
        assert len(violations) > 0
        assert any("Drawdown" in v for v in violations)

    def test_global_risk_check(self, portfolio):
        """Test global risk check and kill switch."""
        # Normal conditions
        is_ok, exit_signals = portfolio.global_risk_check()
        assert is_ok
        assert len(exit_signals) == 0

        # Trigger kill switch with high drawdown
        portfolio.current_drawdown = 0.20  # 20% exceeds 15% limit
        portfolio.peak_equity = 6000
        portfolio.current_equity = 4800  # 20% down

        # Add a position
        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            strategy_id="test",
            direction="LONG",
            quantity=10,
            entry_price=450.0,
            entry_time=datetime.now(),
            current_price=440.0,
        )

        is_ok, exit_signals = portfolio.global_risk_check()
        assert not is_ok
        assert len(exit_signals) == 1
        assert portfolio.is_risk_off
        assert portfolio.risk_off_until is not None

    def test_kelly_fraction_update(self, portfolio):
        """Test Kelly fraction calculation."""
        # Add some mock performance data
        portfolio.strategy_performance["test_strategy"] = {
            "trades": 50,
            "wins": 30,
            "total_pnl": 1000,
            "win_pnl": 1500,
            "loss_pnl": 500,
        }

        portfolio.update_strategy_kelly_fractions()

        assert "test_strategy" in portfolio.strategy_kelly_fractions
        kelly = portfolio.strategy_kelly_fractions["test_strategy"]
        assert 0 <= kelly <= 0.25  # Half-Kelly capped at 25%

    def test_portfolio_metrics(self, portfolio):
        """Test portfolio metrics calculation."""
        metrics = portfolio.calculate_portfolio_metrics()

        assert metrics["total_equity"] == portfolio.initial_capital
        assert metrics["cash"] == portfolio.initial_capital
        assert metrics["positions_value"] == 0
        assert metrics["position_count"] == 0
        assert metrics["current_drawdown"] == 0

    def test_capital_allocation(self, portfolio):
        """Test capital allocation across strategies."""
        # Create signals from different strategies
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="SPY",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=450.0,
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="QQQ",
                direction="LONG",
                strength=0.6,
                strategy_id="strategy2",
                price=380.0,
            ),
        ]

        # Mock market data
        market_data = {
            "SPY": {"returns": [0.01, -0.005, 0.008]},
            "QQQ": {"returns": [0.015, -0.01, 0.012]},
        }

        allocations = portfolio.allocate_capital(signals, market_data)

        assert len(allocations) == 2
        assert sum(allocations.values()) > 0
        assert all(0 <= v <= 1 for v in allocations.values())
