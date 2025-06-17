"""Comprehensive test suite to achieve 100% coverage for portfolio.py module."""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from algostack.core.portfolio import PortfolioEngine, Position
from algostack.strategies.base import Signal
from algostack.utils.constants import (
    DEFAULT_KELLY_FRACTION,
    MAX_KELLY_FRACTION,
)


class TestPosition:
    """Test the Position dataclass comprehensively."""

    def test_position_market_value(self):
        """Test market value calculation."""
        position = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )
        assert position.market_value == 100 * 155.0

        # Test with negative quantity (short)
        short_position = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="SHORT",
            quantity=-100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=145.0,
        )
        assert short_position.market_value == 100 * 145.0  # abs(quantity)

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation for both long and short."""
        # Long position with profit
        long_position = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=160.0,
        )
        assert long_position.unrealized_pnl == 100 * (160.0 - 150.0)

        # Short position with profit
        short_position = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="SHORT",
            quantity=-100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=140.0,
        )
        assert short_position.unrealized_pnl == 100 * (150.0 - 140.0)

        # Short position with loss
        short_loss = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="SHORT",
            quantity=-100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=160.0,
        )
        assert short_loss.unrealized_pnl == 100 * (150.0 - 160.0)

    def test_position_pnl_percentage(self):
        """Test P&L percentage calculation."""
        # Normal case
        position = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=165.0,
        )
        expected_pct = (100 * (165.0 - 150.0)) / (100 * 150.0) * 100
        assert position.pnl_percentage == expected_pct

        # Edge case: zero entry value (should return 0)
        zero_position = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=0,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=165.0,
        )
        assert zero_position.pnl_percentage == 0


class TestPortfolioEngineComprehensive:
    """Comprehensive tests for PortfolioEngine to achieve 100% coverage."""

    @pytest.fixture
    def portfolio_config(self) -> dict[str, Any]:
        """Create test portfolio configuration."""
        return {
            "initial_capital": 100000.0,
            "target_vol": 0.15,
            "max_position_size": 0.25,
            "max_sector_exposure": 0.40,
            "max_drawdown": 0.20,
            "max_correlation": 0.70,
            "margin_buffer": 0.25,
            "use_equal_risk": True,
            "volatility_targets": {"strategy1": 0.10, "strategy2": 0.15},
        }

    @pytest.fixture
    def portfolio(self, portfolio_config) -> PortfolioEngine:
        """Create portfolio engine instance."""
        return PortfolioEngine(portfolio_config)

    def test_update_market_prices(self, portfolio):
        """Test updating market prices for positions."""
        # Add some positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
        )
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            strategy_id="test",
            direction="SHORT",
            quantity=-10,
            entry_price=2800.0,
            entry_time=datetime.now(),
            current_price=2800.0,
        )

        # Update prices
        new_prices = {"AAPL": 155.0, "GOOGL": 2750.0, "MSFT": 300.0}
        portfolio.update_market_prices(new_prices)

        assert portfolio.positions["AAPL"].current_price == 155.0
        assert portfolio.positions["GOOGL"].current_price == 2750.0

    def test_calculate_portfolio_volatility_empty(self, portfolio):
        """Test portfolio volatility with no data."""
        empty_returns = pd.DataFrame()
        vol = portfolio.calculate_portfolio_volatility(empty_returns)
        assert vol == 0.0

        # Also test with positions but no returns data
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
        )
        vol = portfolio.calculate_portfolio_volatility(empty_returns)
        assert vol == 0.0

    def test_calculate_portfolio_volatility_with_positions(self, portfolio):
        """Test portfolio volatility calculation with positions."""
        # Add positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            strategy_id="test",
            direction="SHORT",
            quantity=-10,
            entry_price=2800.0,
            entry_time=datetime.now(),
            current_price=2750.0,
        )

        # Create returns data
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 30),
                "GOOGL": np.random.normal(0.0005, 0.025, 30),
            },
            index=dates,
        )

        vol = portfolio.calculate_portfolio_volatility(returns_data)
        assert vol > 0
        assert vol < 1.0  # Reasonable volatility

    def test_update_correlation_matrix(self, portfolio):
        """Test correlation matrix update."""
        # Single asset - no correlation
        returns_single = pd.DataFrame({"AAPL": [0.01, -0.02, 0.015, -0.005]})
        portfolio.update_correlation_matrix(returns_single)
        assert portfolio.correlation_matrix.empty or len(portfolio.correlation_matrix) == 1

        # Multiple assets
        returns_multi = pd.DataFrame(
            {
                "AAPL": [0.01, -0.02, 0.015, -0.005],
                "GOOGL": [0.008, -0.018, 0.012, -0.003],
                "MSFT": [0.012, -0.015, 0.018, -0.008],
            }
        )
        portfolio.update_correlation_matrix(returns_multi)
        assert not portfolio.correlation_matrix.empty
        assert portfolio.correlation_matrix.shape == (3, 3)

    def test_check_risk_limits_correlation(self, portfolio):
        """Test risk limit checking including correlations."""
        # Add correlated positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=1000,  # Large position
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
        )
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT",
            strategy_id="test",
            direction="LONG",
            quantity=500,
            entry_price=300.0,
            entry_time=datetime.now(),
            current_price=300.0,
        )

        # Set high correlation
        portfolio.correlation_matrix = pd.DataFrame(
            {"AAPL": [1.0, 0.95], "MSFT": [0.95, 1.0]}, index=["AAPL", "MSFT"]
        )

        is_compliant, violations = portfolio.check_risk_limits()
        assert not is_compliant
        assert any("correlation" in v for v in violations)
        assert any("weight" in v for v in violations)  # Position too large

    def test_allocate_capital_non_equal_risk(self, portfolio):
        """Test capital allocation with configured weights."""
        portfolio.config["use_equal_risk"] = False
        portfolio.strategy_allocations = {"strategy1": 0.6, "strategy2": 0.4}

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=150.0,
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="GOOGL",
                direction="LONG",
                strength=0.6,
                strategy_id="strategy2",
                price=2800.0,
            ),
        ]

        market_data = {
            "AAPL": pd.DataFrame({"returns": [0.01, -0.005, 0.008]}),
            "GOOGL": pd.DataFrame({"returns": [0.015, -0.01, 0.012]}),
        }

        allocations = portfolio.allocate_capital(signals, market_data)
        assert allocations["strategy1"] == 0.6
        assert allocations["strategy2"] == 0.4

    def test_allocate_capital_with_kelly(self, portfolio):
        """Test capital allocation with Kelly fractions."""
        portfolio.strategy_kelly_fractions = {"strategy1": 0.5, "strategy2": 0.3}

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=150.0,
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="GOOGL",
                direction="LONG",
                strength=0.6,
                strategy_id="strategy2",
                price=2800.0,
            ),
        ]

        market_data = {
            "AAPL": pd.DataFrame({"close": [150, 151, 149, 152]}),
            "GOOGL": pd.DataFrame({"close": [2800, 2850, 2780, 2820]}),
        }

        allocations = portfolio.allocate_capital(signals, market_data)

        # Check Kelly fractions were applied
        assert "strategy1" in allocations
        assert "strategy2" in allocations
        assert sum(allocations.values()) > 0

    def test_size_position_with_margin_constraints(self, portfolio):
        """Test position sizing with margin constraints."""
        # Fill up margin
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=500,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
        )

        signal = Signal(
            timestamp=datetime.now(),
            symbol="GOOGL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=2800.0,
            atr=50.0,
        )

        position_size, stop_loss = portfolio.size_position(signal, 0.2)

        # Position should be reduced due to margin
        max_position_value = portfolio.current_equity * 0.2
        actual_position_value = position_size * signal.price
        assert actual_position_value <= max_position_value

        # Check ATR-based stop loss
        assert stop_loss == signal.price - (2 * signal.atr)

    def test_size_position_short_stop_loss(self, portfolio):
        """Test position sizing for short with ATR stop loss."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="SHORT",
            strength=-0.8,  # Negative for SHORT
            strategy_id="test",
            price=150.0,
            atr=3.0,
        )

        position_size, stop_loss = portfolio.size_position(signal, 0.1)

        # Stop loss should be above entry for short
        assert stop_loss == signal.price + (2 * signal.atr)

    def test_execute_signal_position_flip(self, portfolio):
        """Test executing signal that flips position direction."""
        # Create existing long position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )
        portfolio.current_equity = 105000  # Account for profit

        # Signal to go short
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="SHORT",
            strength=-0.8,  # Negative for SHORT
            strategy_id="test",
            price=155.0,
        )

        position = portfolio.execute_signal(signal, 100, 160.0)

        # Should have closed long and opened short
        assert position is not None
        assert position.direction == "SHORT"
        assert position.quantity == -100
        assert portfolio.current_equity > 105000  # Profit from closing long

    def test_execute_signal_zero_size(self, portfolio):
        """Test executing signal with zero position size."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=150.0,
        )

        position = portfolio.execute_signal(signal, 0, 145.0)
        assert position is None

    def test_close_position_update_strategy_performance(self, portfolio):
        """Test that closing positions updates strategy performance metrics."""
        # Open and close a winning position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="momentum",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
        )

        result = portfolio.close_position("AAPL", 160.0)

        assert result is not None
        assert "momentum" in portfolio.strategy_performance
        perf = portfolio.strategy_performance["momentum"]
        assert perf["trades"] == 1
        assert perf["wins"] == 1
        assert perf["total_pnl"] == 1000.0
        assert perf["win_pnl"] == 1000.0
        assert perf["loss_pnl"] == 0.0

        # Close a losing position
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            strategy_id="momentum",
            direction="SHORT",
            quantity=-10,
            entry_price=2800.0,
            entry_time=datetime.now(),
            current_price=2800.0,
        )

        result = portfolio.close_position("GOOGL", 2850.0)

        perf = portfolio.strategy_performance["momentum"]
        assert perf["trades"] == 2
        assert perf["wins"] == 1
        assert perf["total_pnl"] == 500.0  # 1000 - 500
        assert perf["loss_pnl"] == 500.0

    def test_update_strategy_kelly_insufficient_trades(self, portfolio):
        """Test Kelly fraction with insufficient trade history."""
        portfolio.strategy_performance["test"] = {
            "trades": 5,  # Less than MIN_TRADES_FOR_KELLY
            "wins": 3,
            "total_pnl": 100,
            "win_pnl": 150,
            "loss_pnl": 50,
        }

        portfolio.update_strategy_kelly_fractions()

        assert portfolio.strategy_kelly_fractions["test"] == DEFAULT_KELLY_FRACTION

    def test_update_strategy_kelly_zero_loss(self, portfolio):
        """Test Kelly fraction with zero average loss."""
        portfolio.strategy_performance["test"] = {
            "trades": 50,
            "wins": 50,  # All wins
            "total_pnl": 5000,
            "win_pnl": 5000,
            "loss_pnl": 0,
        }

        portfolio.update_strategy_kelly_fractions()

        # Kelly fraction is capped at 25% (MAX_KELLY_FRACTION * 0.25)
        assert portfolio.strategy_kelly_fractions["test"] == 0.25

    def test_update_strategy_kelly_normal_case(self, portfolio):
        """Test Kelly fraction calculation with normal win/loss."""
        portfolio.strategy_performance["test"] = {
            "trades": 100,
            "wins": 60,
            "total_pnl": 2000,
            "win_pnl": 4000,
            "loss_pnl": 2000,
        }

        portfolio.update_strategy_kelly_fractions()

        kelly = portfolio.strategy_kelly_fractions["test"]
        assert 0 < kelly <= MAX_KELLY_FRACTION * 0.25

    def test_check_stops_and_targets_long(self, portfolio):
        """Test stop loss and take profit for long positions."""
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
        )

        # Test stop loss hit
        exit_signals = portfolio.check_stops_and_targets({"AAPL": 144.0})
        assert len(exit_signals) == 1
        assert exit_signals[0].metadata["reason"] == "stop_loss"

        # Test take profit hit
        exit_signals = portfolio.check_stops_and_targets({"AAPL": 161.0})
        assert len(exit_signals) == 1
        assert exit_signals[0].metadata["reason"] == "take_profit"

        # Test no exit
        exit_signals = portfolio.check_stops_and_targets({"AAPL": 155.0})
        assert len(exit_signals) == 0

    def test_check_stops_and_targets_short(self, portfolio):
        """Test stop loss and take profit for short positions."""
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="SHORT",
            quantity=-100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
            stop_loss=155.0,
            take_profit=140.0,
        )

        # Test stop loss hit
        exit_signals = portfolio.check_stops_and_targets({"AAPL": 156.0})
        assert len(exit_signals) == 1
        assert exit_signals[0].metadata["reason"] == "stop_loss"

        # Test take profit hit
        exit_signals = portfolio.check_stops_and_targets({"AAPL": 139.0})
        assert len(exit_signals) == 1
        assert exit_signals[0].metadata["reason"] == "take_profit"

    def test_global_risk_check_risk_off_period(self, portfolio):
        """Test global risk check during risk-off period."""
        # Set risk-off mode
        portfolio.is_risk_off = True
        portfolio.risk_off_until = datetime.now() + timedelta(hours=1)

        is_ok, exit_signals = portfolio.global_risk_check()
        assert not is_ok
        assert len(exit_signals) == 0

        # Test risk-off period expired
        portfolio.risk_off_until = datetime.now() - timedelta(hours=1)
        is_ok, exit_signals = portfolio.global_risk_check()
        assert is_ok
        assert not portfolio.is_risk_off
        assert portfolio.risk_off_until is None

    def test_global_risk_check_drawdown_trigger(self, portfolio):
        """Test global risk check with drawdown exceeding limit."""
        # Set high drawdown
        portfolio.current_drawdown = 0.25  # 25% exceeds 20% limit

        # Add positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test1",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=140.0,
        )
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            strategy_id="test2",
            direction="SHORT",
            quantity=-10,
            entry_price=2800.0,
            entry_time=datetime.now(),
            current_price=2850.0,
        )

        is_ok, exit_signals = portfolio.global_risk_check()

        assert not is_ok
        assert len(exit_signals) == 2
        assert all(s.direction == "FLAT" for s in exit_signals)
        assert all(s.metadata["reason"] == "global_risk_off" for s in exit_signals)
        assert portfolio.is_risk_off
        assert portfolio.risk_off_until is not None

    def test_get_portfolio_summary(self, portfolio):
        """Test comprehensive portfolio summary generation."""
        # Setup portfolio state
        portfolio.current_equity = 110000
        portfolio.peak_equity = 115000
        portfolio.current_drawdown = (115000 - 110000) / 115000

        # Add positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="momentum",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            strategy_id="mean_reversion",
            direction="SHORT",
            quantity=-10,
            entry_price=2800.0,
            entry_time=datetime.now(),
            current_price=2750.0,
        )

        # Add strategy performance
        portfolio.strategy_performance["momentum"] = {
            "trades": 50,
            "wins": 30,
            "total_pnl": 5000,
            "win_pnl": 8000,
            "loss_pnl": 3000,
        }
        portfolio.strategy_kelly_fractions["momentum"] = 0.15

        summary = portfolio.get_portfolio_summary()

        # Check all sections exist
        assert "portfolio_metrics" in summary
        assert "strategy_exposure" in summary
        assert "strategy_performance" in summary
        assert "risk_status" in summary
        assert "positions" in summary

        # Check metrics
        metrics = summary["portfolio_metrics"]
        assert metrics["current_equity"] == 110000
        assert metrics["position_count"] == 2
        assert metrics["long_positions"] == 1
        assert metrics["short_positions"] == 1

        # Check strategy exposure
        assert summary["strategy_exposure"]["momentum"] == 100 * 155
        assert summary["strategy_exposure"]["mean_reversion"] == 10 * 2750

        # Check strategy performance
        assert "momentum" in summary["strategy_performance"]
        assert summary["strategy_performance"]["momentum"]["win_rate"] == 0.6
        assert summary["strategy_performance"]["momentum"]["kelly_fraction"] == 0.15

        # Check risk status
        assert summary["risk_status"]["current_drawdown"] == portfolio.current_drawdown

        # Check positions
        assert len(summary["positions"]) == 2
        assert summary["positions"]["AAPL"]["unrealized_pnl"] == 500.0

    def test_calculate_portfolio_metrics_with_daily_pnl(self, portfolio):
        """Test portfolio metrics calculation including daily P&L."""
        # Add some daily P&L history
        portfolio.daily_pnl = [100, -50, 200, -100, 150]

        # Add positions with unrealized P&L
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )

        portfolio.current_equity = 101000
        portfolio.peak_equity = 105000

        metrics = portfolio.calculate_portfolio_metrics()

        assert metrics["unrealized_pnl"] == 500.0
        assert metrics["realized_pnl"] == 500.0  # 101000 - 100000 - 500
        assert metrics["margin_usage"] > 0

    def test_allocate_capital_no_market_data(self, portfolio):
        """Test capital allocation when market data is missing."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=150.0,
            ),
        ]

        # Empty market data
        market_data = {}

        allocations = portfolio.allocate_capital(signals, market_data)

        # Should still allocate with default volatility
        assert len(allocations) > 0
        assert "strategy1" in allocations

    def test_edge_cases(self, portfolio):
        """Test various edge cases for complete coverage."""
        # Test with empty signals
        allocations = portfolio.allocate_capital([], {})
        assert allocations == {}

        # Test position sizing with zero price
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=0.0,
        )
        position_size, _ = portfolio.size_position(signal, 0.1)
        assert position_size == 0

        # Test close position that doesn't exist
        result = portfolio.close_position("NONEXISTENT", 100.0)
        assert result is None

        # Test check stops with missing price
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
            stop_loss=145.0,
        )
        exit_signals = portfolio.check_stops_and_targets({})
        assert len(exit_signals) == 0


    def test_peak_equity_update(self, portfolio):
        """Test peak equity tracking in calculate_portfolio_metrics."""
        # Set initial state
        portfolio.current_equity = 100000
        portfolio.peak_equity = 100000

        # Equity increases - should update peak
        portfolio.current_equity = 110000
        metrics = portfolio.calculate_portfolio_metrics()
        assert portfolio.peak_equity == 110000
        assert metrics["current_drawdown"] == 0.0

        # Equity decreases - peak should remain
        portfolio.current_equity = 105000
        metrics = portfolio.calculate_portfolio_metrics()
        assert portfolio.peak_equity == 110000
        assert metrics["current_drawdown"] == pytest.approx((110000 - 105000) / 110000)

    def test_allocate_capital_returns_from_close(self, portfolio):
        """Test capital allocation using close prices when returns not available."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=150.0,
            ),
        ]

        # Market data with only close prices
        market_data = {
            "AAPL": pd.DataFrame({"close": [148, 150, 149, 151, 152]})
        }

        allocations = portfolio.allocate_capital(signals, market_data)
        assert len(allocations) > 0
        assert "strategy1" in allocations

    def test_allocate_capital_zero_total_allocation(self, portfolio):
        """Test capital allocation when total allocation sums to zero."""
        # Mock a scenario where allocations sum to zero
        portfolio.config["use_equal_risk"] = False
        portfolio.strategy_allocations = {}  # Empty allocations

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=150.0,
            ),
        ]

        market_data = {"AAPL": pd.DataFrame({"returns": [0.01, -0.005]})}

        allocations = portfolio.allocate_capital(signals, market_data)
        # Should handle gracefully
        assert isinstance(allocations, dict)

    def test_size_position_with_metadata_stop_loss(self, portfolio):
        """Test position sizing with stop loss in metadata."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=150.0,
            metadata={"stop_loss": 145.0},
        )

        position_size, stop_loss = portfolio.size_position(signal, 0.1)

        # Should use metadata stop loss
        assert stop_loss == 145.0
        assert position_size > 0

    def test_execute_signal_same_direction_existing(self, portfolio):
        """Test executing signal when position exists in same direction."""
        # Create existing position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )

        # Signal in same direction
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=155.0,
        )

        position = portfolio.execute_signal(signal, 50, 150.0)

        # Should not create new position
        assert position is None
        assert len(portfolio.positions) == 1

    def test_execute_signal_with_take_profit(self, portfolio):
        """Test executing signal creates position with metadata."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=150.0,
            metadata={"take_profit": 160.0, "custom_data": "test"},
        )

        position = portfolio.execute_signal(signal, 100, 145.0)

        assert position is not None
        assert position.metadata == signal.metadata
        assert position.metadata["take_profit"] == 160.0

    def test_update_kelly_fraction_zero_wins(self, portfolio):
        """Test Kelly fraction with no wins (all losses)."""
        portfolio.strategy_performance["test"] = {
            "trades": 50,
            "wins": 0,  # No wins
            "total_pnl": -5000,
            "win_pnl": 0,
            "loss_pnl": 5000,
        }

        # When win_rate = 0, avg_win = 0, b = 0, should handle gracefully
        # The function should catch this case before division
        portfolio.update_strategy_kelly_fractions()

        # With zero wins, Kelly should be 0 (clamped at min)
        kelly = portfolio.strategy_kelly_fractions["test"]
        assert kelly == 0  # Negative Kelly should be clamped to 0

    def test_update_kelly_fraction_zero_avg_loss(self, portfolio):
        """Test Kelly fraction when avg_loss is zero."""
        portfolio.strategy_performance["test"] = {
            "trades": 50,
            "wins": 25,
            "total_pnl": 5000,
            "win_pnl": 5000,
            "loss_pnl": 0,  # No loss amount
        }

        portfolio.update_strategy_kelly_fractions()

        # When avg_loss = 0, should use default 0.5
        assert portfolio.strategy_kelly_fractions["test"] == 0.5

    def test_global_risk_check_with_violations(self, portfolio):
        """Test global risk check logs violations but continues."""
        # Add positions that violate correlation limits
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=1000,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0,
        )
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT",
            strategy_id="test",
            direction="LONG",
            quantity=1000,
            entry_price=300.0,
            entry_time=datetime.now(),
            current_price=300.0,
        )

        # Set high correlation
        portfolio.correlation_matrix = pd.DataFrame(
            {"AAPL": [1.0, 0.95], "MSFT": [0.95, 1.0]},
            index=["AAPL", "MSFT"]
        )

        # Drawdown within limits
        portfolio.current_drawdown = 0.10

        with patch("core.portfolio.logger.warning") as mock_warning:
            is_ok, exit_signals = portfolio.global_risk_check()

            # Should log violations but not trigger risk-off
            assert is_ok
            assert len(exit_signals) == 0
            assert mock_warning.called

    def test_zero_division_edge_cases(self, portfolio):
        """Test various zero division edge cases."""
        # Test with zero peak equity
        portfolio.peak_equity = 0
        portfolio.current_equity = 100000
        metrics = portfolio.calculate_portfolio_metrics()
        assert metrics["current_drawdown"] == 0

        # Test margin usage with zero equity
        portfolio.current_equity = 0
        metrics = portfolio.calculate_portfolio_metrics()
        assert metrics["margin_usage"] == 0

    def test_check_risk_limits_drawdown_violation(self, portfolio):
        """Test risk limit check for drawdown violation specifically."""
        # Set drawdown exceeding limit
        portfolio.current_drawdown = 0.25  # 25% exceeds 20% limit

        is_compliant, violations = portfolio.check_risk_limits()

        assert not is_compliant
        assert len(violations) >= 1
        assert any("Drawdown" in v and "25.0%" in v for v in violations)

    def test_allocate_capital_flat_signals(self, portfolio):
        """Test capital allocation with FLAT signals (should be ignored)."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="FLAT",
                strength=0.0,
                strategy_id="strategy1",
                price=150.0,
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="GOOGL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy2",
                price=2800.0,
            ),
        ]

        market_data = {
            "AAPL": pd.DataFrame({"returns": [0.01, -0.005]}),
            "GOOGL": pd.DataFrame({"returns": [0.015, -0.01]}),
        }

        allocations = portfolio.allocate_capital(signals, market_data)

        # Should only allocate to strategy2
        assert "strategy1" not in allocations
        assert "strategy2" in allocations

    def test_allocate_capital_empty_returns_dataframe(self, portfolio):
        """Test capital allocation when returns DataFrame is empty."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="strategy1",
                price=150.0,
            ),
        ]

        # Market data with empty returns
        market_data = {
            "AAPL": pd.DataFrame({"returns": []})  # Empty
        }

        allocations = portfolio.allocate_capital(signals, market_data)

        # Should still allocate with default volatility
        assert "strategy1" in allocations
        assert allocations["strategy1"] > 0

    def test_allocate_capital_zero_volatility(self, portfolio):
        """Test capital allocation when strategy volatility is zero."""
        # Create multiple signals to trigger volatility calculation
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="zero_vol_strategy",
                price=150.0,
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="GOOGL",
                direction="LONG",
                strength=0.8,
                strategy_id="normal_strategy",
                price=2800.0,
            ),
        ]

        # Market data with zero volatility for AAPL (all returns are 0)
        market_data = {
            "AAPL": pd.DataFrame({"returns": [0.0, 0.0, 0.0, 0.0, 0.0]}),
            "GOOGL": pd.DataFrame({"returns": [0.01, -0.01, 0.015, -0.005, 0.01]}),
        }

        allocations = portfolio.allocate_capital(signals, market_data)

        # Strategy with zero volatility should get 0 allocation
        assert allocations["zero_vol_strategy"] == 0.0
        # Normal strategy should get positive allocation
        assert allocations["normal_strategy"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
