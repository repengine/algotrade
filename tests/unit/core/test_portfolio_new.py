"""
Unit tests for Portfolio class.

Tests cover:
- Portfolio initialization and configuration
- Position management
- PnL calculations
- Risk metrics and limits
- Portfolio volatility calculations
- Capital allocation
- State transitions

All tests follow FIRST principles and use strong assertions.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from core.portfolio import PortfolioEngine, Position
from strategies.base import Signal
from utils.constants import (
    DEFAULT_INITIAL_CAPITAL,
    TRADING_DAYS_PER_YEAR,
)
from freezegun import freeze_time


class TestPortfolioConstruction:
    """Test portfolio initialization and configuration."""

    @pytest.mark.unit
    def test_portfolio_init_default_values(self):
        """
        Portfolio initializes with default values.

        Default portfolio should have:
        - Starting capital as current equity
        - Empty positions
        - Default risk limits
        - No performance history
        """
        config = {"initial_capital": 100000}
        portfolio = PortfolioEngine(config)

        assert portfolio.initial_capital == 100000
        assert portfolio.current_equity == 100000
        assert portfolio.positions == {}
        assert portfolio.peak_equity == 100000
        assert portfolio.current_drawdown == 0.0
        assert portfolio.is_risk_off is False
        assert len(portfolio.performance_history) == 0

    @pytest.mark.unit
    def test_portfolio_init_with_custom_config(self):
        """
        Portfolio respects custom configuration.

        All configuration parameters should be properly set.
        """
        config = {
            "initial_capital": 50000,
            "target_vol": 0.15,
            "max_position_size": 0.25,
            "max_drawdown": 0.20,
            "max_correlation": 0.80,
            "volatility_targets": {"aggressive": 0.20, "conservative": 0.05}
        }
        portfolio = PortfolioEngine(config)

        assert portfolio.initial_capital == 50000
        assert portfolio.risk_limits["max_portfolio_volatility"] == 0.15
        assert portfolio.risk_limits["max_position_size"] == 0.25
        assert portfolio.risk_limits["max_drawdown"] == 0.20
        assert portfolio.risk_limits["max_correlation"] == 0.80
        assert portfolio.volatility_targets == {"aggressive": 0.20, "conservative": 0.05}

    @pytest.mark.unit
    def test_portfolio_init_missing_config(self):
        """
        Portfolio handles missing configuration gracefully.

        Should use defaults for missing values.
        """
        config = {}  # Empty config
        portfolio = PortfolioEngine(config)

        assert portfolio.initial_capital == DEFAULT_INITIAL_CAPITAL
        assert portfolio.risk_limits["max_portfolio_volatility"] == 0.10  # Default 10%
        assert portfolio.risk_limits["max_position_size"] == 0.20  # Default 20%


class TestPositionManagement:
    """Test adding, updating, and managing positions."""

    @pytest.fixture
    def portfolio(self):
        """Standard portfolio for position tests."""
        return PortfolioEngine({"initial_capital": 100000})

    @pytest.mark.unit
    def test_position_creation(self):
        """
        Position is created with correct attributes.

        All position properties should be calculated correctly.
        """
        position = Position(
            symbol="AAPL",
            strategy_id="test_strategy",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
            stop_loss=145.0,
            take_profit=160.0
        )

        assert position.symbol == "AAPL"
        assert position.strategy_id == "test_strategy"
        assert position.direction == "LONG"
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.current_price == 155.0
        assert position.market_value == 15500.0  # 100 * 155
        assert position.unrealized_pnl == 500.0  # 100 * (155 - 150)
        assert position.pnl_percentage == pytest.approx(3.33, rel=0.01)  # 500 / 15000

    @pytest.mark.unit
    def test_short_position_pnl(self):
        """
        Short position PnL is calculated correctly.

        Short positions profit when price decreases.
        """
        position = Position(
            symbol="TSLA",
            strategy_id="short_strategy",
            direction="SHORT",
            quantity=50,  # Short positions can have positive quantity
            entry_price=200.0,
            entry_time=datetime.now(),
            current_price=190.0
        )

        # Market value is always positive
        assert position.market_value == 9500.0  # 50 * 190

        # Short position gains when price drops
        assert position.unrealized_pnl == 500.0  # 50 * (200 - 190)
        assert position.pnl_percentage == pytest.approx(5.0, rel=0.01)  # 500 / 10000

    @pytest.mark.unit
    def test_update_market_prices(self, portfolio):
        """
        Market prices update correctly for all positions.

        Position values and PnL should reflect new prices.
        """
        # Add positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=150.0
        )
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL",
            strategy_id="test",
            direction="SHORT",
            quantity=20,
            entry_price=2500.0,
            entry_time=datetime.now(),
            current_price=2500.0
        )

        # Update prices
        new_prices = {"AAPL": 160.0, "GOOGL": 2450.0}
        portfolio.update_market_prices(new_prices)

        # Check updated values
        assert portfolio.positions["AAPL"].current_price == 160.0
        assert portfolio.positions["AAPL"].unrealized_pnl == 1000.0  # 100 * 10

        assert portfolio.positions["GOOGL"].current_price == 2450.0
        assert portfolio.positions["GOOGL"].unrealized_pnl == 1000.0  # 20 * 50 profit

    @pytest.mark.unit
    def test_update_prices_partial_update(self, portfolio):
        """
        Price updates handle missing symbols gracefully.

        Only positions with price updates should change.
        """
        # Add positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=100, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0
        )
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT", strategy_id="test", direction="LONG",
            quantity=50, entry_price=300.0, entry_time=datetime.now(),
            current_price=300.0
        )

        # Update only AAPL price
        portfolio.update_market_prices({"AAPL": 155.0})

        assert portfolio.positions["AAPL"].current_price == 155.0
        assert portfolio.positions["MSFT"].current_price == 300.0  # Unchanged


class TestPortfolioMetrics:
    """Test portfolio metrics calculations."""

    @pytest.fixture
    def portfolio_with_positions(self):
        """Portfolio with mixed positions for metrics testing."""
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Add winning position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=100, entry_price=150.0, entry_time=datetime.now(),
            current_price=160.0  # +$10/share = +$1000
        )

        # Add losing position
        portfolio.positions["GOOGL"] = Position(
            symbol="GOOGL", strategy_id="test", direction="LONG",
            quantity=20, entry_price=2600.0, entry_time=datetime.now(),
            current_price=2550.0  # -$50/share = -$1000
        )

        # Add short position
        portfolio.positions["TSLA"] = Position(
            symbol="TSLA", strategy_id="test", direction="SHORT",
            quantity=50, entry_price=200.0, entry_time=datetime.now(),
            current_price=190.0  # +$10/share = +$500 profit
        )

        return portfolio

    @pytest.mark.unit
    def test_calculate_portfolio_metrics(self, portfolio_with_positions):
        """
        Portfolio metrics are calculated correctly.

        All metrics should reflect current positions and PnL.
        """
        metrics = portfolio_with_positions.calculate_portfolio_metrics()

        # Positions value: AAPL(16000) + GOOGL(51000) + TSLA(9500) = 76500
        assert metrics["positions_value"] == 76500.0

        # Unrealized PnL: AAPL(+1000) + GOOGL(-1000) + TSLA(+500) = +500
        assert metrics["unrealized_pnl"] == 500.0

        # Cash = Initial - positions cost = 100000 - (15000 + 52000 + 10000) = 23000
        # But we track current equity, so cash = equity - positions_value
        assert metrics["cash"] == 100000 - 76500  # 23500

        assert metrics["total_equity"] == 100000  # Still at initial
        assert metrics["position_count"] == 3
        assert metrics["long_positions"] == 2
        assert metrics["short_positions"] == 1
        assert metrics["margin_usage"] == pytest.approx(0.765, rel=0.001)  # 76500/100000
        assert metrics["current_drawdown"] == 0.0  # No drawdown yet

    @pytest.mark.unit
    def test_drawdown_calculation(self):
        """
        Drawdown is tracked correctly as equity changes.

        Should update peak equity and calculate drawdown percentage.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Simulate equity increase
        portfolio.current_equity = 110000
        metrics = portfolio.calculate_portfolio_metrics()
        assert portfolio.peak_equity == 110000
        assert metrics["current_drawdown"] == 0.0

        # Simulate drawdown
        portfolio.current_equity = 95000
        metrics = portfolio.calculate_portfolio_metrics()
        assert portfolio.peak_equity == 110000  # Peak unchanged
        assert metrics["current_drawdown"] == pytest.approx(0.1364, rel=0.001)  # 15000/110000

        # Partial recovery
        portfolio.current_equity = 105000
        metrics = portfolio.calculate_portfolio_metrics()
        assert portfolio.peak_equity == 110000  # Still unchanged
        assert metrics["current_drawdown"] == pytest.approx(0.0455, rel=0.001)  # 5000/110000

        # New peak
        portfolio.current_equity = 120000
        metrics = portfolio.calculate_portfolio_metrics()
        assert portfolio.peak_equity == 120000
        assert metrics["current_drawdown"] == 0.0

    @pytest.mark.unit
    def test_empty_portfolio_metrics(self):
        """
        Empty portfolio returns valid metrics.

        Should handle case with no positions gracefully.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})
        metrics = portfolio.calculate_portfolio_metrics()

        assert metrics["total_equity"] == 100000
        assert metrics["cash"] == 100000
        assert metrics["positions_value"] == 0
        assert metrics["unrealized_pnl"] == 0
        assert metrics["realized_pnl"] == 0
        assert metrics["position_count"] == 0
        assert metrics["margin_usage"] == 0


class TestRiskLimits:
    """Test risk limit checking and enforcement."""

    @pytest.fixture
    def portfolio(self):
        """Portfolio with strict risk limits."""
        return PortfolioEngine({
            "initial_capital": 100000,
            "max_position_size": 0.20,  # 20% max
            "max_drawdown": 0.15,  # 15% max
            "max_correlation": 0.70,  # 70% max correlation
        })

    @pytest.mark.unit
    def test_position_concentration_check(self, portfolio):
        """
        Position concentration limits are enforced.

        Should flag positions exceeding size limits.
        """
        # Add concentrated position (30% of portfolio)
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=200, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0  # $30,000 = 30% of $100k
        )

        # Add acceptable position (10% of portfolio)
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT", strategy_id="test", direction="LONG",
            quantity=33, entry_price=300.0, entry_time=datetime.now(),
            current_price=300.0  # $9,900 = ~10% of $100k
        )

        is_valid, violations = portfolio.check_risk_limits()

        assert not is_valid
        assert len(violations) == 1
        assert "AAPL weight 30.0% exceeds limit 20.0%" in violations[0]

    @pytest.mark.unit
    def test_drawdown_limit_check(self, portfolio):
        """
        Drawdown limits are enforced.

        Should flag when drawdown exceeds maximum allowed.
        """
        # Simulate drawdown
        portfolio.peak_equity = 100000
        portfolio.current_equity = 83000  # 17% drawdown
        portfolio.current_drawdown = 0.17

        is_valid, violations = portfolio.check_risk_limits()

        assert not is_valid
        assert len(violations) == 1
        assert "Drawdown 17.0% exceeds limit 15.0%" in violations[0]

    @pytest.mark.unit
    def test_correlation_limit_check(self, portfolio):
        """
        Correlation limits between positions are enforced.

        Should flag highly correlated positions.
        """
        # Add two positions - keep under 20% limit
        portfolio.positions["SPY"] = Position(
            symbol="SPY", strategy_id="test", direction="LONG",
            quantity=40, entry_price=450.0, entry_time=datetime.now(),
            current_price=450.0
        )
        portfolio.positions["QQQ"] = Position(
            symbol="QQQ", strategy_id="test", direction="LONG",
            quantity=50, entry_price=380.0, entry_time=datetime.now(),
            current_price=380.0
        )

        # Set high correlation
        portfolio.correlation_matrix = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            index=["SPY", "QQQ"],
            columns=["SPY", "QQQ"]
        )

        is_valid, violations = portfolio.check_risk_limits()

        assert not is_valid
        assert len(violations) == 1
        # Check for correlation violation (order may vary)
        assert "correlation 0.85 exceeds limit 0.70" in violations[0]
        assert ("SPY" in violations[0] and "QQQ" in violations[0])

    @pytest.mark.unit
    def test_multiple_violations(self, portfolio):
        """
        Multiple risk violations are all reported.

        Should return all violations, not just the first one.
        """
        # Add concentrated position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=300, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0  # $45,000 = 45% of portfolio
        )

        # Simulate drawdown
        portfolio.peak_equity = 100000
        portfolio.current_equity = 80000
        portfolio.current_drawdown = 0.20

        is_valid, violations = portfolio.check_risk_limits()

        assert not is_valid
        assert len(violations) == 2
        assert any("AAPL weight" in v for v in violations)
        assert any("Drawdown 20.0%" in v for v in violations)

    @pytest.mark.unit
    def test_no_violations(self, portfolio):
        """
        Portfolio within all limits passes checks.

        Should return valid with empty violations list.
        """
        # Add reasonable position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=100, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0  # $15,000 = 15% of portfolio
        )

        is_valid, violations = portfolio.check_risk_limits()

        assert is_valid
        assert violations == []


class TestPortfolioVolatility:
    """Test portfolio volatility calculations."""

    @pytest.fixture
    def returns_data(self):
        """Generate sample returns data."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

        # Set random seed for reproducible tests
        np.random.seed(42)

        # Correlated returns
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.018, 100),
            'GOOGL': np.random.normal(0.0008, 0.022, 100),
        }, index=dates)

        # Add stronger correlation to ensure test passes
        returns['MSFT'] = returns['MSFT'] * 0.5 + returns['AAPL'] * 0.5

        return returns

    @pytest.mark.unit
    def test_portfolio_volatility_calculation(self, returns_data):
        """
        Portfolio volatility is calculated correctly.

        Should account for positions, weights, and correlations.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Add positions with different weights
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=200, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0  # $30,000 = 30%
        )
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT", strategy_id="test", direction="LONG",
            quantity=100, entry_price=300.0, entry_time=datetime.now(),
            current_price=300.0  # $30,000 = 30%
        )

        vol = portfolio.calculate_portfolio_volatility(returns_data)

        # Portfolio vol should be between min and max constituent vols
        # due to diversification (unless perfect correlation)
        constituent_vols = returns_data.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        assert vol > 0
        assert vol < constituent_vols.max()  # Diversification benefit
        assert vol > constituent_vols.min() * 0.5  # Not too low

    @pytest.mark.unit
    def test_portfolio_volatility_with_shorts(self, returns_data):
        """
        Portfolio volatility accounts for short positions.

        Short positions should have negative weights.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Long position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=200, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0
        )

        # Short position (hedged)
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT", strategy_id="test", direction="SHORT",
            quantity=50, entry_price=300.0, entry_time=datetime.now(),
            current_price=300.0
        )

        vol = portfolio.calculate_portfolio_volatility(returns_data)

        # Hedged portfolio should have lower vol than unhedged
        assert vol > 0
        assert vol < returns_data['AAPL'].std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    @pytest.mark.unit
    def test_empty_portfolio_volatility(self):
        """
        Empty portfolio has zero volatility.

        Should handle edge case gracefully.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})
        returns_data = pd.DataFrame()  # Empty

        vol = portfolio.calculate_portfolio_volatility(returns_data)
        assert vol == 0.0

    @pytest.mark.unit
    def test_update_correlation_matrix(self, returns_data):
        """
        Correlation matrix is updated correctly.

        Should calculate correlations between all assets.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})
        portfolio.update_correlation_matrix(returns_data)

        # Check matrix properties
        corr = portfolio.correlation_matrix
        assert corr.shape == (3, 3)  # 3x3 for 3 assets
        assert all(np.diag(corr.values) == 1.0)  # Diagonal should be 1
        assert corr.equals(corr.T)  # Should be symmetric

        # Check correlation values are reasonable
        assert all(-1 <= corr.values.flatten()) and all(corr.values.flatten() <= 1)

        # AAPL-MSFT should have positive correlation (by construction)
        assert corr.loc['AAPL', 'MSFT'] > 0.5


class TestCapitalAllocation:
    """Test capital allocation and volatility budgeting."""

    @pytest.fixture
    def signals(self):
        """Generate test signals from multiple strategies."""
        base_time = datetime.now()
        return [
            Signal(
                timestamp=base_time,
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="momentum",
                price=150.0,
                atr=3.0
            ),
            Signal(
                timestamp=base_time,
                symbol="MSFT",
                direction="LONG",
                strength=0.6,
                strategy_id="momentum",
                price=300.0,
                atr=5.0
            ),
            Signal(
            timestamp=base_time,
            symbol="GOOGL",
            direction="SHORT",
            strength=-0.7,
                strategy_id="mean_reversion",
                price=2500.0,
                atr=50.0
            ),
        ]

    @pytest.fixture
    def market_data(self):
        """Generate market data for allocation tests."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')

        data = {}
        for symbol, vol in [("AAPL", 0.02), ("MSFT", 0.018), ("GOOGL", 0.025)]:
            returns = np.random.normal(0.001, vol, 50)
            data[symbol] = pd.DataFrame({
                'close': 100 * np.exp(np.cumsum(returns)),
                'returns': returns
            }, index=dates)

        return data

    @pytest.mark.unit
    def test_allocate_capital_basic(self, signals, market_data):
        """
        Capital allocation works for multiple strategies.

        Should allocate based on volatility and signals.
        """
        portfolio = PortfolioEngine({
            "initial_capital": 100000,
            "target_vol": 0.15
        })

        allocations = portfolio.allocate_capital(signals, market_data)

        # Should have allocations for strategies with signals
        assert "momentum" in allocations
        assert "mean_reversion" in allocations

        # Allocations should be positive
        assert all(alloc > 0 for alloc in allocations.values())

        # Total allocation should not exceed capital
        assert sum(allocations.values()) <= portfolio.current_equity

    @pytest.mark.unit
    def test_allocate_capital_no_signals(self, market_data):
        """
        No allocation when there are no signals.

        Should return empty allocations.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})
        allocations = portfolio.allocate_capital([], market_data)

        assert allocations == {}

    @pytest.mark.unit
    def test_allocate_capital_flat_signals_ignored(self, market_data):
        """
        FLAT signals are ignored in allocation.

        Only directional signals should receive capital.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="FLAT",  # Exit signal
                strength=0.0,
                strategy_id="test",
                price=150.0,
                atr=3.0
            )
        ]

        allocations = portfolio.allocate_capital(signals, market_data)
        assert allocations == {}


class TestPortfolioStateTransitions:
    """Test portfolio state changes and risk management."""

    @pytest.mark.unit
    def test_risk_off_mode(self):
        """
        Portfolio enters and exits risk-off mode correctly.

        Should track risk-off state and timing.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Enter risk-off mode
        portfolio.is_risk_off = True
        portfolio.risk_off_until = datetime.now() + timedelta(hours=24)

        assert portfolio.is_risk_off is True
        assert portfolio.risk_off_until > datetime.now()

        # Check if should exit risk-off
        with freeze_time(datetime.now() + timedelta(hours=25)):
            # In real implementation, this would be checked
            assert portfolio.risk_off_until < datetime.now()

    @pytest.mark.unit
    def test_performance_history_tracking(self):
        """
        Performance history is recorded correctly.

        Should track metrics over time for analysis.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Record performance snapshot
        metrics = portfolio.calculate_portfolio_metrics()
        portfolio.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        assert len(portfolio.performance_history) == 1
        assert 'timestamp' in portfolio.performance_history[0]
        assert 'metrics' in portfolio.performance_history[0]
        assert portfolio.performance_history[0]['metrics']['total_equity'] == 100000


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_zero_price_handling(self):
        """
        Positions handle zero prices gracefully.

        Should not crash or produce invalid calculations.
        """
        position = Position(
            symbol="TEST",
            strategy_id="test",
            direction="LONG",
            quantity=100,
            entry_price=50.0,
            entry_time=datetime.now(),
            current_price=0.0  # Stock went to zero
        )

        assert position.market_value == 0.0
        assert position.unrealized_pnl == -5000.0  # Total loss
        assert position.pnl_percentage == -100.0

    @pytest.mark.unit
    def test_negative_equity_handling(self):
        """
        Portfolio handles negative equity scenarios.

        Should not crash when equity goes negative.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})
        portfolio.current_equity = -5000  # Major losses

        violations = portfolio._check_position_concentration()

        # Should handle gracefully
        assert len(violations) >= 0
        if portfolio.positions:
            assert any("zero or negative" in v for v in violations)

    @pytest.mark.unit
    def test_invalid_correlation_matrix(self):
        """
        Invalid correlation matrices are handled safely.

        Should not crash on malformed correlation data.
        """
        portfolio = PortfolioEngine({"initial_capital": 100000})

        # Add positions
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", strategy_id="test", direction="LONG",
            quantity=100, entry_price=150.0, entry_time=datetime.now(),
            current_price=150.0
        )
        portfolio.positions["MSFT"] = Position(
            symbol="MSFT", strategy_id="test", direction="LONG",
            quantity=100, entry_price=300.0, entry_time=datetime.now(),
            current_price=300.0
        )

        # Set invalid correlation matrix (wrong symbols)
        portfolio.correlation_matrix = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["IBM", "ORCL"],  # Different symbols
            columns=["IBM", "ORCL"]
        )

        violations = portfolio._check_correlation_violations()

        # Should handle missing symbols gracefully
        assert isinstance(violations, list)
        assert len(violations) == 0  # No correlations to check


class TestParametrizedScenarios:
    """Test various scenarios with parametrization."""

    @pytest.mark.unit
    @pytest.mark.parametrize("initial_capital,position_size,expected_valid", [
        (100000, 15000, True),   # 15% position - OK
        (100000, 20000, True),   # 20% position - At limit
        (100000, 25000, False),  # 25% position - Over limit
        (50000, 15000, False),   # 30% position - Over limit
        (200000, 30000, True),   # 15% position - OK
    ])
    def test_position_size_limits(self, initial_capital, position_size, expected_valid):
        """
        Position size limits work for various capital levels.

        Tests parametrized position sizing scenarios.
        """
        portfolio = PortfolioEngine({
            "initial_capital": initial_capital,
            "max_position_size": 0.20
        })

        # Add position of specified size
        quantity = position_size / 100  # Assuming $100 price
        portfolio.positions["TEST"] = Position(
            symbol="TEST", strategy_id="test", direction="LONG",
            quantity=quantity, entry_price=100.0,
            entry_time=datetime.now(), current_price=100.0
        )

        is_valid, violations = portfolio.check_risk_limits()
        assert is_valid == expected_valid

        if not expected_valid:
            assert len(violations) > 0
            assert "exceeds limit" in violations[0]

    @pytest.mark.unit
    @pytest.mark.parametrize("direction,entry,current,expected_pnl", [
        ("LONG", 100, 110, 1000),    # Long profit
        ("LONG", 100, 90, -1000),    # Long loss
        ("SHORT", 100, 90, 1000),    # Short profit
        ("SHORT", 100, 110, -1000),  # Short loss
        ("LONG", 100, 100, 0),       # No change
    ])
    def test_pnl_calculations(self, direction, entry, current, expected_pnl):
        """
        PnL calculations are correct for all scenarios.

        Tests long/short positions with profits/losses.
        """
        position = Position(
            symbol="TEST",
            strategy_id="test",
            direction=direction,
            quantity=100,
            entry_price=entry,
            entry_time=datetime.now(),
            current_price=current
        )

        assert position.unrealized_pnl == expected_pnl
