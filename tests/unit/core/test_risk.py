#!/usr/bin/env python3
"""Comprehensive tests for the risk management system."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from core.risk import EnhancedRiskManager, RiskMetrics
from strategies.base import RiskContext, Signal


@pytest.fixture
def risk_config() -> dict[str, Any]:
    """Risk manager configuration fixture."""
    return {
        "max_var_95": 0.02,
        "target_vol": 0.10,
        "max_position_size": 0.20,
        "max_sector_exposure": 0.40,
        "max_drawdown": 0.15,
        "max_correlation": 0.70,
        "min_sharpe": 0.5,
        "concentration_limit": 0.60,
    }


@pytest.fixture
def risk_manager(risk_config: dict[str, Any]) -> EnhancedRiskManager:
    """Create a risk manager instance."""
    return EnhancedRiskManager(risk_config)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Generate sample return data."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    # Generate returns with known properties
    returns = pd.DataFrame(
        {
            "SPY": np.random.normal(0.0005, 0.01, 252),  # ~12% annual return, 16% vol
            "QQQ": np.random.normal(0.0007, 0.015, 252),  # ~17% annual return, 24% vol
            "IWM": np.random.normal(0.0003, 0.012, 252),  # ~7% annual return, 19% vol
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_positions() -> dict[str, Any]:
    """Sample portfolio positions."""
    return {
        "SPY": {"value": 50000, "quantity": 100, "sector": "broad"},
        "QQQ": {"value": 30000, "quantity": 50, "sector": "tech"},
        "IWM": {"value": 20000, "quantity": 80, "sector": "small_cap"},
    }


class TestEnhancedRiskManager:
    """Test suite for EnhancedRiskManager."""

    def test_initialization(self, risk_config):
        """Test risk manager initialization."""
        rm = EnhancedRiskManager(risk_config)

        assert rm.config == risk_config
        assert rm.risk_limits["max_var_95"] == 0.02
        assert rm.risk_limits["max_portfolio_volatility"] == 0.10
        assert rm.risk_limits["max_position_size"] == 0.20
        assert len(rm.historical_metrics) == 0
        assert rm.current_regime == "NORMAL"

    def test_calculate_risk_metrics(self, risk_manager, sample_returns):
        """Test risk metrics calculation."""
        portfolio_value = 100000
        metrics = risk_manager.calculate_risk_metrics(
            sample_returns, portfolio_value, lookback_days=252
        )

        assert isinstance(metrics, RiskMetrics)
        assert 0 < metrics.value_at_risk < 0.05  # VaR should be reasonable
        assert metrics.conditional_var >= metrics.value_at_risk
        assert -2 < metrics.sharpe_ratio < 3  # Reasonable Sharpe range
        assert 0 < metrics.maximum_drawdown < 1
        assert metrics.portfolio_volatility > 0
        assert metrics.portfolio_beta == 1.0  # Default when no benchmark

    def test_calculate_position_var(self, risk_manager, sample_returns):
        """Test position-level VaR calculation."""
        position_value = 10000
        position_returns = sample_returns["SPY"]

        var = risk_manager.calculate_position_var(
            position_returns, position_value, confidence_level=0.95
        )

        assert var > 0
        assert var < position_value * 0.1  # VaR should be less than 10%

    def test_portfolio_optimization(self, risk_manager, sample_returns):
        """Test portfolio optimization."""
        current_weights = np.array([0.5, 0.3, 0.2])
        target_return = 0.10

        optimal_weights = risk_manager.portfolio_optimization(
            sample_returns, current_weights, target_return=target_return
        )

        assert len(optimal_weights) == 3
        assert np.isclose(np.sum(optimal_weights), 1.0)
        assert all(0 <= w <= 1 for w in optimal_weights)

    def test_check_risk_compliance(self, risk_manager, sample_positions):
        """Test risk compliance checking."""
        portfolio_value = 100000
        current_drawdown = 0.08

        is_compliant, violations = risk_manager.check_risk_compliance(
            positions=sample_positions,
            portfolio_value=portfolio_value,
            current_drawdown=current_drawdown,
        )

        assert isinstance(is_compliant, bool)
        assert isinstance(violations, list)

        # Test with excessive concentration
        concentrated_positions = {
            "AAPL": {"value": 70000, "quantity": 400, "sector": "tech"},
            "MSFT": {"value": 30000, "quantity": 100, "sector": "tech"},
        }

        is_compliant, violations = risk_manager.check_risk_compliance(
            positions=concentrated_positions,
            portfolio_value=portfolio_value,
            current_drawdown=current_drawdown,
        )

        assert not is_compliant
        assert len(violations) > 0
        assert any("concentration" in v.lower() for v in violations)

    def test_calculate_stress_scenarios(self, risk_manager, sample_positions):
        """Test stress scenario calculations."""
        # Create mock portfolio
        from unittest.mock import Mock

        # Create position mocks
        positions = {}
        for symbol, data in sample_positions.items():
            pos = Mock()
            pos.direction = "LONG"
            pos.market_value = data["value"]
            positions[symbol] = pos

        portfolio = Mock()
        portfolio.positions = positions
        portfolio.current_equity = 100000

        scenarios = risk_manager.calculate_stress_scenarios(portfolio)

        assert "market_crash" in scenarios
        assert "flash_crash" in scenarios
        assert "liquidity_crisis" in scenarios

        # Market crash should show negative impact
        assert scenarios["market_crash"] < 0

    def test_get_risk_adjusted_sizes(self, risk_manager):
        """Test risk-adjusted position sizing."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="SPY",
                direction="LONG",
                strength=0.8,
                strategy_id="test",
                price=400.0,
                atr=5.0,
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="QQQ",
                direction="SHORT",
                strength=-0.6,
                strategy_id="test",
                price=300.0,
                atr=8.0,
            ),
        ]

        # Create mock portfolio
        from unittest.mock import Mock
        portfolio = Mock()
        portfolio.current_equity = 100000

        # Create market data
        market_data = {
            "SPY": pd.DataFrame({"returns": np.random.normal(0.001, 0.01, 100)}),
            "QQQ": pd.DataFrame({"returns": np.random.normal(0.001, 0.015, 100)})
        }

        adjusted_sizes = risk_manager.get_risk_adjusted_sizes(signals, portfolio, market_data)

        assert len(adjusted_sizes) == 2
        assert "SPY" in adjusted_sizes
        assert "QQQ" in adjusted_sizes

        for _symbol, size in adjusted_sizes.items():
            assert size >= 0
            assert size <= 0.20  # Max position size

    def test_update_regime(self, risk_manager, sample_returns):
        """Test market regime detection."""
        # Normal volatility
        risk_manager.update_regime(sample_returns)
        assert risk_manager.current_regime == "NORMAL"

        # Simulate high volatility
        high_vol_returns = sample_returns * 3  # Triple the volatility
        risk_manager.update_regime(high_vol_returns)
        assert risk_manager.current_regime == "HIGH_VOL"

        # Simulate crisis with correlation breakdown
        crisis_returns = sample_returns.copy()
        crisis_returns["SPY"] = crisis_returns["SPY"] * -2
        crisis_returns["QQQ"] = crisis_returns["QQQ"] * -2.5
        risk_manager.volatility_scalar = 3.0  # Simulate high vol
        risk_manager.update_regime(crisis_returns)
        assert risk_manager.current_regime == "RISK_OFF"

    def test_update_historical_metrics(self, risk_manager):
        """Test historical metrics update."""
        metrics = RiskMetrics(
            value_at_risk=0.015,
            conditional_var=0.02,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            maximum_drawdown=0.12,
            current_drawdown=0.05,
            downside_deviation=0.008,
            portfolio_volatility=0.15,
            portfolio_beta=0.95,
            correlation_risk=0.3,
        )

        risk_manager.update_historical_metrics(metrics)

        assert len(risk_manager.historical_metrics) == 1
        assert risk_manager.historical_metrics[0] == metrics

        # Test rolling window limit
        for _ in range(300):
            risk_manager.update_historical_metrics(metrics)

        assert len(risk_manager.historical_metrics) == 252  # Should maintain window

    def test_get_average_metrics(self, risk_manager):
        """Test average metrics calculation."""
        # Add some historical metrics
        for i in range(30):
            metrics = RiskMetrics(
                value_at_risk=0.01 + i * 0.0001,
                conditional_var=0.015 + i * 0.0001,
                sharpe_ratio=1.0 + i * 0.01,
                sortino_ratio=1.2,
                calmar_ratio=0.8,
                maximum_drawdown=0.1,
                current_drawdown=0.05,
                downside_deviation=0.008,
                portfolio_volatility=0.15,
                portfolio_beta=1.0,
                correlation_risk=0.3,
            )
            risk_manager.update_historical_metrics(metrics)

        avg_metrics = risk_manager.get_average_metrics(lookback_days=20)

        assert isinstance(avg_metrics, dict)
        assert "avg_var" in avg_metrics
        assert "avg_sharpe" in avg_metrics
        assert "max_drawdown" in avg_metrics
        assert avg_metrics["avg_var"] > 0
        assert avg_metrics["avg_sharpe"] > 1.0

    def test_edge_cases(self, risk_manager):
        """Test edge cases and error handling."""
        # Empty returns
        empty_returns = pd.DataFrame()
        metrics = risk_manager.calculate_risk_metrics(empty_returns, 100000)
        assert metrics.value_at_risk == 0

        # Single position
        single_position = {"SPY": {"value": 100000, "quantity": 250, "sector": "broad"}}
        is_compliant, violations = risk_manager.check_risk_compliance(
            single_position, 100000, 0.05
        )
        assert not is_compliant  # Should violate concentration limit

        # Zero volatility
        zero_vol_returns = pd.DataFrame(
            {"SPY": [0.001] * 100}, index=pd.date_range("2023-01-01", periods=100)
        )
        metrics = risk_manager.calculate_risk_metrics(zero_vol_returns, 100000)
        assert metrics.portfolio_volatility > 0  # Should handle gracefully

    def test_risk_parity_allocation(self, risk_manager, sample_returns):
        """Test risk parity allocation."""
        # Calculate equal risk contribution weights
        weights = np.array([0.4, 0.3, 0.3])

        # With risk parity, each asset should contribute equally to risk
        optimal = risk_manager.portfolio_optimization(
            sample_returns, weights, risk_parity=True
        )

        # Verify weights sum to 1
        assert np.isclose(np.sum(optimal), 1.0)

        # Higher volatility assets should have lower weights
        vols = sample_returns.std()
        sorted_idx = np.argsort(vols.values)
        assert optimal.iloc[sorted_idx[0]] > optimal.iloc[sorted_idx[-1]]


class TestRiskMetrics:
    """Test suite for RiskMetrics dataclass."""

    def test_risk_metrics_creation(self):
        """Test RiskMetrics instance creation."""
        metrics = RiskMetrics(
            value_at_risk=0.02,
            conditional_var=0.03,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.2,
            maximum_drawdown=0.15,
            current_drawdown=0.08,
            downside_deviation=0.01,
            portfolio_volatility=0.16,
            portfolio_beta=0.95,
            correlation_risk=0.4,
        )

        assert metrics.value_at_risk == 0.02
        assert metrics.conditional_var == 0.03
        assert metrics.sharpe_ratio == 1.5
        assert metrics.portfolio_beta == 0.95
        assert metrics.correlation_risk == 0.4

    def test_risk_metrics_defaults(self):
        """Test RiskMetrics default values."""
        metrics = RiskMetrics(
            value_at_risk=0.02,
            conditional_var=0.03,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.2,
            maximum_drawdown=0.15,
            current_drawdown=0.08,
            downside_deviation=0.01,
            portfolio_volatility=0.16,
        )

        assert metrics.portfolio_beta == 1.0
        assert metrics.correlation_risk == 0.0


class TestIntegration:
    """Integration tests for risk management system."""

    def test_risk_manager_with_live_signals(self, risk_manager):
        """Test risk manager with realistic trading signals."""
        # Create multiple signals
        signals = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        for i, symbol in enumerate(symbols):
            signal = Signal(
                timestamp=datetime.now(),
                symbol=symbol,
                direction="LONG" if i % 2 == 0 else "SHORT",
                strength=(0.5 + i * 0.1) * (1 if i % 2 == 0 else -1),
                strategy_id="momentum",
                price=100 + i * 50,
                atr=2 + i * 0.5,
            )
            signals.append(signal)

        risk_context = RiskContext(
            account_equity=1000000,
            open_positions=5,
            daily_pnl=5000,
            max_drawdown_pct=0.03,
            volatility_target=0.12,
            max_position_size=0.15,
        )

        # Create mock portfolio and market data
        from unittest.mock import Mock
        portfolio = Mock()
        portfolio.current_equity = risk_context.account_equity

        # Create empty market data
        market_data = {}

        # Get risk-adjusted sizes
        adjusted_sizes = risk_manager.get_risk_adjusted_sizes(signals, portfolio, market_data)

        # Verify constraints
        assert len(adjusted_sizes) == len([s for s in signals if s.direction != "FLAT"])

        for _, size in adjusted_sizes.items():
            assert size >= 0
            assert size <= risk_manager.risk_limits["max_position_size"]

    def test_full_risk_cycle(self, risk_manager, sample_returns, sample_positions):
        """Test complete risk management cycle."""
        portfolio_value = 100000

        # 1. Calculate initial metrics
        metrics = risk_manager.calculate_risk_metrics(sample_returns, portfolio_value)

        # 2. Update historical metrics
        risk_manager.update_historical_metrics(metrics)

        # 3. Check compliance
        is_compliant, violations = risk_manager.check_risk_compliance(
            sample_positions, portfolio_value, metrics.current_drawdown
        )

        # 4. Update regime
        risk_manager.update_regime(sample_returns)

        # 5. Calculate stress scenarios
        scenarios = risk_manager.calculate_stress_scenarios(
            sample_positions, sample_returns.corr()
        )

        # 6. Get average metrics
        avg_metrics = risk_manager.get_average_metrics()

        # Verify full cycle completed successfully
        assert metrics is not None
        assert isinstance(is_compliant, bool)
        assert risk_manager.current_regime in ["NORMAL", "HIGH_VOL", "RISK_OFF"]
        assert len(scenarios) > 0
        assert isinstance(avg_metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
