#!/usr/bin/env python3
"""Comprehensive test suite for risk.py module - 100% coverage."""

from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from core.risk import EnhancedRiskManager, RiskManager, RiskMetrics
from strategies.base import Signal


class TestRiskMetrics:
    """Test suite for RiskMetrics dataclass."""

    def test_risk_metrics_creation(self):
        """Test RiskMetrics instance creation with all fields."""
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
        assert metrics.sortino_ratio == 1.8
        assert metrics.calmar_ratio == 1.2
        assert metrics.maximum_drawdown == 0.15
        assert metrics.current_drawdown == 0.08
        assert metrics.downside_deviation == 0.01
        assert metrics.portfolio_volatility == 0.16
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


class TestEnhancedRiskManager:
    """Test suite for EnhancedRiskManager class."""

    @pytest.fixture
    def risk_config(self) -> dict[str, Any]:
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
            "risk_free_rate": 0.02,
            "use_garch": False,
            "vol_lookback": 60,
            "risk_aversion": 2.0,
            "base_position_size": 0.02,
            "metrics_history_limit": 100,  # Coverage test expects 100
        }

    @pytest.fixture
    def risk_manager(self, risk_config: dict[str, Any]) -> EnhancedRiskManager:
        """Create a risk manager instance."""
        return EnhancedRiskManager(risk_config)

    @pytest.fixture
    def sample_returns(self) -> pd.Series:
        """Generate sample return data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.Series(
            np.random.normal(0.0005, 0.01, 252),
            index=dates,
        )
        return returns

    @pytest.fixture
    def sample_returns_df(self) -> pd.DataFrame:
        """Generate sample return data as DataFrame."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.DataFrame(
            {
                "SPY": np.random.normal(0.0005, 0.01, 252),
                "QQQ": np.random.normal(0.0007, 0.015, 252),
                "IWM": np.random.normal(0.0003, 0.012, 252),
            },
            index=dates,
        )
        return returns

    def test_initialization(self, risk_config):
        """Test risk manager initialization."""
        rm = EnhancedRiskManager(risk_config)

        assert rm.config == risk_config
        assert rm.risk_limits["max_var_95"] == 0.02
        assert rm.risk_limits["max_portfolio_volatility"] == 0.10
        assert rm.risk_limits["max_position_size"] == 0.20
        assert rm.risk_limits["max_sector_exposure"] == 0.40
        assert rm.risk_limits["max_drawdown"] == 0.15
        assert rm.risk_limits["max_correlation"] == 0.70
        assert rm.risk_limits["min_sharpe"] == 0.5
        assert rm.risk_limits["concentration_limit"] == 0.60

        assert isinstance(rm.returns_history, pd.Series)
        assert isinstance(rm.correlation_matrix, pd.DataFrame)
        assert rm.sector_exposures == {}
        assert rm.risk_metrics is None
        assert rm.is_risk_on is True
        assert rm.use_garch is False
        assert rm.vol_lookback == 60
        assert rm.historical_metrics == []
        assert rm.current_regime == "NORMAL"

    def test_calculate_risk_metrics_with_series(self, risk_manager, sample_returns):
        """Test risk metrics calculation with Series input."""
        metrics = risk_manager.calculate_risk_metrics(
            sample_returns, portfolio_value=100000
        )

        assert isinstance(metrics, RiskMetrics)
        assert 0 < metrics.value_at_risk < 0.05
        assert metrics.conditional_var >= metrics.value_at_risk
        assert -2 < metrics.sharpe_ratio < 3
        assert 0 < metrics.maximum_drawdown < 1
        assert metrics.portfolio_volatility > 0
        assert metrics.portfolio_beta == 1.0

    def test_calculate_risk_metrics_with_dataframe(
        self, risk_manager, sample_returns_df
    ):
        """Test risk metrics calculation with DataFrame input."""
        metrics = risk_manager.calculate_risk_metrics(
            sample_returns_df, portfolio_value=100000
        )

        assert isinstance(metrics, RiskMetrics)
        assert metrics.value_at_risk > 0
        assert metrics.conditional_var >= metrics.value_at_risk

    def test_calculate_risk_metrics_insufficient_data(self, risk_manager):
        """Test risk metrics with insufficient data."""
        short_returns = pd.Series([0.01, 0.02, -0.01, 0.0, 0.01])
        metrics = risk_manager.calculate_risk_metrics(short_returns)

        assert metrics.value_at_risk == 0.0
        assert metrics.conditional_var == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.portfolio_volatility == 0.15

    def test_calculate_risk_metrics_with_benchmark(
        self, risk_manager, sample_returns, sample_returns_df
    ):
        """Test risk metrics calculation with benchmark."""
        portfolio_returns = sample_returns
        benchmark_returns = sample_returns_df["SPY"]

        metrics = risk_manager.calculate_risk_metrics(
            portfolio_returns,
            portfolio_value=100000,
            benchmark_returns=benchmark_returns,
        )

        assert isinstance(metrics, RiskMetrics)
        assert metrics.portfolio_beta != 1.0  # Should be calculated vs benchmark

    def test_calculate_risk_metrics_edge_cases(self, risk_manager):
        """Test risk metrics edge cases."""
        # All positive returns - no downside
        positive_returns = pd.Series([0.01] * 100)
        metrics = risk_manager.calculate_risk_metrics(positive_returns)
        # With all positive returns, sortino should be high but might equal sharpe due to numerical precision
        assert metrics.sortino_ratio >= metrics.sharpe_ratio

        # Zero volatility
        zero_vol_returns = pd.Series([0.001] * 100)
        metrics = risk_manager.calculate_risk_metrics(zero_vol_returns)
        assert metrics.sharpe_ratio > 0

        # Maximum drawdown at end
        declining_returns = pd.Series(
            [0.01] * 50 + [-0.02] * 50, index=pd.date_range("2023-01-01", periods=100)
        )
        metrics = risk_manager.calculate_risk_metrics(declining_returns)
        assert metrics.current_drawdown > 0

    def test_default_risk_metrics(self, risk_manager):
        """Test default risk metrics generation."""
        metrics = risk_manager._default_risk_metrics()

        assert metrics.value_at_risk == 0.0
        assert metrics.conditional_var == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0
        assert metrics.maximum_drawdown == 0.0
        assert metrics.current_drawdown == 0.0
        assert metrics.downside_deviation == 0.15
        assert metrics.portfolio_volatility == 0.15
        assert metrics.portfolio_beta == 1.0

    def test_forecast_volatility_insufficient_data(self, risk_manager):
        """Test volatility forecast with insufficient data."""
        short_returns = pd.Series([0.01, 0.02, -0.01])
        vol = risk_manager.forecast_volatility(short_returns)

        assert vol > 0
        assert vol == short_returns.std() * np.sqrt(252)

    def test_forecast_volatility_rolling_window(self, risk_manager, sample_returns):
        """Test volatility forecast with rolling window (no GARCH)."""
        vol = risk_manager.forecast_volatility(sample_returns)

        assert isinstance(vol, float)
        assert vol > 0
        assert vol < 1.0  # Reasonable annualized volatility

    def test_forecast_volatility_with_garch(self, risk_manager, sample_returns):
        """Test volatility forecast with GARCH enabled."""
        risk_manager.use_garch = True
        vol = risk_manager.forecast_volatility(sample_returns)

        assert isinstance(vol, float)
        assert vol > 0
        assert vol < 1.0

    def test_calculate_position_var(self, risk_manager, sample_returns):
        """Test position VaR calculation."""
        position_value = 10000
        var = risk_manager.calculate_position_var(
            sample_returns, position_value, confidence_level=0.95
        )

        assert var > 0
        assert var < position_value * 0.1

    def test_portfolio_optimization_success(self, risk_manager, sample_returns_df):
        """Test successful portfolio optimization."""
        current_weights = pd.Series([0.4, 0.3, 0.3], index=sample_returns_df.columns)

        # Mock successful optimization
        with patch("scipy.optimize.minimize") as mock_minimize:
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array([0.5, 0.3, 0.2])
            mock_minimize.return_value = mock_result

            optimal_weights = risk_manager.portfolio_optimization(
                sample_returns_df, current_weights, target_return=0.10
            )

            assert len(optimal_weights) == 3
            assert abs(optimal_weights.sum() - 1.0) < 0.001
            assert optimal_weights.iloc[0] == 0.5
            assert optimal_weights.iloc[1] == 0.3
            assert optimal_weights.iloc[2] == 0.2

    def test_portfolio_optimization_empty_weights(self, risk_manager, sample_returns_df):
        """Test portfolio optimization with empty current weights."""
        current_weights = pd.Series(dtype=float)

        optimal_weights = risk_manager.portfolio_optimization(
            sample_returns_df, current_weights
        )

        assert len(optimal_weights) == 3
        assert abs(optimal_weights.sum() - 1.0) < 0.001

    @patch("scipy.optimize.minimize")
    def test_portfolio_optimization_failure(self, mock_minimize, risk_manager, sample_returns_df):
        """Test portfolio optimization when optimization fails."""
        mock_result = Mock()
        mock_result.success = False
        mock_minimize.return_value = mock_result

        current_weights = pd.Series([0.4, 0.3, 0.3], index=sample_returns_df.columns)

        with patch("core.risk.logger") as mock_logger:
            optimal_weights = risk_manager.portfolio_optimization(
                sample_returns_df, current_weights
            )

            mock_logger.warning.assert_called_once()
            assert all(w == 1.0 / 3 for w in optimal_weights.values)

    def test_check_risk_compliance_no_portfolio(self, risk_manager):
        """Test risk compliance check without portfolio."""
        is_compliant, violations = risk_manager.check_risk_compliance()

        assert is_compliant is True
        assert violations == []

    def test_check_risk_compliance_with_portfolio(self, risk_manager, sample_returns):
        """Test risk compliance check with portfolio."""
        # Create mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.get_returns_history.return_value = sample_returns

        risk_manager.returns_history = sample_returns

        is_compliant, violations = risk_manager.check_risk_compliance(
            portfolio=mock_portfolio
        )

        assert isinstance(is_compliant, bool)
        assert isinstance(violations, list)

    def test_check_risk_compliance_violations(self, risk_manager):
        """Test risk compliance with various violations."""
        # Set up risk metrics that violate limits
        risk_manager.risk_metrics = RiskMetrics(
            value_at_risk=0.03,  # Exceeds 2% limit
            conditional_var=0.04,
            sharpe_ratio=0.3,  # Below 0.5 minimum
            sortino_ratio=0.4,
            calmar_ratio=0.2,
            maximum_drawdown=0.10,
            current_drawdown=0.20,  # Exceeds 15% limit
            downside_deviation=0.02,
            portfolio_volatility=0.20,  # Exceeds 10% limit
            portfolio_beta=1.0,
        )

        is_compliant, violations = risk_manager.check_risk_compliance()

        assert is_compliant is False
        assert len(violations) == 4
        assert any("VaR" in v for v in violations)
        assert any("volatility" in v for v in violations)
        assert any("Drawdown" in v for v in violations)
        assert any("Sharpe" in v for v in violations)

    def test_check_risk_compliance_with_proposed_trade(self, risk_manager):
        """Test risk compliance with proposed trade."""
        # Set up existing risk metrics
        risk_manager.risk_metrics = RiskMetrics(
            value_at_risk=0.015,  # Just under limit
            conditional_var=0.02,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.8,
            maximum_drawdown=0.10,
            current_drawdown=0.05,
            downside_deviation=0.01,
            portfolio_volatility=0.08,
            portfolio_beta=1.0,
        )

        # Create a proposed trade
        proposed_trade = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=150.0,
        )

        is_compliant, violations = risk_manager.check_risk_compliance(
            portfolio_value=100000, proposed_trade=proposed_trade
        )

        assert isinstance(is_compliant, bool)
        assert isinstance(violations, list)

    def test_check_risk_compliance_proposed_trade_breach(self, risk_manager):
        """Test risk compliance when proposed trade would breach limits."""
        # Set risk metrics at limit
        risk_manager.risk_metrics = RiskMetrics(
            value_at_risk=0.019,  # Close to 2% limit
            conditional_var=0.025,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.8,
            maximum_drawdown=0.10,
            current_drawdown=0.05,
            downside_deviation=0.01,
            portfolio_volatility=0.08,
            portfolio_beta=1.0,
        )

        # Add some return history for volatility calculation
        risk_manager.returns_history = pd.Series(np.random.normal(0, 0.01, 100))

        proposed_trade = Signal(
            timestamp=datetime.now(),
            symbol="TSLA",
            direction="LONG",
            strength=1.0,
            strategy_id="test",
            price=200.0,
        )

        is_compliant, violations = risk_manager.check_risk_compliance(
            portfolio_value=100000, proposed_trade=proposed_trade
        )

        assert is_compliant is False
        assert len(violations) > 0
        assert any("Proposed trade" in v for v in violations)

    def test_calculate_stress_scenarios_default(self, risk_manager):
        """Test stress scenarios with default scenarios."""
        mock_portfolio = Mock()
        mock_portfolio.positions = {
            "AAPL": Mock(direction="LONG", market_value=50000),
            "TSLA": Mock(direction="SHORT", market_value=30000),
        }
        mock_portfolio.current_equity = 100000

        stress_results = risk_manager.calculate_stress_scenarios(mock_portfolio)

        assert "market_crash" in stress_results
        assert "flash_crash" in stress_results
        assert "correlation_breakdown" in stress_results
        assert "liquidity_crisis" in stress_results

        # Market crash should show negative for long positions
        assert stress_results["market_crash"] < 0

    def test_calculate_stress_scenarios_custom(self, risk_manager):
        """Test stress scenarios with custom scenarios."""
        mock_portfolio = Mock()
        mock_portfolio.positions = {
            "AAPL": Mock(direction="LONG", market_value=50000),
        }
        mock_portfolio.current_equity = 100000

        custom_scenarios = {
            "tech_crash": {"equity": -0.30, "volatility": 4.0, "spread": 0.10},
            "rate_shock": {"equity": -0.05},
        }

        stress_results = risk_manager.calculate_stress_scenarios(
            mock_portfolio, custom_scenarios
        )

        assert "tech_crash" in stress_results
        assert "rate_shock" in stress_results
        assert stress_results["tech_crash"] < stress_results["rate_shock"]

    def test_get_risk_adjusted_sizes_flat_signals(self, risk_manager):
        """Test risk-adjusted sizing with flat signals."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="FLAT",
                strength=0.0,
                strategy_id="test",
                price=150.0,
            )
        ]

        mock_portfolio = Mock()
        market_data = {}

        sizes = risk_manager.get_risk_adjusted_sizes(signals, mock_portfolio, market_data)

        assert sizes == {}

    def test_get_risk_adjusted_sizes_with_market_data(self, risk_manager):
        """Test risk-adjusted sizing with market data."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=0.8,
                strategy_id="test",
                price=150.0,
            )
        ]

        mock_portfolio = Mock()
        market_data = {
            "AAPL": pd.DataFrame(
                {"returns": np.random.normal(0, 0.02, 100)},
                index=pd.date_range("2023-01-01", periods=100),
            )
        }

        sizes = risk_manager.get_risk_adjusted_sizes(signals, mock_portfolio, market_data)

        assert "AAPL" in sizes
        assert 0 < sizes["AAPL"] <= risk_manager.risk_limits["max_position_size"]

    def test_get_risk_adjusted_sizes_no_market_data(self, risk_manager):
        """Test risk-adjusted sizing without market data."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=1.0,
                strategy_id="test",
                price=150.0,
            )
        ]

        mock_portfolio = Mock()
        market_data = {}

        sizes = risk_manager.get_risk_adjusted_sizes(signals, mock_portfolio, market_data)

        assert "AAPL" in sizes
        assert sizes["AAPL"] > 0

    def test_get_risk_adjusted_sizes_with_correlation(self, risk_manager):
        """Test risk-adjusted sizing with correlation adjustment."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=1.0,
                strategy_id="test",
                price=150.0,
            )
        ]

        # Set up correlation matrix
        risk_manager.correlation_matrix = pd.DataFrame(
            {"AAPL": [1.0, 0.8], "MSFT": [0.8, 1.0]}, index=["AAPL", "MSFT"]
        )

        mock_portfolio = Mock()
        market_data = {}

        sizes = risk_manager.get_risk_adjusted_sizes(signals, mock_portfolio, market_data)

        assert "AAPL" in sizes
        # High correlation should reduce size
        assert sizes["AAPL"] <= risk_manager.risk_limits["max_position_size"]

    def test_get_risk_adjusted_sizes_risk_off(self, risk_manager):
        """Test risk-adjusted sizing in risk-off mode."""
        risk_manager.is_risk_on = False

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction="LONG",
                strength=1.0,
                strategy_id="test",
                price=150.0,
            )
        ]

        mock_portfolio = Mock()
        market_data = {}

        sizes = risk_manager.get_risk_adjusted_sizes(signals, mock_portfolio, market_data)

        assert "AAPL" in sizes
        # Risk-off should halve the size
        assert sizes["AAPL"] <= risk_manager.risk_limits["max_position_size"] * 0.5

    def test_update_risk_state(self, risk_manager, sample_returns):
        """Test risk state update."""
        mock_portfolio = Mock()
        mock_portfolio.get_returns_history.return_value = sample_returns

        # Test normal state
        risk_manager.update_risk_state(mock_portfolio)
        assert risk_manager.risk_metrics is not None
        assert risk_manager.is_risk_on is True

        # Test risk-off trigger (high drawdown)
        # We need to mock the calculate_risk_metrics to return high drawdown metrics
        high_dd_metrics = RiskMetrics(
            value_at_risk=0.02,
            conditional_var=0.03,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.8,
            maximum_drawdown=0.20,
            current_drawdown=0.13,  # 13% > 12% (80% of 15% limit)
            downside_deviation=0.01,
            portfolio_volatility=0.15,
        )

        with patch.object(risk_manager, 'calculate_risk_metrics', return_value=high_dd_metrics):
            with patch("core.risk.logger") as mock_logger:
                risk_manager.update_risk_state(mock_portfolio)
                # Check if warning was logged
                mock_logger.warning.assert_called_once()
                # Risk state should be off after this update
                assert risk_manager.is_risk_on is False

        # Test risk-on restoration (low drawdown)
        low_dd_metrics = RiskMetrics(
            value_at_risk=0.02,
            conditional_var=0.03,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.8,
            maximum_drawdown=0.20,
            current_drawdown=0.06,  # 6% < 7.5% (50% of 15% limit)
            downside_deviation=0.01,
            portfolio_volatility=0.15,
        )

        with patch.object(risk_manager, 'calculate_risk_metrics', return_value=low_dd_metrics):
            risk_manager.update_risk_state(mock_portfolio)
            assert risk_manager.is_risk_on is True

    def test_get_risk_report(self, risk_manager):
        """Test risk report generation."""
        # Test without metrics
        report = risk_manager.get_risk_report()
        assert report["timestamp"] is not None
        assert report["risk_state"] == "RISK_ON"
        assert report["metrics"] is None
        assert report["limits"] == risk_manager.risk_limits
        assert report["violations"] == []

        # Test with metrics and violations
        risk_manager.risk_metrics = RiskMetrics(
            value_at_risk=0.025,  # Exceeds limit
            conditional_var=0.03,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.8,
            maximum_drawdown=0.20,
            current_drawdown=0.18,  # Exceeds limit
            downside_deviation=0.01,
            portfolio_volatility=0.12,  # Exceeds limit
        )
        risk_manager.is_risk_on = False

        report = risk_manager.get_risk_report()
        assert report["risk_state"] == "RISK_OFF"
        assert report["metrics"] is not None
        assert len(report["violations"]) == 3
        assert "VaR limit exceeded" in report["violations"]
        assert "Drawdown limit exceeded" in report["violations"]
        assert "Volatility limit exceeded" in report["violations"]

    def test_update_regime_series(self, risk_manager, sample_returns):
        """Test regime update with Series input."""
        # Normal volatility
        risk_manager.update_regime(sample_returns)
        assert risk_manager.current_regime == "NORMAL"

        # High volatility
        high_vol_returns = sample_returns * 3
        risk_manager.update_regime(high_vol_returns)
        assert risk_manager.current_regime == "HIGH_VOL"

        # Low volatility
        low_vol_returns = sample_returns * 0.2
        risk_manager.update_regime(low_vol_returns)
        assert risk_manager.current_regime == "LOW_VOL"

    def test_update_regime_dataframe(self, risk_manager, sample_returns_df):
        """Test regime update with DataFrame input."""
        risk_manager.update_regime(sample_returns_df)
        assert risk_manager.current_regime in ["NORMAL", "HIGH_VOL", "LOW_VOL"]

    def test_update_historical_metrics(self, risk_manager):
        """Test historical metrics storage."""
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

        risk_manager.update_historical_metrics(metrics)
        assert len(risk_manager.historical_metrics) == 1
        assert risk_manager.historical_metrics[0] == metrics

        # Test limit enforcement
        for _i in range(150):
            risk_manager.update_historical_metrics(metrics)

        assert len(risk_manager.historical_metrics) == 100

    def test_get_average_metrics(self, risk_manager):
        """Test average metrics calculation."""
        # Test with insufficient data
        assert risk_manager.get_average_metrics(30) is None

        # Add metrics
        for i in range(50):
            metrics = RiskMetrics(
                value_at_risk=0.02,
                conditional_var=0.03,
                sharpe_ratio=1.5 + i * 0.01,
                sortino_ratio=1.8,
                calmar_ratio=1.2,
                maximum_drawdown=0.15,
                current_drawdown=0.08,
                downside_deviation=0.01,
                portfolio_volatility=0.16,
            )
            risk_manager.update_historical_metrics(metrics)

        avg_metrics = risk_manager.get_average_metrics(30)
        assert avg_metrics is not None
        assert isinstance(avg_metrics, dict)
        assert "avg_var" in avg_metrics
        assert "avg_sharpe" in avg_metrics

    def test_can_trade(self, risk_manager):
        """Test trading permission checks."""
        # Test risk-off state
        risk_manager.is_risk_on = False
        assert risk_manager.can_trade(Mock()) is False

        # Test with excessive drawdown
        risk_manager.is_risk_on = True
        risk_manager.risk_metrics = RiskMetrics(
            value_at_risk=0.02,
            conditional_var=0.03,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.2,
            maximum_drawdown=0.20,
            current_drawdown=0.18,  # Exceeds 15% limit
            downside_deviation=0.01,
            portfolio_volatility=0.08,
        )

        with patch("core.risk.logger") as mock_logger:
            assert risk_manager.can_trade(Mock()) is False
            mock_logger.warning.assert_called()

        # Test with excessive volatility
        risk_manager.risk_metrics.current_drawdown = 0.05
        risk_manager.risk_metrics.portfolio_volatility = 0.15  # Exceeds 10% limit

        with patch("core.risk.logger") as mock_logger:
            assert risk_manager.can_trade(Mock()) is False
            mock_logger.warning.assert_called()

        # Test normal conditions
        risk_manager.risk_metrics.portfolio_volatility = 0.08
        assert risk_manager.can_trade(Mock()) is True

    def test_size_orders(self, risk_manager):
        """Test order sizing."""
        signals = {
            "strategy1": {
                "AAPL": Mock(strength=0.8, direction="LONG"),
                "MSFT": Mock(strength=0.6, direction="SHORT"),
            },
            "strategy2": {},  # Empty signals
        }

        mock_portfolio = Mock()
        mock_portfolio.get_position_weight.return_value = 0.05

        sized_orders = risk_manager.size_orders(signals, mock_portfolio)

        assert "strategy1" in sized_orders
        assert "AAPL" in sized_orders["strategy1"]
        assert "MSFT" in sized_orders["strategy1"]
        assert "strategy2" not in sized_orders  # Empty strategies excluded

    def test_calculate_position_size(self, risk_manager):
        """Test position size calculation."""
        signal = Mock(strength=0.5)
        portfolio = Mock()

        size = risk_manager._calculate_position_size(signal, portfolio)
        assert size == 0.02 * 0.5  # base_size * strength

        # Test without strength attribute
        signal_no_strength = Mock(spec=[])
        size = risk_manager._calculate_position_size(signal_no_strength, portfolio)
        assert size == 0.02

    def test_apply_risk_limits(self, risk_manager):
        """Test risk limit application."""
        # Test basic limit without get_position_weight method
        mock_portfolio = Mock(spec=[])  # No get_position_weight method
        size = risk_manager._apply_risk_limits(0.25, "AAPL", mock_portfolio)
        assert size == 0.20  # Capped at max_position_size

        # Test concentration limit
        mock_portfolio = Mock()
        mock_portfolio.get_position_weight = Mock(return_value=0.15)

        size = risk_manager._apply_risk_limits(0.10, "AAPL", mock_portfolio)
        assert abs(size - 0.05) < 0.0001  # 0.20 - 0.15 = 0.05

        # Test when already at limit
        mock_portfolio.get_position_weight = Mock(return_value=0.20)
        size = risk_manager._apply_risk_limits(0.10, "AAPL", mock_portfolio)
        assert size == 0  # Already at max

        # Test when position would exceed limit (line 598)
        mock_portfolio.get_position_weight = Mock(return_value=0.25)  # Already over limit
        size = risk_manager._apply_risk_limits(0.10, "AAPL", mock_portfolio)
        assert size == 0  # Can't add more when already over limit

    def test_portfolio_optimization_risk_parity(self, risk_manager, sample_returns_df):
        """Test portfolio optimization with risk parity flag."""
        current_weights = pd.Series([0.4, 0.3, 0.3], index=sample_returns_df.columns)

        # Test with risk_parity=True
        optimal_weights = risk_manager.portfolio_optimization(
            sample_returns_df, current_weights, risk_parity=True
        )

        assert len(optimal_weights) == 3
        assert abs(optimal_weights.sum() - 1.0) < 0.001

    def test_trigger_kill_switch(self, risk_manager):
        """Test emergency kill switch."""
        with patch("core.risk.logger") as mock_logger:
            risk_manager.trigger_kill_switch()
            assert risk_manager.is_risk_on is False
            mock_logger.critical.assert_called_once()

        # Test the full emergency stop behavior
        risk_manager.is_risk_on = True  # Reset
        risk_manager.trigger_kill_switch()
        assert risk_manager.is_risk_on is False


class TestRiskManagerAlias:
    """Test RiskManager alias for backward compatibility."""

    def test_risk_manager_alias(self):
        """Test that RiskManager is an alias for EnhancedRiskManager."""
        assert RiskManager is EnhancedRiskManager

        # Test instantiation
        config = {"max_var_95": 0.02}
        rm = RiskManager(config)
        assert isinstance(rm, EnhancedRiskManager)


class TestModulePlaceholders:
    """Test module-level placeholder types."""

    def test_placeholder_types(self):
        """Test that placeholder types exist."""
        from core.risk import (
            PortfolioRisk,
            PositionRisk,
            RiskAlert,
            RiskLimits,
            RiskViolation,
        )

        # Check types - RiskLimits is a dataclass, others are dict aliases or dataclasses
        from dataclasses import is_dataclass
        assert is_dataclass(RiskLimits)  # RiskLimits is a dataclass
        assert PositionRisk is dict  # Dict alias
        assert PortfolioRisk is dict  # Dict alias
        assert is_dataclass(RiskAlert)  # RiskAlert is a dataclass
        assert is_dataclass(RiskViolation)  # RiskViolation is a dataclass


class TestMissingCoverage:
    """Additional tests to achieve 100% coverage."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with standard config."""
        config = {
            "max_var_95": 0.02,
            "target_vol": 0.10,
            "max_position_size": 0.20,
            "max_drawdown": 0.15,
            "min_sharpe": 0.5,
            "base_position_size": 0.02,
        }
        return EnhancedRiskManager(config)

    def test_get_risk_adjusted_sizes_empty_market_data_with_correlation(self, risk_manager):
        """Test risk-adjusted sizing when symbol not in correlation matrix."""
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="NEW_SYMBOL",
                direction="LONG",
                strength=1.0,
                strategy_id="test",
                price=150.0,
            )
        ]

        # Set up correlation matrix without NEW_SYMBOL
        risk_manager.correlation_matrix = pd.DataFrame(
            {"AAPL": [1.0, 0.8], "MSFT": [0.8, 1.0]}, index=["AAPL", "MSFT"]
        )

        mock_portfolio = Mock()
        market_data = {}

        sizes = risk_manager.get_risk_adjusted_sizes(signals, mock_portfolio, market_data)

        assert "NEW_SYMBOL" in sizes
        assert sizes["NEW_SYMBOL"] > 0

    def test_portfolio_optimization_with_risk_parity_flag(self, risk_manager):
        """Test that risk_parity parameter is handled but not implemented."""
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            columns=['A', 'B', 'C']
        )
        current_weights = pd.Series([0.4, 0.3, 0.3], index=['A', 'B', 'C'])

        # This should work even though risk_parity is ignored
        weights = risk_manager.portfolio_optimization(
            returns, current_weights, risk_parity=True
        )

        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.risk", "--cov-report=term-missing"])
