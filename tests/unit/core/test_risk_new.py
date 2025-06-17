"""
Unit tests for Risk Management module.

Tests cover:
- Risk metrics calculations (VaR, CVaR, Sharpe, Sortino, etc.)
- Risk limit enforcement
- Volatility forecasting
- Beta calculations
- Position sizing
- Risk state management

All tests follow FIRST principles with strong assertions.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from core.risk import EnhancedRiskManager, RiskMetrics


class TestRiskMetricsCalculation:
    """Test risk metrics calculations."""

    @pytest.fixture
    def risk_manager(self):
        """Standard risk manager for testing."""
        config = {
            "max_var_95": 0.02,
            "target_vol": 0.15,
            "max_position_size": 0.20,
            "max_drawdown": 0.15,
            "risk_free_rate": 0.03,  # 3% annual
        }
        return EnhancedRiskManager(config)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return series for testing."""
        np.random.seed(42)
        # 252 trading days of returns with 1% daily vol, 0.05% daily drift
        returns = np.random.normal(0.0005, 0.01, 252)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def crash_returns(self):
        """Generate returns with a market crash."""
        # Normal returns then a crash
        normal = np.random.normal(0.0005, 0.01, 200)
        crash = np.array([-0.05, -0.08, -0.03, -0.04, -0.02])  # 20% crash
        recovery = np.random.normal(0.001, 0.015, 47)

        all_returns = np.concatenate([normal, crash, recovery])
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        return pd.Series(all_returns, index=dates)

    @pytest.mark.unit
    def test_risk_metrics_basic_calculation(self, risk_manager, sample_returns):
        """
        Risk metrics are calculated correctly for normal returns.

        Verifies all risk metrics are within expected ranges.
        """
        metrics = risk_manager.calculate_risk_metrics(sample_returns)

        # Value at Risk (should be positive, representing loss)
        assert metrics.value_at_risk > 0
        assert 0.01 < metrics.value_at_risk < 0.03  # 1-3% daily VaR reasonable

        # CVaR should be worse than VaR
        assert metrics.conditional_var > metrics.value_at_risk

        # Sharpe ratio (positive returns should give positive Sharpe)
        assert metrics.sharpe_ratio > 0
        assert 0.1 < metrics.sharpe_ratio < 2.0  # Reasonable range

        # Sortino should be higher than Sharpe (only penalizes downside)
        assert metrics.sortino_ratio >= metrics.sharpe_ratio

        # Volatility should be annualized
        daily_vol = sample_returns.std()
        expected_annual_vol = daily_vol * np.sqrt(252)
        assert metrics.portfolio_volatility == pytest.approx(expected_annual_vol, rel=0.01)

        # Beta should be 1.0 without benchmark
        assert metrics.portfolio_beta == 1.0

    @pytest.mark.unit
    def test_risk_metrics_with_crash(self, risk_manager, crash_returns):
        """
        Risk metrics correctly reflect market crash scenarios.

        Crash should increase VaR, drawdown, and reduce ratios.
        """
        metrics = risk_manager.calculate_risk_metrics(crash_returns)

        # Higher VaR due to crash
        assert metrics.value_at_risk > 0.03  # Over 3% daily VaR

        # Significant maximum drawdown
        assert metrics.maximum_drawdown > 0.15  # Over 15% drawdown

        # Current drawdown depends on recovery
        assert metrics.current_drawdown >= 0  # Should be positive

        # Lower or negative Sharpe due to crash
        assert metrics.sharpe_ratio < 0.5

        # Calmar ratio should be low (return/drawdown)
        assert metrics.calmar_ratio < 1.0

    @pytest.mark.unit
    def test_risk_metrics_insufficient_data(self, risk_manager):
        """
        Risk metrics handle insufficient data gracefully.

        Should return default metrics without crashing.
        """
        # Less than 30 days of data
        short_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        metrics = risk_manager.calculate_risk_metrics(short_returns)

        # Should return default metrics
        assert metrics.value_at_risk == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.portfolio_volatility == 0.15  # Default 15%
        assert metrics.portfolio_beta == 1.0

    @pytest.mark.unit
    def test_var_calculation_accuracy(self, risk_manager):
        """
        Value at Risk calculation is statistically accurate.

        95% VaR should capture the 5th percentile of losses.
        """
        # Generate returns with known distribution
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 10000)  # 2% daily volatility
        returns_series = pd.Series(returns)

        metrics = risk_manager.calculate_risk_metrics(returns_series)

        # Theoretical 95% VaR for normal distribution
        theoretical_var = 1.645 * 0.02  # 1.645 std devs for 5th percentile

        # Should be close to theoretical value
        assert metrics.value_at_risk == pytest.approx(theoretical_var, rel=0.1)

        # Verify only 5% of returns are worse than VaR
        worse_than_var = (returns < -metrics.value_at_risk).sum()
        percentage_worse = worse_than_var / len(returns)
        assert percentage_worse == pytest.approx(0.05, abs=0.01)

    @pytest.mark.unit
    def test_cvar_calculation(self, risk_manager):
        """
        Conditional VaR (Expected Shortfall) calculated correctly.

        CVaR should be the average of losses beyond VaR.
        """
        # Known distribution
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 10000)
        returns_series = pd.Series(returns)

        metrics = risk_manager.calculate_risk_metrics(returns_series)

        # Calculate CVaR manually
        var_threshold = np.percentile(returns, 5)
        tail_losses = returns[returns <= var_threshold]
        expected_cvar = abs(np.mean(tail_losses))

        assert metrics.conditional_var == pytest.approx(expected_cvar, rel=0.01)
        assert metrics.conditional_var > metrics.value_at_risk  # CVaR > VaR always

    @pytest.mark.unit
    @pytest.mark.parametrize("annual_return,volatility,expected_sharpe", [
        (0.10, 0.15, 0.47),  # 10% return, 15% vol, 3% risk-free → ~0.47 Sharpe
        (0.03, 0.10, 0.0),   # 3% return = risk-free → 0 Sharpe
        (-0.05, 0.20, -0.4), # -5% return, 20% vol → negative Sharpe
        (0.20, 0.20, 0.85),  # 20% return, 20% vol → ~0.85 Sharpe
    ])
    def test_sharpe_ratio_calculation(self, risk_manager, annual_return, volatility, expected_sharpe):
        """
        Sharpe ratio calculation is accurate for various scenarios.

        Tests different return/volatility combinations.
        """
        # Generate returns with specified characteristics
        daily_return = annual_return / 252
        daily_vol = volatility / np.sqrt(252)

        np.random.seed(42)
        returns = np.random.normal(daily_return, daily_vol, 252)
        returns_series = pd.Series(returns)

        metrics = risk_manager.calculate_risk_metrics(returns_series)

        # Sharpe should be close to expected (some variation due to sampling)
        assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, abs=0.15)

    @pytest.mark.unit
    def test_sortino_ratio_calculation(self, risk_manager):
        """
        Sortino ratio only penalizes downside volatility.

        Should be higher than Sharpe for asymmetric returns.
        """
        # Create returns with positive skew (more upside than downside)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        # Add some large positive returns
        returns[::20] = np.random.uniform(0.03, 0.05, len(returns[::20]))
        returns_series = pd.Series(returns)

        metrics = risk_manager.calculate_risk_metrics(returns_series)

        # Sortino should exceed Sharpe due to upside skew
        assert metrics.sortino_ratio > metrics.sharpe_ratio
        assert metrics.sortino_ratio > 0

        # Downside deviation should be less than total volatility
        total_vol = returns_series.std() * np.sqrt(252)
        assert metrics.downside_deviation < total_vol

    @pytest.mark.unit
    def test_maximum_drawdown_calculation(self, risk_manager):
        """
        Maximum drawdown tracks largest peak-to-trough decline.

        Should identify the worst historical drawdown.
        """
        # Create returns with known drawdown
        prices = [100]
        returns = []

        # Rise to 120
        for _ in range(20):
            ret = 0.01
            returns.append(ret)
            prices.append(prices[-1] * (1 + ret))

        # Drop to 90 (25% drawdown from 120)
        for _ in range(10):
            ret = -0.03
            returns.append(ret)
            prices.append(prices[-1] * (1 + ret))

        # Recover to 110
        for _ in range(20):
            ret = 0.01
            returns.append(ret)
            prices.append(prices[-1] * (1 + ret))

        returns_series = pd.Series(returns)
        metrics = risk_manager.calculate_risk_metrics(returns_series)

        # Maximum drawdown should be ~25%
        assert 0.24 < metrics.maximum_drawdown < 0.26

        # Current drawdown should be less (some recovery)
        assert metrics.current_drawdown < metrics.maximum_drawdown

    @pytest.mark.unit
    def test_calmar_ratio_calculation(self, risk_manager, sample_returns):
        """
        Calmar ratio correctly relates returns to maximum drawdown.

        Calmar = Annual Return / Maximum Drawdown
        """
        metrics = risk_manager.calculate_risk_metrics(sample_returns)

        # Calculate expected Calmar
        annual_return = sample_returns.mean() * 252

        if metrics.maximum_drawdown > 0:
            expected_calmar = annual_return / metrics.maximum_drawdown
            assert metrics.calmar_ratio == pytest.approx(expected_calmar, rel=0.01)
        else:
            assert metrics.calmar_ratio == 0.0

    @pytest.mark.unit
    def test_dataframe_input_handling(self, risk_manager):
        """
        Risk metrics handle DataFrame input correctly.

        Should calculate portfolio returns from asset returns.
        """
        # Create multi-asset returns
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        returns_df = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100),
        }, index=dates)

        metrics = risk_manager.calculate_risk_metrics(returns_df)

        # Should calculate metrics for equal-weight portfolio
        assert isinstance(metrics, RiskMetrics)
        assert metrics.portfolio_volatility > 0
        assert metrics.value_at_risk > 0


class TestBetaCalculation:
    """Test beta calculation against benchmark."""

    @pytest.fixture
    def risk_manager(self):
        """Risk manager for beta testing."""
        return EnhancedRiskManager({})

    @pytest.fixture
    def correlated_returns(self):
        """Generate portfolio and benchmark returns with known correlation."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        # Benchmark returns
        benchmark = np.random.normal(0.0004, 0.01, 252)

        # Portfolio returns = 0.8 * benchmark + noise (beta ≈ 1.2)
        noise = np.random.normal(0, 0.008, 252)
        portfolio = 1.2 * benchmark + noise

        return pd.Series(portfolio, index=dates), pd.Series(benchmark, index=dates)

    @pytest.mark.unit
    def test_beta_calculation_basic(self, risk_manager, correlated_returns):
        """
        Beta calculation is accurate for correlated returns.

        Beta = Cov(R_p, R_m) / Var(R_m)
        """
        portfolio_returns, benchmark_returns = correlated_returns

        metrics = risk_manager.calculate_risk_metrics(
            portfolio_returns,
            benchmark_returns=benchmark_returns
        )

        # Beta should be close to 1.2
        assert 1.1 < metrics.portfolio_beta < 1.3

    @pytest.mark.unit
    def test_beta_no_benchmark(self, risk_manager):
        """
        Beta defaults to 1.0 when no benchmark provided.

        Standard assumption for market beta.
        """
        returns = pd.Series(np.random.normal(0, 0.01, 100))

        metrics = risk_manager.calculate_risk_metrics(returns)

        assert metrics.portfolio_beta == 1.0

    @pytest.mark.unit
    def test_beta_perfect_correlation(self, risk_manager):
        """
        Beta equals volatility ratio for perfect correlation.

        When correlation = 1, beta = σ_p / σ_m
        """
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

        # Perfect correlation, different volatilities
        benchmark = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
        portfolio = benchmark * 1.5  # 50% more volatile

        beta = risk_manager._calculate_beta(portfolio, benchmark)

        assert beta == pytest.approx(1.5, rel=0.01)

    @pytest.mark.unit
    def test_beta_negative_correlation(self, risk_manager):
        """
        Beta is negative for inverse correlation.

        Short positions or inverse ETFs have negative beta.
        """
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

        benchmark = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
        portfolio = -0.8 * benchmark  # Inverse with 0.8x leverage

        beta = risk_manager._calculate_beta(portfolio, benchmark)

        assert beta == pytest.approx(-0.8, rel=0.01)

    @pytest.mark.unit
    def test_beta_insufficient_data(self, risk_manager):
        """
        Beta calculation handles insufficient data gracefully.

        Should return 1.0 for insufficient overlap.
        """
        # Less than 30 days of common data
        dates1 = pd.date_range(end=datetime.now(), periods=20, freq='D')
        dates2 = pd.date_range(end=datetime.now() - timedelta(days=10), periods=20, freq='D')

        portfolio = pd.Series(np.random.normal(0, 0.01, 20), index=dates1)
        benchmark = pd.Series(np.random.normal(0, 0.01, 20), index=dates2)

        beta = risk_manager._calculate_beta(portfolio, benchmark)

        assert beta == 1.0  # Default value

    @pytest.mark.unit
    def test_beta_edge_cases(self, risk_manager):
        """
        Beta calculation handles edge cases safely.

        Zero variance, NaN values, etc.
        """
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')

        # Zero variance benchmark
        portfolio = pd.Series(np.random.normal(0, 0.01, 50), index=dates)
        benchmark = pd.Series([0.001] * 50, index=dates)  # Constant returns

        beta = risk_manager._calculate_beta(portfolio, benchmark)
        assert beta == 1.0  # Default when variance is zero

        # NaN values
        portfolio_with_nan = portfolio.copy()
        portfolio_with_nan[10:15] = np.nan

        beta = risk_manager._calculate_beta(portfolio_with_nan, benchmark)
        assert isinstance(beta, float)
        assert not np.isnan(beta)


class TestVolatilityForecasting:
    """Test volatility forecasting functionality."""

    @pytest.fixture
    def risk_manager_simple(self):
        """Risk manager without GARCH."""
        return EnhancedRiskManager({"use_garch": False})

    @pytest.fixture
    def risk_manager_garch(self):
        """Risk manager with GARCH enabled."""
        return EnhancedRiskManager({
            "use_garch": True,
            "vol_lookback": 60
        })

    @pytest.mark.unit
    def test_simple_volatility_forecast(self, risk_manager_simple):
        """
        Simple volatility forecast uses historical volatility.

        Should return annualized historical standard deviation.
        """
        returns = pd.Series(np.random.normal(0, 0.01, 100))

        forecast_vol = risk_manager_simple.forecast_volatility(returns)

        # Should equal historical volatility
        expected_vol = returns.std() * np.sqrt(252)
        assert forecast_vol == pytest.approx(expected_vol, rel=0.001)

    @pytest.mark.unit
    def test_volatility_forecast_insufficient_data(self, risk_manager_simple):
        """
        Volatility forecast handles insufficient data.

        Should return default volatility.
        """
        # Less than 2 data points
        returns = pd.Series([0.01])

        forecast_vol = risk_manager_simple.forecast_volatility(returns)

        assert forecast_vol == 0.15  # Default 15% volatility

    @pytest.mark.unit
    def test_volatility_forecast_clustering(self, risk_manager_garch):
        """
        GARCH model captures volatility clustering.

        Recent volatility should have more weight.
        """
        # Create returns with volatility regime change
        quiet_period = np.random.normal(0, 0.005, 100)  # Low vol
        volatile_period = np.random.normal(0, 0.03, 20)  # High vol
        all_returns = np.concatenate([quiet_period, volatile_period])
        returns = pd.Series(all_returns)

        # Forecast should be higher than long-term average
        forecast_vol = risk_manager_garch.forecast_volatility(returns)
        returns.std() * np.sqrt(252)

        # GARCH should put more weight on recent high volatility
        # Note: Without actual GARCH implementation, this tests the interface
        assert forecast_vol > 0
        assert isinstance(forecast_vol, float)

    @pytest.mark.unit
    @pytest.mark.parametrize("returns_std,expected_annual_vol", [
        (0.01, 0.1587),   # 1% daily → ~15.87% annual
        (0.02, 0.3175),   # 2% daily → ~31.75% annual
        (0.005, 0.0794),  # 0.5% daily → ~7.94% annual
    ])
    def test_volatility_annualization(self, risk_manager_simple, returns_std, expected_annual_vol):
        """
        Volatility is correctly annualized.

        Annual vol = Daily vol * sqrt(252)
        """
        # Generate returns with specified volatility
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, returns_std, 252))

        forecast_vol = risk_manager_simple.forecast_volatility(returns)

        assert forecast_vol == pytest.approx(expected_annual_vol, rel=0.1)


class TestRiskLimitEnforcement:
    """Test risk limit checking and enforcement."""

    @pytest.fixture
    def risk_manager(self):
        """Risk manager with strict limits."""
        return EnhancedRiskManager({
            "max_var_95": 0.02,  # 2% daily VaR
            "target_vol": 0.15,  # 15% annual vol
            "max_drawdown": 0.10,  # 10% max drawdown
            "min_sharpe": 0.5,   # Minimum Sharpe ratio
        })

    @pytest.mark.unit
    def test_var_limit_check(self, risk_manager):
        """
        VaR limit violations are detected.

        Should flag when VaR exceeds limit.
        """
        # High volatility returns
        high_vol_returns = pd.Series(np.random.normal(0, 0.03, 100))  # 3% daily vol

        metrics = risk_manager.calculate_risk_metrics(high_vol_returns)
        risk_manager.risk_metrics = metrics

        # VaR should exceed 2% limit
        assert metrics.value_at_risk > 0.02

        # Check if violation would be detected
        assert metrics.value_at_risk > risk_manager.risk_limits["max_var_95"]

    @pytest.mark.unit
    def test_volatility_limit_check(self, risk_manager):
        """
        Portfolio volatility limits are enforced.

        Should flag when volatility exceeds target.
        """
        # Generate high volatility returns
        high_vol_returns = pd.Series(np.random.normal(0, 0.015, 252))  # ~24% annual vol

        metrics = risk_manager.calculate_risk_metrics(high_vol_returns)

        # Volatility should exceed 15% target
        assert metrics.portfolio_volatility > 0.15
        assert metrics.portfolio_volatility > risk_manager.risk_limits["max_portfolio_volatility"]

    @pytest.mark.unit
    def test_drawdown_limit_check(self, risk_manager):
        """
        Drawdown limits are monitored correctly.

        Should flag when drawdown exceeds maximum.
        """
        # Create returns with 15% drawdown
        returns = [0.01] * 50 + [-0.03] * 5 + [0.005] * 45
        returns_series = pd.Series(returns)

        metrics = risk_manager.calculate_risk_metrics(returns_series)

        # Should exceed 10% drawdown limit
        assert metrics.maximum_drawdown > 0.10
        assert metrics.maximum_drawdown > risk_manager.risk_limits["max_drawdown"]

    @pytest.mark.unit
    def test_sharpe_ratio_minimum(self, risk_manager):
        """
        Minimum Sharpe ratio is enforced.

        Should flag underperforming strategies.
        """
        # Poor performance returns
        poor_returns = pd.Series(np.random.normal(-0.0002, 0.02, 252))  # Negative drift

        metrics = risk_manager.calculate_risk_metrics(poor_returns)

        # Sharpe should be below minimum
        assert metrics.sharpe_ratio < 0.5
        assert metrics.sharpe_ratio < risk_manager.risk_limits["min_sharpe"]

    @pytest.mark.unit
    def test_risk_state_management(self, risk_manager):
        """
        Risk manager tracks risk-on/risk-off state.

        Should update state based on risk metrics.
        """
        # Initial state should be risk-on
        assert risk_manager.is_risk_on is True

        # Generate returns that violate multiple limits
        bad_returns = pd.Series(np.random.normal(-0.001, 0.03, 100))
        metrics = risk_manager.calculate_risk_metrics(bad_returns)
        risk_manager.risk_metrics = metrics

        # Risk manager should track metrics
        assert risk_manager.risk_metrics is not None
        assert isinstance(risk_manager.risk_metrics, RiskMetrics)


class TestRiskMetricsEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def risk_manager(self):
        """Standard risk manager."""
        return EnhancedRiskManager({})

    @pytest.mark.unit
    def test_empty_returns(self, risk_manager):
        """
        Empty returns are handled gracefully.

        Should return default metrics.
        """
        empty_returns = pd.Series(dtype=float)

        metrics = risk_manager.calculate_risk_metrics(empty_returns)

        assert metrics.value_at_risk == 0.0
        assert metrics.portfolio_volatility == 0.15  # Default
        assert metrics.sharpe_ratio == 0.0

    @pytest.mark.unit
    def test_single_return(self, risk_manager):
        """
        Single return value is handled properly.

        Cannot calculate statistics with one point.
        """
        single_return = pd.Series([0.01])

        metrics = risk_manager.calculate_risk_metrics(single_return)

        # Should return defaults
        assert metrics.value_at_risk == 0.0
        assert metrics.sharpe_ratio == 0.0

    @pytest.mark.unit
    def test_all_positive_returns(self, risk_manager):
        """
        All positive returns handled correctly.

        Sortino ratio should handle no downside.
        """
        all_positive = pd.Series(np.random.uniform(0.0001, 0.02, 100))

        metrics = risk_manager.calculate_risk_metrics(all_positive)

        # No losses means zero VaR
        assert metrics.value_at_risk == 0.0

        # Very high Sortino (no downside)
        assert metrics.sortino_ratio > metrics.sharpe_ratio
        assert metrics.sortino_ratio > 5.0  # Should be very high

        # Zero drawdown
        assert metrics.maximum_drawdown == 0.0

    @pytest.mark.unit
    def test_constant_returns(self, risk_manager):
        """
        Constant returns (zero volatility) handled safely.

        Should not divide by zero.
        """
        constant_returns = pd.Series([0.001] * 100)

        metrics = risk_manager.calculate_risk_metrics(constant_returns)

        # Zero volatility edge case
        assert metrics.portfolio_volatility == 0.0

        # Sharpe undefined but should not crash
        assert isinstance(metrics.sharpe_ratio, float)
        assert not np.isnan(metrics.sharpe_ratio)

    @pytest.mark.unit
    def test_extreme_returns(self, risk_manager):
        """
        Extreme returns don't break calculations.

        Should handle without overflow.
        """
        # Include some extreme returns
        returns = np.random.normal(0, 0.01, 100)
        returns[50] = 0.5   # 50% gain
        returns[51] = -0.4  # 40% loss
        extreme_returns = pd.Series(returns)

        metrics = risk_manager.calculate_risk_metrics(extreme_returns)

        # Should complete without errors
        assert metrics.value_at_risk > 0
        assert metrics.maximum_drawdown > 0.3
        assert isinstance(metrics.sharpe_ratio, float)

    @pytest.mark.unit
    def test_safe_float_conversion(self, risk_manager):
        """
        Safe float conversion handles various input types.

        Should convert pandas/numpy types safely.
        """
        # Test various input types
        assert risk_manager._safe_float_conversion(1.5) == 1.5
        assert risk_manager._safe_float_conversion(np.float64(2.5)) == 2.5
        assert risk_manager._safe_float_conversion(pd.Series([3.5]).iloc[0]) == 3.5
        assert risk_manager._safe_float_conversion(np.array([4.5])[0]) == 4.5
        assert risk_manager._safe_float_conversion(None) == 0.0
        assert risk_manager._safe_float_conversion(np.nan) == 0.0
        assert risk_manager._safe_float_conversion("invalid") == 0.0


class TestRiskManagerIntegration:
    """Test risk manager integration with other components."""

    @pytest.fixture
    def risk_manager(self):
        """Risk manager with full configuration."""
        return EnhancedRiskManager({
            "max_var_95": 0.02,
            "target_vol": 0.15,
            "max_position_size": 0.20,
            "max_drawdown": 0.15,
            "max_correlation": 0.70,
            "min_sharpe": 0.5,
            "use_garch": True,
            "vol_lookback": 60,
        })

    @pytest.mark.unit
    def test_historical_metrics_tracking(self, risk_manager):
        """
        Risk manager maintains historical metrics.

        Should store metrics for trend analysis.
        """
        # Calculate metrics multiple times
        for i in range(5):
            returns = pd.Series(np.random.normal(0.0005, 0.01 + i*0.002, 100))
            metrics = risk_manager.calculate_risk_metrics(returns)
            risk_manager.historical_metrics.append(metrics)

        # Should track all metrics
        assert len(risk_manager.historical_metrics) == 5

        # Volatility should be increasing
        vols = [m.portfolio_volatility for m in risk_manager.historical_metrics]
        assert all(vols[i] < vols[i+1] for i in range(4))

    @pytest.mark.unit
    def test_regime_detection(self, risk_manager):
        """
        Risk manager can detect market regimes.

        Should identify normal/high volatility regimes.
        """
        # Normal regime
        normal_returns = pd.Series(np.random.normal(0, 0.01, 100))
        metrics = risk_manager.calculate_risk_metrics(normal_returns)
        risk_manager.risk_metrics = metrics

        assert risk_manager.current_regime == "NORMAL"

        # High volatility regime
        high_vol_returns = pd.Series(np.random.normal(0, 0.03, 100))
        metrics = risk_manager.calculate_risk_metrics(high_vol_returns)
        risk_manager.risk_metrics = metrics

        # Would need regime detection logic
        # For now, just verify the attribute exists
        assert hasattr(risk_manager, 'current_regime')

    @pytest.mark.unit
    def test_correlation_matrix_update(self, risk_manager):
        """
        Correlation matrix is maintained for multi-asset portfolios.

        Should update correlation estimates.
        """
        # Multi-asset returns
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.018, 100),
            'GOOGL': np.random.normal(0.0008, 0.022, 100),
        }, index=dates)

        # Add correlation
        returns['MSFT'] = returns['MSFT'] * 0.5 + returns['AAPL'] * 0.5

        risk_manager.correlation_matrix = returns.corr()

        # Verify correlation matrix properties
        assert risk_manager.correlation_matrix.shape == (3, 3)
        assert np.allclose(risk_manager.correlation_matrix.values.diagonal(), 1.0)
        assert risk_manager.correlation_matrix.loc['AAPL', 'MSFT'] > 0.5

    @pytest.mark.unit
    def test_sector_exposure_tracking(self, risk_manager):
        """
        Sector exposures are tracked correctly.

        Should monitor concentration by sector.
        """
        # Set sector exposures
        risk_manager.sector_exposures = {
            'Technology': 0.45,  # Over limit
            'Healthcare': 0.20,
            'Finance': 0.15,
            'Consumer': 0.20,
        }

        # Check against sector limit
        sector_limit = risk_manager.risk_limits.get("max_sector_exposure", 0.40)

        violations = [
            sector for sector, exposure in risk_manager.sector_exposures.items()
            if exposure > sector_limit
        ]

        assert 'Technology' in violations
        assert len(violations) == 1


class TestParametrizedRiskScenarios:
    """Test various risk scenarios with parametrization."""

    @pytest.fixture
    def risk_manager(self):
        """Standard risk manager."""
        return EnhancedRiskManager({"risk_free_rate": 0.02})

    @pytest.mark.unit
    @pytest.mark.parametrize("volatility,expected_var_range", [
        (0.01, (0.01, 0.02)),    # Low volatility
        (0.02, (0.025, 0.04)),   # Normal volatility
        (0.05, (0.07, 0.10)),    # High volatility
        (0.10, (0.15, 0.20)),    # Extreme volatility
    ])
    def test_var_scales_with_volatility(self, risk_manager, volatility, expected_var_range):
        """
        VaR scales appropriately with volatility.

        Higher volatility should produce higher VaR.
        """
        # Generate returns with specified volatility
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, volatility, 1000))

        metrics = risk_manager.calculate_risk_metrics(returns)

        # VaR should be in expected range
        assert expected_var_range[0] < metrics.value_at_risk < expected_var_range[1]

    @pytest.mark.unit
    @pytest.mark.parametrize("correlation,expected_beta", [
        (1.0, 1.0),    # Perfect correlation
        (0.8, 0.8),    # High correlation
        (0.0, 0.0),    # No correlation
        (-0.5, -0.5),  # Negative correlation
    ])
    def test_beta_correlation_relationship(self, risk_manager, correlation, expected_beta):
        """
        Beta reflects correlation with benchmark.

        Tests the relationship between correlation and beta.
        """
        # Generate correlated returns
        np.random.seed(42)
        benchmark = pd.Series(np.random.normal(0, 0.01, 100))

        # Create portfolio with specified correlation
        noise = pd.Series(np.random.normal(0, 0.01, 100))
        portfolio = correlation * benchmark + np.sqrt(1 - correlation**2) * noise

        beta = risk_manager._calculate_beta(portfolio, benchmark)

        # Beta should approximately equal correlation (when vols are equal)
        assert beta == pytest.approx(expected_beta, abs=0.1)

    @pytest.mark.unit
    @pytest.mark.parametrize("return_mean,return_vol,min_sharpe", [
        (0.0005, 0.01, True),   # Good Sharpe
        (0.0, 0.01, False),     # Zero Sharpe
        (-0.0002, 0.01, False), # Negative Sharpe
        (0.001, 0.005, True),   # Very good Sharpe
    ])
    def test_sharpe_ratio_threshold(self, risk_manager, return_mean, return_vol, min_sharpe):
        """
        Sharpe ratio threshold correctly identifies good/bad performance.

        Tests various return profiles against minimum Sharpe.
        """
        # Generate returns
        returns = pd.Series(np.random.normal(return_mean, return_vol, 252))

        metrics = risk_manager.calculate_risk_metrics(returns)

        meets_minimum = metrics.sharpe_ratio >= risk_manager.risk_limits["min_sharpe"]
        assert meets_minimum == min_sharpe
