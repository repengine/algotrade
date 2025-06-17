"""Comprehensive test suite for risk management."""


import numpy as np
import pandas as pd
import pytest

from algostack.core.risk import (
    RiskLimits,
    RiskManager,
)


class TestRiskManager:
    """Test suite for RiskManager class."""

    @pytest.fixture
    def risk_limits(self):
        """Create sample risk limits."""
        return RiskLimits(
            max_position_size=0.20,  # 20% of portfolio
            max_portfolio_risk=0.06,  # 6% total risk
            max_single_loss=0.02,     # 2% max loss per trade
            max_daily_loss=0.03,      # 3% max daily loss
            max_leverage=1.5,         # 1.5x leverage
            max_correlation=0.7,      # 70% max correlation
            max_concentration=0.40    # 40% max sector concentration
        )

    @pytest.fixture
    def risk_manager(self, risk_limits):
        """Create risk manager instance."""
        return RiskManager(
            risk_limits=risk_limits,
            portfolio_value=100000
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return {
            'AAPL': {'quantity': 100, 'price': 150.0, 'value': 15000, 'sector': 'Technology'},
            'GOOGL': {'quantity': 10, 'price': 2800.0, 'value': 28000, 'sector': 'Technology'},
            'JPM': {'quantity': 200, 'price': 140.0, 'value': 28000, 'sector': 'Finance'},
            'XOM': {'quantity': 300, 'price': 80.0, 'value': 24000, 'sector': 'Energy'}
        }

    def test_initialization(self, risk_limits):
        """Test risk manager initialization."""
        rm = RiskManager(
            risk_limits=risk_limits,
            portfolio_value=50000,
            volatility_lookback=30
        )

        assert rm.risk_limits == risk_limits
        assert rm.portfolio_value == 50000
        assert rm.volatility_lookback == 30
        assert rm.risk_alerts == []
        assert rm.violations == []

    def test_check_position_size_within_limit(self, risk_manager):
        """Test position size check within limits."""
        # 10% position - should pass
        is_valid = risk_manager.check_position_size(
            symbol='AAPL',
            position_value=10000,
            portfolio_value=100000
        )

        assert is_valid
        assert len(risk_manager.risk_alerts) == 0

    def test_check_position_size_exceeds_limit(self, risk_manager):
        """Test position size check exceeding limits."""
        # 25% position - should fail (limit is 20%)
        is_valid = risk_manager.check_position_size(
            symbol='AAPL',
            position_value=25000,
            portfolio_value=100000
        )

        assert not is_valid
        assert len(risk_manager.violations) == 1
        assert risk_manager.violations[0].rule == 'MAX_POSITION_SIZE'

    def test_check_portfolio_risk(self, risk_manager, sample_positions):
        """Test portfolio risk calculation."""
        # Calculate portfolio risk
        portfolio_risk = risk_manager.calculate_portfolio_risk(
            positions=sample_positions,
            market_volatilities={'AAPL': 0.02, 'GOOGL': 0.03, 'JPM': 0.025, 'XOM': 0.035}
        )

        assert isinstance(portfolio_risk, float)
        assert portfolio_risk > 0

        # Check if within limits
        is_valid = risk_manager.check_portfolio_risk(portfolio_risk)
        assert isinstance(is_valid, bool)

    def test_check_daily_loss_limit(self, risk_manager):
        """Test daily loss limit check."""
        # Simulate daily P&L
        daily_pnl = -2500  # 2.5% loss

        is_valid = risk_manager.check_daily_loss(
            daily_pnl=daily_pnl,
            portfolio_value=100000
        )

        assert is_valid  # Within 3% limit

        # Test exceeding limit
        daily_pnl = -3500  # 3.5% loss
        is_valid = risk_manager.check_daily_loss(
            daily_pnl=daily_pnl,
            portfolio_value=100000
        )

        assert not is_valid
        assert len(risk_manager.violations) == 1

    def test_check_leverage(self, risk_manager):
        """Test leverage check."""
        # Test within leverage limit
        total_exposure = 120000  # 1.2x leverage
        is_valid = risk_manager.check_leverage(
            total_exposure=total_exposure,
            portfolio_value=100000
        )

        assert is_valid

        # Test exceeding leverage limit
        total_exposure = 180000  # 1.8x leverage
        is_valid = risk_manager.check_leverage(
            total_exposure=total_exposure,
            portfolio_value=100000
        )

        assert not is_valid
        assert any(v.rule == 'MAX_LEVERAGE' for v in risk_manager.violations)

    def test_check_correlation_risk(self, risk_manager):
        """Test correlation risk check."""
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.6, 0.3, 0.2],
            'GOOGL': [0.6, 1.0, 0.2, 0.1],
            'JPM': [0.3, 0.2, 1.0, 0.4],
            'XOM': [0.2, 0.1, 0.4, 1.0]
        }, index=['AAPL', 'GOOGL', 'JPM', 'XOM'])

        # Check correlation risk
        is_valid = risk_manager.check_correlation_risk(
            correlation_matrix=correlation_matrix,
            positions=['AAPL', 'GOOGL', 'JPM', 'XOM']
        )

        assert is_valid  # Max correlation is 0.6, below 0.7 limit

        # Test with high correlation
        correlation_matrix.loc['AAPL', 'GOOGL'] = 0.85
        correlation_matrix.loc['GOOGL', 'AAPL'] = 0.85

        is_valid = risk_manager.check_correlation_risk(
            correlation_matrix=correlation_matrix,
            positions=['AAPL', 'GOOGL']
        )

        assert not is_valid

    def test_check_concentration_risk(self, risk_manager, sample_positions):
        """Test concentration risk check."""
        # Calculate sector concentration
        sector_allocation = risk_manager.calculate_sector_allocation(sample_positions)

        # Technology sector has AAPL (15k) + GOOGL (28k) = 43k out of 95k total
        # That's 45.3% - exceeds 40% limit
        is_valid = risk_manager.check_concentration_risk(sector_allocation)

        assert not is_valid
        assert any(v.rule == 'MAX_CONCENTRATION' for v in risk_manager.violations)

    def test_calculate_var(self, risk_manager):
        """Test Value at Risk calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns

        var_95 = risk_manager.calculate_var(
            returns=returns,
            confidence_level=0.95,
            portfolio_value=100000
        )

        assert var_95 < 0  # VaR is negative (potential loss)
        assert abs(var_95) < 10000  # Reasonable VaR for 100k portfolio

    def test_calculate_cvar(self, risk_manager):
        """Test Conditional Value at Risk calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        cvar_95 = risk_manager.calculate_cvar(
            returns=returns,
            confidence_level=0.95,
            portfolio_value=100000
        )

        var_95 = risk_manager.calculate_var(returns, 0.95, 100000)

        assert cvar_95 < var_95  # CVaR should be more negative than VaR

    def test_calculate_sharpe_ratio(self, risk_manager):
        """Test Sharpe ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.015, 252))

        sharpe = risk_manager.calculate_sharpe_ratio(
            returns=returns,
            risk_free_rate=0.02
        )

        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range

    def test_calculate_max_drawdown(self, risk_manager):
        """Test maximum drawdown calculation."""
        # Create equity curve with drawdown
        prices = pd.Series([100, 105, 110, 105, 95, 100, 105, 110, 115])

        max_dd = risk_manager.calculate_max_drawdown(prices)

        # Max drawdown from 110 to 95 = 13.6%
        assert max_dd == pytest.approx(-0.136, rel=0.01)

    def test_position_risk_metrics(self, risk_manager):
        """Test individual position risk metrics."""
        position_risk = risk_manager.calculate_position_risk(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            volatility=0.02,
            stop_loss=145.0
        )

        assert position_risk['value_at_risk'] < 0
        assert position_risk['stop_loss_risk'] == -500.0  # (145-150) * 100
        assert position_risk['volatility_risk'] == pytest.approx(100 * 155 * 0.02)
        assert position_risk['position_value'] == 15500.0

    def test_pre_trade_risk_check(self, risk_manager, sample_positions):
        """Test pre-trade risk validation."""
        # Simulate new trade
        new_trade = {
            'symbol': 'MSFT',
            'quantity': 100,
            'price': 300.0,
            'value': 30000,
            'sector': 'Technology'
        }

        # Check if trade passes risk checks
        checks = risk_manager.pre_trade_risk_check(
            new_trade=new_trade,
            current_positions=sample_positions,
            portfolio_value=100000
        )

        assert 'position_size' in checks
        assert 'portfolio_risk' in checks
        assert 'concentration' in checks
        assert 'approved' in checks

        # This trade would make Tech sector too concentrated
        assert not checks['approved']

    def test_risk_alert_generation(self, risk_manager):
        """Test risk alert generation."""
        # Generate various risk conditions
        risk_manager.add_alert(
            level='WARNING',
            message='Position approaching size limit',
            metric='position_size',
            value=0.18,
            threshold=0.20
        )

        risk_manager.add_alert(
            level='CRITICAL',
            message='Daily loss limit exceeded',
            metric='daily_loss',
            value=-0.035,
            threshold=-0.03
        )

        assert len(risk_manager.risk_alerts) == 2
        assert any(a.level == 'CRITICAL' for a in risk_manager.risk_alerts)

        # Test alert filtering
        critical_alerts = risk_manager.get_alerts_by_level('CRITICAL')
        assert len(critical_alerts) == 1

    def test_risk_metrics_calculation(self, risk_manager, sample_positions):
        """Test comprehensive risk metrics calculation."""
        market_data = {
            'volatilities': {'AAPL': 0.02, 'GOOGL': 0.03, 'JPM': 0.025, 'XOM': 0.035},
            'correlations': pd.DataFrame(
                np.random.rand(4, 4) * 0.5 + 0.25,
                index=['AAPL', 'GOOGL', 'JPM', 'XOM'],
                columns=['AAPL', 'GOOGL', 'JPM', 'XOM']
            )
        }

        # Set diagonal to 1
        for symbol in market_data['correlations'].index:
            market_data['correlations'].loc[symbol, symbol] = 1.0

        metrics = risk_manager.calculate_risk_metrics(
            positions=sample_positions,
            market_data=market_data,
            portfolio_value=100000
        )

        assert 'total_exposure' in metrics
        assert 'leverage' in metrics
        assert 'portfolio_var' in metrics
        assert 'portfolio_volatility' in metrics
        assert 'max_position_size' in metrics
        assert 'concentration_by_sector' in metrics

    def test_stress_testing(self, risk_manager, sample_positions):
        """Test portfolio stress testing."""
        scenarios = {
            'market_crash': {'AAPL': -0.20, 'GOOGL': -0.25, 'JPM': -0.15, 'XOM': -0.30},
            'tech_selloff': {'AAPL': -0.30, 'GOOGL': -0.35, 'JPM': -0.05, 'XOM': -0.02},
            'rate_hike': {'AAPL': -0.10, 'GOOGL': -0.12, 'JPM': 0.05, 'XOM': -0.08}
        }

        stress_results = risk_manager.run_stress_tests(
            positions=sample_positions,
            scenarios=scenarios
        )

        assert len(stress_results) == 3
        assert all(result < 0 for result in stress_results.values())  # All scenarios show losses
        assert stress_results['tech_selloff'] < stress_results['rate_hike']  # Tech selloff worse

    def test_risk_report_generation(self, risk_manager, sample_positions):
        """Test risk report generation."""
        report = risk_manager.generate_risk_report(
            positions=sample_positions,
            portfolio_value=100000,
            daily_pnl=-1500
        )

        assert 'timestamp' in report
        assert 'portfolio_value' in report
        assert 'risk_metrics' in report
        assert 'violations' in report
        assert 'alerts' in report
        assert 'recommendations' in report

    def test_dynamic_risk_adjustment(self, risk_manager):
        """Test dynamic risk limit adjustment based on market conditions."""
        # Simulate high volatility environment
        market_volatility = 0.04  # 4% daily vol (high)

        adjusted_limits = risk_manager.adjust_risk_limits(
            base_limits=risk_manager.risk_limits,
            market_volatility=market_volatility,
            recent_performance=0.05  # 5% recent return
        )

        # Should reduce risk limits in high volatility
        assert adjusted_limits.max_position_size < risk_manager.risk_limits.max_position_size
        assert adjusted_limits.max_leverage < risk_manager.risk_limits.max_leverage

    def test_margin_calculation(self, risk_manager):
        """Test margin requirement calculations."""
        positions = {
            'AAPL': {'quantity': 100, 'price': 150.0, 'type': 'LONG'},
            'TSLA': {'quantity': -50, 'price': 200.0, 'type': 'SHORT'},  # Short position
        }

        margin_req = risk_manager.calculate_margin_requirements(
            positions=positions,
            margin_rates={'LONG': 0.25, 'SHORT': 0.30}
        )

        expected_margin = (100 * 150 * 0.25) + (50 * 200 * 0.30)
        assert margin_req == expected_margin

    def test_liquidity_risk(self, risk_manager):
        """Test liquidity risk assessment."""
        position_sizes = {
            'AAPL': 1000,  # shares
            'SMALL_CAP': 50000  # shares of illiquid stock
        }

        avg_daily_volumes = {
            'AAPL': 50000000,
            'SMALL_CAP': 100000
        }

        liquidity_risk = risk_manager.assess_liquidity_risk(
            position_sizes=position_sizes,
            avg_daily_volumes=avg_daily_volumes,
            max_participation=0.10  # Max 10% of daily volume
        )

        assert liquidity_risk['AAPL']['days_to_liquidate'] < 1
        assert liquidity_risk['SMALL_CAP']['days_to_liquidate'] > 1
        assert liquidity_risk['SMALL_CAP']['liquidity_score'] < liquidity_risk['AAPL']['liquidity_score']
