"""Comprehensive test suite for EnhancedRiskManager."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.risk import EnhancedRiskManager, RiskMetrics


class TestRiskMetrics:
    """Test suite for RiskMetrics class."""
    
    def test_risk_metrics_initialization(self):
        """Test RiskMetrics initialization."""
        metrics = RiskMetrics()
        
        assert metrics.var_95 == 0.0
        assert metrics.cvar_95 == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.current_drawdown == 0.0
        assert metrics.portfolio_volatility == 0.0
        assert metrics.portfolio_beta == 0.0
        assert metrics.correlation_risk == 0.0
        assert metrics.concentration_risk == 0.0
        assert metrics.liquidity_score == 1.0


class TestEnhancedRiskManager:
    """Test suite for EnhancedRiskManager class."""
    
    @pytest.fixture
    def risk_config(self):
        """Create risk configuration."""
        return {
            'max_position_size': 0.20,
            'max_portfolio_risk': 0.06,
            'max_drawdown': 0.15,
            'max_leverage': 1.5,
            'position_limits': {
                'AAPL': 0.15,
                'TSLA': 0.10
            },
            'sector_limits': {
                'Technology': 0.40,
                'Finance': 0.30
            },
            'var_limit': 10000,
            'risk_free_rate': 0.03
        }
    
    @pytest.fixture
    def risk_manager(self, risk_config):
        """Create risk manager instance."""
        return EnhancedRiskManager(risk_config)
    
    @pytest.fixture
    def portfolio_state(self):
        """Create sample portfolio state."""
        return {
            'cash': 50000,
            'positions': {
                'AAPL': {'quantity': 100, 'market_value': 15000, 'cost_basis': 14000},
                'GOOGL': {'quantity': 10, 'market_value': 28000, 'cost_basis': 27000},
                'JPM': {'quantity': 200, 'market_value': 28000, 'cost_basis': 30000}
            },
            'total_value': 121000,
            'leverage': 1.42
        }
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data."""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        returns = {
            'AAPL': pd.Series(np.random.normal(0.001, 0.02, 252), index=dates),
            'GOOGL': pd.Series(np.random.normal(0.0008, 0.025, 252), index=dates),
            'JPM': pd.Series(np.random.normal(0.0005, 0.018, 252), index=dates),
            'SPY': pd.Series(np.random.normal(0.0007, 0.015, 252), index=dates)
        }
        
        return {
            'returns': pd.DataFrame(returns),
            'volatilities': {'AAPL': 0.02, 'GOOGL': 0.025, 'JPM': 0.018},
            'correlations': pd.DataFrame(
                [[1.0, 0.6, 0.3], [0.6, 1.0, 0.2], [0.3, 0.2, 1.0]],
                index=['AAPL', 'GOOGL', 'JPM'],
                columns=['AAPL', 'GOOGL', 'JPM']
            ),
            'betas': {'AAPL': 1.2, 'GOOGL': 1.4, 'JPM': 0.9}
        }
    
    def test_initialization(self, risk_config):
        """Test risk manager initialization."""
        rm = EnhancedRiskManager(risk_config)
        
        assert rm.config == risk_config
        assert rm.current_metrics is not None
        assert rm.risk_limits['var_95'] == risk_config['var_limit']
        assert rm.sector_allocations == {}
    
    def test_check_position_limit(self, risk_manager, portfolio_state):
        """Test position limit checking."""
        # Check within limit
        result = risk_manager.check_position_limit(
            'MSFT',
            proposed_value=20000,
            portfolio_value=portfolio_state['total_value']
        )
        
        assert result['allowed'] is True
        assert result['max_allowed'] > 20000
        
        # Check exceeding limit
        result = risk_manager.check_position_limit(
            'MSFT',
            proposed_value=30000,
            portfolio_value=portfolio_state['total_value']
        )
        
        assert result['allowed'] is False
        assert 'exceeds limit' in result['reason'].lower()
    
    def test_check_position_limit_specific_symbol(self, risk_manager, portfolio_state):
        """Test position limit for specific symbols."""
        # TSLA has 10% limit
        result = risk_manager.check_position_limit(
            'TSLA',
            proposed_value=15000,
            portfolio_value=portfolio_state['total_value']
        )
        
        assert result['allowed'] is False
        assert result['limit'] == 0.10
    
    def test_calculate_portfolio_metrics(self, risk_manager, portfolio_state, market_data):
        """Test portfolio metrics calculation."""
        metrics = risk_manager.calculate_portfolio_metrics(
            portfolio_state,
            market_data['returns']
        )
        
        assert isinstance(metrics.var_95, float)
        assert metrics.var_95 < 0  # VaR is negative
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.portfolio_volatility, float)
        assert metrics.portfolio_volatility > 0
    
    def test_check_risk_limits(self, risk_manager, portfolio_state, market_data):
        """Test comprehensive risk limit checking."""
        # Calculate metrics first
        risk_manager.calculate_portfolio_metrics(portfolio_state, market_data['returns'])
        
        # Check limits
        violations = risk_manager.check_risk_limits(portfolio_state)
        
        assert isinstance(violations, list)
        # Check for any violations
        for violation in violations:
            assert 'rule' in violation
            assert 'current' in violation
            assert 'limit' in violation
            assert 'severity' in violation
    
    def test_calculate_position_risk(self, risk_manager, market_data):
        """Test individual position risk calculation."""
        position_risk = risk_manager.calculate_position_risk(
            symbol='AAPL',
            position_value=15000,
            returns=market_data['returns']['AAPL'],
            portfolio_value=100000
        )
        
        assert 'var_95' in position_risk
        assert 'volatility' in position_risk
        assert 'concentration' in position_risk
        assert position_risk['concentration'] == 0.15
    
    def test_pre_trade_check(self, risk_manager, portfolio_state, market_data):
        """Test pre-trade risk validation."""
        proposed_trade = {
            'symbol': 'MSFT',
            'side': 'BUY',
            'quantity': 100,
            'price': 300,
            'value': 30000
        }
        
        # Update metrics
        risk_manager.calculate_portfolio_metrics(portfolio_state, market_data['returns'])
        
        # Check trade
        result = risk_manager.pre_trade_check(
            proposed_trade,
            portfolio_state,
            market_data.get('returns', {}).get('MSFT', pd.Series())
        )
        
        assert 'approved' in result
        assert 'checks' in result
        assert 'position_limit' in result['checks']
        assert 'portfolio_risk' in result['checks']
    
    def test_calculate_optimal_position_size(self, risk_manager, market_data):
        """Test optimal position size calculation."""
        size_info = risk_manager.calculate_optimal_position_size(
            symbol='AAPL',
            signal_strength=0.8,
            volatility=market_data['volatilities']['AAPL'],
            portfolio_value=100000,
            current_positions=2
        )
        
        assert 'shares' in size_info
        assert 'position_value' in size_info
        assert 'risk_allocation' in size_info
        assert size_info['position_value'] <= 20000  # Max 20% position
    
    def test_stress_test(self, risk_manager, portfolio_state):
        """Test portfolio stress testing."""
        scenarios = {
            'market_crash': {
                'AAPL': -0.20,
                'GOOGL': -0.25,
                'JPM': -0.15
            },
            'tech_crash': {
                'AAPL': -0.30,
                'GOOGL': -0.35,
                'JPM': -0.05
            },
            'rate_shock': {
                'AAPL': -0.10,
                'GOOGL': -0.12,
                'JPM': 0.05
            }
        }
        
        results = risk_manager.stress_test(portfolio_state, scenarios)
        
        assert len(results) == 3
        assert all(scenario in results for scenario in scenarios)
        
        # Market crash should be worst for this portfolio
        assert results['tech_crash']['portfolio_impact'] < results['rate_shock']['portfolio_impact']
    
    def test_update_risk_metrics(self, risk_manager, portfolio_state, market_data):
        """Test risk metrics update."""
        risk_manager.update_risk_metrics(
            portfolio_state,
            market_data['returns'],
            market_data['correlations']
        )
        
        metrics = risk_manager.current_metrics
        
        assert metrics.portfolio_volatility > 0
        assert metrics.var_95 < 0
        assert 0 <= metrics.correlation_risk <= 1
    
    def test_get_risk_report(self, risk_manager, portfolio_state, market_data):
        """Test risk report generation."""
        # Update metrics
        risk_manager.calculate_portfolio_metrics(portfolio_state, market_data['returns'])
        
        report = risk_manager.get_risk_report(portfolio_state)
        
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert 'timestamp' in report
    
    def test_sector_concentration(self, risk_manager):
        """Test sector concentration limit checking."""
        portfolio_state = {
            'positions': {
                'AAPL': {'market_value': 30000, 'sector': 'Technology'},
                'GOOGL': {'market_value': 25000, 'sector': 'Technology'},
                'MSFT': {'market_value': 20000, 'sector': 'Technology'},
                'JPM': {'market_value': 25000, 'sector': 'Finance'}
            },
            'total_value': 100000
        }
        
        risk_manager.update_sector_allocations(portfolio_state)
        violations = risk_manager.check_sector_limits()
        
        # Technology sector is 75% - exceeds 40% limit
        assert len(violations) > 0
        assert any(v['sector'] == 'Technology' for v in violations)
    
    def test_correlation_risk(self, risk_manager, market_data):
        """Test correlation risk assessment."""
        positions = ['AAPL', 'GOOGL', 'JPM']
        weights = np.array([0.3, 0.4, 0.3])
        
        corr_risk = risk_manager.calculate_correlation_risk(
            market_data['correlations'],
            positions,
            weights
        )
        
        assert 0 <= corr_risk <= 1
        # With positive correlations, risk should be > 0
        assert corr_risk > 0
    
    def test_liquidity_assessment(self, risk_manager):
        """Test liquidity risk assessment."""
        position_sizes = {
            'AAPL': 1000,
            'SMALL_CAP': 50000
        }
        
        daily_volumes = {
            'AAPL': 50000000,
            'SMALL_CAP': 100000
        }
        
        liquidity_scores = risk_manager.assess_liquidity(
            position_sizes,
            daily_volumes,
            impact_threshold=0.05
        )
        
        assert liquidity_scores['AAPL'] > liquidity_scores['SMALL_CAP']
        assert liquidity_scores['SMALL_CAP'] < 0.5  # Poor liquidity
    
    def test_dynamic_adjustment(self, risk_manager, market_data):
        """Test dynamic risk limit adjustment."""
        # High volatility environment
        high_vol_returns = market_data['returns'] * 2  # Double volatility
        
        adjusted_limits = risk_manager.adjust_limits_dynamic(
            market_volatility=0.04,  # 4% daily vol
            recent_drawdown=0.10     # 10% recent drawdown
        )
        
        # Limits should be tightened
        assert adjusted_limits['max_position_size'] < risk_manager.config['max_position_size']
        assert adjusted_limits['max_leverage'] < risk_manager.config['max_leverage']
    
    def test_marginal_risk_contribution(self, risk_manager, portfolio_state, market_data):
        """Test marginal risk contribution calculation."""
        # Calculate how much risk a new position would add
        marginal_risk = risk_manager.calculate_marginal_risk(
            'TSLA',
            proposed_value=10000,
            portfolio_state=portfolio_state,
            returns=pd.Series(np.random.normal(0.002, 0.03, 252))
        )
        
        assert 'marginal_var' in marginal_risk
        assert 'risk_contribution' in marginal_risk
        assert marginal_risk['risk_contribution'] > 0