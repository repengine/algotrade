"""
Phase 1 Day 3: Risk Manager Tests

Tests for:
1. Order side type handling (string vs enum)
2. Position concentration checks
3. Additional risk validations
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
from enum import Enum

from src.core.risk import EnhancedRiskManager


class OrderSide(Enum):
    """Mock OrderSide enum for testing."""
    BUY = "buy"
    SELL = "sell"


class TestOrderSideTypeHandling:
    """Test handling of different order side formats."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        config = {
            "max_position_size": 0.20,
            "margin_buffer": 0.25,
            "max_daily_loss": 0.02
        }
        return EnhancedRiskManager(config)
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        portfolio.cash = 50000
        portfolio.daily_pnl = [0]
        return portfolio
    
    def test_order_with_enum_side(self, risk_manager, mock_portfolio):
        """Test order with enum side (has .value attribute)."""
        order = Mock()
        order.side = OrderSide.BUY
        order.quantity = 100
        order.price = 150.0
        
        # Should pass - position is 15% of equity
        assert risk_manager.check_order(order, mock_portfolio) is True
        
        # Large order should fail
        order.quantity = 200  # 30% of equity
        assert risk_manager.check_order(order, mock_portfolio) is False
    
    def test_order_with_string_side_lowercase(self, risk_manager, mock_portfolio):
        """Test order with lowercase string side."""
        order = Mock()
        order.side = "buy"
        order.quantity = 100
        order.price = 150.0
        
        assert risk_manager.check_order(order, mock_portfolio) is True
    
    def test_order_with_string_side_uppercase(self, risk_manager, mock_portfolio):
        """Test order with uppercase string side."""
        order = Mock()
        order.side = "BUY"
        order.quantity = 100
        order.price = 150.0
        
        assert risk_manager.check_order(order, mock_portfolio) is True
    
    def test_order_with_mixed_case_side(self, risk_manager, mock_portfolio):
        """Test order with mixed case string side."""
        order = Mock()
        order.side = "Buy"
        order.quantity = 100
        order.price = 150.0
        
        assert risk_manager.check_order(order, mock_portfolio) is True
    
    def test_order_with_sell_side_enum(self, risk_manager, mock_portfolio):
        """Test sell order with enum side."""
        order = Mock()
        order.side = OrderSide.SELL
        order.quantity = 100
        order.price = 150.0
        
        # Sell orders don't require cash check
        assert risk_manager.check_order(order, mock_portfolio) is True
    
    def test_order_with_sell_side_string(self, risk_manager, mock_portfolio):
        """Test sell order with string side."""
        order = Mock()
        order.side = "sell"
        order.quantity = 100
        order.price = 150.0
        
        assert risk_manager.check_order(order, mock_portfolio) is True
    
    def test_order_with_insufficient_cash(self, risk_manager, mock_portfolio):
        """Test buy order with insufficient cash."""
        mock_portfolio.cash = 10000  # Only $10k cash
        
        order = Mock()
        order.side = "buy"
        order.quantity = 100
        order.price = 150.0  # Needs $15k + 25% margin = $18.75k
        
        assert risk_manager.check_order(order, mock_portfolio) is False
    
    def test_order_with_daily_loss_exceeded(self, risk_manager, mock_portfolio):
        """Test order rejection due to daily loss limit."""
        mock_portfolio.daily_pnl = [-2500]  # 2.5% loss
        
        order = Mock()
        order.side = "buy"
        order.quantity = 10
        order.price = 150.0
        
        assert risk_manager.check_order(order, mock_portfolio) is False


class TestPositionConcentrationChecks:
    """Test position concentration limit checks."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with concentration limits."""
        config = {
            "concentration_limit": 0.30,  # 30% max per position
            "max_sector_exposure": 0.40   # 40% max per sector
        }
        rm = EnhancedRiskManager(config)
        # Set risk limits properly
        rm.risk_limits = {
            "concentration_limit": 0.30,
            "max_sector_exposure": 0.40
        }
        return rm
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio with positions."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        
        # Mock positions - use dict format expected by check_risk_compliance
        positions = {
            "AAPL": {"value": 25000, "sector": "Technology"},
            "MSFT": {"value": 20000, "sector": "Technology"},
            "JPM": {"value": 15000, "sector": "Finance"},
            "XOM": {"value": 10000, "sector": "Energy"}
        }
        portfolio.positions = positions
        
        # Helper method for tests
        def get_position(symbol):
            return positions.get(symbol)
        portfolio.get_position = get_position
        
        return portfolio
    
    def test_single_position_concentration_check(self, risk_manager, mock_portfolio):
        """Test single position concentration limit."""
        # check_risk_compliance returns tuple (bool, list)
        compliant, violations = risk_manager.check_risk_compliance(
            positions=mock_portfolio.positions,
            portfolio_value=mock_portfolio.current_equity
        )
        
        # AAPL is 25% of portfolio - should be OK with 30% limit
        assert not any("AAPL" in str(v) for v in violations)
        
        # Increase AAPL position to 35%
        mock_portfolio.positions["AAPL"]["value"] = 35000
        compliant, violations = risk_manager.check_risk_compliance(
            positions=mock_portfolio.positions,
            portfolio_value=mock_portfolio.current_equity
        )
        
        # Should now have a violation
        assert any("AAPL" in str(v) and "concentration" in str(v) for v in violations)
    
    def test_sector_concentration_check(self, risk_manager, mock_portfolio):
        """Test sector concentration limit."""
        # Technology sector is 45% (AAPL 25% + MSFT 20%)
        compliant, violations = risk_manager.check_risk_compliance(
            positions=mock_portfolio.positions,
            portfolio_value=mock_portfolio.current_equity
        )
        
        # Should have sector concentration violation
        assert any("Technology" in str(v) for v in violations)
    
    def test_multiple_concentration_violations(self, risk_manager, mock_portfolio):
        """Test multiple concentration violations."""
        # Increase JPM to 35% of portfolio
        mock_portfolio.positions["JPM"]["value"] = 35000
        
        compliant, violations = risk_manager.check_risk_compliance(
            positions=mock_portfolio.positions,
            portfolio_value=mock_portfolio.current_equity
        )
        
        # Should have violations for:
        # 1. JPM position concentration
        # 2. Technology sector concentration
        assert len(violations) >= 2
        assert any("JPM" in str(v) for v in violations)
        assert any("Technology" in str(v) for v in violations)
    
    def test_no_violations_within_limits(self, risk_manager, mock_portfolio):
        """Test no violations when within limits."""
        # Adjust positions to be within limits
        mock_portfolio.positions = {
            "AAPL": {"value": 20000, "sector": "Technology"},
            "MSFT": {"value": 15000, "sector": "Technology"},
            "JPM": {"value": 20000, "sector": "Finance"},
            "XOM": {"value": 15000, "sector": "Energy"},
            "GE": {"value": 10000, "sector": "Industrial"}
        }
        
        # Set proper risk metrics to avoid default violations
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.10  # Within limits
        risk_manager.risk_metrics.sharpe_ratio = 1.0  # Good Sharpe
        risk_manager.risk_metrics.value_at_risk = 0.01  # Low VaR
        risk_manager.risk_metrics.maximum_drawdown = 0.05  # Low drawdown
        risk_manager.risk_metrics.current_drawdown = 0.05
        
        compliant, violations = risk_manager.check_risk_compliance(
            positions=mock_portfolio.positions,
            portfolio_value=mock_portfolio.current_equity
        )
        
        # Should have no violations
        assert len(violations) == 0


class TestAdditionalRiskValidations:
    """Test additional risk validation features."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager with comprehensive limits."""
        config = {
            "max_portfolio_volatility": 0.20,
            "max_var": 10000,
            "max_drawdown": 0.15,
            "min_sharpe_ratio": 0.5,
            "max_correlation_risk": 0.80
        }
        return EnhancedRiskManager(config)
    
    def test_pre_trade_risk_validation(self, risk_manager):
        """Test comprehensive pre-trade risk validation."""
        # Create mock portfolio
        portfolio = Mock()
        portfolio.current_equity = 100000
        portfolio.positions = {}
        
        # Create mock order
        order = Mock()
        order.symbol = "AAPL"
        order.side = "buy"
        order.quantity = 100
        order.price = 150.0
        
        # Mock current risk metrics
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.15
        risk_manager.risk_metrics.value_at_risk = 0.08  # 8% of portfolio value = $8000
        risk_manager.risk_metrics.maximum_drawdown = 0.10
        risk_manager.risk_metrics.sharpe_ratio = 1.2
        
        # Should pass all checks
        result = risk_manager.validate_pre_trade_risk(order, portfolio)
        assert result["approved"] is True
        assert len(result["violations"]) == 0
    
    def test_volatility_limit_breach(self, risk_manager):
        """Test order rejection due to volatility limit."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        portfolio.positions = {}
        
        order = Mock()
        order.symbol = "TSLA"  # High volatility stock
        order.side = "buy"
        order.quantity = 100
        order.price = 200.0
        
        # Set high portfolio volatility
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.25  # Above 20% limit
        risk_manager.risk_metrics.value_at_risk = 8000
        risk_manager.risk_metrics.maximum_drawdown = 0.10
        risk_manager.risk_metrics.sharpe_ratio = 1.2
        
        result = risk_manager.validate_pre_trade_risk(order, portfolio)
        assert result["approved"] is False
        assert any("volatility" in v.lower() for v in result["violations"])
    
    def test_var_limit_breach(self, risk_manager):
        """Test order rejection due to VaR limit."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        portfolio.positions = {}
        
        order = Mock()
        order.symbol = "AAPL"
        order.side = "buy"
        order.quantity = 100
        order.price = 150.0
        
        # Set high VaR
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.15
        risk_manager.risk_metrics.value_at_risk = 0.12  # 12% of portfolio value = $12000
        risk_manager.risk_metrics.maximum_drawdown = 0.10
        risk_manager.risk_metrics.sharpe_ratio = 1.2
        
        result = risk_manager.validate_pre_trade_risk(order, portfolio)
        assert result["approved"] is False
        assert any("var" in v.lower() or "value at risk" in v.lower() for v in result["violations"])
    
    def test_drawdown_limit_breach(self, risk_manager):
        """Test order rejection due to drawdown limit."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        portfolio.positions = {}
        
        order = Mock()
        order.symbol = "AAPL"
        order.side = "buy"
        order.quantity = 100
        order.price = 150.0
        
        # Set high drawdown
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.15
        risk_manager.risk_metrics.value_at_risk = 8000
        risk_manager.risk_metrics.maximum_drawdown = 0.18  # Above 15% limit
        risk_manager.risk_metrics.sharpe_ratio = 1.2
        
        result = risk_manager.validate_pre_trade_risk(order, portfolio)
        assert result["approved"] is False
        assert any("drawdown" in v.lower() for v in result["violations"])
    
    def test_low_sharpe_ratio_warning(self, risk_manager):
        """Test warning for low Sharpe ratio."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        portfolio.positions = {}
        
        order = Mock()
        order.symbol = "AAPL"
        order.side = "buy"
        order.quantity = 100
        order.price = 150.0
        
        # Set low Sharpe ratio
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.15
        risk_manager.risk_metrics.value_at_risk = 0.08  # 8% of portfolio value
        risk_manager.risk_metrics.maximum_drawdown = 0.10
        risk_manager.risk_metrics.sharpe_ratio = 0.3  # Below 0.5 threshold
        
        result = risk_manager.validate_pre_trade_risk(order, portfolio)
        # Low Sharpe is a warning, not a rejection
        assert result["approved"] is True
        assert any("sharpe" in w.lower() for w in result.get("warnings", []))
    
    def test_correlation_risk_check(self, risk_manager):
        """Test correlation risk validation."""
        portfolio = Mock()
        portfolio.current_equity = 100000
        
        # Mock highly correlated positions
        positions = {
            "AAPL": Mock(value=20000, sector="Technology"),
            "MSFT": Mock(value=20000, sector="Technology"),
            "GOOGL": Mock(value=20000, sector="Technology")
        }
        portfolio.positions = positions
        
        order = Mock()
        order.symbol = "META"  # Another tech stock
        order.side = "buy"
        order.quantity = 100
        order.price = 300.0
        
        # Mock correlation calculation
        risk_manager.risk_metrics = Mock()
        risk_manager.risk_metrics.portfolio_volatility = 0.15
        risk_manager.risk_metrics.value_at_risk = 8000
        risk_manager.risk_metrics.maximum_drawdown = 0.10
        risk_manager.risk_metrics.sharpe_ratio = 1.2
        
        # Set high correlation
        risk_manager.calculate_order_correlation_risk = Mock(return_value=0.85)
        
        result = risk_manager.validate_pre_trade_risk(order, portfolio)
        assert result["approved"] is False
        assert any("correlation" in v.lower() for v in result["violations"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])