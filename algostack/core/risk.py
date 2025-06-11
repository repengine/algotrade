#!/usr/bin/env python3
"""Risk management with volatility scaling and portfolio controls."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics."""

    value_at_risk: float  # 95% VaR
    conditional_var: float  # CVaR/Expected Shortfall
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    maximum_drawdown: float
    current_drawdown: float
    downside_deviation: float
    portfolio_volatility: float
    portfolio_beta: float = 1.0
    correlation_risk: float = 0.0


class EnhancedRiskManager:
    """Advanced risk management system with multiple risk measures."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.risk_limits = {
            "max_var_95": config.get("max_var_95", 0.02),  # 2% daily VaR
            "max_portfolio_volatility": config.get("target_vol", 0.10),
            "max_position_size": config.get("max_position_size", 0.20),
            "max_sector_exposure": config.get("max_sector_exposure", 0.40),
            "max_drawdown": config.get("max_drawdown", 0.15),
            "max_correlation": config.get("max_correlation", 0.70),
            "min_sharpe": config.get("min_sharpe", 0.5),
            "concentration_limit": config.get("concentration_limit", 0.60),
        }

        # Risk tracking
        self.returns_history = pd.Series(dtype=float)
        self.correlation_matrix = pd.DataFrame()
        self.sector_exposures: dict[str, float] = {}
        self.risk_metrics: Optional[RiskMetrics] = None
        self.is_risk_on = True

        # Volatility forecasting
        self.use_garch = config.get("use_garch", False)
        self.vol_lookback = config.get("vol_lookback", 60)

        # Additional attributes expected by tests
        self.historical_metrics: list[RiskMetrics] = []
        self.current_regime = "NORMAL"

    def calculate_risk_metrics(
        self,
        portfolio_returns: Union[pd.Series, pd.DataFrame],
        portfolio_value: float = 100000,
        lookback_days: int = 252,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Handle DataFrame input by converting to portfolio returns
        if isinstance(portfolio_returns, pd.DataFrame):
            # Simple equal-weight portfolio
            portfolio_returns = portfolio_returns.mean(axis='columns')

        if len(portfolio_returns) < 30:
            return self._default_risk_metrics()

        # Basic statistics
        daily_returns = portfolio_returns.dropna()
        if daily_returns.empty:
            return self._default_risk_metrics()
            
        mean_return = float(daily_returns.mean())
        std_return = float(daily_returns.std())

        # Value at Risk (95% confidence) - return as positive value
        var_95 = float(abs(np.percentile(daily_returns, 5)))

        # Conditional VaR (Expected Shortfall) - return as positive value
        percentile_5 = float(np.percentile(daily_returns, 5))
        cvar_returns = daily_returns[daily_returns <= percentile_5]
        cvar = float(abs(cvar_returns.mean())) if not cvar_returns.empty else 0.0

        # Sharpe Ratio (annualized)
        risk_free_rate = self.config.get("risk_free_rate", 0.02) / 252
        sharpe = float(
            (mean_return - risk_free_rate) / std_return * np.sqrt(252)
            if std_return > 0
            else 0.0
        )

        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = float(
            downside_returns.std() if not downside_returns.empty else std_return
        )
        sortino = float(
            (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
            if downside_std > 0
            else 0.0
        )

        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).fillna(0)
        max_drawdown = float(abs(drawdown.min()))  # Return as positive value
        current_drawdown = float(abs(drawdown.iloc[-1]) if not drawdown.empty else 0.0)  # Return as positive value

        # Calmar Ratio
        annual_return = mean_return * 252
        calmar = float(annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0)

        # Portfolio Volatility (annualized)
        portfolio_vol = float(std_return * np.sqrt(252))

        # Beta (if benchmark provided)
        beta = self._calculate_beta(daily_returns, benchmark_returns)

        # Ensure downside_std is annualized for RiskMetrics
        annualized_downside_std = float(downside_std * np.sqrt(252))

        return RiskMetrics(
            value_at_risk=var_95,
            conditional_var=cvar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            maximum_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            downside_deviation=annualized_downside_std,
            portfolio_volatility=portfolio_vol,
            portfolio_beta=beta,
        )

    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when insufficient data."""
        return RiskMetrics(
            value_at_risk=0.0,
            conditional_var=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            maximum_drawdown=0.0,
            current_drawdown=0.0,
            downside_deviation=0.15,
            portfolio_volatility=0.15,
            portfolio_beta=1.0,
        )

    def _calculate_beta(
        self, 
        daily_returns: pd.Series, 
        benchmark_returns: Optional[pd.Series]
    ) -> float:
        """Calculate portfolio beta against benchmark with robust error handling."""
        if benchmark_returns is None or len(benchmark_returns) == 0 or len(daily_returns) == 0:
            return 1.0
        
        try:
            # Align the series and find common dates
            aligned_benchmark = benchmark_returns.reindex(daily_returns.index).dropna()
            # Convert the second index to a list to satisfy stricter type hints for intersection
            common_index = daily_returns.index.intersection(list(aligned_benchmark.index))
            
            if len(common_index) < 30:  # Minimum data points
                return 1.0
            
            # Get aligned data
            returns_aligned = daily_returns.loc[common_index]
            benchmark_aligned = aligned_benchmark.loc[common_index]
            
            # Option 1: Covariance-based calculation
            covariance_result = returns_aligned.cov(benchmark_aligned)
            variance_result = benchmark_aligned.var()
            
            # Type guard: Ensure results are numeric before further processing
            if not isinstance(covariance_result, (int, float, complex, np.number)):
                logger.warning(
                    f"Covariance result is not a number: {covariance_result} (type: {type(covariance_result)}). Defaulting beta to 1.0."
                )
                return 1.0
            
            if not isinstance(variance_result, (int, float, complex, np.number)):
                logger.warning(
                    f"Variance result is not a number: {variance_result} (type: {type(variance_result)}). Defaulting beta to 1.0."
                )
                return 1.0
            
            # Handle potential NaN or zero variance
            if pd.isna(covariance_result) or pd.isna(variance_result) or variance_result == 0:
                return 1.0
                
            # Use safe conversion method
            covariance = self._safe_float_conversion(covariance_result)
            variance = self._safe_float_conversion(variance_result)
            
            # Final check on converted variance
            if variance <= 0:
                return 1.0
            
            return covariance / variance
            
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating beta: {e}")
            return 1.0

    def _safe_float_conversion(self, value: Any) -> float:
        """Safely convert pandas/numpy values to Python float."""
        try:
            # Handle NaN/None cases
            if pd.isna(value):
                return 0.0
            
            # For pandas Scalar types, use .item() to get Python scalar
            if hasattr(value, 'item'):
                return float(value.item())
            
            # For numpy arrays with single values
            if hasattr(value, 'size') and value.size == 1:
                return float(value.flat[0])
            
            # Direct conversion for basic types
            return float(value)
            
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _calculate_beta_correlation(
        self, 
        daily_returns: pd.Series, 
        benchmark_returns: Optional[pd.Series]
    ) -> float:
        """Calculate portfolio beta using correlation method (more stable)."""
        if benchmark_returns is None or len(benchmark_returns) == 0 or len(daily_returns) == 0:
            return 1.0
        
        try:
            # Align the series
            aligned_data = daily_returns.align(benchmark_returns, join='inner')
            returns_aligned = aligned_data[0]
            benchmark_aligned = aligned_data[1]
            
            if len(returns_aligned) < 30:
                return 1.0
            
            # Calculate using correlation method
            correlation = returns_aligned.corr(benchmark_aligned)
            portfolio_std = returns_aligned.std()
            benchmark_std = benchmark_aligned.std()
            
            # Safely convert to floats
            corr_val = self._safe_float_conversion(correlation)
            port_std = self._safe_float_conversion(portfolio_std) 
            bench_std = self._safe_float_conversion(benchmark_std)
            
            if bench_std > 0 and port_std > 0 and not pd.isna(corr_val):
                return corr_val * (port_std / bench_std)
            
            return 1.0
            
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating correlation-based beta: {e}")
            return 1.0

    def forecast_volatility(self, returns: pd.Series) -> float:
        """Forecast next-period volatility."""
        # Need at least 2 data points for std calculation
        if len(returns) < 2:
            return 0.15
        
        # Calculate standard deviation safely
        try:
            std_value = self._safe_float_conversion(returns.std())
            if std_value <= 0:
                return 0.15
        except (TypeError, ValueError, AttributeError):
            return 0.15
        
        # Calculate annualized volatility
        annual_vol = std_value * np.sqrt(252)
        
        # Return simple volatility if not using GARCH or insufficient data
        if not self.use_garch or len(returns) < self.vol_lookback:
            return annual_vol

        # GARCH/EWMA calculation
        try:
            lambda_param = 0.94
            squared_returns = returns**2
            ewma_var = squared_returns.ewm(alpha=1 - lambda_param, adjust=False).mean()
            last_var = ewma_var.iloc[-1]
            
            if pd.notna(last_var) and last_var > 0:
                return float(np.sqrt(last_var * 252))
        except Exception as e:
            logger.warning(f"GARCH calculation failed: {e}")
        
        return annual_vol

    def calculate_position_var(
        self,
        position_returns: pd.Series,
        position_value: float,
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Value at Risk for a position."""
        # Calculate VaR from actual returns
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(position_returns, percentile)
        var = abs(var_return * position_value)
        return float(var)

    def portfolio_optimization(
        self,
        returns: pd.DataFrame,
        current_weights: pd.Series,
        target_return: Optional[float] = None,
        risk_parity: bool = False,
    ) -> pd.Series:
        """
        Optimize portfolio weights using mean-variance optimization.
        This is a simplified version - for production use cvxpy or similar.
        """
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean()
        covariance_matrix = returns.cov()
        n_assets = len(expected_returns)

        # Risk aversion parameter
        risk_aversion = self.config.get("risk_aversion", 2.0)

        # Objective: maximize expected return - risk_aversion * variance
        # Subject to: sum(weights) = 1, 0 <= weights <= max_position_size

        from scipy.optimize import minimize

        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return float(-(portfolio_return - risk_aversion * portfolio_variance))

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Sum to 1

        # Bounds
        bounds = [(0, self.risk_limits["max_position_size"]) for _ in range(n_assets)]

        # Initial guess
        initial_weights = (
            current_weights.values
            if not current_weights.empty
            else np.ones(n_assets) / n_assets
        )

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            logger.warning("Portfolio optimization failed, using equal weights")
            return pd.Series(1.0 / n_assets, index=expected_returns.index)

    def check_risk_compliance(
        self,
        positions: Optional[dict] = None,
        portfolio_value: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        proposed_trade: Optional[Signal] = None,
        portfolio: Optional[Any] = None,
    ) -> tuple[bool, list[str]]:
        """Check if portfolio (with proposed trade) meets risk limits."""
        violations = []

        # Calculate current metrics
        if portfolio and hasattr(portfolio, "get_returns_history"):
            returns = portfolio.get_returns_history()
            if not returns.empty:  # Add check for empty returns
                self.risk_metrics = self.calculate_risk_metrics(returns)
            else:
                logger.warning("Portfolio returns history is empty, cannot calculate risk metrics.")
                self.risk_metrics = self._default_risk_metrics()
        elif not self.risk_metrics:  # If not calculated and not already set
            logger.warning("No portfolio data to calculate risk metrics, using defaults.")
            self.risk_metrics = self._default_risk_metrics()
            # Return early if using defaults - no violations on default metrics
            return (True, [])

        # Check VaR limit
        if (
            self.risk_metrics
            and abs(self.risk_metrics.value_at_risk) > self.risk_limits["max_var_95"]
        ):
            violations.append(
                f"VaR {abs(self.risk_metrics.value_at_risk):.1%} exceeds limit {self.risk_limits['max_var_95']:.1%}"
            )

        # Check volatility limit
        if (
            self.risk_metrics
            and self.risk_metrics.portfolio_volatility
            > self.risk_limits["max_portfolio_volatility"]
        ):
            violations.append(
                f"Portfolio volatility {self.risk_metrics.portfolio_volatility:.1%} exceeds limit"
            )

        # Check drawdown
        if (
            self.risk_metrics
            and abs(self.risk_metrics.current_drawdown)
            > self.risk_limits["max_drawdown"]
        ):
            violations.append(
                f"Drawdown {abs(self.risk_metrics.current_drawdown):.1%} exceeds limit"
            )

        # Check Sharpe ratio
        if (
            self.risk_metrics
            and self.risk_metrics.sharpe_ratio < self.risk_limits["min_sharpe"]
        ):
            violations.append(
                f"Sharpe ratio {self.risk_metrics.sharpe_ratio:.2f} below minimum"
            )

        # Check proposed trade impact
        if proposed_trade and proposed_trade.direction != "FLAT":
            # Estimate position VaR
            position_value = (portfolio_value or 100000) * 0.1  # Assume 10% allocation
            # Use simple VaR calculation based on volatility
            volatility = (
                self.forecast_volatility(self.returns_history)
                if len(self.returns_history) > 0
                else 0.15
            )
            # 95% VaR approximation: 1.65 * volatility * position_value
            position_var = 1.65 * (volatility / np.sqrt(252)) * position_value

            # Check if adding position would breach VaR limit
            current_var = (
                abs(self.risk_metrics.value_at_risk * (portfolio_value or 100000))
                if self.risk_metrics
                else 0
            )
            total_var = current_var + position_var
            portfolio_var_pct = total_var / (portfolio_value or 100000)

            if portfolio_var_pct > self.risk_limits["max_var_95"]:
                violations.append(
                    f"Proposed trade would increase VaR to {portfolio_var_pct:.1%}"
                )

        return len(violations) == 0, violations

    def calculate_stress_scenarios(
        self, portfolio: Any, scenarios: Optional[dict[str, dict[str, float]]] = None
    ) -> dict[str, float]:
        """Calculate portfolio impact under stress scenarios."""
        if scenarios is None:
            # Default stress scenarios
            scenarios = {
                "market_crash": {"equity": -0.20, "volatility": 2.0},
                "flash_crash": {"equity": -0.10, "volatility": 3.0},
                "correlation_breakdown": {"correlation": 1.0},
                "liquidity_crisis": {"spread": 0.05, "volume": 0.2},
            }

        stress_results: dict[str, float] = {}

        for scenario_name, shocks in scenarios.items():
            scenario_pnl = 0

            # Apply equity shock
            if "equity" in shocks:
                for position in portfolio.positions.values():
                    if position.direction == "LONG":
                        scenario_pnl += position.market_value * shocks["equity"]
                    else:  # SHORT positions benefit from drops
                        scenario_pnl -= position.market_value * shocks["equity"]

            # Apply volatility shock (increases likely losses)
            if "volatility" in shocks:
                vol_impact = -abs(
                    portfolio.current_equity * 0.01 * shocks["volatility"]
                )
                scenario_pnl += vol_impact

            # Apply spread costs
            if "spread" in shocks:
                spread_cost = sum(
                    pos.market_value * shocks["spread"]
                    for pos in portfolio.positions.values()
                )
                scenario_pnl -= spread_cost

            stress_results[scenario_name] = scenario_pnl

        return stress_results

    def get_risk_adjusted_sizes(
        self, signals: list[Signal], portfolio: Any, market_data: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Calculate risk-adjusted position sizes for signals."""
        risk_sizes = {}

        # Get portfolio risk metrics

        for signal in signals:
            if signal.direction == "FLAT":
                continue

            symbol = signal.symbol

            # Get asset volatility
            if symbol in market_data and "returns" in market_data[symbol].columns:
                asset_vol = self.forecast_volatility(market_data[symbol]["returns"])
            else:
                asset_vol = 0.20  # Default 20%

            # Base allocation from volatility scaling
            vol_scaled_size = self.risk_limits["max_portfolio_volatility"] / asset_vol

            # Adjust for correlation
            if (
                not self.correlation_matrix.empty
                and symbol in self.correlation_matrix.index
            ):
                avg_correlation = self.correlation_matrix[symbol].mean()
                correlation_adj = 1 - (
                    avg_correlation * 0.5
                )  # Reduce size for high correlation
                vol_scaled_size *= correlation_adj

            # Apply signal strength
            vol_scaled_size *= abs(signal.strength)

            # Apply position limit
            vol_scaled_size = min(
                vol_scaled_size, self.risk_limits["max_position_size"]
            )

            # Apply portfolio-level scaling if in risk-off mode
            if not self.is_risk_on:
                vol_scaled_size *= 0.5  # Half size in risk-off

            risk_sizes[symbol] = vol_scaled_size

        return risk_sizes

    def update_risk_state(self, portfolio: Any) -> None:
        """Update risk manager state based on portfolio."""
        # Update returns history
        if hasattr(portfolio, "get_returns_history"):
            self.returns_history = portfolio.get_returns_history()

        # Recalculate risk metrics
        self.risk_metrics = self.calculate_risk_metrics(self.returns_history)

        # Update risk-on/risk-off state
        if self.risk_metrics:
            # Enter risk-off if drawdown exceeds threshold
            if (
                abs(self.risk_metrics.current_drawdown)
                > self.risk_limits["max_drawdown"] * 0.8
            ):
                self.is_risk_on = False
                logger.warning("Entering risk-off mode due to drawdown")
            elif (
                abs(self.risk_metrics.current_drawdown)
                < self.risk_limits["max_drawdown"] * 0.5
            ):
                self.is_risk_on = True

    def get_risk_report(self) -> dict[str, Any]:
        """Generate comprehensive risk report."""
        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "risk_state": "RISK_ON" if self.is_risk_on else "RISK_OFF",
            "metrics": None,
            "limits": self.risk_limits,
            "violations": [],
        }

        if self.risk_metrics:
            report["metrics"] = {
                "var_95": f"{self.risk_metrics.value_at_risk:.2%}",
                "cvar": f"{self.risk_metrics.conditional_var:.2%}",
                "sharpe_ratio": f"{self.risk_metrics.sharpe_ratio:.2f}",
                "sortino_ratio": f"{self.risk_metrics.sortino_ratio:.2f}",
                "max_drawdown": f"{self.risk_metrics.maximum_drawdown:.2%}",
                "current_drawdown": f"{self.risk_metrics.current_drawdown:.2%}",
                "portfolio_volatility": f"{self.risk_metrics.portfolio_volatility:.2%}",
            }

            # Check for violations
            if abs(self.risk_metrics.value_at_risk) > self.risk_limits["max_var_95"]:
                report["violations"].append("VaR limit exceeded")
            if (
                abs(self.risk_metrics.current_drawdown)
                > self.risk_limits["max_drawdown"]
            ):
                report["violations"].append("Drawdown limit exceeded")
            if (
                self.risk_metrics.portfolio_volatility
                > self.risk_limits["max_portfolio_volatility"]
            ):
                report["violations"].append("Volatility limit exceeded")

        return report

    def update_regime(self, returns: Union[pd.Series, pd.DataFrame]) -> None:
        """Update current market regime based on volatility."""
        # Handle DataFrame input by converting to portfolio returns
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis='columns')

        if returns.empty or len(returns) < 2:  # Need at least 2 points for std
            self.current_regime = "NORMAL"  # Default if not enough data
            return
            
        volatility = float(returns.std() * np.sqrt(252))
        if pd.isna(volatility):  # Handle NaN volatility
            self.current_regime = "NORMAL"
        elif volatility > 0.25:
            self.current_regime = "HIGH_VOL"
        elif volatility < 0.10:
            self.current_regime = "LOW_VOL"
        else:
            self.current_regime = "NORMAL"

    def update_historical_metrics(self, metrics: RiskMetrics) -> None:
        """Update historical metrics storage."""
        self.historical_metrics.append(metrics)
        # Keep only last 100 metrics
        if len(self.historical_metrics) > 100:
            self.historical_metrics = self.historical_metrics[-100:]

    def get_average_metrics(self, lookback: int = 30) -> Optional[RiskMetrics]:
        """Get average of historical metrics over lookback period."""
        if len(self.historical_metrics) < lookback:
            return None
        recent_metrics = self.historical_metrics[-lookback:]
        # Return most recent for simplicity
        return recent_metrics[-1] if recent_metrics else None

    def can_trade(self, strategy: Any) -> bool:
        """Check if a strategy is allowed to trade based on risk limits."""
        # Check if risk system is enabled
        if not self.is_risk_on:
            return False

        # Check if we're within drawdown limits
        if (
            self.risk_metrics
            and self.risk_metrics.current_drawdown > self.risk_limits["max_drawdown"]
        ):
            logger.warning(
                f"Trading disabled - drawdown {self.risk_metrics.current_drawdown:.2%} exceeds limit"
            )
            return False

        # Check if volatility is acceptable
        if (
            self.risk_metrics
            and self.risk_metrics.portfolio_volatility
            > self.risk_limits["max_portfolio_volatility"]
        ):
            logger.warning(
                f"Trading disabled - volatility {self.risk_metrics.portfolio_volatility:.2%} exceeds limit"
            )
            return False

        return True

    def size_orders(self, signals: dict[str, Any], portfolio: Any) -> dict[str, Any]:
        """Size orders based on risk constraints and portfolio state."""
        sized_orders: dict[str, Any] = {}

        for strategy_name, strategy_signals in signals.items():
            if not strategy_signals:
                continue

            for symbol, signal in strategy_signals.items():
                # Calculate position size based on volatility
                position_size = self._calculate_position_size(signal, portfolio)

                # Apply risk limits
                position_size = self._apply_risk_limits(
                    position_size, symbol, portfolio
                )

                if position_size > 0:
                    sized_orders.setdefault(strategy_name, {})[symbol] = {
                        "signal": signal,
                        "size": position_size,
                    }

        return sized_orders

    def _calculate_position_size(self, signal: Any, portfolio: Any) -> float:
        """Calculate position size based on Kelly criterion or fixed sizing."""
        # Simple fixed position sizing for now
        base_size = self.config.get("base_position_size", 0.02)  # 2% of portfolio

        # Adjust for signal strength if available
        if hasattr(signal, "strength"):
            base_size *= signal.strength

        return float(base_size)

    def _apply_risk_limits(self, size: float, symbol: str, portfolio: Any) -> float:
        """Apply risk limits to position size."""
        # Maximum position size
        max_size = self.risk_limits["max_position_size"]
        size = min(size, max_size)

        # Check concentration limits
        if hasattr(portfolio, "get_position_weight"):
            current_weight = portfolio.get_position_weight(symbol)
            if current_weight + size > max_size:
                size = max(0, max_size - current_weight)

        return size

    def trigger_kill_switch(self) -> None:
        """Emergency stop - disable all trading."""
        logger.critical("RISK KILL SWITCH ACTIVATED - All trading disabled")
        self.is_risk_on = False

        # Additional emergency actions could be added here:
        # - Send alerts
        # - Close all positions
        # - Save state to disk

    def check_limits(self) -> list[dict[str, Any]]:
        """
        Check all risk limits and return violations.
        
        Returns:
            List of risk limit violations, each containing:
            - limit_type: Type of limit violated
            - current_value: Current value
            - limit_value: Limit that was exceeded
            - severity: 'warning' or 'critical'
        """
        violations = []
        
        # Check drawdown
        if self.risk_metrics:
            current_dd = self.risk_metrics.current_drawdown
            if abs(current_dd) > self.risk_limits["max_drawdown"]:
                violations.append({
                    "limit_type": "max_drawdown",
                    "current_value": current_dd,
                    "limit_value": self.risk_limits["max_drawdown"],
                    "severity": "critical"
                })
            
            # Check VaR
            if self.risk_metrics.value_at_risk > self.risk_limits["max_var_95"]:
                violations.append({
                    "limit_type": "value_at_risk",
                    "current_value": self.risk_metrics.value_at_risk,
                    "limit_value": self.risk_limits["max_var_95"],
                    "severity": "warning"
                })
            
            # Check volatility
            if self.risk_metrics.portfolio_volatility > self.risk_limits["max_portfolio_volatility"]:
                violations.append({
                    "limit_type": "portfolio_volatility",
                    "current_value": self.risk_metrics.portfolio_volatility,
                    "limit_value": self.risk_limits["max_portfolio_volatility"],
                    "severity": "warning"
                })
        
        return violations


# Alias for backward compatibility
RiskManager = EnhancedRiskManager

# Additional aliases and placeholder classes for test compatibility
RiskLimits = dict  # Placeholder
PositionRisk = dict  # Placeholder
PortfolioRisk = dict  # Placeholder
RiskAlert = dict  # Placeholder
RiskViolation = dict  # Placeholder
