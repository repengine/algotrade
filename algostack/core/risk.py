#!/usr/bin/env python3
"""Risk management with volatility scaling and portfolio controls."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

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
        self.sector_exposures = {}
        self.risk_metrics = None
        self.is_risk_on = True

        # Volatility forecasting
        self.use_garch = config.get("use_garch", False)
        self.vol_lookback = config.get("vol_lookback", 60)

        # Additional attributes expected by tests
        self.historical_metrics = []
        self.current_regime = "NORMAL"

    def calculate_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: float = 100000,
        lookback_days: int = 252,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Handle DataFrame input by converting to portfolio returns
        if isinstance(portfolio_returns, pd.DataFrame):
            # Simple equal-weight portfolio
            portfolio_returns = portfolio_returns.mean(axis=1)

        if len(portfolio_returns) < 30:
            return self._default_risk_metrics()

        # Basic statistics
        daily_returns = portfolio_returns.dropna()
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()

        # Value at Risk (95% confidence) - return as positive value
        var_95 = abs(np.percentile(daily_returns, 5))

        # Conditional VaR (Expected Shortfall) - return as positive value
        percentile_5 = np.percentile(daily_returns, 5)
        cvar = abs(daily_returns[daily_returns <= percentile_5].mean())

        # Sharpe Ratio (annualized)
        risk_free_rate = self.config.get("risk_free_rate", 0.02) / 252
        sharpe = (
            (mean_return - risk_free_rate) / std_return * np.sqrt(252)
            if std_return > 0
            else 0
        )

        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = (
            downside_returns.std() if len(downside_returns) > 0 else std_return
        )
        sortino = (
            (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
            if downside_std > 0
            else 0
        )

        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).fillna(0)
        max_drawdown = abs(drawdown.min())  # Return as positive value
        current_drawdown = abs(drawdown.iloc[-1])  # Return as positive value

        # Calmar Ratio
        annual_return = mean_return * 252
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Portfolio Volatility (annualized)
        portfolio_vol = std_return * np.sqrt(252)

        # Beta (if benchmark provided)
        beta = 1.0
        if benchmark_returns is not None and len(benchmark_returns) >= len(
            daily_returns
        ):
            aligned_benchmark = benchmark_returns.reindex(daily_returns.index)
            covariance = daily_returns.cov(aligned_benchmark)
            benchmark_var = aligned_benchmark.var()
            beta = covariance / benchmark_var if benchmark_var > 0 else 1.0

        return RiskMetrics(
            value_at_risk=var_95,
            conditional_var=cvar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            maximum_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            downside_deviation=downside_std * np.sqrt(252),
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

    def forecast_volatility(self, returns: pd.Series) -> float:
        """Forecast next-period volatility."""
        if len(returns) < self.vol_lookback:
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.15

        if self.use_garch:
            # Simplified EWMA approximation of GARCH
            # For production, use arch library
            lambda_param = 0.94
            squared_returns = returns**2

            # EWMA variance
            ewma_var = squared_returns.ewm(alpha=1 - lambda_param, adjust=False).mean()
            forecast_vol = np.sqrt(ewma_var.iloc[-1] * 252)
        else:
            # Simple rolling window
            rolling_vol = returns.rolling(self.vol_lookback).std()
            forecast_vol = rolling_vol.iloc[-1] * np.sqrt(252)

        return forecast_vol

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
        return var

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
            return -(portfolio_return - risk_aversion * portfolio_variance)

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
            self.risk_metrics = self.calculate_risk_metrics(returns)

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
        self, portfolio, scenarios: Optional[dict[str, dict[str, float]]] = None
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

        stress_results = {}

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
        self, signals: list[Signal], portfolio, market_data: dict[str, pd.DataFrame]
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

    def update_risk_state(self, portfolio) -> None:
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
        report = {
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

    def update_regime(self, returns: pd.Series) -> None:
        """Update current market regime based on volatility."""
        # Handle DataFrame input by converting to portfolio returns
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)

        volatility = returns.std() * np.sqrt(252)
        if volatility > 0.25:
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
        sized_orders = {}

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

        return base_size

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
