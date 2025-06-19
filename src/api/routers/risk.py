from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ..dependencies import get_trading_engine, get_risk_manager, require_admin_auth
from ..models import RiskMetrics, RiskLimits, RiskLimitUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/risk/metrics", response_model=RiskMetrics)
async def get_risk_metrics(
    engine = Depends(get_trading_engine)
) -> RiskMetrics:
    """Get current risk metrics - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        risk_manager = engine.risk_manager
        portfolio = engine.portfolio
        
        # Calculate current risk metrics
        metrics = risk_manager.calculate_portfolio_metrics()
        
        # Get position concentration
        position_concentrations = {}
        total_value = portfolio.total_equity
        
        for position in portfolio.positions.values():
            concentration = abs(position.market_value) / total_value * 100
            position_concentrations[position.symbol] = concentration
        
        # Get sector concentration (would need sector mapping)
        sector_concentrations = {}  # Would calculate from positions and sector data
        
        # Check limit usage
        limits = risk_manager.config
        limit_usage = {
            "position_size": max(position_concentrations.values()) / limits.max_position_size if position_concentrations else 0.0,
            "portfolio_var": metrics.get("var_95", 0) / limits.max_portfolio_var,
            "daily_loss": abs(metrics.get("daily_pnl", 0)) / limits.max_daily_loss if metrics.get("daily_pnl", 0) < 0 else 0.0,
            "sector_concentration": max(sector_concentrations.values()) / limits.max_sector_concentration if sector_concentrations else 0.0
        }
        
        # Risk warnings
        warnings = []
        if limit_usage["position_size"] > 0.8:
            warnings.append("Approaching position size limit")
        if limit_usage["portfolio_var"] > 0.8:
            warnings.append("Approaching VaR limit")
        if limit_usage["daily_loss"] > 0.8:
            warnings.append("Approaching daily loss limit")
        if metrics.get("sharpe_ratio", 0) < 0.5:
            warnings.append("Low Sharpe ratio")
        
        return RiskMetrics(
            portfolio_var_95=metrics.get("var_95", 0.0),
            portfolio_var_99=metrics.get("var_99", 0.0),
            portfolio_volatility=metrics.get("volatility", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            current_drawdown=metrics.get("current_drawdown", 0.0),
            position_concentrations=position_concentrations,
            sector_concentrations=sector_concentrations,
            correlation_matrix=metrics.get("correlation_matrix", {}),
            limit_usage=limit_usage,
            warnings=warnings,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/limits", response_model=RiskLimits)
async def get_risk_limits(
    engine = Depends(get_trading_engine)
) -> RiskLimits:
    """Get configured risk limits - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        risk_manager = engine.risk_manager
        config = risk_manager.config
        
        return RiskLimits(
            max_position_size=config.max_position_size,
            max_portfolio_var=config.max_portfolio_var,
            max_daily_loss=config.max_daily_loss,
            max_sector_concentration=config.max_sector_concentration,
            max_correlation=config.max_correlation,
            max_leverage=config.max_leverage,
            stop_loss_pct=config.stop_loss_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            circuit_breaker_threshold=getattr(config, "circuit_breaker_threshold", 0.2),
            last_updated=getattr(config, "last_updated", datetime.utcnow())
        )
        
    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/risk/limits", response_model=RiskLimits, dependencies=[Depends(require_admin_auth)])
async def update_risk_limits(
    limits: RiskLimitUpdate,
    engine = Depends(get_trading_engine)
) -> RiskLimits:
    """Update risk limits (requires admin auth) - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        risk_manager = engine.risk_manager
        config = risk_manager.config
        
        # Validate new limits
        updates = limits.dict(exclude_unset=True)
        
        # Sanity checks
        if "max_position_size" in updates:
            if not 0.01 <= updates["max_position_size"] <= 1.0:
                raise HTTPException(status_code=400, detail="max_position_size must be between 1% and 100%")
        
        if "max_daily_loss" in updates:
            if not 0.001 <= updates["max_daily_loss"] <= 0.5:
                raise HTTPException(status_code=400, detail="max_daily_loss must be between 0.1% and 50%")
        
        if "max_leverage" in updates:
            if not 0.1 <= updates["max_leverage"] <= 10.0:
                raise HTTPException(status_code=400, detail="max_leverage must be between 0.1 and 10")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.last_updated = datetime.utcnow()
        
        # Log the change
        logger.warning(f"Risk limits updated: {updates}")
        
        # Return updated limits
        return RiskLimits(
            max_position_size=config.max_position_size,
            max_portfolio_var=config.max_portfolio_var,
            max_daily_loss=config.max_daily_loss,
            max_sector_concentration=config.max_sector_concentration,
            max_correlation=config.max_correlation,
            max_leverage=config.max_leverage,
            stop_loss_pct=config.stop_loss_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            circuit_breaker_threshold=getattr(config, "circuit_breaker_threshold", 0.2),
            last_updated=config.last_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/exposure")
async def get_risk_exposure(
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Get current risk exposure by various dimensions - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        portfolio = engine.portfolio
        total_equity = portfolio.total_equity
        
        # Calculate exposures
        exposures = {
            "total_exposure": sum(abs(p.market_value) for p in portfolio.positions.values()),
            "long_exposure": sum(p.market_value for p in portfolio.positions.values() if p.quantity > 0),
            "short_exposure": sum(abs(p.market_value) for p in portfolio.positions.values() if p.quantity < 0),
            "net_exposure": sum(p.market_value for p in portfolio.positions.values()),
            "gross_leverage": sum(abs(p.market_value) for p in portfolio.positions.values()) / total_equity,
            "net_leverage": abs(sum(p.market_value for p in portfolio.positions.values())) / total_equity,
        }
        
        # By symbol
        symbol_exposure = {}
        for position in portfolio.positions.values():
            symbol_exposure[position.symbol] = {
                "market_value": position.market_value,
                "percent_of_portfolio": position.market_value / total_equity * 100,
                "quantity": position.quantity
            }
        
        # By sector (would need sector mapping)
        sector_exposure = {}
        
        return {
            "summary": exposures,
            "by_symbol": symbol_exposure,
            "by_sector": sector_exposure,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/stress-test")
async def run_stress_test(
    scenarios: Dict[str, float] = Body(..., example={"SPY": -0.10, "QQQ": -0.15}),
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Run stress test with custom scenarios - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        portfolio = engine.portfolio
        current_equity = portfolio.total_equity
        
        # Calculate impact of scenarios
        total_impact = 0.0
        position_impacts = {}
        
        for position in portfolio.positions.values():
            if position.symbol in scenarios:
                price_change = scenarios[position.symbol]
                impact = position.market_value * price_change
                total_impact += impact
                position_impacts[position.symbol] = {
                    "current_value": position.market_value,
                    "scenario_change": price_change,
                    "impact": impact,
                    "new_value": position.market_value * (1 + price_change)
                }
        
        new_equity = current_equity + total_impact
        drawdown_pct = (total_impact / current_equity) * 100
        
        return {
            "current_equity": current_equity,
            "scenario_impact": total_impact,
            "new_equity": new_equity,
            "drawdown_pct": drawdown_pct,
            "position_impacts": position_impacts,
            "risk_limit_breaches": drawdown_pct < -engine.risk_manager.config.max_daily_loss * 100,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))