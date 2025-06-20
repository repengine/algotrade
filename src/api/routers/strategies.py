import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_trading_engine
from ..models import StrategyInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/strategies", response_model=List[StrategyInfo])
async def get_strategies(
    engine = Depends(get_trading_engine)
) -> List[StrategyInfo]:
    """Get all loaded strategies - PILLAR 3: OPERATIONAL STABILITY"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        strategies = []
        for strategy in engine.strategies:
            strategy_info = StrategyInfo(
                strategy_id=strategy.__class__.__name__,
                name=getattr(strategy, "name", strategy.__class__.__name__),
                description=getattr(strategy, "__doc__", "No description"),
                active=getattr(strategy, "active", False),
                parameters=getattr(strategy, "parameters", {}),
                performance={
                    "total_pnl": 0.0,  # Would get from performance tracking
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0
                },
                last_signal=getattr(strategy, "last_signal_time", None)
            )
            strategies.append(strategy_info)

        return strategies

    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/strategies/{strategy_id}", response_model=StrategyInfo)
async def get_strategy(
    strategy_id: str,
    engine = Depends(get_trading_engine)
) -> StrategyInfo:
    """Get specific strategy details - PILLAR 3: OPERATIONAL STABILITY"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        # Find strategy
        strategy = None
        for s in engine.strategies:
            if s.__class__.__name__ == strategy_id:
                strategy = s
                break

        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        return StrategyInfo(
            strategy_id=strategy.__class__.__name__,
            name=getattr(strategy, "name", strategy.__class__.__name__),
            description=getattr(strategy, "__doc__", "No description"),
            active=getattr(strategy, "active", False),
            parameters=getattr(strategy, "parameters", {}),
            performance={
                "total_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0
            },
            last_signal=getattr(strategy, "last_signal_time", None)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/strategies/{strategy_id}/enable")
async def enable_strategy(
    strategy_id: str,
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Enable a strategy - PILLAR 1: CAPITAL PRESERVATION (controlled activation)"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        # Find strategy
        strategy = None
        for s in engine.strategies:
            if s.__class__.__name__ == strategy_id:
                strategy = s
                break

        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Check if already active
        if getattr(strategy, "active", False):
            return {
                "message": f"Strategy {strategy_id} is already enabled",
                "strategy_id": strategy_id,
                "active": True
            }

        # Enable strategy
        strategy.active = True

        # Log the change
        logger.info(f"Strategy {strategy_id} enabled")

        return {
            "message": f"Strategy {strategy_id} enabled successfully",
            "strategy_id": strategy_id,
            "active": True,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/strategies/{strategy_id}/disable")
async def disable_strategy(
    strategy_id: str,
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Disable a strategy - PILLAR 1: CAPITAL PRESERVATION (controlled deactivation)"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        # Find strategy
        strategy = None
        for s in engine.strategies:
            if s.__class__.__name__ == strategy_id:
                strategy = s
                break

        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Check if already inactive
        if not getattr(strategy, "active", True):
            return {
                "message": f"Strategy {strategy_id} is already disabled",
                "strategy_id": strategy_id,
                "active": False
            }

        # Disable strategy
        strategy.active = False

        # Cancel any pending orders from this strategy
        order_manager = engine.order_manager
        cancelled_orders = []
        for order in order_manager.orders.values():
            if order.strategy_id == strategy_id and order.status == "PENDING":
                # Cancel the order
                order.status = "CANCELLED"
                cancelled_orders.append(order.order_id)

        # Log the change
        logger.info(f"Strategy {strategy_id} disabled, cancelled {len(cancelled_orders)} orders")

        return {
            "message": f"Strategy {strategy_id} disabled successfully",
            "strategy_id": strategy_id,
            "active": False,
            "cancelled_orders": cancelled_orders,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/strategies/{strategy_id}/parameters")
async def update_strategy_parameters(
    strategy_id: str,
    parameters: Dict[str, Any],
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Update strategy parameters - PILLAR 2: PROFIT GENERATION (parameter tuning)"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        # Find strategy
        strategy = None
        for s in engine.strategies:
            if s.__class__.__name__ == strategy_id:
                strategy = s
                break

        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Update parameters
        current_params = getattr(strategy, "parameters", {})
        for key, value in parameters.items():
            if key in current_params:
                # Validate parameter type matches
                if not isinstance(value, type(current_params[key])):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Parameter {key} type mismatch: expected {type(current_params[key])}, got {type(value)}"
                    )
                current_params[key] = value
            else:
                raise HTTPException(status_code=400, detail=f"Unknown parameter: {key}")

        strategy.parameters = current_params

        # Log the change
        logger.info(f"Strategy {strategy_id} parameters updated: {parameters}")

        return {
            "message": f"Strategy {strategy_id} parameters updated successfully",
            "strategy_id": strategy_id,
            "parameters": current_params,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id} parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
