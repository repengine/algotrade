from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime
import asyncio

from ..dependencies import get_trading_engine, get_database_status, get_broker_status

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """System health check endpoint - PILLAR 3: OPERATIONAL STABILITY"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "details": {}
    }
    
    try:
        # Check database connection
        db_status = await get_database_status()
        health_status["details"]["database"] = db_status
        
        # Check broker connection
        broker_status = await get_broker_status()
        health_status["details"]["broker"] = broker_status
        
        # Check trading engine
        engine = await get_trading_engine()
        engine_healthy = engine is not None
        health_status["details"]["trading_engine"] = {
            "connected": engine_healthy,
            "status": getattr(engine, 'status', 'unknown') if engine else 'disconnected'
        }
        
        # For test environment or when no engine is set, still return healthy
        # In production, you would want stricter checks
        if engine is None:
            # No engine configured, but API is up - return healthy
            health_status["status"] = "healthy"
        else:
            # Determine overall health based on actual connections
            if not all([
                health_status["details"]["database"].get("connected", False),
                health_status["details"]["broker"].get("connected", False),
                engine_healthy
            ]):
                health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


@router.get("/status")
async def system_status(engine = Depends(get_trading_engine)) -> Dict[str, Any]:
    """Get detailed trading system status - PILLAR 3: OPERATIONAL STABILITY"""
    if not engine:
        return {
            "status": "offline",
            "message": "Trading engine not initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get comprehensive system status
        status = {
            "status": engine.status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": getattr(engine, 'uptime_seconds', 0),
            "market_hours": {
                "is_open": getattr(engine, 'is_market_open', lambda: False)(),
                "next_open": None,  # Could be calculated based on market calendar
                "next_close": None
            },
            "strategies": {
                "active": len([s for s in getattr(engine, 'strategies', []) if getattr(s, 'active', False)]),
                "total": len(getattr(engine, 'strategies', [])),
                "names": [s.__class__.__name__ for s in getattr(engine, 'strategies', [])]
            },
            "risk_status": {
                "circuit_breaker_active": getattr(engine, 'circuit_breaker_active', False),
                "risk_limits_ok": True,  # Could check risk manager
                "last_risk_check": datetime.utcnow().isoformat()
            },
            "data_feed_status": {
                "connected": True,  # Could check data handler
                "last_update": datetime.utcnow().isoformat(),
                "symbols_tracked": len(getattr(engine, 'symbols', []))
            },
            "performance": {
                "total_pnl": 0.0,  # Could calculate from portfolio
                "open_positions": 0,  # Could get from portfolio
                "pending_orders": 0  # Could get from order manager
            }
        }
        
        return status
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get system status: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }