from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends
import asyncio

# Global instances - in production these would be properly managed
_trading_engine = None
_risk_manager = None
_portfolio_engine = None
_order_manager = None


def set_trading_engine(engine):
    """Set the global trading engine instance"""
    global _trading_engine
    _trading_engine = engine


async def get_trading_engine():
    """Get the trading engine instance - PILLAR 3: OPERATIONAL STABILITY"""
    return _trading_engine


async def get_risk_manager():
    """Get the risk manager instance - PILLAR 1: CAPITAL PRESERVATION"""
    if _trading_engine:
        return _trading_engine.risk_manager
    return None


async def get_portfolio_engine():
    """Get the portfolio engine instance - PILLAR 4: VERIFIABLE CORRECTNESS"""
    if _trading_engine:
        return _trading_engine.portfolio
    return None


async def get_order_manager():
    """Get the order manager instance - PILLAR 1: CAPITAL PRESERVATION"""
    if _trading_engine:
        return _trading_engine.order_manager
    return None


async def get_database_status() -> Dict[str, Any]:
    """Check database connection status"""
    # In production, would check actual database connection
    return {
        "connected": True,
        "latency_ms": 5,
        "last_check": "2024-01-01T00:00:00Z"
    }


async def get_broker_status() -> Dict[str, Any]:
    """Check broker connection status"""
    # In production, would check actual broker connection
    if _trading_engine and hasattr(_trading_engine, 'broker'):
        broker = _trading_engine.broker
        return {
            "connected": getattr(broker, 'connected', False),
            "account_id": getattr(broker, 'account_id', None),
            "last_heartbeat": getattr(broker, 'last_heartbeat', None)
        }
    return {
        "connected": False,
        "account_id": None,
        "last_heartbeat": None
    }


async def require_admin_auth(authorization: Optional[str] = Header(None)):
    """Require admin authentication for sensitive operations - PILLAR 1: CAPITAL PRESERVATION"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # In production, would validate JWT token or API key
    # For now, just check for a specific token
    if authorization != "Bearer admin-token":
        raise HTTPException(
            status_code=403,
            detail="Invalid authorization token"
        )
    
    return True


async def get_client_id(x_forwarded_for: Optional[str] = Header(None)) -> str:
    """Extract client ID from request headers"""
    if x_forwarded_for:
        # Get first IP from X-Forwarded-For header
        return x_forwarded_for.split(",")[0].strip()
    return "default"


class RateLimiter:
    """Simple in-memory rate limiter - PILLAR 3: OPERATIONAL STABILITY"""
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.requests = {}
    
    async def __call__(self, client_id: str = Depends(get_client_id)):
        now = asyncio.get_event_loop().time()
        
        # Clean old requests
        self.requests = {
            k: v for k, v in self.requests.items()
            if now - v[-1] < self.period
        }
        
        # Check rate limit
        if client_id in self.requests:
            timestamps = self.requests[client_id]
            timestamps = [t for t in timestamps if now - t < self.period]
            
            if len(timestamps) >= self.calls:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.calls} calls per {self.period} seconds"
                )
            
            timestamps.append(now)
            self.requests[client_id] = timestamps
        else:
            self.requests[client_id] = [now]
        
        return True


# Rate limiters for different endpoints
rate_limit_health = RateLimiter(calls=60, period=60)  # 60 calls per minute
rate_limit_orders = RateLimiter(calls=10, period=60)  # 10 orders per minute
rate_limit_api = RateLimiter(calls=100, period=60)    # 100 calls per minute