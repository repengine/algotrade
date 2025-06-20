import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.executor import OrderStatus
from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_trading_engine
from ..models import (
    Order,
    OrderCreate,
    OrderResponse,
    OrderUpdate,
    PositionResponse,
    TradeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    engine = Depends(get_trading_engine)
) -> List[PositionResponse]:
    """Get all open positions - PILLAR 4: VERIFIABLE CORRECTNESS"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        portfolio = engine.portfolio
        positions = list(portfolio.positions.values())

        # Apply filters
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        if strategy_id:
            positions = [p for p in positions if p.strategy_id == strategy_id]

        # Convert to response model
        return [
            PositionResponse(
                position_id=p.position_id,
                symbol=p.symbol,
                quantity=p.quantity,
                entry_price=p.entry_price,
                current_price=p.current_price,
                market_value=p.market_value,
                pnl=p.pnl,
                pnl_percentage=p.pnl_percentage,
                strategy_id=p.strategy_id,
                opened_at=p.opened_at,
                updated_at=p.updated_at
            )
            for p in positions
        ]
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/positions/{position_id}", response_model=PositionResponse)
async def get_position(
    position_id: str,
    engine = Depends(get_trading_engine)
) -> PositionResponse:
    """Get single position details - PILLAR 4: VERIFIABLE CORRECTNESS"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        portfolio = engine.portfolio
        position = portfolio.positions.get(position_id)

        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

        return PositionResponse(
            position_id=position.position_id,
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=position.current_price,
            market_value=position.market_value,
            pnl=position.pnl,
            pnl_percentage=position.pnl_percentage,
            strategy_id=position.strategy_id,
            opened_at=position.opened_at,
            updated_at=position.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position {position_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    status: Optional[OrderStatus] = Query(None, description="Filter by status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    engine = Depends(get_trading_engine)
) -> List[OrderResponse]:
    """Get orders with pagination - PILLAR 4: VERIFIABLE CORRECTNESS"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        order_manager = engine.order_manager
        orders = list(order_manager.orders.values())

        # Apply filters
        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]

        # Sort by created_at descending
        orders.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        len(orders)
        orders = orders[offset:offset + limit]

        # Convert to response model
        return [
            OrderResponse(
                order_id=o.order_id,
                symbol=o.symbol,
                side=o.side,
                quantity=o.quantity,
                order_type=o.order_type,
                limit_price=o.limit_price,
                stop_price=o.stop_price,
                status=o.status,
                filled_quantity=o.filled_quantity,
                average_fill_price=o.average_fill_price,
                strategy_id=o.strategy_id,
                created_at=o.created_at,
                updated_at=o.updated_at,
                broker_order_id=o.broker_order_id
            )
            for o in orders
        ]
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order: OrderCreate,
    engine = Depends(get_trading_engine)
) -> OrderResponse:
    """Submit new order with risk checks - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        # Validate market hours
        if not engine.is_market_open():
            raise HTTPException(status_code=400, detail="Market is closed")

        # Create order object
        new_order = Order(
            order_id=f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            strategy_id=order.strategy_id or "manual",
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Risk checks - CRITICAL FOR CAPITAL PRESERVATION
        risk_check = await engine.risk_manager.check_order(new_order)
        if not risk_check["allowed"]:
            raise HTTPException(
                status_code=403,
                detail=f"Order rejected by risk manager: {risk_check.get('reason', 'Unknown')}"
            )

        # Submit order
        await engine.submit_order(new_order)

        # Get updated order from order manager
        submitted_order = engine.order_manager.orders.get(new_order.order_id)
        if not submitted_order:
            raise HTTPException(status_code=500, detail="Order submission failed")

        return OrderResponse(
            order_id=submitted_order.order_id,
            symbol=submitted_order.symbol,
            side=submitted_order.side,
            quantity=submitted_order.quantity,
            order_type=submitted_order.order_type,
            limit_price=submitted_order.limit_price,
            stop_price=submitted_order.stop_price,
            status=submitted_order.status,
            filled_quantity=submitted_order.filled_quantity,
            average_fill_price=submitted_order.average_fill_price,
            strategy_id=submitted_order.strategy_id,
            created_at=submitted_order.created_at,
            updated_at=submitted_order.updated_at,
            broker_order_id=submitted_order.broker_order_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/orders/{order_id}", response_model=OrderResponse)
async def update_order(
    order_id: str,
    update: OrderUpdate,
    engine = Depends(get_trading_engine)
) -> OrderResponse:
    """Modify pending order - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        order_manager = engine.order_manager
        order = order_manager.orders.get(order_id)

        if not order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Only pending orders can be modified
        if order.status != OrderStatus.PENDING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot modify order with status {order.status}"
            )

        # Apply updates
        if update.quantity is not None:
            order.quantity = update.quantity
        if update.limit_price is not None:
            order.limit_price = update.limit_price

        # Re-run risk checks
        risk_check = await engine.risk_manager.check_order(order)
        if not risk_check["allowed"]:
            raise HTTPException(
                status_code=403,
                detail=f"Updated order rejected by risk manager: {risk_check.get('reason', 'Unknown')}"
            )

        # Update with broker
        await engine.broker.modify_order(order_id, update.dict(exclude_unset=True))

        order.updated_at = datetime.utcnow()

        return OrderResponse(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            strategy_id=order.strategy_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
            broker_order_id=order.broker_order_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Cancel pending order - PILLAR 1: CAPITAL PRESERVATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        order_manager = engine.order_manager
        order = order_manager.orders.get(order_id)

        if not order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Only pending orders can be cancelled
        if order.status != OrderStatus.PENDING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel order with status {order.status}"
            )

        # Cancel with broker
        await engine.broker.cancel_order(order_id)

        # Update order status
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()

        return {
            "message": f"Order {order_id} cancelled successfully",
            "order_id": order_id,
            "status": "cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    engine = Depends(get_trading_engine)
) -> List[TradeResponse]:
    """Get trade history - PILLAR 4: VERIFIABLE CORRECTNESS"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")

    try:
        # Get trades from portfolio or trade history
        trades = []  # Would get from portfolio.trade_history or similar

        # Apply filters
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if strategy_id:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]

        # Sort by timestamp descending
        trades.sort(key=lambda x: x.timestamp, reverse=True)

        # Apply pagination
        len(trades)
        trades = trades[offset:offset + limit]

        # Convert to response model
        return [
            TradeResponse(
                trade_id=t.trade_id,
                symbol=t.symbol,
                side=t.side,
                quantity=t.quantity,
                price=t.price,
                commission=t.commission,
                timestamp=t.timestamp,
                order_id=t.order_id,
                strategy_id=t.strategy_id,
                pnl=t.pnl
            )
            for t in trades
        ]
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
