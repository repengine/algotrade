import csv
import io
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from ..dependencies import get_trading_engine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/export")
async def export_data(
    data_type: str = Query("trades", regex="^(trades|positions|orders|performance)$"),
    format: str = Query("csv", regex="^(csv|json)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    engine = Depends(get_trading_engine)
) -> Response:
    """Export trading data in various formats - PILLAR 4: VERIFIABLE CORRECTNESS"""
    if not engine:
        return Response(
            content=json.dumps({"error": "Trading engine not available"}),
            media_type="application/json",
            status_code=503
        )

    try:
        # Get data based on type
        if data_type == "trades":
            # Get trades from portfolio or trade history
            trades = []  # Would get from portfolio.trade_history

            if format == "csv":
                output = io.StringIO()
                if trades:
                    fieldnames = ["trade_id", "symbol", "side", "quantity", "price",
                                  "commission", "timestamp", "pnl", "strategy_id"]
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for trade in trades:
                        writer.writerow({
                            "trade_id": trade.trade_id,
                            "symbol": trade.symbol,
                            "side": trade.side,
                            "quantity": trade.quantity,
                            "price": trade.price,
                            "commission": trade.commission,
                            "timestamp": trade.timestamp.isoformat(),
                            "pnl": trade.pnl,
                            "strategy_id": trade.strategy_id
                        })

                return Response(
                    content=output.getvalue(),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=trades_{datetime.now().strftime('%Y%m%d')}.csv"
                    }
                )
            else:
                return Response(
                    content=json.dumps([t.dict() for t in trades], default=str),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f"attachment; filename=trades_{datetime.now().strftime('%Y%m%d')}.json"
                    }
                )

        elif data_type == "positions":
            positions = list(engine.portfolio.positions.values())

            if format == "csv":
                output = io.StringIO()
                if positions:
                    fieldnames = ["symbol", "quantity", "entry_price", "current_price",
                                  "market_value", "pnl", "pnl_percentage", "strategy_id"]
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for pos in positions:
                        writer.writerow({
                            "symbol": pos.symbol,
                            "quantity": pos.quantity,
                            "entry_price": pos.entry_price,
                            "current_price": pos.current_price,
                            "market_value": pos.market_value,
                            "pnl": pos.pnl,
                            "pnl_percentage": pos.pnl_percentage,
                            "strategy_id": pos.strategy_id
                        })

                return Response(
                    content=output.getvalue(),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=positions_{datetime.now().strftime('%Y%m%d')}.csv"
                    }
                )
            else:
                return Response(
                    content=json.dumps([p.__dict__ for p in positions], default=str),
                    media_type="application/json"
                )

        elif data_type == "orders":
            orders = list(engine.order_manager.orders.values())

            if format == "csv":
                output = io.StringIO()
                if orders:
                    fieldnames = ["order_id", "symbol", "side", "quantity", "order_type",
                                  "status", "limit_price", "filled_quantity", "average_fill_price"]
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for order in orders:
                        writer.writerow({
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "side": order.side,
                            "quantity": order.quantity,
                            "order_type": order.order_type,
                            "status": order.status,
                            "limit_price": order.limit_price,
                            "filled_quantity": order.filled_quantity,
                            "average_fill_price": order.average_fill_price
                        })

                return Response(
                    content=output.getvalue(),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=orders_{datetime.now().strftime('%Y%m%d')}.csv"
                    }
                )
            else:
                return Response(
                    content=json.dumps([o.__dict__ for o in orders], default=str),
                    media_type="application/json"
                )

        elif data_type == "performance":
            # Generate performance report
            perf_data = {
                "generated_at": datetime.now().isoformat(),
                "total_equity": engine.portfolio.total_equity,
                "starting_capital": engine.portfolio.starting_capital,
                "total_pnl": engine.portfolio.total_equity - engine.portfolio.starting_capital,
                "open_positions": len(engine.portfolio.positions),
                "total_trades": 0,  # Would count from trade history
                "win_rate": 0.0,  # Would calculate
                "sharpe_ratio": 0.0,  # Would calculate
                "max_drawdown": 0.0  # Would calculate
            }

            if format == "csv":
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=perf_data.keys())
                writer.writeheader()
                writer.writerow(perf_data)

                return Response(
                    content=output.getvalue(),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename=performance_{datetime.now().strftime('%Y%m%d')}.csv"
                    }
                )
            else:
                return Response(
                    content=json.dumps(perf_data, default=str),
                    media_type="application/json"
                )

    except Exception as e:
        logger.error(f"Error exporting {data_type}: {e}")
        return Response(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500
        )
