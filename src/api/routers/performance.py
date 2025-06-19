from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from ..dependencies import get_trading_engine, get_portfolio_engine
from ..models import PerformanceMetrics, PnLTimeSeries

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    timeframe: str = Query("1D", regex="^(1D|1W|1M|3M|YTD|ALL)$", description="Time period"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    engine = Depends(get_trading_engine)
) -> PerformanceMetrics:
    """Get performance metrics - PILLAR 2: PROFIT GENERATION & PILLAR 4: VERIFIABLE CORRECTNESS"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        portfolio = engine.portfolio
        
        # Calculate time range
        end_date = datetime.utcnow()
        if timeframe == "1D":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "1W":
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == "1M":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "3M":
            start_date = end_date - timedelta(days=90)
        elif timeframe == "YTD":
            start_date = datetime(end_date.year, 1, 1)
        else:  # ALL
            start_date = datetime(2020, 1, 1)  # Or get from first trade
        
        # Get equity curve data
        equity_curve = portfolio.get_equity_curve(start_date, end_date)
        
        # Apply filters if needed
        if strategy_id or symbol:
            # Filter trades/positions by strategy or symbol
            pass
        
        # Calculate metrics
        if len(equity_curve) < 2:
            # Not enough data
            return PerformanceMetrics(
                total_pnl=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                volatility=0.0,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        
        # Calculate returns
        equity_values = equity_curve["equity"].values
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Calculate metrics
        total_pnl = equity_values[-1] - equity_values[0]
        total_return_pct = (equity_values[-1] / equity_values[0] - 1) * 100
        
        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        # Trade statistics (would get from trade history)
        trades = []  # portfolio.get_trades(start_date, end_date)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0.0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = abs(min([t.pnl for t in losing_trades])) if losing_trades else 0.0
        
        gross_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0.0
        gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0.0
        
        return PerformanceMetrics(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            volatility=volatility,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/pnl", response_model=PnLTimeSeries)
async def get_pnl_timeseries(
    granularity: str = Query("1h", regex="^(1m|5m|1h|1d)$", description="Data granularity"),
    timeframe: str = Query("1D", regex="^(1D|1W|1M|3M|YTD|ALL)$", description="Time period"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    engine = Depends(get_trading_engine)
) -> PnLTimeSeries:
    """Get P&L time series data - PILLAR 2: PROFIT GENERATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        portfolio = engine.portfolio
        
        # Calculate time range
        end_date = datetime.utcnow()
        if timeframe == "1D":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "1W":
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == "1M":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "3M":
            start_date = end_date - timedelta(days=90)
        elif timeframe == "YTD":
            start_date = datetime(end_date.year, 1, 1)
        else:  # ALL
            start_date = datetime(2020, 1, 1)
        
        # Get P&L data at specified granularity
        pnl_data = portfolio.get_pnl_series(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            strategy_id=strategy_id,
            symbol=symbol
        )
        
        # Convert to response format
        timestamps = []
        realized_pnl = []
        unrealized_pnl = []
        total_pnl = []
        fees = []
        
        for row in pnl_data:
            timestamps.append(row["timestamp"])
            realized_pnl.append(row["realized_pnl"])
            unrealized_pnl.append(row["unrealized_pnl"])
            total_pnl.append(row["realized_pnl"] + row["unrealized_pnl"])
            fees.append(row["fees"])
        
        return PnLTimeSeries(
            timestamps=timestamps,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            fees=fees,
            granularity=granularity,
            timeframe=timeframe
        )
        
    except Exception as e:
        logger.error(f"Error getting P&L timeseries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary")
async def get_performance_summary(
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Get quick performance summary - PILLAR 2: PROFIT GENERATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        portfolio = engine.portfolio
        
        # Get current metrics
        current_equity = portfolio.total_equity
        starting_capital = portfolio.starting_capital
        
        return {
            "current_equity": current_equity,
            "starting_capital": starting_capital,
            "total_pnl": current_equity - starting_capital,
            "total_return_pct": ((current_equity / starting_capital) - 1) * 100,
            "open_positions": len(portfolio.positions),
            "total_trades_today": 0,  # Would count from trade history
            "daily_pnl": 0.0,  # Would calculate from today's trades
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/by-strategy")
async def get_performance_by_strategy(
    timeframe: str = Query("1M", regex="^(1D|1W|1M|3M|YTD|ALL)$", description="Time period"),
    engine = Depends(get_trading_engine)
) -> Dict[str, Any]:
    """Get performance breakdown by strategy - PILLAR 2: PROFIT GENERATION"""
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        # Would aggregate performance by strategy
        strategy_performance = {}
        
        for strategy in engine.strategies:
            strategy_id = strategy.__class__.__name__
            # Calculate strategy-specific metrics
            strategy_performance[strategy_id] = {
                "total_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "active": getattr(strategy, "active", False)
            }
        
        return {
            "timeframe": timeframe,
            "strategies": strategy_performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))