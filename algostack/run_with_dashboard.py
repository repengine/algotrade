"""
Run AlgoStack with Monitoring Dashboard.

This script starts both the trading engine and the FastAPI monitoring dashboard.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI

from algostack.api import create_app
from algostack.core.live_engine import LiveTradingEngine, TradingMode
from algostack.strategies.mean_reversion_equity import MeanReversionEquityStrategy


# Global engine instance for signal handling
engine_instance: Optional[LiveTradingEngine] = None
shutdown_event = asyncio.Event()


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\nShutdown signal received...")
    shutdown_event.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("Starting AlgoStack Trading System...")
    yield
    # Shutdown
    print("Shutting down AlgoStack Trading System...")
    if engine_instance:
        await engine_instance.stop()


async def run_trading_engine(config: dict) -> LiveTradingEngine:
    """Run the trading engine."""
    global engine_instance
    
    engine = LiveTradingEngine(config)
    engine_instance = engine
    
    # Start engine in background
    engine_task = asyncio.create_task(engine.start())
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # Stop engine
    await engine.stop()
    
    return engine


async def run_dashboard(engine: LiveTradingEngine, host: str = "0.0.0.0", port: int = 8000):
    """Run the monitoring dashboard."""
    app = create_app(engine)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
    )
    
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Trading engine configuration
    engine_config = {
        "mode": TradingMode.PAPER,
        
        "strategies": [
            {
                "class": MeanReversionEquityStrategy,
                "id": "mean_reversion_spy",
                "params": {
                    "symbol": "SPY",
                    "lookback": 20,
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "stop_loss": 0.03,
                }
            }
        ],
        
        "portfolio_config": {
            "initial_capital": 100000,
        },
        
        "risk_config": {
            "max_position_size": 0.2,
            "max_leverage": 1.0,
            "max_drawdown": 0.15,
        },
        
        "executor_config": {
            "paper": {
                "initial_capital": 100000,
                "commission": 1.0,
                "slippage": 0.0001,
                "fill_delay": 0.1,
            }
        },
        
        "schedule": {
            "market_open": "09:30",
            "market_close": "16:00",
            "pre_market": "09:00",
            "post_market": "16:30",
            "timezone": "US/Eastern",
        },
        
        "update_interval": 1.0,
        "min_signal_strength": 0.6,
        "risk_per_trade": 0.02,
        "max_position_size": 1000,
    }
    
    print("\n" + "=" * 60)
    print("AlgoStack Trading System with Monitoring Dashboard")
    print("=" * 60)
    print(f"Trading Mode: {engine_config['mode']}")
    print(f"Dashboard URL: http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    # Create engine
    engine = LiveTradingEngine(engine_config)
    engine_instance = engine
    
    # Create tasks
    engine_task = asyncio.create_task(run_trading_engine(engine_config))
    
    # Create and configure dashboard app
    app = create_app(engine)
    
    # Run dashboard server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
    
    server = uvicorn.Server(config)
    
    # Start server
    server_task = asyncio.create_task(server.serve())
    
    print("System started. Press Ctrl+C to stop.\n")
    
    try:
        # Wait for shutdown
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        
        # Cancel tasks
        engine_task.cancel()
        server_task.cancel()
        
        # Wait for cleanup
        await asyncio.gather(engine_task, server_task, return_exceptions=True)
        
        print("Shutdown complete.")


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main
    asyncio.run(main())