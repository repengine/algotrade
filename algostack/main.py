#!/usr/bin/env python3
"""Main entry point for AlgoStack."""

import asyncio
import logging
import sys
from pathlib import Path

import click
import yaml

from core.engine import TradingEngine, EngineConfig
from strategies.mean_reversion_equity import MeanReversionEquity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """AlgoStack - Algorithmic Trading Framework"""
    pass


@cli.command()
@click.option('--config', '-c', default='config/base.yaml', help='Configuration file')
@click.option('--mode', '-m', type=click.Choice(['paper', 'live']), default='paper')
def run(config: str, mode: str) -> None:
    """Run the trading engine."""
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Error: Config file {config} not found")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Override mode if specified
    cfg['engine']['mode'] = mode
    
    # Create engine
    engine_config = EngineConfig(
        mode=mode,
        data_providers=cfg['data']['providers'],
        strategies=list(cfg['strategies'].keys()),
        risk_params=cfg['risk']
    )
    
    engine = TradingEngine(engine_config)
    
    # Add strategies
    if cfg['strategies']['mean_reversion']['enabled']:
        strategy = MeanReversionEquity(cfg['strategies']['mean_reversion']['params'])
        engine.add_strategy('mean_reversion', strategy)
        
    # Run engine
    click.echo(f"Starting AlgoStack in {mode} mode...")
    
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        engine.stop()
        

@cli.command()
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--start', default='2020-01-01', help='Start date')
@click.option('--end', default='2023-12-31', help='End date')
@click.option('--config', '-c', default='config/base.yaml', help='Configuration file')
@click.option('--walk-forward', is_flag=True, help='Run walk-forward analysis')
def backtest(strategy: str, start: str, end: str, config: str, walk_forward: bool) -> None:
    """Run backtest for a strategy."""
    from backtests.run_backtests import BacktestEngine, run_walk_forward_optimization
    from strategies.mean_reversion_equity import MeanReversionEquity
    from strategies.trend_following_multi import TrendFollowingMulti
    from strategies.pairs_stat_arb import PairsStatArb
    from strategies.intraday_orb import IntradayORB
    from strategies.overnight_drift import OvernightDrift
    from strategies.hybrid_regime import HybridRegime
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Error: Config file {config} not found")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get strategy configuration
    if strategy not in cfg['strategies']:
        click.echo(f"Error: Strategy {strategy} not found in config")
        sys.exit(1)
    
    strategy_config = cfg['strategies'][strategy]
    
    # Create strategy instance
    strategy_map = {
        'mean_reversion': MeanReversionEquity,
        'trend_following': TrendFollowingMulti,
        'pairs_stat_arb': PairsStatArb,
        'intraday_orb': IntradayORB,
        'overnight_drift': OvernightDrift,
        'hybrid_regime': HybridRegime
    }
    
    if strategy not in strategy_map:
        click.echo(f"Error: Strategy {strategy} not implemented")
        click.echo(f"Available strategies: {', '.join(strategy_map.keys())}")
        sys.exit(1)
        
    strategy_class = strategy_map[strategy]
    strategy_instance = strategy_class(strategy_config['params'])
    symbols = strategy_config['symbols']
    
    click.echo(f"Running backtest for {strategy} from {start} to {end}")
    
    if walk_forward:
        # Run walk-forward optimization
        results_df = run_walk_forward_optimization(
            MeanReversionEquity,
            strategy_config['params'],
            symbols,
            start,
            end
        )
        
        # Display walk-forward results
        click.echo("\nWalk-Forward Analysis Results:")
        click.echo(results_df.to_string())
        
        # Summary statistics
        click.echo(f"\nAverage Annual Return: {results_df['annual_return'].mean():.2f}%")
        click.echo(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}")
        click.echo(f"Worst Drawdown: {results_df['max_drawdown'].min():.2f}%")
    else:
        # Run single backtest
        engine = BacktestEngine(initial_capital=cfg['portfolio']['initial_capital'])
        metrics = engine.run_backtest(
            strategy_instance,
            symbols,
            start,
            end,
            commission=cfg['backtest'].get('commission', 0.0),
            slippage=cfg['backtest'].get('slippage', 0.0005)
        )
        
        # Display results
        engine.print_summary()
        
        # Save results
        output_file = f"backtest_results/{strategy}_{start}_{end}.json"
        engine.save_results(output_file)
    

@cli.command()
@click.option('--symbol', '-s', required=True, help='Symbol to fetch')
@click.option('--start', default='2023-01-01', help='Start date')
@click.option('--provider', '-p', default='yfinance', help='Data provider')
def fetch_data(symbol: str, start: str, provider: str) -> None:
    """Fetch and cache historical data."""
    from datetime import datetime
    from core.data_handler import DataHandler
    
    click.echo(f"Fetching {symbol} data from {provider}...")
    
    handler = DataHandler([provider])
    asyncio.run(handler.initialize())
    
    df = handler.get_historical(
        symbol,
        datetime.strptime(start, '%Y-%m-%d'),
        datetime.now()
    )
    
    if not df.empty:
        click.echo(f"Fetched {len(df)} bars")
        click.echo(f"Date range: {df.index[0]} to {df.index[-1]}")
        click.echo(f"Latest close: ${df['close'].iloc[-1]:.2f}")
    else:
        click.echo("No data fetched")
        

@cli.command()
def test() -> None:
    """Run test suite."""
    import subprocess
    
    click.echo("Running tests...")
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])
    sys.exit(result.returncode)
    

if __name__ == '__main__':
    cli()