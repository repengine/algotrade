#!/usr/bin/env python3
"""Backtesting engine with Backtrader integration."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

import pandas as pd
import numpy as np
import backtrader as bt
from backtrader.analyzers import SharpeRatio, DrawDown, Returns, TradeAnalyzer

from core.data_handler import DataHandler
from strategies.base import BaseStrategy, Signal, RiskContext


logger = logging.getLogger(__name__)


class AlgoStackStrategy(bt.Strategy):
    """Backtrader adapter for AlgoStack strategies."""
    
    params = (
        ('algostack_strategy', None),  # Our strategy instance
        ('risk_context', None),  # Risk parameters
    )
    
    def __init__(self):
        """Initialize the Backtrader strategy wrapper."""
        self.algostack_strategy = self.params.algostack_strategy
        self.risk_context = self.params.risk_context
        self.signals_history = []
        self.trades_history = []
        
        # Initialize the AlgoStack strategy
        if self.algostack_strategy:
            self.algostack_strategy.init()
    
    def next(self):
        """Called on each bar."""
        # Prepare data for AlgoStack strategy
        lookback = self.algostack_strategy.config.get('lookback_period', 252)
        
        # Get historical data
        data_dict = {
            'open': [self.data.open[i] for i in range(-lookback, 1)],
            'high': [self.data.high[i] for i in range(-lookback, 1)],
            'low': [self.data.low[i] for i in range(-lookback, 1)],
            'close': [self.data.close[i] for i in range(-lookback, 1)],
            'volume': [self.data.volume[i] for i in range(-lookback, 1)],
        }
        
        # Create DataFrame
        dates = [self.data.datetime.datetime(i) for i in range(-lookback, 1)]
        df = pd.DataFrame(data_dict, index=dates)
        df.attrs['symbol'] = self.data._name
        
        # Get signal from AlgoStack strategy
        signal = self.algostack_strategy.next(df)
        
        if signal:
            self.signals_history.append({
                'datetime': self.data.datetime.datetime(0),
                'signal': signal.dict()
            })
            
            # Execute based on signal
            self._execute_signal(signal)
    
    def _execute_signal(self, signal: Signal):
        """Execute trading signal."""
        current_position = self.getposition(self.data).size
        
        if signal.direction == 'LONG' and current_position == 0:
            # Calculate position size
            position_size, stop_loss = self.algostack_strategy.size(
                signal, self._get_current_risk_context()
            )
            
            if position_size > 0:
                # Place buy order
                self.buy(size=int(position_size))
                
                # Set stop loss
                if stop_loss > 0:
                    self.sell(exectype=bt.Order.Stop, price=stop_loss)
                    
        elif signal.direction == 'FLAT' and current_position > 0:
            # Close position
            self.close()
    
    def _get_current_risk_context(self) -> RiskContext:
        """Get current risk context."""
        return RiskContext(
            account_equity=self.broker.getvalue(),
            open_positions=len([p for p in self.broker.positions.values() if p.size != 0]),
            daily_pnl=0,  # TODO: Calculate from recent trades
            max_drawdown_pct=0.15,
            volatility_target=self.risk_context.volatility_target,
            max_position_size=self.risk_context.max_position_size
        )
    
    def notify_trade(self, trade):
        """Track completed trades."""
        if trade.isclosed:
            self.trades_history.append({
                'datetime': self.data.datetime.datetime(0),
                'symbol': self.data._name,
                'size': trade.size,
                'entry_price': trade.price,
                'exit_price': self.data.close[0],
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / (trade.price * abs(trade.size)) * 100,
                'commission': trade.commission
            })
            
            # Update strategy performance
            self.algostack_strategy.update_performance({
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / (trade.price * abs(trade.size)) * 100
            })


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, initial_capital: float = 5000.0):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        commission: float = 0.0,
        slippage: float = 0.0005,
        data_provider: str = "yfinance"
    ) -> Dict[str, Any]:
        """Run backtest for a strategy."""
        logger.info(f"Running backtest for {strategy.name} from {start_date} to {end_date}")
        
        # Initialize Backtrader
        cerebro = bt.Cerebro()
        
        # Set initial capital
        cerebro.broker.setcash(self.initial_capital)
        
        # Set commission
        cerebro.broker.setcommission(commission=commission)
        
        # Add slippage
        if slippage > 0:
            cerebro.broker.set_slippage_perc(slippage)
        
        # Load data for each symbol
        data_handler = DataHandler([data_provider], premium_av=True)
        
        for symbol in symbols:
            # Fetch historical data
            df = data_handler.get_historical(
                symbol,
                datetime.strptime(start_date, '%Y-%m-%d'),
                datetime.strptime(end_date, '%Y-%m-%d')
            )
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            # Convert to Backtrader format
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=None
            )
            data._name = symbol
            
            # Add data to cerebro
            cerebro.adddata(data)
        
        # Add strategy with risk context
        risk_context = RiskContext(
            account_equity=self.initial_capital,
            open_positions=0,
            daily_pnl=0,
            max_drawdown_pct=0.15,
            volatility_target=0.10,
            max_position_size=0.20
        )
        
        cerebro.addstrategy(
            AlgoStackStrategy,
            algostack_strategy=strategy,
            risk_context=risk_context
        )
        
        # Add analyzers
        cerebro.addanalyzer(SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addanalyzer(DrawDown, _name='drawdown')
        cerebro.addanalyzer(Returns, _name='returns')
        cerebro.addanalyzer(TradeAnalyzer, _name='trades')
        
        # Run backtest
        results = cerebro.run()
        strategy_results = results[0]
        
        # Extract metrics
        metrics = self._extract_metrics(cerebro, strategy_results)
        
        # Add strategy-specific metrics
        if hasattr(strategy, 'backtest_metrics'):
            trades_df = pd.DataFrame(strategy_results.trades_history)
            if not trades_df.empty:
                strategy_metrics = strategy.backtest_metrics(trades_df)
                metrics.update(strategy_metrics)
        
        # Store results
        self.results[strategy.name] = {
            'metrics': metrics,
            'signals': strategy_results.signals_history,
            'trades': strategy_results.trades_history
        }
        
        return metrics
    
    def _extract_metrics(self, cerebro, strategy_results) -> Dict[str, float]:
        """Extract performance metrics from backtest results."""
        # Get analyzers
        sharpe = strategy_results.analyzers.sharpe.get_analysis()
        drawdown = strategy_results.analyzers.drawdown.get_analysis()
        returns = strategy_results.analyzers.returns.get_analysis()
        trades = strategy_results.analyzers.trades.get_analysis()
        
        # Calculate metrics
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe.get('sharperatio', 0),
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
            'total_trades': trades.get('total', {}).get('total', 0),
            'winning_trades': trades.get('won', {}).get('total', 0),
            'losing_trades': trades.get('lost', {}).get('total', 0),
            'avg_trade': trades.get('pnl', {}).get('average', 0),
        }
        
        # Calculate additional metrics
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
            metrics['profit_factor'] = abs(
                trades.get('won', {}).get('pnl', {}).get('total', 0) / 
                trades.get('lost', {}).get('pnl', {}).get('total', 1)
            )
        else:
            metrics['win_rate'] = 0
            metrics['profit_factor'] = 0
        
        # Annualized metrics
        years = (datetime.strptime(returns.get('end'), '%Y-%m-%d %H:%M:%S') - 
                datetime.strptime(returns.get('start'), '%Y-%m-%d %H:%M:%S')).days / 365.25
        
        if years > 0:
            metrics['annual_return'] = (pow(final_value / self.initial_capital, 1/years) - 1) * 100
        else:
            metrics['annual_return'] = 0
        
        return metrics
    
    def save_results(self, filepath: str):
        """Save backtest results to file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary of all backtest results."""
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_name}")
            print(f"{'='*60}")
            print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
            print(f"Final Value: ${metrics['final_value']:,.2f}")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Annual Return: {metrics['annual_return']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")


def run_walk_forward_optimization(
    strategy_class,
    config: Dict[str, Any],
    symbols: List[str],
    start_date: str,
    end_date: str,
    window_size: int = 252,  # Trading days in a year
    step_size: int = 63,     # Quarter
    optimization_params: Dict[str, List[Any]] = None
) -> pd.DataFrame:
    """Run walk-forward optimization."""
    results = []
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Generate windows
    current_start = start
    while current_start + pd.Timedelta(days=window_size) < end:
        # In-sample period
        is_start = current_start
        is_end = current_start + pd.Timedelta(days=window_size)
        
        # Out-of-sample period
        oos_start = is_end
        oos_end = min(oos_start + pd.Timedelta(days=step_size), end)
        
        logger.info(f"Walk-forward: IS {is_start.date()} to {is_end.date()}, "
                   f"OOS {oos_start.date()} to {oos_end.date()}")
        
        # TODO: Implement parameter optimization on in-sample
        # For now, use provided config
        strategy = strategy_class(config)
        
        # Run out-of-sample backtest
        engine = BacktestEngine()
        metrics = engine.run_backtest(
            strategy,
            symbols,
            str(oos_start.date()),
            str(oos_end.date())
        )
        
        results.append({
            'window_start': oos_start,
            'window_end': oos_end,
            **metrics
        })
        
        # Move to next window
        current_start += pd.Timedelta(days=step_size)
    
    return pd.DataFrame(results)