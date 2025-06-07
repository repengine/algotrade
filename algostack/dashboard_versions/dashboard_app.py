#!/usr/bin/env python3
"""
AlgoStack Dashboard - Fully Integrated Trading Strategy Dashboard
Provides real integration with strategies, backtesting, and configuration
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import importlib
import inspect
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Type
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the algostack directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Handle TA-Lib import gracefully
try:
    import talib
except ImportError:
    # Use mock TA-Lib if not installed
    import mock_talib as talib
    sys.modules['talib'] = talib

# Import AlgoStack components
from strategies.base import BaseStrategy
from backtests.run_backtests import BacktestEngine, StrategyBacktraderAdapter
from core.data_handler import DataHandler
from core.portfolio import Portfolio
from core.risk import RiskManager
from utils.logging import get_logger
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="AlgoStack Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'strategies' not in st.session_state:
    st.session_state.strategies = {}
if 'config' not in st.session_state:
    st.session_state.config = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}


class StrategyRegistry:
    """Dynamic strategy discovery and management"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self._discover_strategies()
    
    def _discover_strategies(self):
        """Discover all strategy classes in the strategies directory"""
        strategies_dir = Path(__file__).parent / "strategies"
        
        for file_path in strategies_dir.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name == "base.py":
                continue
                
            module_name = file_path.stem
            try:
                module = importlib.import_module(f"strategies.{module_name}")
                
                # Find all classes that inherit from BaseStrategy
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                        # Use a more friendly name
                        friendly_name = name.replace("Strategy", "").replace("_", " ").title()
                        self.strategies[friendly_name] = obj
                        logger.info(f"Discovered strategy: {friendly_name} ({name})")
                        
            except Exception as e:
                logger.error(f"Failed to load strategy from {module_name}: {e}")
    
    def get_strategy_class(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class by name"""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names"""
        return list(self.strategies.keys())
    
    def get_strategy_parameters(self, name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            return {}
        
        # Get __init__ signature
        sig = inspect.signature(strategy_class.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'config']:
                continue
                
            # Try to get default value
            if param.default != inspect.Parameter.empty:
                params[param_name] = param.default
            else:
                # Set reasonable defaults based on parameter name
                if 'period' in param_name:
                    params[param_name] = 20
                elif 'threshold' in param_name:
                    params[param_name] = 30.0
                elif 'lookback' in param_name:
                    params[param_name] = 50
                else:
                    params[param_name] = None
                    
        return params


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "base.yaml"
    
    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return None


def create_strategy_config_ui(strategy_name: str, registry: StrategyRegistry) -> Dict[str, Any]:
    """Create UI for configuring strategy parameters"""
    params = registry.get_strategy_parameters(strategy_name)
    configured_params = {}
    
    if params:
        st.subheader(f"Configure {strategy_name}")
        
        cols = st.columns(min(len(params), 3))
        for i, (param_name, default_value) in enumerate(params.items()):
            col = cols[i % len(cols)]
            
            with col:
                # Create appropriate input widget based on parameter type
                if isinstance(default_value, bool):
                    configured_params[param_name] = st.checkbox(
                        param_name.replace("_", " ").title(),
                        value=default_value
                    )
                elif isinstance(default_value, int):
                    configured_params[param_name] = st.number_input(
                        param_name.replace("_", " ").title(),
                        value=default_value,
                        step=1
                    )
                elif isinstance(default_value, float):
                    configured_params[param_name] = st.number_input(
                        param_name.replace("_", " ").title(),
                        value=default_value,
                        step=0.1,
                        format="%.2f"
                    )
                else:
                    configured_params[param_name] = st.text_input(
                        param_name.replace("_", " ").title(),
                        value=str(default_value) if default_value else ""
                    )
    
    return configured_params


def aggregate_backtest_results(all_results: Dict[str, Dict]) -> Dict:
    """Aggregate results from multiple strategy-symbol combinations"""
    
    # Initialize aggregated metrics
    total_trades = 0
    total_return = 0
    all_trades = []
    all_signals = []
    strategy_metrics = {}
    
    # Aggregate by strategy
    for result_key, result in all_results.items():
        strategy_name = result['strategy']
        metrics = result['metrics']
        
        # Initialize strategy metrics if not exists
        if strategy_name not in strategy_metrics:
            strategy_metrics[strategy_name] = {
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'symbols': []
            }
        
        # Update strategy metrics
        strategy_metrics[strategy_name]['total_return'] += metrics.get('total_return', 0)
        strategy_metrics[strategy_name]['total_trades'] += metrics.get('total_trades', 0)
        strategy_metrics[strategy_name]['winning_trades'] += metrics.get('winning_trades', 0)
        strategy_metrics[strategy_name]['losing_trades'] += metrics.get('losing_trades', 0)
        strategy_metrics[strategy_name]['symbols'].append(result['symbol'])
        
        # Update overall metrics
        total_trades += metrics.get('total_trades', 0)
        total_return += metrics.get('total_return', 0)
        
        # Collect all trades and signals
        trades = result.get('trades', [])
        signals = result.get('signals', [])
        
        # Add strategy and symbol info to trades
        for trade in trades:
            trade['strategy'] = strategy_name
            trade['symbol'] = result['symbol']
            all_trades.append(trade)
        
        all_signals.extend(signals)
    
    # Calculate aggregate metrics
    num_strategies = len(strategy_metrics)
    avg_return = total_return / num_strategies if num_strategies > 0 else 0
    
    # Calculate win rate
    total_winners = sum(s['winning_trades'] for s in strategy_metrics.values())
    total_losers = sum(s['losing_trades'] for s in strategy_metrics.values())
    win_rate = total_winners / (total_winners + total_losers) if (total_winners + total_losers) > 0 else 0
    
    # Build final results
    aggregated = {
        'metrics': {
            'total_return': avg_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': np.mean([r['metrics'].get('sharpe_ratio', 0) for r in all_results.values()]),
            'max_drawdown': np.max([r['metrics'].get('max_drawdown', 0) for r in all_results.values()]),
            'sortino_ratio': np.mean([r['metrics'].get('sortino_ratio', 0) for r in all_results.values()]),
            'calmar_ratio': np.mean([r['metrics'].get('calmar_ratio', 0) for r in all_results.values()]),
            'profit_factor': np.mean([r['metrics'].get('profit_factor', 0) for r in all_results.values()])
        },
        'trades': all_trades,
        'signals': all_signals,
        'strategy_metrics': strategy_metrics,
        'individual_results': all_results
    }
    
    return aggregated


def run_backtest(strategies: Dict[str, Dict[str, Any]], symbols: List[str], 
                 start_date: datetime, end_date: datetime, 
                 initial_cash: float = 100000) -> Dict:
    """Run backtest with selected strategies"""
    
    # Create a temporary config for backtest
    config = st.session_state.config
    if not config:
        st.error("Configuration not loaded")
        return None
    
    # Initialize backtest engine with data handler
    data_handler = DataHandler(config)
    engine = BacktestEngine(
        data_handler=data_handler,
        initial_capital=initial_cash
    )
    
    # Create strategy instances
    registry = StrategyRegistry()
    all_results = {}
    
    for strategy_name, params in strategies.items():
        strategy_class = registry.get_strategy_class(strategy_name)
        if strategy_class:
            try:
                # Create strategy config by merging params with base config
                strategy_config = {
                    'name': strategy_name,
                    'symbols': symbols,
                    **params
                }
                
                # Create strategy instance
                strategy = strategy_class(config=strategy_config)
                logger.info(f"Created strategy instance: {strategy_name}")
                
                # Run backtest for this strategy
                for symbol in symbols:
                    result_key = f"{strategy_name}_{symbol}"
                    st.text(f"Backtesting {strategy_name} on {symbol}...")
                    
                    metrics = engine.run_backtest(
                        strategy=strategy,
                        symbols=[symbol],
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    
                    all_results[result_key] = {
                        'strategy': strategy_name,
                        'symbol': symbol,
                        'metrics': metrics,
                        'signals': engine.results.get(strategy.name, {}).get('signals', []),
                        'trades': engine.results.get(strategy.name, {}).get('trades', [])
                    }
                    
            except Exception as e:
                st.error(f"Failed to backtest {strategy_name}: {e}")
                logger.error(f"Strategy backtest error: {e}", exc_info=True)
    
    if not all_results:
        st.error("No valid backtest results")
        return None
    
    # Aggregate results
    aggregated_results = aggregate_backtest_results(all_results)
    
    return aggregated_results


def display_backtest_results(results: Dict):
    """Display comprehensive backtest results"""
    if not results:
        return
    
    # Performance metrics
    st.header("Performance Metrics")
    
    metrics = results.get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
    with col2:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
        
    with col3:
        st.metric("Total Trades", metrics.get('total_trades', 0))
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        
    with col4:
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
    
    # Equity curve
    st.header("Equity Curve")
    
    if 'equity_curve' in results:
        equity_df = pd.DataFrame(results['equity_curve'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    if 'trades' in results and results['trades']:
        st.header("Trade Analysis")
        
        trades_df = pd.DataFrame(results['trades'])
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Winners vs Losers")
            winners = trades_df[trades_df['pnl'] > 0]
            losers = trades_df[trades_df['pnl'] <= 0]
            
            fig = go.Figure(data=[
                go.Bar(name='Winners', x=['Count', 'Avg P&L'], 
                      y=[len(winners), winners['pnl'].mean() if len(winners) > 0 else 0]),
                go.Bar(name='Losers', x=['Count', 'Avg P&L'], 
                      y=[len(losers), losers['pnl'].mean() if len(losers) > 0 else 0])
            ])
            fig.update_layout(barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("P&L Distribution")
            fig = go.Figure(data=[go.Histogram(x=trades_df['pnl'], nbinsx=30)])
            fig.update_layout(
                xaxis_title="P&L ($)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade details
        st.subheader("Recent Trades")
        st.dataframe(
            trades_df.tail(20)[['datetime', 'symbol', 'side', 'quantity', 'price', 'pnl']],
            use_container_width=True
        )
    
    # Strategy breakdown
    if 'strategy_metrics' in results:
        st.header("Strategy Performance Breakdown")
        
        strategy_metrics = results['strategy_metrics']
        
        # Create comparison table
        comparison_data = []
        for strategy_name, metrics in strategy_metrics.items():
            win_rate = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.get('total_return', 0):.2%}",
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate': f"{win_rate:.2%}",
                'Symbols': ', '.join(metrics.get('symbols', []))
            })
        
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
        # Show individual results per strategy-symbol combination
        if 'individual_results' in results:
            st.subheader("Detailed Results by Strategy-Symbol")
            
            for result_key, result in results['individual_results'].items():
                with st.expander(f"{result['strategy']} - {result['symbol']}"):
                    metrics = result['metrics']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Return", f"{metrics.get('total_return', 0):.2%}")
                        st.metric("Trades", metrics.get('total_trades', 0))
                    
                    with col2:
                        st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        st.metric("Max DD", f"{metrics.get('max_drawdown', 0):.2%}")
                    
                    with col3:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
                        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")


def main():
    """Main dashboard application"""
    
    st.title("🚀 AlgoStack Trading Dashboard")
    st.markdown("### Fully Integrated Strategy Testing & Configuration Platform")
    
    # Initialize
    registry = StrategyRegistry()
    
    # Load configuration
    if st.session_state.config is None:
        st.session_state.config = load_config()
    
    # Sidebar for strategy selection and configuration
    with st.sidebar:
        st.header("Strategy Configuration")
        
        # Multi-strategy selection
        available_strategies = registry.list_strategies()
        selected_strategies = st.multiselect(
            "Select Strategies",
            available_strategies,
            help="Choose one or more strategies to test"
        )
        
        # Configure each selected strategy
        strategy_configs = {}
        for strategy_name in selected_strategies:
            with st.expander(f"Configure {strategy_name}"):
                params = create_strategy_config_ui(strategy_name, registry)
                strategy_configs[strategy_name] = params
        
        st.divider()
        
        # Symbol selection
        st.header("Market Selection")
        
        # Get symbols from config or use defaults
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']
        if st.session_state.config:
            config_symbols = st.session_state.config.get('symbols', {})
            if config_symbols:
                default_symbols = list(config_symbols.keys())
        
        symbols = st.multiselect(
            "Select Symbols",
            default_symbols,
            default=['AAPL', 'MSFT', 'SPY']
        )
        
        st.divider()
        
        # Backtest parameters
        st.header("Backtest Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        initial_cash = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        # Run backtest button
        run_backtest_btn = st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True,
            disabled=not (selected_strategies and symbols)
        )
    
    # Main content area
    if not selected_strategies:
        st.info("👈 Select one or more strategies from the sidebar to begin")
        
        # Show available strategies
        st.header("Available Strategies")
        
        for strategy_name in available_strategies:
            strategy_class = registry.get_strategy_class(strategy_name)
            if strategy_class and strategy_class.__doc__:
                st.subheader(strategy_name)
                st.write(strategy_class.__doc__.strip())
    
    elif run_backtest_btn:
        # Run backtest
        st.header("Running Backtest...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        progress_bar.progress(25)
        status_text.text("Initializing strategies...")
        
        # Run the backtest
        results = run_backtest(
            strategies=strategy_configs,
            symbols=symbols,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
            initial_cash=initial_cash
        )
        
        progress_bar.progress(100)
        status_text.text("Backtest complete!")
        
        # Store results
        if results:
            st.session_state.backtest_results = results
            st.success("✅ Backtest completed successfully!")
        else:
            st.error("❌ Backtest failed. Check the logs for details.")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    # Display results if available
    if st.session_state.backtest_results:
        display_backtest_results(st.session_state.backtest_results)
        
        # Export options
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Results (CSV)"):
                # Export trades to CSV
                if 'trades' in st.session_state.backtest_results:
                    trades_df = pd.DataFrame(st.session_state.backtest_results['trades'])
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        "Download Trades CSV",
                        csv,
                        "backtest_trades.csv",
                        "text/csv"
                    )
        
        with col2:
            if st.button("📈 Export Metrics (JSON)"):
                import json
                metrics_json = json.dumps(st.session_state.backtest_results.get('metrics', {}), indent=2)
                st.download_button(
                    "Download Metrics JSON",
                    metrics_json,
                    "backtest_metrics.json",
                    "application/json"
                )
        
        with col3:
            if st.button("🗑️ Clear Results"):
                st.session_state.backtest_results = {}
                st.rerun()


if __name__ == "__main__":
    main()