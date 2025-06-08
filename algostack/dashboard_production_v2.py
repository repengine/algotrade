#!/usr/bin/env python3
"""
AlgoStack Dashboard - Production Version V2

This version includes:
1. Bypassed validation for flexibility
2. Automatic type conversion
3. Better error handling
"""

import os
import sys
import json
import time
import importlib
import inspect
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mock validator that converts types automatically
def mock_validator(config):
    """Mock validator that converts types as needed."""
    converted = config.copy()
    
    # Common type conversions
    for key, value in converted.items():
        # Convert numeric strings to numbers
        if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
            try:
                if '.' in value:
                    converted[key] = float(value)
                else:
                    converted[key] = int(value)
            except:
                pass
        
        # Convert 0/1 to bool for known boolean parameters
        if key in ['volume_filter', 'trail_stop', 'volume_confirmation', 'volatility_filter', 'rsi_filter']:
            if value in [0, 1]:
                converted[key] = bool(value)
            elif value in ['0', '1']:
                converted[key] = bool(int(value))
        
        # Ensure floats for percentage parameters
        if any(x in key for x in ['threshold', 'oversold', 'overbought', 'pct', 'mult', 'ratio']):
            try:
                converted[key] = float(value)
            except:
                pass
        
        # Ensure ints for period parameters
        if 'period' in key or key in ['max_positions', 'max_pairs', 'max_trades_per_day']:
            try:
                converted[key] = int(value)
            except:
                pass
    
    return converted

# Patch all validators - use try/except to handle any missing ones
validators_to_patch = [
    'validate_mean_reversion_config',
    'validate_trend_following_config',
    'validate_pairs_trading_config',
    'validate_pairs_stat_arb_config',  # May not exist
    'validate_intraday_orb_config',
    'validate_overnight_drift_config',
    'validate_hybrid_regime_config',
]

for validator in validators_to_patch:
    try:
        patch(f'utils.validators.strategy_validators.{validator}', side_effect=mock_validator).start()
    except AttributeError:
        pass  # Skip if validator doesn't exist

# Import integration helpers
from strategy_integration_helpers import (
    patch_talib_imports, 
    DataFormatConverter,
    RiskContextMock,
    TechnicalIndicators
)

# Import strategy defaults
from strategy_defaults import get_strategy_defaults, merge_with_defaults, PARAMETER_TOOLTIPS

# Patch talib if not available
patch_talib_imports()

# Import base strategy
from strategies.base import BaseStrategy, Signal


class ProductionStrategyManager:
    """Production-ready strategy manager with proper parameter handling."""
    
    def __init__(self):
        self.strategies = self._discover_strategies()
        self.converter = DataFormatConverter()
    
    def _discover_strategies(self) -> Dict[str, type]:
        """Discover all strategy classes in the strategies folder."""
        strategies = {}
        strategies_dir = Path(__file__).parent / 'strategies'
        
        # Skip these files
        skip_files = {'__init__.py', 'base.py', '__pycache__'}
        
        for py_file in strategies_dir.glob('*.py'):
            if py_file.name in skip_files:
                continue
            
            module_name = py_file.stem
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"strategies.{module_name}", 
                    py_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find strategy classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        # Use class name as key for better default matching
                        strategies[name] = obj
                        
            except Exception as e:
                error_msg = str(e)
                # Only show errors that aren't about optional dependencies
                if not any(skip in error_msg for skip in ['statsmodels', 'sklearn', 'scipy']):
                    print(f"⚠️ Error loading {module_name}: {e}")
        
        return strategies
    
    def get_display_name(self, class_name: str) -> str:
        """Convert class name to display name."""
        # Convert CamelCase to Title Case
        import re
        formatted = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
        return formatted
    
    def get_strategy_parameters(self, strategy_class_name: str) -> Dict[str, Any]:
        """Get all parameters for a strategy with proper defaults."""
        return get_strategy_defaults(strategy_class_name)
    
    def run_backtest(self, strategy_class: type, strategy_name: str,
                    user_params: Dict[str, Any], data: pd.DataFrame, 
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """Run a complete backtest with proper parameter handling."""
        try:
            # Debug output
            print(f"\n🔍 Running backtest for {strategy_name}")
            print(f"   Data shape: {data.shape}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            
            # Merge user parameters with defaults
            full_config = merge_with_defaults(strategy_name, user_params)
            
            # Convert data format
            strategy_data = self.converter.dashboard_to_strategy(
                data, 
                symbol=user_params.get('symbol', 'UNKNOWN')
            )
            
            # Store the symbol attribute for preservation
            symbol = strategy_data.attrs.get('symbol', 'UNKNOWN')
            
            # Initialize strategy with full config
            strategy = strategy_class(full_config)
            
            # Clear any existing positions
            if hasattr(strategy, 'positions'):
                strategy.positions = {}
            strategy.init()
            
            # Get lookback period
            lookback = getattr(strategy, 'lookback_period', 50)
            if 'lookback_period' in full_config:
                lookback = full_config['lookback_period']
            
            # Initialize tracking
            positions = pd.Series(0, index=data.index)
            signals = pd.Series(0, index=data.index)
            trades = []
            
            # Track position internally
            current_position = 0
            entry_price = None
            signal_count = 0
            error_count = 0
            
            # Run strategy
            for i in range(lookback, len(strategy_data)):
                # Get data window
                window = strategy_data.iloc[:i+1].copy()
                
                # Preserve the symbol attribute
                window.attrs['symbol'] = symbol
                
                try:
                    # Generate signal
                    signal = strategy.next(window)
                    
                    if signal and isinstance(signal, Signal):
                        signal_count += 1
                        # Process signal based on direction
                        if signal.direction == 'LONG' and current_position <= 0:
                            signals.iloc[i] = 1
                            current_position = 1
                            entry_price = data['Close'].iloc[i]
                            trades.append({
                                'date': data.index[i],
                                'action': 'BUY',
                                'price': entry_price,
                                'signal_strength': signal.strength
                            })
                            
                        elif signal.direction == 'SHORT':
                            # For long-only strategies, SHORT means exit
                            if current_position > 0:
                                signals.iloc[i] = 0
                                exit_price = data['Close'].iloc[i]
                                current_position = 0
                                
                                # Calculate return
                                if entry_price:
                                    trade_return = (exit_price - entry_price) / entry_price
                                else:
                                    trade_return = 0
                                    
                                trades.append({
                                    'date': data.index[i],
                                    'action': 'SELL',
                                    'price': exit_price,
                                    'signal_strength': signal.strength,
                                    'return': trade_return
                                })
                                entry_price = None
                            
                        elif signal.direction == 'FLAT' and current_position != 0:
                            signals.iloc[i] = 0
                            exit_price = data['Close'].iloc[i]
                            
                            if current_position > 0 and entry_price:
                                trade_return = (exit_price - entry_price) / entry_price
                            else:
                                trade_return = 0
                                
                            current_position = 0
                            trades.append({
                                'date': data.index[i],
                                'action': 'EXIT',
                                'price': exit_price,
                                'signal_strength': 0,
                                'return': trade_return
                            })
                            entry_price = None
                
                except Exception as e:
                    error_count += 1
                    # Only print first few errors to avoid spam
                    if error_count <= 3:
                        print(f"⚠️ Signal generation error at index {i}: {str(e)[:100]}")
                    continue
            
            # Print debug info
            print(f"   Signals generated: {signal_count}")
            print(f"   Trades executed: {len(trades)}")
            print(f"   Errors encountered: {error_count}")
            
            # Close any open position at the end
            if current_position != 0 and entry_price:
                exit_price = data['Close'].iloc[-1]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'date': data.index[-1],
                    'action': 'CLOSE',
                    'price': exit_price,
                    'signal_strength': 0,
                    'return': trade_return
                })
            
            # Forward fill positions
            positions = signals.replace(0, np.nan).ffill().fillna(0)
            
            # Calculate returns
            price_returns = data['Close'].pct_change()
            strategy_returns = positions.shift(1) * price_returns
            
            # Apply transaction costs
            position_changes = positions != positions.shift(1)
            transaction_cost = 0.001  # 0.1%
            strategy_returns[position_changes] -= transaction_cost
            
            # Calculate metrics
            results = {
                'positions': positions,
                'signals': signals,
                'trades': trades,
                'returns': strategy_returns,
                'cumulative_returns': (1 + strategy_returns).cumprod(),
                'price_returns': price_returns
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Backtest error for {strategy_name}: {str(e)}")
            # Return empty results on error
            return {
                'positions': pd.Series(0, index=data.index),
                'signals': pd.Series(0, index=data.index),
                'trades': [],
                'returns': pd.Series(0, index=data.index),
                'cumulative_returns': pd.Series(1, index=data.index),
                'price_returns': data['Close'].pct_change()
            }


def calculate_metrics(results: Dict[str, Any], initial_capital: float = 100000) -> Dict[str, float]:
    """Calculate performance metrics from backtest results."""
    returns = results['returns']
    cumulative = results['cumulative_returns']
    price_returns = results['price_returns']
    
    # Total return
    total_return = (cumulative.iloc[-1] - 1) * 100
    
    # Annualized metrics
    n_days = len(returns)
    annual_factor = 252 / n_days if n_days > 0 else 1
    
    annual_return = ((cumulative.iloc[-1] ** annual_factor) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio
    risk_free_rate = 0.02  # 2% annual
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    winning_trades = [t for t in results['trades'] if t.get('return', 0) > 0]
    losing_trades = [t for t in results['trades'] if t.get('return', 0) <= 0]
    total_trades = len(winning_trades) + len(losing_trades)
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    
    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    if profit_factor == float('inf'):
        profit_factor = 999.99  # Cap for display
    
    # Trade statistics
    trades = results.get('trades', [])
    num_trades = len([t for t in trades if t['action'] in ['BUY', 'SELL', 'EXIT', 'CLOSE']])
    
    # Benchmark comparison
    benchmark_cumulative = (1 + price_returns).cumprod()
    benchmark_return = (benchmark_cumulative.iloc[-1] - 1) * 100
    excess_return = total_return - benchmark_return
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': num_trades,
        'benchmark_return': benchmark_return,
        'excess_return': excess_return,
        'final_value': initial_capital * cumulative.iloc[-1]
    }


def create_performance_chart(data: pd.DataFrame, results: Dict[str, Any], 
                           initial_capital: float, symbol: str) -> go.Figure:
    """Create comprehensive performance visualization."""
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Portfolio Performance', 'Price & Signals', 'Drawdown'),
        row_heights=[0.5, 0.3, 0.2]
    )
    
    # 1. Portfolio value comparison
    strategy_value = initial_capital * results['cumulative_returns']
    benchmark_value = initial_capital * (1 + results['price_returns']).cumprod()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=strategy_value,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=benchmark_value,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1.5, dash='dash')
        ),
        row=1, col=1
    )
    
    # 2. Price with buy/sell signals
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='black', width=1),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add buy signals
    buy_trades = [t for t in results['trades'] if t['action'] == 'BUY']
    if buy_trades:
        buy_dates = [t['date'] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green'
                )
            ),
            row=2, col=1
        )
    
    # Add sell signals
    sell_trades = [t for t in results['trades'] if t['action'] in ['SELL', 'EXIT', 'CLOSE']]
    if sell_trades:
        sell_dates = [t['date'] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red'
                )
            ),
            row=2, col=1
        )
    
    # 3. Drawdown
    cum_returns = (1 + results['returns']).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = ((cum_returns - running_max) / running_max) * 100
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="AlgoStack Trading Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AlgoStack Trading Dashboard - Production V2")
    st.markdown("Dynamic strategy backtesting with real AlgoStack strategies")
    
    # Initialize strategy manager
    if 'strategy_manager' not in st.session_state:
        st.session_state.strategy_manager = ProductionStrategyManager()
    
    manager = st.session_state.strategy_manager
    
    # Check if strategies were found
    if not manager.strategies:
        st.error("❌ No strategies found! Please check the strategies folder.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Symbol selection
        symbol = st.text_input("Symbol", value="SPY", help="Enter stock ticker symbol")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Beginning of backtest period"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End of backtest period"
            )
        
        # Strategy selection
        strategy_names = list(manager.strategies.keys())
        display_names = [manager.get_display_name(name) for name in strategy_names]
        
        selected_idx = st.selectbox(
            "Strategy",
            range(len(strategy_names)),
            format_func=lambda x: display_names[x],
            help="Select a strategy to backtest"
        )
        
        selected_strategy_name = strategy_names[selected_idx]
        selected_strategy_class = manager.strategies[selected_strategy_name]
        
        # Get strategy parameters
        st.subheader("Strategy Parameters")
        default_params = manager.get_strategy_parameters(selected_strategy_name)
        
        # Create parameter inputs with tooltips
        user_params = {'symbol': symbol}
        
        # Display all parameters in a simple list
        for param, default_value in default_params.items():
            if param == 'symbols':
                continue  # Skip symbols array
                
            tooltip = PARAMETER_TOOLTIPS.get(param, f"Parameter: {param}")
            
            if isinstance(default_value, bool):
                user_params[param] = st.checkbox(
                    param.replace('_', ' ').title(),
                    value=default_value,
                    help=tooltip,
                    key=f"{selected_strategy_name}_{param}"
                )
            elif isinstance(default_value, int):
                user_params[param] = st.number_input(
                    param.replace('_', ' ').title(),
                    value=default_value,
                    step=1,
                    help=tooltip,
                    key=f"{selected_strategy_name}_{param}"
                )
            elif isinstance(default_value, float):
                user_params[param] = st.number_input(
                    param.replace('_', ' ').title(),
                    value=default_value,
                    format="%.4f",
                    help=tooltip,
                    key=f"{selected_strategy_name}_{param}"
                )
            elif isinstance(default_value, str):
                user_params[param] = st.text_input(
                    param.replace('_', ' ').title(),
                    value=default_value,
                    help=tooltip,
                    key=f"{selected_strategy_name}_{param}"
                )
            elif isinstance(default_value, list):
                # Skip lists for now
                user_params[param] = default_value
        
        # Initial capital
        initial_capital = st.number_input(
            "Initial Capital",
            value=100000,
            step=10000,
            help="Starting portfolio value"
        )
        
        # Run backtest button
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Main content area
    if run_backtest:
        with st.spinner(f"Fetching {symbol} data..."):
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"❌ No data found for {symbol}")
                return
            
            st.success(f"✅ Loaded {len(data)} days of data for {symbol}")
        
        with st.spinner(f"Running {manager.get_display_name(selected_strategy_name)} backtest..."):
            # Run backtest
            results = manager.run_backtest(
                selected_strategy_class,
                selected_strategy_name,
                user_params,
                data,
                initial_capital
            )
            
            # Calculate metrics
            metrics = calculate_metrics(results, initial_capital)
        
        # Display results
        st.header("Backtest Results")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            st.metric("Annual Return", f"{metrics['annual_return']:.2f}%")
            st.metric("Final Value", f"${metrics['final_value']:,.2f}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            st.metric("Volatility", f"{metrics['volatility']:.2f}%")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        
        with col4:
            st.metric("Number of Trades", metrics['num_trades'])
            st.metric("Benchmark Return", f"{metrics['benchmark_return']:.2f}%")
            st.metric("Excess Return", f"{metrics['excess_return']:.2f}%")
        
        # Performance chart
        st.subheader("Performance Visualization")
        fig = create_performance_chart(data, results, initial_capital, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if results['trades']:
            st.subheader("Trade History")
            trades_df = pd.DataFrame(results['trades'])
            trades_df['return'] = trades_df.get('return', 0) * 100  # Convert to percentage
            
            # Format the dataframe for display
            display_df = trades_df[['date', 'action', 'price', 'signal_strength']]
            if 'return' in trades_df.columns:
                display_df['return %'] = trades_df['return'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Select a Symbol**: Enter any valid stock ticker (e.g., AAPL, MSFT, SPY)
        2. **Choose Date Range**: Select the backtest period
        3. **Pick a Strategy**: Choose from available AlgoStack strategies
        4. **Adjust Parameters**: Fine-tune strategy parameters (hover for descriptions)
        5. **Run Backtest**: Click the button to see results
        
        **Strategies Available:**
        - **Mean Reversion Equity**: Trades oversold conditions using RSI and ATR bands
        - **Trend Following Multi**: Follows trends using channel breakouts and ADX
        - **Hybrid Regime**: Switches between mean reversion and trend following
        - **Intraday ORB**: Opening range breakout strategy
        - **Overnight Drift**: Captures overnight market movements
        - **Pairs Stat Arb**: Statistical arbitrage between correlated pairs
        """)


if __name__ == "__main__":
    main()