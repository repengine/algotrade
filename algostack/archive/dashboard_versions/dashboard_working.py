#!/usr/bin/env python3
"""
AlgoStack Dashboard - Working Version with Fixes
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
import yfinance as yf

# Add the algostack directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Handle TA-Lib import gracefully
try:
    import talib
except ImportError:
    import mock_talib as talib
    sys.modules['talib'] = talib

# Import strategy base class
from strategies.base import BaseStrategy

# Page configuration
st.set_page_config(
    page_title="AlgoStack Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'strategy_configs' not in st.session_state:
    st.session_state.strategy_configs = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'enabled_strategies' not in st.session_state:
    st.session_state.enabled_strategies = set()


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
                        
            except Exception as e:
                st.error(f"Failed to load strategy from {module_name}: {e}")
    
    def get_strategy_class(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class by name"""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names"""
        return sorted(list(self.strategies.keys()))
    
    def get_strategy_parameters(self, name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        # Strategy-specific default parameters
        if "Mean Reversion" in name:
            return {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        elif "Trend Following" in name:
            return {
                'fast_period': 20,
                'slow_period': 50
            }
        else:
            return {}


def fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical data using yfinance"""
    try:
        # Download data with explicit date range
        data = yf.download(
            symbol, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            st.error(f"No data available for {symbol} in the selected date range")
            return pd.DataFrame()
        
        # Handle MultiIndex columns from yfinance (when downloading single symbol)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        st.info(f"Fetched {len(data)} days of data for {symbol}")
        
        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()


def simulate_strategy_signals(strategy_name: str, data: pd.DataFrame, params: Dict[str, Any], 
                            commission: float = 0.001, slippage: float = 0.0005) -> pd.DataFrame:
    """Simulate strategy signals on historical data"""
    # Create signals dataframe
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    signals['position'] = 0.0
    
    # Initialize columns as float
    signals['transaction_costs'] = 0.0
    signals['trade_entry'] = 0
    signals['trade_exit'] = 0
    signals['trade_type'] = ''
    
    # Ensure we have enough data
    if len(data) < 50:
        st.warning(f"Not enough data for {strategy_name} (only {len(data)} days)")
        signals['returns'] = 0
        signals['strategy_returns'] = 0
        signals['cumulative_returns'] = 1
        signals['buy_hold_returns'] = 1
        return signals
    
    # Simple example strategies
    if "Mean Reversion" in strategy_name:
        # RSI-based mean reversion
        period = params.get('rsi_period', 14)
        oversold = params.get('rsi_oversold', 30)
        overbought = params.get('rsi_overbought', 70)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals - use shift to avoid look-ahead bias
        signals.loc[rsi < oversold, 'signal'] = 1  # Buy signal
        signals.loc[rsi > overbought, 'signal'] = -1  # Sell signal
        
        # Debug info
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        st.text(f"RSI Strategy: {buy_signals} buy signals, {sell_signals} sell signals")
        
    elif "Trend Following" in strategy_name:
        # Moving average crossover
        fast_period = params.get('fast_period', 20)
        slow_period = params.get('slow_period', 50)
        
        fast_ma = data['Close'].rolling(window=fast_period).mean()
        slow_ma = data['Close'].rolling(window=slow_period).mean()
        
        # Generate signals when trend changes
        signals.loc[fast_ma > slow_ma, 'signal'] = 1
        signals.loc[fast_ma <= slow_ma, 'signal'] = -1
        
        # Debug info
        long_signals = (signals['signal'] == 1).sum()
        short_signals = (signals['signal'] == -1).sum()
        st.text(f"MA Strategy: {long_signals} long periods, {short_signals} short periods")
    
    # Forward fill positions
    signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Identify trade points - only when position actually changes
    position_changes = signals['position'].diff()
    
    # Entry: position changes from 0 to non-zero
    entry_mask = (signals['position'].shift(1).fillna(0) == 0) & (signals['position'] != 0)
    signals.loc[entry_mask, 'trade_entry'] = 1
    signals.loc[entry_mask & (signals['position'] > 0), 'trade_type'] = 'BUY'
    signals.loc[entry_mask & (signals['position'] < 0), 'trade_type'] = 'SHORT'
    
    # Exit: position changes from non-zero to 0 or flips
    exit_mask = ((signals['position'].shift(1).fillna(0) != 0) & (signals['position'] == 0)) | \
                ((signals['position'].shift(1).fillna(0) * signals['position']) < 0)
    signals.loc[exit_mask, 'trade_exit'] = 1
    signals.loc[exit_mask, 'trade_type'] = 'EXIT'
    
    # Calculate returns
    signals['returns'] = data['Close'].pct_change().fillna(0)
    
    # Calculate strategy returns with transaction costs
    signals['gross_returns'] = signals['position'].shift(1).fillna(0) * signals['returns']
    
    # Apply transaction costs on trade days
    trade_days = entry_mask | exit_mask
    total_cost = commission + slippage
    signals.loc[trade_days, 'transaction_costs'] = total_cost
    
    # Net returns after costs
    signals['strategy_returns'] = signals['gross_returns'] - signals['transaction_costs']
    
    # Calculate cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    signals['buy_hold_returns'] = (1 + signals['returns']).cumprod()
    
    # Ensure first values are 1.0
    if len(signals) > 0:
        signals.loc[signals.index[0], 'cumulative_returns'] = 1.0
        signals.loc[signals.index[0], 'buy_hold_returns'] = 1.0
    
    # Fill any remaining NaN values
    signals = signals.fillna(method='ffill').fillna(0)
    
    # Debug final stats
    total_trades = entry_mask.sum()
    final_return = (signals['cumulative_returns'].iloc[-1] - 1) * 100
    st.text(f"Total trades: {total_trades}, Final return: {final_return:.2f}%")
    
    return signals


def calculate_metrics(signals: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics from signals"""
    # Check if we have valid data
    if len(signals) < 2:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }
    
    strategy_returns = signals['strategy_returns'].dropna()
    
    # Total return
    total_return = (signals['cumulative_returns'].iloc[-1] - 1) * 100
    
    # Sharpe ratio
    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
        sharpe_ratio = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
    else:
        sharpe_ratio = 0.0
    
    # Max drawdown
    cumulative = signals['cumulative_returns']
    if len(cumulative) > 0:
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(float(drawdown.min() * 100))
    else:
        max_drawdown = 0.0
    
    # Win rate and trade counting
    entries = signals[signals['trade_entry'] == 1]
    total_trades = len(entries)
    
    if total_trades > 0:
        # Calculate P&L for each completed trade
        trade_results = []
        
        entry_points = signals[signals['trade_entry'] == 1].copy()
        exit_points = signals[signals['trade_exit'] == 1].copy()
        
        for entry_idx in entry_points.index:
            entry_price = signals.loc[entry_idx, 'price']
            entry_position = signals.loc[entry_idx, 'position']
            
            # Find corresponding exit
            future_exits = exit_points[exit_points.index > entry_idx]
            if len(future_exits) > 0:
                exit_idx = future_exits.index[0]
                exit_price = signals.loc[exit_idx, 'price']
                
                # Calculate return based on position type
                if entry_position > 0:  # Long trade
                    trade_return = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    trade_return = (entry_price - exit_price) / entry_price
                
                trade_results.append(trade_return)
        
        if trade_results:
            winning_trades = sum(1 for r in trade_results if r > 0)
            win_rate = (winning_trades / len(trade_results)) * 100
            total_trades = len(trade_results)
        else:
            win_rate = 0.0
    else:
        win_rate = 0.0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_trades': int(total_trades)
    }


def display_strategy_configuration(registry: StrategyRegistry):
    """Display strategy selection and configuration interface"""
    st.header("Strategy Configuration")
    
    available_strategies = registry.list_strategies()
    
    # Use columns only in main area
    cols = st.columns(2)
    
    for i, strategy_name in enumerate(available_strategies):
        col = cols[i % 2]
        
        with col:
            # Strategy container
            with st.container():
                # On/Off toggle with strategy name
                enabled = st.checkbox(
                    f"**{strategy_name}**",
                    key=f"enable_{strategy_name}",
                    value=strategy_name in st.session_state.enabled_strategies
                )
                
                if enabled:
                    st.session_state.enabled_strategies.add(strategy_name)
                    # Force a rerun to update the button state
                    if strategy_name not in st.session_state.enabled_strategies:
                        st.rerun()
                else:
                    st.session_state.enabled_strategies.discard(strategy_name)
                    # Force a rerun to update the button state
                    if strategy_name in st.session_state.enabled_strategies:
                        st.rerun()
                
                # Show parameters if enabled
                if enabled:
                    # Get default parameters
                    default_params = registry.get_strategy_parameters(strategy_name)
                    
                    if default_params:
                        # Create parameter inputs
                        params = {}
                        
                        for param_name, default_value in default_params.items():
                            if isinstance(default_value, int):
                                params[param_name] = st.number_input(
                                    param_name.replace("_", " ").title(),
                                    value=default_value,
                                    step=1,
                                    key=f"{strategy_name}_{param_name}"
                                )
                            elif isinstance(default_value, float):
                                params[param_name] = st.number_input(
                                    param_name.replace("_", " ").title(),
                                    value=default_value,
                                    step=0.1,
                                    format="%.2f",
                                    key=f"{strategy_name}_{param_name}"
                                )
                        
                        # Store configuration
                        st.session_state.strategy_configs[strategy_name] = params
                
                # Add spacing
                st.markdown("")


def run_backtest(symbols: List[str], start_date: datetime, end_date: datetime, 
                 initial_capital: float = 100000, commission: float = 0.001, 
                 slippage: float = 0.0005) -> Dict:
    """Run backtest with selected strategies"""
    
    enabled_strategies = list(st.session_state.enabled_strategies)
    
    if not enabled_strategies:
        st.warning("No strategies enabled. Please enable at least one strategy.")
        return {}
    
    results = {}
    
    progress_bar = st.progress(0)
    total_tests = len(symbols) * len(enabled_strategies)
    current_test = 0
    
    for symbol in symbols:
        # Fetch data once per symbol
        data = fetch_data(symbol, start_date, end_date)
        
        if data.empty:
            continue
        
        symbol_results = {
            'data': data,
            'strategies': {}
        }
        
        for strategy_name in enabled_strategies:
            current_test += 1
            progress_bar.progress(current_test / total_tests)
            
            st.text(f"Testing {strategy_name} on {symbol}...")
            
            # Get strategy parameters
            params = st.session_state.strategy_configs.get(strategy_name, {})
            
            # Simulate strategy
            signals = simulate_strategy_signals(strategy_name, data, params, commission, slippage)
            metrics = calculate_metrics(signals)
            
            symbol_results['strategies'][strategy_name] = {
                'signals': signals,
                'metrics': metrics,
                'params': params
            }
        
        results[symbol] = symbol_results
    
    progress_bar.empty()
    
    return results


def display_results(results: Dict):
    """Display backtest results with properly scaled charts"""
    if not results:
        return
    
    # Overall Performance Summary
    st.header("📊 Overall Performance")
    
    # Aggregate metrics across all strategies and symbols
    all_metrics = []
    for symbol, symbol_data in results.items():
        for strategy_name, strategy_data in symbol_data['strategies'].items():
            metrics = strategy_data['metrics']
            metrics['symbol'] = symbol
            metrics['strategy'] = strategy_name
            all_metrics.append(metrics)
    
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        
        # Display average metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_return = df_metrics['total_return'].mean()
            st.metric("Avg Return", f"{avg_return:.2f}%")
        
        with col2:
            avg_sharpe = df_metrics['sharpe_ratio'].mean()
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
        
        with col3:
            max_dd = df_metrics['max_drawdown'].max()
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        with col4:
            avg_win_rate = df_metrics['win_rate'].mean()
            st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
    
    # Individual Symbol Charts
    st.header("📈 Price Charts with Strategy Performance")
    
    for symbol, symbol_data in results.items():
        st.subheader(f"{symbol}")
        
        # Create figure with subplots - PROPERLY CONFIGURED FOR SECONDARY Y-AXIS
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price & Strategy Performance', 'Position Indicators'),
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Get price data
        data = symbol_data['data']
        
        # Plot price on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1,
            secondary_y=False
        )
        
        # Calculate and plot buy & hold returns on secondary y-axis
        buy_hold_returns = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=buy_hold_returns,
                mode='lines',
                name='Buy & Hold %',
                line=dict(color='gray', width=2, dash='dash')
            ),
            row=1, col=1,
            secondary_y=True
        )
        
        # Color palette for strategies
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        # Plot each strategy's performance
        for i, (strategy_name, strategy_data) in enumerate(symbol_data['strategies'].items()):
            signals = strategy_data['signals']
            color = colors[i % len(colors)]
            
            # Plot strategy cumulative returns on secondary y-axis
            strategy_returns = (signals['cumulative_returns'] - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=strategy_returns,
                    mode='lines',
                    name=f'{strategy_name} %',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1,
                secondary_y=True
            )
            
            # Plot trade entry points on price chart
            entries = signals[signals['trade_entry'] == 1]
            if not entries.empty:
                long_entries = entries[signals.loc[entries.index, 'position'] > 0]
                short_entries = entries[signals.loc[entries.index, 'position'] < 0]
                
                if not long_entries.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=long_entries.index,
                            y=long_entries['price'],
                            mode='markers',
                            name=f'{strategy_name} Buy',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color=color,
                                line=dict(width=2, color='white')
                            ),
                            showlegend=False
                        ),
                        row=1, col=1,
                        secondary_y=False
                    )
                
                if not short_entries.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=short_entries.index,
                            y=short_entries['price'],
                            mode='markers',
                            name=f'{strategy_name} Short',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color=color,
                                line=dict(width=2, color='white')
                            ),
                            showlegend=False
                        ),
                        row=1, col=1,
                        secondary_y=False
                    )
            
            # Plot exit points
            exits = signals[signals['trade_exit'] == 1]
            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits.index,
                        y=exits['price'],
                        mode='markers',
                        name=f'{strategy_name} Exit',
                        marker=dict(
                            symbol='x',
                            size=12,
                            color=color,
                            line=dict(width=2)
                        ),
                        showlegend=False
                    ),
                    row=1, col=1,
                    secondary_y=False
                )
            
            # Plot position indicator in bottom panel
            fig.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=signals['position'],
                    mode='lines',
                    name=f'{strategy_name} Pos',
                    line=dict(color=color, width=2),
                    fill='tozeroy'
                ),
                row=2, col=1
            )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Returns (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Position", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics table
        st.subheader(f"{symbol} Strategy Metrics")
        
        symbol_metrics = []
        for strategy_name, strategy_data in symbol_data['strategies'].items():
            metrics = strategy_data['metrics'].copy()
            metrics['Strategy'] = strategy_name
            symbol_metrics.append(metrics)
        
        if symbol_metrics:
            df_symbol = pd.DataFrame(symbol_metrics)
            df_symbol = df_symbol[['Strategy', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']]
            df_symbol.columns = ['Strategy', 'Return (%)', 'Sharpe', 'Max DD (%)', 'Win Rate (%)', 'Trades']
            st.dataframe(df_symbol)


def main():
    """Main dashboard application"""
    
    st.title("🚀 AlgoStack Trading Dashboard")
    st.markdown("### Strategy Testing & Analysis Platform")
    
    # Initialize
    registry = StrategyRegistry()
    
    # Debug info
    if st.checkbox("Show Debug Info", value=False):
        st.write("Enabled strategies:", st.session_state.enabled_strategies)
        st.write("Number of enabled strategies:", len(st.session_state.enabled_strategies))
    
    # Sidebar
    with st.sidebar:
        # Market selection
        st.header("Market Selection")
        
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']
        symbols = st.multiselect(
            "Select Symbols",
            default_symbols,
            default=['SPY']
        )
        
        st.divider()
        
        # Time period
        st.header("Time Period")
        
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
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        st.divider()
        
        # Trading costs
        st.header("Trading Costs")
        
        commission = st.number_input(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f"
        ) / 100
        
        slippage = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f"
        ) / 100
        
        # Run button - check both symbols and strategies
        button_disabled = not (symbols and len(st.session_state.enabled_strategies) > 0)
        
        run_backtest_btn = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=button_disabled
        )
        
        if button_disabled:
            if not symbols:
                st.warning("Select at least one symbol")
            if len(st.session_state.enabled_strategies) == 0:
                st.warning("Enable at least one strategy")
    
    # Main content area
    if not run_backtest_btn:
        display_strategy_configuration(registry)
    
    # Show message if no strategies enabled
    if not st.session_state.enabled_strategies and not run_backtest_btn:
        st.info("👆 Enable at least one strategy above to begin analysis")
    
    elif run_backtest_btn:
        st.header("Running Analysis...")
        
        # Run backtest
        results = run_backtest(
            symbols=symbols,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        if results:
            st.session_state.backtest_results = results
            st.success("✅ Analysis completed!")
        else:
            st.error("❌ Analysis failed")
    
    # Display results
    if st.session_state.backtest_results:
        display_results(st.session_state.backtest_results)


if __name__ == "__main__":
    main()