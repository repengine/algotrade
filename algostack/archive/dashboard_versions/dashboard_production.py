#!/usr/bin/env python3
"""
AlgoStack Dashboard - Production Version
Uses proven strategy implementations with full UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AlgoStack Trading Dashboard",
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


class BuiltInStrategies:
    """Built-in working strategies"""
    
    @staticmethod
    def list_strategies() -> List[str]:
        return ["MA Crossover", "RSI Mean Reversion", "Momentum", "Bollinger Bands"]
    
    @staticmethod
    def get_parameters(name: str) -> Dict[str, Any]:
        if name == "MA Crossover":
            return {
                'fast_period': 10,
                'slow_period': 30
            }
        elif name == "RSI Mean Reversion":
            return {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        elif name == "Momentum":
            return {
                'lookback_period': 20,
                'threshold': 0.05
            }
        elif name == "Bollinger Bands":
            return {
                'bb_period': 20,
                'bb_std': 2.0
            }
        return {}


def fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical data using yfinance"""
    try:
        data = yf.download(
            symbol, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            st.error(f"No data available for {symbol}")
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        data = data.sort_index()
        st.info(f"Fetched {len(data)} days of data for {symbol}")
        
        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()


def simulate_strategy(strategy_name: str, data: pd.DataFrame, params: Dict[str, Any], 
                     commission: float = 0.001, slippage: float = 0.0005) -> pd.DataFrame:
    """Simulate strategy signals on historical data"""
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    signals['position'] = 0.0
    signals['transaction_costs'] = 0.0
    signals['trade_entry'] = 0
    signals['trade_exit'] = 0
    signals['trade_type'] = ''
    
    # Ensure we have enough data
    if len(data) < 50:
        st.warning(f"Not enough data for {strategy_name}")
        signals['returns'] = 0
        signals['strategy_returns'] = 0
        signals['cumulative_returns'] = 1
        signals['buy_hold_returns'] = 1
        return signals
    
    # Strategy implementations
    if strategy_name == "MA Crossover":
        fast_period = params.get('fast_period', 10)
        slow_period = params.get('slow_period', 30)
        
        fast_ma = data['Close'].rolling(window=fast_period).mean()
        slow_ma = data['Close'].rolling(window=slow_period).mean()
        
        # Detect crossovers
        ma_diff = fast_ma - slow_ma
        ma_diff_prev = ma_diff.shift(1)
        
        signals.loc[(ma_diff > 0) & (ma_diff_prev <= 0), 'signal'] = 1
        signals.loc[(ma_diff < 0) & (ma_diff_prev >= 0), 'signal'] = -1
        
    elif strategy_name == "RSI Mean Reversion":
        period = params.get('rsi_period', 14)
        oversold = params.get('rsi_oversold', 30)
        overbought = params.get('rsi_overbought', 70)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals at crossovers
        signals.loc[(rsi < oversold) & (rsi.shift(1) >= oversold), 'signal'] = 1
        signals.loc[(rsi > overbought) & (rsi.shift(1) <= overbought), 'signal'] = -1
        
    elif strategy_name == "Momentum":
        lookback = params.get('lookback_period', 20)
        threshold = params.get('threshold', 0.05)
        
        # Calculate momentum
        momentum = data['Close'].pct_change(lookback)
        
        # Generate signals
        signals.loc[momentum > threshold, 'signal'] = 1
        signals.loc[momentum < -threshold, 'signal'] = -1
        
    elif strategy_name == "Bollinger Bands":
        period = params.get('bb_period', 20)
        std_dev = params.get('bb_std', 2.0)
        
        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Generate signals
        signals.loc[data['Close'] < lower_band, 'signal'] = 1
        signals.loc[data['Close'] > upper_band, 'signal'] = -1
    
    # Forward fill positions
    signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Identify trade points
    entry_mask = ((signals['position'] != 0) & (signals['position'].shift(1).fillna(0) == 0)) | \
                 ((signals['position'] * signals['position'].shift(1).fillna(0)) < 0)
    
    signals.loc[entry_mask, 'trade_entry'] = 1
    signals.loc[entry_mask & (signals['position'] > 0), 'trade_type'] = 'BUY'
    signals.loc[entry_mask & (signals['position'] < 0), 'trade_type'] = 'SHORT'
    
    exit_mask = (signals['position'] == 0) & (signals['position'].shift(1).fillna(0) != 0)
    signals.loc[exit_mask, 'trade_exit'] = 1
    signals.loc[exit_mask, 'trade_type'] = 'EXIT'
    
    # Calculate returns
    signals['returns'] = data['Close'].pct_change().fillna(0)
    signals['gross_returns'] = signals['position'].shift(1).fillna(0) * signals['returns']
    
    # Apply transaction costs
    trade_days = entry_mask | exit_mask
    total_cost = commission + slippage
    signals.loc[trade_days, 'transaction_costs'] = total_cost
    
    # Net returns
    signals['strategy_returns'] = signals['gross_returns'] - signals['transaction_costs']
    
    # Cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    signals['buy_hold_returns'] = (1 + signals['returns']).cumprod()
    
    # Ensure first values are 1.0
    if len(signals) > 0:
        signals.iloc[0, signals.columns.get_loc('cumulative_returns')] = 1.0
        signals.iloc[0, signals.columns.get_loc('buy_hold_returns')] = 1.0
    
    # Debug info
    total_trades = entry_mask.sum()
    final_return = (signals['cumulative_returns'].iloc[-1] - 1) * 100
    st.text(f"{strategy_name}: {total_trades} trades, {final_return:.2f}% return")
    
    return signals


def calculate_metrics(signals: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics"""
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
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(float(drawdown.min() * 100))
    
    # Win rate
    entries = signals[signals['trade_entry'] == 1]
    total_trades = len(entries)
    
    if total_trades > 0:
        # Simple win rate calculation
        profitable_days = (signals['strategy_returns'] > 0).sum()
        total_days = (signals['strategy_returns'] != 0).sum()
        win_rate = (profitable_days / total_days * 100) if total_days > 0 else 0
    else:
        win_rate = 0.0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_trades': int(total_trades)
    }


def display_strategy_configuration():
    """Display strategy selection and configuration"""
    st.header("Strategy Configuration")
    
    strategies = BuiltInStrategies()
    available_strategies = strategies.list_strategies()
    
    cols = st.columns(2)
    
    for i, strategy_name in enumerate(available_strategies):
        col = cols[i % 2]
        
        with col:
            with st.container():
                # Enable/disable strategy
                enabled = st.checkbox(
                    f"**{strategy_name}**",
                    key=f"enable_{strategy_name}",
                    value=strategy_name in st.session_state.enabled_strategies
                )
                
                if enabled:
                    st.session_state.enabled_strategies.add(strategy_name)
                else:
                    st.session_state.enabled_strategies.discard(strategy_name)
                
                # Show parameters if enabled
                if enabled:
                    default_params = strategies.get_parameters(strategy_name)
                    
                    if default_params:
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
                                    step=0.01,
                                    format="%.2f",
                                    key=f"{strategy_name}_{param_name}"
                                )
                        
                        st.session_state.strategy_configs[strategy_name] = params
                
                st.markdown("")


def run_backtest(symbols: List[str], start_date: datetime, end_date: datetime, 
                 commission: float = 0.001, slippage: float = 0.0005) -> Dict:
    """Run backtest with selected strategies"""
    
    enabled_strategies = list(st.session_state.enabled_strategies)
    
    if not enabled_strategies:
        st.warning("No strategies enabled.")
        return {}
    
    results = {}
    
    progress_bar = st.progress(0)
    total_tests = len(symbols) * len(enabled_strategies)
    current_test = 0
    
    for symbol in symbols:
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
            
            params = st.session_state.strategy_configs.get(strategy_name, {})
            signals = simulate_strategy(strategy_name, data, params, commission, slippage)
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
    """Display backtest results"""
    if not results:
        return
    
    # Overall Performance Summary
    st.header("📊 Overall Performance")
    
    all_metrics = []
    for symbol, symbol_data in results.items():
        for strategy_name, strategy_data in symbol_data['strategies'].items():
            metrics = strategy_data['metrics']
            metrics['symbol'] = symbol
            metrics['strategy'] = strategy_name
            all_metrics.append(metrics)
    
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        
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
            avg_trades = df_metrics['total_trades'].mean()
            st.metric("Avg Trades", f"{avg_trades:.0f}")
    
    # Individual Symbol Charts
    st.header("📈 Price Charts with Strategy Performance")
    
    for symbol, symbol_data in results.items():
        st.subheader(f"{symbol}")
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price & Strategy Performance', 'Position Indicators'),
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        data = symbol_data['data']
        
        # Plot price
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
        
        # Plot buy & hold
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
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        # Plot each strategy
        for i, (strategy_name, strategy_data) in enumerate(symbol_data['strategies'].items()):
            signals = strategy_data['signals']
            color = colors[i % len(colors)]
            
            # Plot strategy returns
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
            
            # Plot entry points
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
                            marker=dict(symbol='triangle-up', size=10, color=color),
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
                            marker=dict(symbol='triangle-down', size=10, color=color),
                            showlegend=False
                        ),
                        row=1, col=1,
                        secondary_y=False
                    )
            
            # Plot positions
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
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Returns (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Position", row=2, col=1)
        
        fig.update_layout(height=800, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
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
    """Main application"""
    
    st.title("🚀 AlgoStack Trading Dashboard")
    st.markdown("### Strategy Testing & Analysis Platform")
    
    # Sidebar
    with st.sidebar:
        st.header("Market Selection")
        
        symbols = st.multiselect(
            "Select Symbols",
            ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'QQQ'],
            default=['SPY']
        )
        
        st.divider()
        
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
        
        st.divider()
        
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
        
        # Run button
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
    
    # Main content
    if not run_backtest_btn:
        display_strategy_configuration()
    
    if not st.session_state.enabled_strategies and not run_backtest_btn:
        st.info("👆 Enable at least one strategy above to begin analysis")
    
    elif run_backtest_btn:
        st.header("Running Analysis...")
        
        results = run_backtest(
            symbols=symbols,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
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