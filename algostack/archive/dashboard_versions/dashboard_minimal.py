#!/usr/bin/env python3
"""
AlgoStack Dashboard - Minimal Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AlgoStack Dashboard (Minimal)",
    page_icon="📈",
    layout="wide"
)

def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    return data

def ma_crossover_strategy(data: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    """Simple MA crossover strategy"""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate MAs
    fast_ma = data['Close'].rolling(window=fast).mean()
    slow_ma = data['Close'].rolling(window=slow).mean()
    
    # Detect crossovers
    ma_diff = fast_ma - slow_ma
    ma_diff_prev = ma_diff.shift(1)
    
    signals['signal'] = 0
    signals.loc[(ma_diff > 0) & (ma_diff_prev <= 0), 'signal'] = 1
    signals.loc[(ma_diff < 0) & (ma_diff_prev >= 0), 'signal'] = -1
    
    # Forward fill positions
    signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Calculate returns
    signals['returns'] = data['Close'].pct_change().fillna(0)
    signals['strategy_returns'] = signals['position'].shift(1).fillna(0) * signals['returns']
    
    # Cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    signals['buy_hold_returns'] = (1 + signals['returns']).cumprod()
    
    return signals

def rsi_strategy(data: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """RSI mean reversion strategy"""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals at crossovers
    signals['signal'] = 0
    signals.loc[(rsi < oversold) & (rsi.shift(1) >= oversold), 'signal'] = 1
    signals.loc[(rsi > overbought) & (rsi.shift(1) <= overbought), 'signal'] = -1
    
    # Forward fill positions
    signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
    
    # Calculate returns
    signals['returns'] = data['Close'].pct_change().fillna(0)
    signals['strategy_returns'] = signals['position'].shift(1).fillna(0) * signals['returns']
    
    # Cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    signals['buy_hold_returns'] = (1 + signals['returns']).cumprod()
    
    return signals

def plot_results(symbol: str, data: pd.DataFrame, strategies: dict):
    """Plot results"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price & Strategy Performance', 'Positions'),
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
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
    buy_hold_pct = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=buy_hold_pct,
            mode='lines',
            name='Buy & Hold %',
            line=dict(color='gray', width=2, dash='dash')
        ),
        row=1, col=1,
        secondary_y=True
    )
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (name, signals) in enumerate(strategies.items()):
        color = colors[i % len(colors)]
        
        # Plot strategy returns
        strategy_pct = ((signals['cumulative_returns']) - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=strategy_pct,
                mode='lines',
                name=f'{name} %',
                line=dict(color=color, width=2)
            ),
            row=1, col=1,
            secondary_y=True
        )
        
        # Plot entry/exit points
        entries = signals[signals['signal'] == 1]
        exits = signals[signals['signal'] == -1]
        
        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries.index,
                    y=entries['price'],
                    mode='markers',
                    name=f'{name} Buy',
                    marker=dict(symbol='triangle-up', size=10, color=color),
                    showlegend=False
                ),
                row=1, col=1,
                secondary_y=False
            )
        
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits.index,
                    y=exits['price'],
                    mode='markers',
                    name=f'{name} Sell',
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
                name=f'{name} Position',
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
    
    return fig

def main():
    st.title("📈 AlgoStack Dashboard - Minimal Version")
    st.markdown("### Simple Strategy Testing")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        symbol = st.selectbox("Symbol", ['SPY', 'AAPL', 'MSFT', 'GOOGL'], index=0)
        
        days = st.slider("Days of History", 30, 365*2, 365)
        
        st.divider()
        
        st.header("Strategies")
        
        use_ma = st.checkbox("MA Crossover", value=True)
        if use_ma:
            fast_ma = st.number_input("Fast MA", 5, 50, 10)
            slow_ma = st.number_input("Slow MA", 20, 200, 30)
        
        use_rsi = st.checkbox("RSI", value=True)
        if use_rsi:
            rsi_period = st.number_input("RSI Period", 5, 30, 14)
            rsi_oversold = st.number_input("RSI Oversold", 10, 40, 30)
            rsi_overbought = st.number_input("RSI Overbought", 60, 90, 70)
        
        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Main area
    if run_btn:
        with st.spinner("Fetching data..."):
            data = fetch_data(symbol, days)
        
        if data.empty:
            st.error("No data available")
            return
        
        st.success(f"Fetched {len(data)} days of data for {symbol}")
        
        strategies = {}
        
        if use_ma:
            with st.spinner("Running MA Crossover strategy..."):
                ma_signals = ma_crossover_strategy(data, fast_ma, slow_ma)
                strategies['MA Crossover'] = ma_signals
        
        if use_rsi:
            with st.spinner("Running RSI strategy..."):
                rsi_signals = rsi_strategy(data, rsi_period, rsi_oversold, rsi_overbought)
                strategies['RSI'] = rsi_signals
        
        if strategies:
            # Show results
            st.header("Results")
            
            # Metrics
            cols = st.columns(len(strategies) + 1)
            
            # Buy & Hold
            buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            cols[0].metric("Buy & Hold", f"{buy_hold_return:.2f}%")
            
            # Strategy metrics
            for i, (name, signals) in enumerate(strategies.items()):
                strategy_return = (signals['cumulative_returns'].iloc[-1] - 1) * 100
                trades = (signals['signal'] != 0).sum()
                cols[i+1].metric(name, f"{strategy_return:.2f}%", f"{trades} trades")
            
            # Chart
            st.subheader("Performance Chart")
            fig = plot_results(symbol, data, strategies)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade details
            st.subheader("Trade Details")
            for name, signals in strategies.items():
                trades = signals[signals['signal'] != 0][['price', 'signal']]
                if not trades.empty:
                    st.write(f"**{name} Trades:**")
                    trades['Action'] = trades['signal'].map({1: 'BUY', -1: 'SELL'})
                    trades = trades[['price', 'Action']].rename(columns={'price': 'Price'})
                    st.dataframe(trades, use_container_width=True)
                    st.write("")

if __name__ == "__main__":
    main()