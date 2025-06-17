#!/usr/bin/env python3
"""
Interactive Mean Reversion Strategy Dashboard
Visualize backtests and tweak parameters in real-time
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Mean Reversion Strategy Dashboard", page_icon="📊", layout="wide"
)

st.title("🎯 Mean Reversion Strategy Dashboard")
st.markdown("Interactive backtesting with real-time parameter optimization")


@st.cache_data(ttl=3600)
def load_market_data(symbol, interval, days):
    """Load market data with caching."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    data.columns = data.columns.str.lower()

    return data


def calculate_indicators(data, lookback, rsi_period):
    """Calculate technical indicators."""
    df = data.copy()

    # Z-score
    df["sma"] = df["close"].rolling(lookback).mean()
    df["std"] = df["close"].rolling(lookback).std()
    df["zscore"] = (df["close"] - df["sma"]) / df["std"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # Bollinger Bands
    df["bb_upper"] = df["sma"] + (df["std"] * 2)
    df["bb_lower"] = df["sma"] - (df["std"] * 2)

    return df


def run_backtest(data, params):
    """Run backtest with given parameters."""
    df = calculate_indicators(data, params["lookback_period"], params["rsi_period"])

    # Initialize
    initial_capital = 10000
    cash = initial_capital
    position = 0
    shares = 0
    trades = []
    equity_curve = []
    signals = []

    for i in range(params["lookback_period"], len(df)):
        current_price = df["close"].iloc[i]
        current_zscore = df["zscore"].iloc[i]
        current_rsi = df["rsi"].iloc[i]
        current_time = df.index[i]
        current_atr = df["atr"].iloc[i]

        # Skip invalid values
        if pd.isna(current_zscore) or pd.isna(current_rsi):
            continue

        # Track equity
        if position > 0:
            current_equity = cash + (shares * current_price)
        else:
            current_equity = cash

        equity_curve.append(
            {"time": current_time, "equity": current_equity, "price": current_price}
        )

        # Trading logic
        if position == 0:
            # Entry signal
            if (
                current_zscore < -params["zscore_threshold"]
                and current_rsi < params["rsi_oversold"]
            ):

                shares = int(cash * params["position_size"] / current_price)
                if shares > 0:
                    cash -= shares * current_price
                    position = 1
                    entry_price = current_price
                    stop_price = current_price - (current_atr * params["stop_loss_atr"])

                    trades.append(
                        {
                            "time": current_time,
                            "action": "BUY",
                            "price": current_price,
                            "shares": shares,
                            "zscore": current_zscore,
                            "rsi": current_rsi,
                        }
                    )

                    signals.append(
                        {"time": current_time, "price": current_price, "type": "BUY"}
                    )

        else:  # position == 1
            # Exit conditions
            exit_signal = False
            exit_reason = ""

            if current_zscore > -params["exit_zscore"]:
                exit_signal = True
                exit_reason = "mean_reversion"
            elif current_price <= stop_price:
                exit_signal = True
                exit_reason = "stop_loss"

            if exit_signal:
                cash += shares * current_price
                pnl_pct = (current_price - entry_price) / entry_price * 100

                trades.append(
                    {
                        "time": current_time,
                        "action": "SELL",
                        "price": current_price,
                        "shares": shares,
                        "pnl_pct": pnl_pct,
                        "reason": exit_reason,
                    }
                )

                signals.append(
                    {"time": current_time, "price": current_price, "type": "SELL"}
                )

                position = 0
                shares = 0

    # Close final position
    if position > 0:
        final_price = df["close"].iloc[-1]
        cash += shares * final_price
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append(
            {
                "action": "SELL",
                "price": final_price,
                "pnl_pct": pnl_pct,
                "reason": "end_of_test",
            }
        )

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    total_return = (cash - initial_capital) / initial_capital * 100

    # Trade statistics
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    num_trades = len(buy_trades)

    if num_trades > 0:
        winning_trades = len([t for t in trades if t.get("pnl_pct", 0) > 0])
        win_rate = winning_trades / num_trades * 100

        wins = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) > 0]
        losses = [t["pnl_pct"] for t in trades if t.get("pnl_pct", 0) < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else 0

        # Sharpe ratio
        returns = equity_df["equity"].pct_change().dropna()
        periods_per_year = 252 * 78 if params["interval"] == "5m" else 252
        sharpe = np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-6)
    else:
        win_rate = avg_win = avg_loss = profit_factor = sharpe = 0

    return {
        "equity_curve": equity_df,
        "trades": trades,
        "signals": signals,
        "metrics": {
            "total_return": total_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "final_value": cash,
        },
        "indicators": df,
    }


# Sidebar for parameters
st.sidebar.header("📈 Strategy Parameters")

# Data settings
st.sidebar.subheader("Data Settings")
symbol = st.sidebar.selectbox("Symbol", ["SPY", "QQQ", "IWM", "DIA"], index=0)
interval = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=0)
lookback_days = st.sidebar.slider("Lookback Days", 7, 60, 30)

# Strategy parameters
st.sidebar.subheader("Mean Reversion Parameters")
lookback_period = st.sidebar.slider("Lookback Period (bars)", 5, 50, 15)
zscore_threshold = st.sidebar.slider("Z-Score Entry Threshold", 0.5, 3.0, 1.5, 0.1)
exit_zscore = st.sidebar.slider("Z-Score Exit", -1.0, 1.0, 0.0, 0.1)

st.sidebar.subheader("RSI Parameters")
rsi_period = st.sidebar.slider("RSI Period", 2, 14, 3)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10.0, 40.0, 25.0, 1.0)

st.sidebar.subheader("Risk Management")
stop_loss_atr = st.sidebar.slider("Stop Loss (ATR)", 1.0, 5.0, 2.5, 0.5)
position_size = st.sidebar.slider("Position Size", 0.5, 1.0, 0.95, 0.05)

# Load data
data = load_market_data(symbol, interval, lookback_days)

# Prepare parameters
params = {
    "lookback_period": lookback_period,
    "zscore_threshold": zscore_threshold,
    "exit_zscore": exit_zscore,
    "rsi_period": rsi_period,
    "rsi_oversold": rsi_oversold,
    "stop_loss_atr": stop_loss_atr,
    "position_size": position_size,
    "interval": interval,
}

# Run backtest
results = run_backtest(data, params)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Return",
        f"{results['metrics']['total_return']:.2f}%",
        delta=f"${results['metrics']['final_value'] - 10000:,.2f}",
    )

with col2:
    st.metric(
        "Win Rate",
        f"{results['metrics']['win_rate']:.1f}%",
        delta=f"{results['metrics']['num_trades']} trades",
    )

with col3:
    st.metric(
        "Profit Factor",
        f"{results['metrics']['profit_factor']:.2f}",
        delta=f"Sharpe: {results['metrics']['sharpe_ratio']:.2f}",
    )

with col4:
    # Annualized return
    days = (data.index[-1] - data.index[0]).days
    if days > 0 and results["metrics"]["total_return"] > 0:
        annual_return = (1 + results["metrics"]["total_return"] / 100) ** (
            365 / days
        ) - 1
        st.metric(
            "Annual Return", f"{annual_return*100:.1f}%", delta=f"from {days} days"
        )
    else:
        st.metric("Annual Return", "N/A")

# Main chart
st.subheader("📊 Equity Curve & Trading Signals")

# Create main figure with subplots
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Price & Signals", "Z-Score", "RSI"),
)

# Price chart with signals
df_plot = results["indicators"].iloc[-500:]  # Last 500 bars
fig.add_trace(
    go.Candlestick(
        x=df_plot.index,
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="Price",
    ),
    row=1,
    col=1,
)

# Add Bollinger Bands
fig.add_trace(
    go.Scatter(
        x=df_plot.index,
        y=df_plot["bb_upper"],
        line={"color": "gray", "width": 1, "dash": "dash"},
        name="BB Upper",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df_plot.index,
        y=df_plot["bb_lower"],
        line={"color": "gray", "width": 1, "dash": "dash"},
        name="BB Lower",
    ),
    row=1,
    col=1,
)

# Add SMA
fig.add_trace(
    go.Scatter(
        x=df_plot.index,
        y=df_plot["sma"],
        line={"color": "orange", "width": 2},
        name="SMA",
    ),
    row=1,
    col=1,
)

# Add buy/sell signals
buy_signals = [
    s for s in results["signals"] if s["type"] == "BUY" and s["time"] in df_plot.index
]
sell_signals = [
    s for s in results["signals"] if s["type"] == "SELL" and s["time"] in df_plot.index
]

if buy_signals:
    fig.add_trace(
        go.Scatter(
            x=[s["time"] for s in buy_signals],
            y=[s["price"] for s in buy_signals],
            mode="markers",
            marker={"symbol": "triangle-up", "size": 12, "color": "green"},
            name="Buy",
        ),
        row=1,
        col=1,
    )

if sell_signals:
    fig.add_trace(
        go.Scatter(
            x=[s["time"] for s in sell_signals],
            y=[s["price"] for s in sell_signals],
            mode="markers",
            marker={"symbol": "triangle-down", "size": 12, "color": "red"},
            name="Sell",
        ),
        row=1,
        col=1,
    )

# Z-Score
fig.add_trace(
    go.Scatter(
        x=df_plot.index,
        y=df_plot["zscore"],
        line={"color": "blue", "width": 2},
        name="Z-Score",
    ),
    row=2,
    col=1,
)

# Z-Score thresholds
fig.add_hline(y=-zscore_threshold, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=-exit_zscore, line_dash="dash", line_color="red", row=2, col=1)

# RSI
fig.add_trace(
    go.Scatter(
        x=df_plot.index,
        y=df_plot["rsi"],
        line={"color": "purple", "width": 2},
        name="RSI",
    ),
    row=3,
    col=1,
)

# RSI threshold
fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)

# Update layout
fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# Equity curve
st.subheader("💰 Equity Curve")

equity_fig = go.Figure()
equity_fig.add_trace(
    go.Scatter(
        x=results["equity_curve"]["time"],
        y=results["equity_curve"]["equity"],
        mode="lines",
        name="Portfolio Value",
        line={"color": "green", "width": 2},
    )
)

equity_fig.add_hline(y=10000, line_dash="dash", line_color="gray")
equity_fig.update_layout(
    height=400, xaxis_title="Date", yaxis_title="Portfolio Value ($)"
)

st.plotly_chart(equity_fig, use_container_width=True)

# Trade analysis
st.subheader("📋 Trade Analysis")

col1, col2 = st.columns(2)

with col1:
    # Trade distribution
    if results["metrics"]["num_trades"] > 0:
        trade_pnls = [t["pnl_pct"] for t in results["trades"] if "pnl_pct" in t]

        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(x=trade_pnls, nbinsx=20, name="Trade P&L Distribution")
        )
        hist_fig.update_layout(height=300, xaxis_title="P&L %", yaxis_title="Frequency")
        st.plotly_chart(hist_fig, use_container_width=True)

with col2:
    # Recent trades table
    if results["trades"]:
        recent_trades = results["trades"][-10:]
        trade_df = pd.DataFrame(recent_trades)

        # Format for display
        display_cols = ["action", "price"]
        if "pnl_pct" in trade_df.columns:
            display_cols.append("pnl_pct")
        if "reason" in trade_df.columns:
            display_cols.append("reason")

        st.dataframe(trade_df[display_cols].round(2), use_container_width=True)

# Export configuration
st.subheader("💾 Export Configuration")

config_to_export = {
    "strategy": "Mean Reversion",
    "symbol": symbol,
    "interval": interval,
    "parameters": params,
    "performance": results["metrics"],
    "generated_at": datetime.now().isoformat(),
}

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Download Config JSON"):
        json_str = json.dumps(config_to_export, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"mean_reversion_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

with col2:
    st.code(
        f"""
# Python code to use this configuration:

config = {{
    'symbols': ['{symbol}'],
    'lookback_period': {lookback_period},
    'zscore_threshold': {zscore_threshold},
    'exit_zscore': {exit_zscore},
    'rsi_period': {rsi_period},
    'rsi_oversold': {rsi_oversold},
    'stop_loss_atr': {stop_loss_atr},
    'position_size': {position_size}
}}

# Expected performance:
# Total Return: {results['metrics']['total_return']:.2f}%
# Win Rate: {results['metrics']['win_rate']:.1f}%
# Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}
"""
    )

# Footer
st.markdown("---")
st.markdown(
    "🚀 **Next Steps:** Use these parameters in your live trading system or continue optimizing!"
)
