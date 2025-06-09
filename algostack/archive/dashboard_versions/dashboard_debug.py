#!/usr/bin/env python3
"""
AlgoStack Dashboard - Debug Version
Enhanced with logging and debugging features
"""

import importlib
import inspect
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the algostack directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Handle TA-Lib import gracefully
try:
    import talib
except ImportError:
    import mock_talib as talib

    sys.modules["talib"] = talib

# Import strategy base class
from strategies.base import BaseStrategy

# Page configuration
st.set_page_config(
    page_title="AlgoStack Dashboard (Debug)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "strategy_configs" not in st.session_state:
    st.session_state.strategy_configs = {}
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = {}
if "enabled_strategies" not in st.session_state:
    st.session_state.enabled_strategies = set()
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []


def add_debug_log(message: str, level: str = "INFO"):
    """Add a debug log entry"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {level}: {message}"
    st.session_state.debug_logs.append(log_entry)
    logger.log(getattr(logging, level), message)


class StrategyRegistry:
    """Dynamic strategy discovery and management"""

    def __init__(self):
        self.strategies: dict[str, type[BaseStrategy]] = {}
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
                        friendly_name = (
                            name.replace("Strategy", "").replace("_", " ").title()
                        )
                        self.strategies[friendly_name] = obj
                        add_debug_log(f"Discovered strategy: {friendly_name}", "INFO")

            except Exception as e:
                add_debug_log(
                    f"Failed to load strategy from {module_name}: {e}", "ERROR"
                )

    def get_strategy_class(self, name: str) -> type[BaseStrategy]:
        """Get strategy class by name"""
        return self.strategies.get(name)

    def list_strategies(self) -> list[str]:
        """List all available strategy names"""
        return sorted(self.strategies.keys())

    def get_strategy_parameters(self, name: str) -> dict[str, Any]:
        """Get default parameters for a strategy"""
        strategy_class = self.get_strategy_class(name)
        if not strategy_class:
            return {}

        # Strategy-specific default parameters
        if "Hybrid" in name:
            return {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "trend_period": 50,
                "volatility_lookback": 20,
            }
        elif "Mean Reversion" in name:
            return {
                "rsi_period": 2,
                "rsi_oversold": 10,
                "rsi_overbought": 90,
                "atr_period": 14,
                "atr_multiplier": 2.5,
            }
        elif "Trend Following" in name:
            return {
                "fast_period": 20,
                "slow_period": 50,
                "atr_period": 14,
                "risk_per_trade": 0.02,
            }
        elif "Intraday" in name or "Orb" in name:
            return {
                "lookback_period": 20,
                "breakout_threshold": 1.5,
                "stop_loss_atr": 2.0,
            }
        elif "Overnight" in name:
            return {
                "holding_period": 1,
                "entry_time": 15,  # 3 PM
                "exit_time": 9,  # 9 AM
            }
        elif "Pairs" in name:
            return {
                "lookback_period": 60,
                "z_score_threshold": 2.0,
                "correlation_threshold": 0.8,
            }
        else:
            # Try to get from __init__ signature
            sig = inspect.signature(strategy_class.__init__)
            params = {}

            for param_name, param in sig.parameters.items():
                if param_name in ["self", "config"]:
                    continue

                # Try to get default value
                if param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
                else:
                    # Set reasonable defaults based on parameter name
                    if "period" in param_name:
                        params[param_name] = 20
                    elif "threshold" in param_name:
                        params[param_name] = 30.0
                    elif "lookback" in param_name:
                        params[param_name] = 50
                    else:
                        params[param_name] = None

            return params


def fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical data using yfinance"""
    try:
        add_debug_log(
            f"Fetching data for {symbol} from {start_date} to {end_date}", "INFO"
        )

        # Download data with explicit date range
        data = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            add_debug_log(f"No data available for {symbol}", "ERROR")
            st.error(f"No data available for {symbol} in the selected date range")
            return pd.DataFrame()

        # Handle MultiIndex columns from yfinance (when downloading single symbol)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            add_debug_log(f"Dropped MultiIndex level from {symbol} data", "DEBUG")

        # Ensure data is sorted by date
        data = data.sort_index()

        add_debug_log(f"Fetched {len(data)} days of data for {symbol}", "INFO")

        return data
    except Exception as e:
        add_debug_log(f"Failed to fetch data for {symbol}: {e}", "ERROR")
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()


def simulate_strategy_signals(
    strategy_name: str,
    data: pd.DataFrame,
    params: dict[str, Any],
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """Simulate strategy signals on historical data with debugging"""
    add_debug_log(f"Starting signal simulation for {strategy_name}", "INFO")

    # Create signals dataframe
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data["Close"]
    signals["signal"] = 0
    signals["position"] = 0.0

    # Initialize transaction costs columns as float
    signals["transaction_costs"] = 0.0
    signals["trade_entry"] = 0
    signals["trade_exit"] = 0
    signals["trade_type"] = ""

    # Ensure we have enough data
    if len(data) < 50:
        add_debug_log(
            f"Not enough data for {strategy_name} (only {len(data)} days)", "WARNING"
        )
        signals["returns"] = 0
        signals["strategy_returns"] = 0
        signals["cumulative_returns"] = 1
        signals["buy_hold_returns"] = 1
        signals["portfolio_value"] = 1
        return signals

    # Simple example strategies
    if "Mean Reversion" in strategy_name or "Hybrid" in strategy_name:
        # RSI-based mean reversion
        period = params.get("rsi_period", 14) if params else 14
        oversold = params.get("rsi_oversold", 30) if params else 30
        overbought = params.get("rsi_overbought", 70) if params else 70

        # Calculate RSI
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        buy_condition = rsi < oversold
        sell_condition = rsi > overbought

        # Apply conditions where they are True
        signals.loc[buy_condition, "signal"] = 1  # Buy
        signals.loc[sell_condition, "signal"] = -1  # Sell

        add_debug_log(
            f"RSI strategy generated {(signals['signal'] != 0).sum()} signals", "DEBUG"
        )

    elif "Trend Following" in strategy_name:
        # Moving average crossover
        fast_period = params.get("fast_period", 20) if params else 20
        slow_period = params.get("slow_period", 50) if params else 50

        fast_ma = data["Close"].rolling(window=fast_period).mean()
        slow_ma = data["Close"].rolling(window=slow_period).mean()

        # Generate signals at crossovers
        signals["signal"] = 0
        long_condition = fast_ma > slow_ma
        short_condition = fast_ma < slow_ma

        signals.loc[long_condition, "signal"] = 1
        signals.loc[short_condition, "signal"] = -1

        add_debug_log(
            f"MA crossover strategy generated {(signals['signal'] != 0).sum()} signals",
            "DEBUG",
        )

    elif "Intraday" in strategy_name or "Orb" in strategy_name:
        # Opening Range Breakout
        lookback = params.get("lookback_period", 20) if params else 20

        # Simple version: buy if price breaks above 20-day high
        rolling_high = data["High"].rolling(window=lookback).max()
        rolling_low = data["Low"].rolling(window=lookback).min()

        buy_breakout = data["Close"] > rolling_high.shift(1)
        sell_breakout = data["Close"] < rolling_low.shift(1)

        signals.loc[buy_breakout, "signal"] = 1
        signals.loc[sell_breakout, "signal"] = -1

        add_debug_log(
            f"Breakout strategy generated {(signals['signal'] != 0).sum()} signals",
            "DEBUG",
        )

    else:
        # Default strategy - simple momentum
        returns = data["Close"].pct_change(20)

        buy_momentum = returns > 0.05
        sell_momentum = returns < -0.05

        signals.loc[buy_momentum, "signal"] = 1
        signals.loc[sell_momentum, "signal"] = -1

        add_debug_log(
            f"Momentum strategy generated {(signals['signal'] != 0).sum()} signals",
            "DEBUG",
        )

    # Forward fill positions between signals
    signals["position"] = signals["signal"].replace(0, np.nan).ffill().fillna(0)

    # Debug position changes
    position_changes = signals["position"].diff() != 0
    add_debug_log(f"Position changes: {position_changes.sum()}", "DEBUG")

    # Identify trade points
    signals["position"].diff()

    # Mark entries (position changes from 0 to non-zero)
    entry_mask = (signals["position"].shift(1).fillna(0) == 0) & (
        signals["position"] != 0
    )
    signals.loc[entry_mask, "trade_entry"] = 1
    signals.loc[entry_mask & (signals["position"] > 0), "trade_type"] = "BUY"
    signals.loc[entry_mask & (signals["position"] < 0), "trade_type"] = "SHORT"

    # Mark exits (position changes from non-zero to 0)
    exit_mask = (signals["position"].shift(1).fillna(0) != 0) & (
        signals["position"] == 0
    )
    signals.loc[exit_mask, "trade_exit"] = 1
    signals.loc[exit_mask, "trade_type"] = "EXIT"

    add_debug_log(
        f"Trade entries: {entry_mask.sum()}, Trade exits: {exit_mask.sum()}", "INFO"
    )

    # Calculate returns
    signals["returns"] = data["Close"].pct_change().fillna(0)

    # Calculate strategy returns with transaction costs
    signals["gross_returns"] = (
        signals["position"].shift(1).fillna(0) * signals["returns"]
    )

    # Apply transaction costs on trade days
    trade_days = entry_mask | exit_mask
    total_cost = commission + slippage
    signals.loc[trade_days, "transaction_costs"] = total_cost

    # Net returns after costs
    signals["strategy_returns"] = (
        signals["gross_returns"] - signals["transaction_costs"]
    )

    # Calculate cumulative returns
    signals["cumulative_returns"] = (1 + signals["strategy_returns"]).cumprod()
    signals["buy_hold_returns"] = (1 + signals["returns"]).cumprod()

    # Calculate portfolio value (starting with 1.0 = 100%)
    signals["portfolio_value"] = signals["cumulative_returns"]

    # Ensure first values are 1.0
    if len(signals) > 0:
        signals.loc[signals.index[0], "cumulative_returns"] = 1.0
        signals.loc[signals.index[0], "buy_hold_returns"] = 1.0
        signals.loc[signals.index[0], "portfolio_value"] = 1.0

    # Fill any remaining NaN values
    signals = signals.fillna(0)

    # Log final statistics
    final_return = (signals["cumulative_returns"].iloc[-1] - 1) * 100
    add_debug_log(f"Strategy {strategy_name} final return: {final_return:.2f}%", "INFO")

    return signals


def calculate_metrics(signals: pd.DataFrame) -> dict[str, float]:
    """Calculate performance metrics from signals with debugging"""
    add_debug_log("Calculating metrics", "DEBUG")

    # Check if we have valid data
    if len(signals) < 2:
        add_debug_log("Not enough data for metrics calculation", "WARNING")
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

    strategy_returns = signals["strategy_returns"].dropna()

    # Total return
    total_return = (signals["cumulative_returns"].iloc[-1] - 1) * 100

    # Sharpe ratio
    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
        sharpe_ratio = float(
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        )
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = signals["cumulative_returns"]
    if len(cumulative) > 0:
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(float(drawdown.min() * 100))
    else:
        max_drawdown = 0.0

    # Win rate and trade counting
    # Count entries as trades
    entries = signals[signals["trade_entry"] == 1]
    total_trades = len(entries)

    add_debug_log(f"Found {total_trades} trade entries", "DEBUG")

    if total_trades > 0:
        # Calculate P&L for each completed trade
        trade_results = []

        # Get entry and exit points
        entry_points = signals[signals["trade_entry"] == 1].copy()
        exit_points = signals[signals["trade_exit"] == 1].copy()

        for entry_idx in entry_points.index:
            entry_price = signals.loc[entry_idx, "price"]
            entry_position = signals.loc[entry_idx, "position"]

            # Find corresponding exit
            future_exits = exit_points[exit_points.index > entry_idx]
            if len(future_exits) > 0:
                exit_idx = future_exits.index[0]
                exit_price = signals.loc[exit_idx, "price"]

                # Calculate return based on position type
                if entry_position > 0:  # Long trade
                    trade_return = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    trade_return = (entry_price - exit_price) / entry_price

                trade_results.append(trade_return)
                add_debug_log(
                    f"Trade: Entry={entry_price:.2f}, Exit={exit_price:.2f}, Return={trade_return:.2%}",
                    "DEBUG",
                )

        if trade_results:
            winning_trades = sum(1 for r in trade_results if r > 0)
            win_rate = (winning_trades / len(trade_results)) * 100
            total_trades = len(trade_results)  # Only count completed trades
            add_debug_log(
                f"Completed trades: {total_trades}, Win rate: {win_rate:.1f}%", "INFO"
            )
        else:
            win_rate = 0.0
            add_debug_log("No completed trades found", "WARNING")
    else:
        win_rate = 0.0

    metrics = {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "total_trades": int(total_trades),
    }

    add_debug_log(f"Metrics calculated: {metrics}", "DEBUG")

    return metrics


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
                    value=strategy_name in st.session_state.enabled_strategies,
                )

                if enabled:
                    st.session_state.enabled_strategies.add(strategy_name)
                else:
                    st.session_state.enabled_strategies.discard(strategy_name)

                # Force rerun to update button state
                if enabled != (strategy_name in st.session_state.enabled_strategies):
                    st.rerun()

                # Show parameters if enabled
                if enabled:
                    # Get default parameters
                    default_params = registry.get_strategy_parameters(strategy_name)

                    if default_params:
                        # Create parameter inputs
                        params = {}

                        for param_name, default_value in default_params.items():
                            if isinstance(default_value, bool):
                                params[param_name] = st.checkbox(
                                    param_name.replace("_", " ").title(),
                                    value=default_value,
                                    key=f"{strategy_name}_{param_name}",
                                    help=f"Parameter for {strategy_name}",
                                )
                            elif isinstance(default_value, int):
                                params[param_name] = st.number_input(
                                    param_name.replace("_", " ").title(),
                                    value=default_value,
                                    step=1,
                                    key=f"{strategy_name}_{param_name}",
                                    help=f"Parameter for {strategy_name}",
                                )
                            elif isinstance(default_value, float):
                                params[param_name] = st.number_input(
                                    param_name.replace("_", " ").title(),
                                    value=default_value,
                                    step=0.1,
                                    format="%.2f",
                                    key=f"{strategy_name}_{param_name}",
                                    help=f"Parameter for {strategy_name}",
                                )

                        # Store configuration
                        st.session_state.strategy_configs[strategy_name] = params

                # Add spacing
                st.markdown("")


def run_backtest(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
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

        symbol_results = {"data": data, "strategies": {}}

        for strategy_name in enabled_strategies:
            current_test += 1
            progress_bar.progress(current_test / total_tests)

            st.text(f"Testing {strategy_name} on {symbol}...")

            # Get strategy parameters
            params = st.session_state.strategy_configs.get(strategy_name, {})

            # Simulate strategy
            signals = simulate_strategy_signals(
                strategy_name, data, params, commission, slippage
            )
            metrics = calculate_metrics(signals)

            symbol_results["strategies"][strategy_name] = {
                "signals": signals,
                "metrics": metrics,
                "params": params,
            }

        results[symbol] = symbol_results

    progress_bar.empty()

    return results


def display_results(results: dict):
    """Display backtest results with strategy overlays and debugging info"""
    if not results:
        return

    # Debug Information Panel
    with st.expander("🔍 Debug Information", expanded=False):
        st.text_area(
            "Debug Logs", value="\n".join(st.session_state.debug_logs), height=300
        )

    # Overall Performance Summary
    st.header("📊 Overall Performance")

    # Aggregate metrics across all strategies and symbols
    all_metrics = []
    for symbol, symbol_data in results.items():
        for strategy_name, strategy_data in symbol_data["strategies"].items():
            metrics = strategy_data["metrics"]
            metrics["symbol"] = symbol
            metrics["strategy"] = strategy_name
            all_metrics.append(metrics)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)

        # Display average metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_return = df_metrics["total_return"].mean()
            st.metric("Avg Return", f"{avg_return:.2f}%")

        with col2:
            avg_sharpe = df_metrics["sharpe_ratio"].mean()
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")

        with col3:
            max_dd = df_metrics["max_drawdown"].max()
            st.metric("Max Drawdown", f"{max_dd:.2f}%")

        with col4:
            avg_win_rate = df_metrics["win_rate"].mean()
            st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")

    # Strategy Performance Comparison
    st.header("🎯 Strategy Performance Comparison")

    strategy_summary = (
        df_metrics.groupby("strategy")
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
                "total_trades": "sum",
            }
        )
        .round(2)
    )

    st.dataframe(strategy_summary)

    # Individual Symbol Charts with Strategy Overlays
    st.header("📈 Price Charts with Strategy Performance")

    for symbol, symbol_data in results.items():
        st.subheader(f"{symbol}")

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f"{symbol} Price & Strategy Performance",
                "Strategy Signals",
            ),
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        # Get price data
        data = symbol_data["data"]

        # Plot price on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Price",
                line={"color": "black", "width": 2},
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        # Plot buy & hold performance (as percentage from start) on secondary y-axis
        buy_hold_returns = ((data["Close"] / data["Close"].iloc[0]) - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=buy_hold_returns,
                mode="lines",
                name="Buy & Hold",
                line={"color": "gray", "width": 2, "dash": "dash"},
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Color palette for strategies
        colors = ["blue", "red", "green", "purple", "orange"]

        # Plot each strategy's performance
        for i, (strategy_name, strategy_data) in enumerate(
            symbol_data["strategies"].items()
        ):
            signals = strategy_data["signals"]
            color = colors[i % len(colors)]

            # Check if we have valid signal data
            if "cumulative_returns" in signals.columns:
                # Plot strategy cumulative returns (as percentage from start)
                strategy_returns = (signals["cumulative_returns"] - 1) * 100
                fig.add_trace(
                    go.Scatter(
                        x=signals.index,
                        y=strategy_returns,
                        mode="lines",
                        name=f"{strategy_name}",
                        line={"color": color, "width": 2},
                    ),
                    row=1,
                    col=1,
                    secondary_y=True,
                )

                # Debug: Log the strategy performance
                add_debug_log(
                    f"{strategy_name} on {symbol}: Final return = {strategy_returns.iloc[-1]:.2f}%",
                    "INFO",
                )
            else:
                add_debug_log(
                    f"No cumulative_returns column in {strategy_name} signals", "ERROR"
                )

            # Plot trade entry and exit points
            entries = signals[signals["trade_entry"] == 1]
            exits = signals[signals["trade_exit"] == 1]

            if not entries.empty:
                # Plot entry points
                for _, entry in entries.iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[entry.name],
                            y=[entry["price"]],
                            mode="markers+text",
                            name=f"{strategy_name} Entry",
                            marker={
                                "symbol": (
                                    "triangle-up"
                                    if entry["position"] > 0
                                    else "triangle-down"
                                ),
                                "size": 12,
                                "color": color,
                                "line": {"width": 2, "color": "white"},
                            },
                            text=[entry["trade_type"]],
                            textposition="bottom center",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                        secondary_y=False,
                    )

            if not exits.empty:
                # Plot exit points
                for _, exit in exits.iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[exit.name],
                            y=[exit["price"]],
                            mode="markers+text",
                            name=f"{strategy_name} Exit",
                            marker={
                                "symbol": "x", "size": 12, "color": color, "line": {"width": 2}
                            },
                            text=["EXIT"],
                            textposition="top center",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                        secondary_y=False,
                    )

            # Add shaded regions for positions
            if len(entries) > 0:
                for _, entry in entries.iterrows():
                    # Find corresponding exit
                    future_exits = exits[exits.index > entry.name]
                    if len(future_exits) > 0:
                        exit_idx = future_exits.index[0]

                        # Add shaded region
                        fig.add_vrect(
                            x0=entry.name,
                            x1=exit_idx,
                            fillcolor=color,
                            opacity=0.1,
                            layer="below",
                            line_width=0,
                            row=1,
                            col=1,
                        )

            # Plot position indicator
            if "position" in signals.columns:
                fig.add_trace(
                    go.Scatter(
                        x=signals.index,
                        y=signals["position"],
                        mode="lines",
                        name=f"{strategy_name} Position",
                        line={"color": color, "width": 1},
                        fill="tozeroy",
                        opacity=0.3,
                    ),
                    row=2,
                    col=1,
                )

        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Returns (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Position", row=2, col=1)

        fig.update_layout(
            height=800,
            hovermode="x unified",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics table for this symbol
        st.subheader(f"{symbol} Strategy Metrics")

        symbol_metrics = []
        for strategy_name, strategy_data in symbol_data["strategies"].items():
            metrics = strategy_data["metrics"].copy()
            metrics["Strategy"] = strategy_name
            symbol_metrics.append(metrics)

        if symbol_metrics:
            df_symbol = pd.DataFrame(symbol_metrics)
            df_symbol = df_symbol[
                [
                    "Strategy",
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                    "total_trades",
                ]
            ]
            df_symbol.columns = [
                "Strategy",
                "Return (%)",
                "Sharpe",
                "Max DD (%)",
                "Win Rate (%)",
                "Trades",
            ]
            st.dataframe(df_symbol)


def main():
    """Main dashboard application"""

    st.title("🔍 AlgoStack Trading Dashboard (Debug Mode)")
    st.markdown(
        "### Integrated Strategy Testing & Analysis Platform with Enhanced Debugging"
    )

    # Initialize
    registry = StrategyRegistry()

    # Clear debug logs button
    if st.button("Clear Debug Logs", type="secondary"):
        st.session_state.debug_logs = []
        st.rerun()

    # Sidebar
    with st.sidebar:
        # Market selection
        st.header("Market Selection")

        default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]
        symbols = st.multiselect(
            "Select Symbols", default_symbols, default=["SPY", "AAPL"]
        )

        st.divider()

        # Time period
        st.header("Time Period")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now(),
            )

        with col2:
            end_date = st.date_input(
                "End Date", value=datetime.now(), max_value=datetime.now()
            )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
        )

        st.divider()

        # Trading costs
        st.header("Trading Costs")

        commission = (
            st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                help="Commission per trade as percentage",
            )
            / 100
        )

        slippage = (
            st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.3f",
                help="Slippage per trade as percentage",
            )
            / 100
        )

        # Run button
        run_backtest_btn = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not (symbols and st.session_state.enabled_strategies),
        )

    # Main content area - Strategy configuration
    if not run_backtest_btn:
        display_strategy_configuration(registry)

    # Show message if no strategies enabled
    if not st.session_state.enabled_strategies and not run_backtest_btn:
        st.info("👆 Enable at least one strategy above to begin analysis")

    elif run_backtest_btn:
        st.header("Running Analysis...")

        # Clear previous debug logs for this run
        st.session_state.debug_logs = []

        # Run backtest
        results = run_backtest(
            symbols=symbols,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
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
