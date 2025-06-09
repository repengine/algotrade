#!/usr/bin/env python3
"""
Simplified AlgoStack Dashboard - Works without TA-Lib
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
import yfinance as yf

# Add the algostack directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="AlgoStack Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = {}


def load_config() -> dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "base.yaml"

    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return {}


def get_available_strategies() -> list[str]:
    """Get list of available strategies"""
    strategies_dir = Path(__file__).parent / "strategies"
    strategies = []

    for file_path in strategies_dir.glob("*.py"):
        if file_path.name.startswith("_") or file_path.name == "base.py":
            continue
        module_name = file_path.stem
        # Convert to friendly name
        friendly_name = module_name.replace("_", " ").title()
        strategies.append(friendly_name)

    return strategies


def fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical data using yfinance"""
    try:
        # Debug info
        st.text(f"Fetching {symbol} from {start_date.date()} to {end_date.date()}")

        # Use Ticker object for more reliable data fetching
        ticker = yf.Ticker(symbol)

        # Calculate period in days
        days_diff = (end_date - start_date).days

        # Use appropriate method based on date range
        if days_diff <= 7:
            data = ticker.history(period="1wk")
        elif days_diff <= 30:
            data = ticker.history(period="1mo")
        elif days_diff <= 90:
            data = ticker.history(period="3mo")
        elif days_diff <= 180:
            data = ticker.history(period="6mo")
        elif days_diff <= 365:
            data = ticker.history(period="1y")
        elif days_diff <= 730:
            data = ticker.history(period="2y")
        else:
            # For longer periods, use start/end dates
            data = ticker.history(start=start_date, end=end_date)

        # Check if we got data
        if data.empty:
            st.warning(f"No data returned for {symbol}")
        else:
            actual_start = data.index[0].date()
            actual_end = data.index[-1].date()
            st.success(
                f"✓ Fetched {len(data)} days of data for {symbol} ({actual_start} to {actual_end})"
            )

        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        import traceback

        st.text(traceback.format_exc())
        return pd.DataFrame()


def calculate_simple_metrics(
    data: pd.DataFrame, initial_capital: float = 100000
) -> dict[str, float]:
    """Calculate basic performance metrics"""
    if data.empty or len(data) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
        }

    # Simple buy and hold returns
    returns = data["Close"].pct_change().dropna()

    # Calculate metrics
    total_return = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100

    # Annual return - handle short periods
    days = len(data)
    if days > 0:
        annual_return = (
            ((1 + total_return / 100) ** (252 / days) - 1) * 100
            if days < 252
            else total_return
        )
    else:
        annual_return = 0.0

    # Sharpe ratio (simplified)
    if len(returns) > 1:
        std_returns = float(returns.std())
        mean_returns = float(returns.mean())
        sharpe_ratio = (
            (mean_returns / std_returns * np.sqrt(252)) if std_returns > 0 else 0.0
        )
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min() * 100) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0

    # Volatility
    volatility = float(returns.std()) * np.sqrt(252) * 100 if len(returns) > 1 else 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "volatility": float(volatility),
    }


def run_simple_backtest(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
) -> dict:
    """Run a simple backtest without complex dependencies"""
    results = {}

    progress_bar = st.progress(0)

    for i, symbol in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        st.text(f"Analyzing {symbol}...")

        # Fetch data
        data = fetch_data(symbol, start_date, end_date)

        if not data.empty:
            # Calculate metrics
            metrics = calculate_simple_metrics(data, initial_capital)

            results[symbol] = {"data": data, "metrics": metrics}

    progress_bar.empty()

    # Aggregate results
    if results:
        avg_return = np.mean([r["metrics"]["total_return"] for r in results.values()])
        avg_sharpe = np.mean([r["metrics"]["sharpe_ratio"] for r in results.values()])
        max_dd = np.max([r["metrics"]["max_drawdown"] for r in results.values()])

        return {
            "individual_results": results,
            "aggregate_metrics": {
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "max_drawdown": max_dd,
                "symbols_analyzed": len(results),
            },
        }

    return {}


def display_results(results: dict):
    """Display backtest results"""
    if not results:
        return

    # Aggregate metrics
    st.header("📊 Overall Performance")

    agg_metrics = results.get("aggregate_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Return", f"{agg_metrics.get('avg_return', 0):.2f}%")

    with col2:
        st.metric("Average Sharpe", f"{agg_metrics.get('avg_sharpe', 0):.2f}")

    with col3:
        st.metric("Max Drawdown", f"{agg_metrics.get('max_drawdown', 0):.2f}%")

    with col4:
        st.metric("Symbols Analyzed", agg_metrics.get("symbols_analyzed", 0))

    # Individual results
    st.header("📈 Individual Symbol Performance")

    individual_results = results.get("individual_results", {})

    # Create comparison table
    comparison_data = []
    for symbol, data in individual_results.items():
        metrics = data["metrics"]
        comparison_data.append(
            {
                "Symbol": symbol,
                "Total Return": f"{float(metrics.get('total_return', 0)):.2f}%",
                "Annual Return": f"{float(metrics.get('annual_return', 0)):.2f}%",
                "Sharpe Ratio": f"{float(metrics.get('sharpe_ratio', 0)):.2f}",
                "Max Drawdown": f"{float(metrics.get('max_drawdown', 0)):.2f}%",
                "Volatility": f"{float(metrics.get('volatility', 0)):.2f}%",
            }
        )

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Price charts
    st.header("💹 Price Charts")

    # Create subplots
    num_symbols = len(individual_results)
    cols = st.columns(min(2, num_symbols))

    for i, (symbol, data) in enumerate(individual_results.items()):
        col = cols[i % len(cols)]

        with col:
            df = data["data"]
            if not df.empty:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Close"],
                        mode="lines",
                        name=symbol,
                        line={"width": 2},
                    )
                )

                # Add price statistics
                start_price = float(df["Close"].iloc[0])
                end_price = float(df["Close"].iloc[-1])
                price_change = end_price - start_price
                pct_change = (price_change / start_price) * 100

                fig.update_layout(
                    title=f"{symbol} Price<br><sub>Start: ${start_price:.2f} | End: ${end_price:.2f} | Change: {pct_change:+.2f}%</sub>",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    xaxis={"rangeslider": {"visible": False}, "type": "date"},
                    yaxis={"tickformat": "$,.2f"},
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {symbol}")


def main():
    """Main dashboard application"""

    st.title("🚀 AlgoStack Trading Dashboard (Simplified)")
    st.markdown("### Strategy Testing & Analysis Platform")

    # Load configuration
    config = load_config()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Get available strategies (for display only)
        strategies = get_available_strategies()
        if strategies:
            st.info(
                f"Found {len(strategies)} strategies:\n"
                + "\n".join(f"• {s}" for s in strategies[:5])
            )
            if len(strategies) > 5:
                st.text(f"... and {len(strategies) - 5} more")

        st.divider()

        # Symbol selection
        st.header("Market Selection")

        default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]

        symbols = st.multiselect(
            "Select Symbols",
            default_symbols,
            default=["AAPL", "MSFT", "SPY"],
            help="Choose symbols to analyze",
        )

        st.divider()

        # Date range
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

        # Run analysis button
        run_analysis = st.button(
            "📊 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not symbols,
        )

    # Main content
    if not symbols:
        st.info("👈 Select symbols from the sidebar to begin analysis")

        # Show config info
        if config:
            st.header("📋 Configuration Loaded")
            st.json(config.get("engine", {}))

    elif run_analysis:
        st.header("Running Analysis...")

        # Run simple backtest
        results = run_simple_backtest(
            symbols=symbols,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
            initial_capital=initial_capital,
        )

        # Store results
        if results:
            st.session_state.backtest_results = results
            st.success("✅ Analysis completed!")
        else:
            st.error("❌ Analysis failed")

    # Display results
    if st.session_state.backtest_results:
        display_results(st.session_state.backtest_results)

        # Export options
        st.divider()

        if st.button("🗑️ Clear Results"):
            st.session_state.backtest_results = {}
            st.rerun()


if __name__ == "__main__":
    main()
