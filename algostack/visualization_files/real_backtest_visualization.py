#!/usr/bin/env python3
"""
Real Backtest Visualization for AlgoStack

This script runs backtests using REAL libraries (no mocking) with comprehensive visualizations.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import REAL libraries directly (bypass mock system)
try:
    import yfinance as yf

    print("✅ Using REAL yfinance")
except ImportError:
    print("❌ yfinance not available - install with: pip install yfinance")
    sys.exit(1)

try:
    import scipy.stats

    print("✅ Using REAL scipy")
except ImportError:
    print("❌ scipy not available")

try:
    import backtrader as bt

    print("✅ Using REAL backtrader")
except ImportError:
    print("❌ backtrader not available")

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SimpleDataHandler:
    """Simple data handler using yfinance directly."""

    def get_historical(self, symbol, start_date, end_date):
        """Fetch historical data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                return pd.DataFrame()

            # Standardize column names
            data.columns = [col.lower() for col in data.columns]

            # Calculate additional indicators
            data = self.calculate_indicators(data)

            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """Add common technical indicators."""
        if df.empty:
            return df

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift())

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Simple RSI approximation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        return df


class SimpleMeanReversionStrategy:
    """Simple mean reversion strategy implementation."""

    def __init__(self, config):
        self.config = config
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.position = 0

    def init(self):
        """Initialize strategy."""
        pass

    def next(self, data):
        """Generate trading signal based on current data."""
        if len(data) < self.rsi_period + 1:
            return None

        current_row = data.iloc[-1]
        rsi = current_row.get("rsi", 50)

        # Simple mean reversion logic
        if rsi < self.rsi_oversold and self.position <= 0:
            # Oversold - buy signal
            return SimpleSignal("LONG", 0.8, current_row["close"])
        elif rsi > self.rsi_overbought and self.position >= 0:
            # Overbought - sell signal
            return SimpleSignal("SHORT", -0.8, current_row["close"])

        return None


class SimpleSignal:
    """Simple signal class."""

    def __init__(self, direction, strength, price):
        self.direction = direction
        self.strength = strength
        self.price = price
        self.timestamp = datetime.now()


class VisualizedBacktester:
    """Backtester with comprehensive visualizations."""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.data_handler = SimpleDataHandler()

    def run_strategy_backtest(self, strategy, symbol, start_date, end_date):
        """Run backtest for a strategy with detailed tracking."""
        print(f"📊 Running backtest for {symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print("-" * 50)

        # Get data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        print("📈 Fetching market data...")
        data = self.data_handler.get_historical(symbol, start_dt, end_dt)

        if data.empty:
            print("❌ No data available for the specified period")
            return None

        print(f"✅ Fetched {len(data)} days of data")

        # Initialize strategy
        strategy.init()

        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0
        position_value = 0

        # Track portfolio over time
        portfolio_history = []
        trade_history = []
        signals_history = []

        print("🔄 Running strategy simulation...")

        # Run strategy simulation
        for i, (date, row) in enumerate(data.iterrows()):
            if i < 50:  # Need enough data for indicators
                portfolio_history.append(
                    {
                        "date": date,
                        "portfolio_value": portfolio_value,
                        "cash": cash,
                        "position_value": position_value,
                        "returns": 0.0,
                        "drawdown": 0.0,
                    }
                )
                continue

            # Get recent data for strategy
            recent_data = data.iloc[max(0, i - 100) : i + 1].copy()

            # Generate signal
            signal = strategy.next(recent_data)

            current_price = row["close"]

            if signal:
                signals_history.append(
                    {
                        "date": date,
                        "signal": signal.direction,
                        "price": current_price,
                        "strength": signal.strength,
                    }
                )

                # Simple position sizing (using 95% of available cash)
                if signal.direction == "LONG" and position <= 0:
                    # Buy signal
                    shares_to_buy = int((cash * 0.95) / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        position += shares_to_buy
                        strategy.position = position

                        trade_history.append(
                            {
                                "date": date,
                                "action": "BUY",
                                "shares": shares_to_buy,
                                "price": current_price,
                                "value": cost,
                            }
                        )

                elif signal.direction == "SHORT" and position >= 0:
                    # Sell signal (close long position)
                    if position > 0:
                        proceeds = position * current_price
                        cash += proceeds

                        trade_history.append(
                            {
                                "date": date,
                                "action": "SELL",
                                "shares": position,
                                "price": current_price,
                                "value": proceeds,
                            }
                        )

                        position = 0
                        strategy.position = position

            # Update portfolio value
            position_value = position * current_price
            portfolio_value = cash + position_value

            # Calculate returns and drawdown
            if len(portfolio_history) > 0:
                prev_value = portfolio_history[-1]["portfolio_value"]
                daily_return = (portfolio_value - prev_value) / prev_value
            else:
                daily_return = 0.0

            # Calculate running maximum for drawdown
            if len(portfolio_history) > 0:
                running_max = max(
                    [h["portfolio_value"] for h in portfolio_history]
                    + [portfolio_value]
                )
                drawdown = (portfolio_value - running_max) / running_max
            else:
                drawdown = 0.0

            portfolio_history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position_value": position_value,
                    "returns": daily_return,
                    "drawdown": drawdown,
                    "price": current_price,
                }
            )

        # Convert to DataFrames
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index("date", inplace=True)

        trades_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
        signals_df = (
            pd.DataFrame(signals_history) if signals_history else pd.DataFrame()
        )

        # Calculate performance metrics
        metrics = self._calculate_metrics(portfolio_df, data)

        # Create visualizations
        self._create_visualizations(
            portfolio_df, trades_df, signals_df, data, symbol, metrics
        )

        return {
            "portfolio": portfolio_df,
            "trades": trades_df,
            "signals": signals_df,
            "metrics": metrics,
        }

    def _calculate_metrics(self, portfolio_df, price_data):
        """Calculate performance metrics."""
        if len(portfolio_df) < 2:
            return {}

        # Basic metrics
        total_return = (
            portfolio_df["portfolio_value"].iloc[-1] / self.initial_capital - 1
        ) * 100

        # Benchmark (buy and hold) - align dates with portfolio
        aligned_price_data = price_data.loc[portfolio_df.index]
        if len(aligned_price_data) > 0:
            benchmark_return = (
                aligned_price_data["close"].iloc[-1]
                / aligned_price_data["close"].iloc[0]
                - 1
            ) * 100
        else:
            benchmark_return = 0

        # Risk metrics
        returns = portfolio_df["returns"].dropna()
        if len(returns) > 0:
            sharpe_ratio = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            )
            max_drawdown = portfolio_df["drawdown"].min() * 100
            volatility = returns.std() * np.sqrt(252) * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            volatility = 0

        # Win rate (from daily returns)
        win_rate = (returns > 0).mean() * 100 if len(returns) > 0 else 0

        return {
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": total_return - benchmark_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "win_rate": win_rate,
            "total_days": len(portfolio_df),
        }

    def _create_visualizations(
        self, portfolio_df, trades_df, signals_df, price_data, symbol, metrics
    ):
        """Create comprehensive performance visualizations."""
        # Create figure with subplots
        plt.figure(figsize=(16, 12))

        # Align benchmark data with portfolio dates
        aligned_price_data = price_data.loc[portfolio_df.index]
        benchmark_value = self.initial_capital * (
            aligned_price_data["close"] / aligned_price_data["close"].iloc[0]
        )

        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(
            portfolio_df.index,
            portfolio_df["portfolio_value"],
            label="Strategy Portfolio",
            linewidth=2,
            color="blue",
        )

        # Add benchmark
        ax1.plot(
            benchmark_value.index,
            benchmark_value,
            label=f"{symbol} Buy & Hold",
            linewidth=2,
            color="gray",
            alpha=0.7,
        )

        ax1.set_title("📈 Portfolio Value Over Time", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # 2. Cumulative Returns
        ax2 = plt.subplot(3, 2, 2)
        strategy_returns = (
            portfolio_df["portfolio_value"] / self.initial_capital - 1
        ) * 100
        benchmark_returns = (benchmark_value / self.initial_capital - 1) * 100

        ax2.plot(
            portfolio_df.index,
            strategy_returns,
            label="Strategy",
            linewidth=2,
            color="green",
        )
        ax2.plot(
            benchmark_returns.index,
            benchmark_returns,
            label=f"{symbol} Benchmark",
            linewidth=2,
            color="orange",
            alpha=0.7,
        )

        ax2.set_title("📊 Cumulative Returns (%)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Returns (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Drawdown
        ax3 = plt.subplot(3, 2, 3)
        ax3.fill_between(
            portfolio_df.index,
            portfolio_df["drawdown"] * 100,
            0,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax3.plot(
            portfolio_df.index, portfolio_df["drawdown"] * 100, color="red", linewidth=1
        )

        ax3.set_title("📉 Drawdown Analysis", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Drawdown (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Price with Signals
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(
            aligned_price_data.index,
            aligned_price_data["close"],
            label=f"{symbol} Price",
            linewidth=1,
            color="black",
        )

        if not signals_df.empty:
            buy_signals = signals_df[signals_df["signal"] == "LONG"]
            sell_signals = signals_df[signals_df["signal"] == "SHORT"]

            if not buy_signals.empty:
                ax4.scatter(
                    buy_signals["date"],
                    buy_signals["price"],
                    color="green",
                    marker="^",
                    s=100,
                    label="Buy Signals",
                    zorder=5,
                )

            if not sell_signals.empty:
                ax4.scatter(
                    sell_signals["date"],
                    sell_signals["price"],
                    color="red",
                    marker="v",
                    s=100,
                    label="Sell Signals",
                    zorder=5,
                )

        ax4.set_title(
            f"💹 {symbol} Price with Trading Signals", fontsize=14, fontweight="bold"
        )
        ax4.set_ylabel("Price ($)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Daily Returns Distribution
        ax5 = plt.subplot(3, 2, 5)
        returns = portfolio_df["returns"].dropna() * 100
        if len(returns) > 0:
            ax5.hist(returns, bins=50, alpha=0.7, color="blue", edgecolor="black")
            ax5.axvline(
                returns.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {returns.mean():.2f}%",
            )
            ax5.axvline(0, color="gray", linestyle="-", alpha=0.5)

        ax5.set_title("📊 Daily Returns Distribution", fontsize=14, fontweight="bold")
        ax5.set_xlabel("Daily Returns (%)")
        ax5.set_ylabel("Frequency")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Performance Metrics Table
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis("off")

        # Create metrics table
        metrics_text = f"""
🏆 PERFORMANCE SUMMARY
{'='*35}

💰 Total Return:        {metrics['total_return']:>8.1f}%
📈 Benchmark Return:    {metrics['benchmark_return']:>8.1f}%
⚡ Excess Return:       {metrics['excess_return']:>8.1f}%

📊 Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}
📉 Max Drawdown:        {metrics['max_drawdown']:>8.1f}%
🎯 Win Rate:            {metrics['win_rate']:>8.1f}%
📈 Volatility:          {metrics['volatility']:>8.1f}%

📅 Trading Days:        {metrics['total_days']:>8}
🏁 Final Value:    ${portfolio_df['portfolio_value'].iloc[-1]:>12,.2f}
"""

        ax6.text(
            0.1,
            0.9,
            metrics_text,
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.1},
        )

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{symbol}_{timestamp}.png"
        filepath = Path("backtest_results") / filename
        filepath.parent.mkdir(exist_ok=True)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"📊 Visualization saved to: {filepath}")

        plt.show()


def main():
    """Run the visualized backtest."""
    print("🚀 AlgoStack REAL Backtest Visualization")
    print("=" * 50)

    # Create strategy
    strategy_config = {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    }

    strategy = SimpleMeanReversionStrategy(strategy_config)

    # Create backtester
    backtester = VisualizedBacktester(initial_capital=100000)

    # Run backtest
    symbol = "SPY"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    results = backtester.run_strategy_backtest(
        strategy=strategy, symbol=symbol, start_date=start_date, end_date=end_date
    )

    if results:
        print("\n✅ Backtest completed successfully!")
        print("\n📊 Key Results:")
        metrics = results["metrics"]
        print(f"   💰 Total Return: {metrics['total_return']:.1f}%")
        print(f"   📈 Benchmark Return: {metrics['benchmark_return']:.1f}%")
        print(f"   ⚡ Excess Return: {metrics['excess_return']:.1f}%")
        print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"   🎯 Win Rate: {metrics['win_rate']:.1f}%")

        if not results["trades"].empty:
            print(f"   🔄 Total Trades: {len(results['trades'])}")

            # Show trade details
            print("\n📋 Recent Trades:")
            recent_trades = results["trades"].tail(5)
            for _, trade in recent_trades.iterrows():
                print(
                    f"   {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} {trade['shares']} shares @ ${trade['price']:.2f}"
                )
    else:
        print("❌ Backtest failed!")


if __name__ == "__main__":
    main()
