#!/usr/bin/env python3
"""
5-Minute Volatility Crusher - Detailed Analysis and Implementation
Ultra-high frequency mean reversion with 3x leverage
"""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


class VolatilityCrusher5Min:
    """
    5-Minute Volatility Crusher Strategy

    Core Concept:
    - Exploits micro mean reversions in 5-minute timeframe
    - Uses 3x leverage to amplify small moves
    - Tight stop losses to control risk
    - Targets quick profits from volatility spikes
    """

    def __init__(self):
        self.config = {
            "lookback_periods": 15,  # 15 * 5min = 75 minutes
            "z_entry": -1.25,  # Enter when 1.25 std dev below mean
            "z_exit": -0.25,  # Exit when price recovers to -0.25 std dev
            "rsi_period": 2,  # Ultra-short RSI (10 minutes)
            "rsi_oversold": 25,  # Aggressive oversold level
            "leverage": 3.0,  # 3x leverage
            "stop_loss_pct": -2.0,  # -2% stop loss (becomes -6% with leverage)
            "position_size": 0.95,  # Use 95% of available buying power
            "max_positions": 1,  # Focus on one position at a time
            "min_volume_ratio": 1.2,  # Only trade when volume > 1.2x average
        }

    def download_5min_data(self, symbol="SPY", days=30):
        """Download 5-minute data (limited to last 60 days by yfinance)."""
        print(f"Downloading {days} days of 5-minute data for {symbol}...")

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d", interval="5m")
        data.columns = data.columns.str.lower()

        print(f"Downloaded {len(data)} 5-minute bars")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")

        return data

    def calculate_indicators(self, data):
        """Calculate all technical indicators."""
        df = data.copy()

        # Price-based indicators
        df["returns"] = df["close"].pct_change()

        # Z-score (standardized price)
        df["sma"] = df["close"].rolling(window=self.config["lookback_periods"]).mean()
        df["std"] = df["close"].rolling(window=self.config["lookback_periods"]).std()
        df["zscore"] = (df["close"] - df["sma"]) / df["std"]

        # RSI
        df["rsi"] = self.calculate_rsi(df["close"], self.config["rsi_period"])

        # Volume analysis
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Volatility bands for dynamic stops
        df["atr"] = self.calculate_atr(df, period=10)
        df["upper_band"] = df["sma"] + (2 * df["std"])
        df["lower_band"] = df["sma"] - (2 * df["std"])

        # Time-based features
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["is_first_hour"] = (df["hour"] == 9) & (df["minute"] >= 30)
        df["is_last_hour"] = (df["hour"] == 15) & (df["minute"] >= 0)

        return df

    def calculate_rsi(self, prices, period):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period):
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()

        return atr

    def backtest(self, data):
        """Run detailed backtest with trade analysis."""
        df = self.calculate_indicators(data)

        # Initialize tracking variables
        position = 0
        cash = 10000
        initial_capital = cash
        shares = 0
        trades = []
        equity_curve = []

        # Performance tracking
        entry_price = 0
        entry_time = None
        max_equity = initial_capital

        # Skip initial periods needed for indicators
        start_idx = max(self.config["lookback_periods"], 20)

        for i in range(start_idx, len(df)):
            current_time = df.index[i]
            current_price = df["close"].iloc[i]
            current_zscore = df["zscore"].iloc[i]
            current_rsi = df["rsi"].iloc[i]
            current_volume_ratio = df["volume_ratio"].iloc[i]

            # Skip if indicators are invalid
            if pd.isna(current_zscore) or pd.isna(current_rsi):
                continue

            # Calculate current equity
            if position == 1:
                current_equity = cash + shares * current_price
            else:
                current_equity = cash

            equity_curve.append(
                {"time": current_time, "equity": current_equity, "position": position}
            )

            # Update max equity for drawdown calculation
            max_equity = max(max_equity, current_equity)

            # ENTRY LOGIC
            if position == 0:
                # Check entry conditions
                entry_signal = (
                    current_zscore < self.config["z_entry"]
                    and current_rsi < self.config["rsi_oversold"]
                    and current_volume_ratio > self.config["min_volume_ratio"]
                    and not df["is_last_hour"].iloc[i]  # Don't enter in last hour
                )

                if entry_signal:
                    # Calculate position size with leverage
                    buying_power = cash * self.config["leverage"]
                    shares = int(
                        buying_power * self.config["position_size"] / current_price
                    )

                    if shares > 0:
                        # Execute trade
                        cash -= shares * current_price / self.config["leverage"]
                        position = 1
                        entry_price = current_price
                        entry_time = current_time

                        trades.append(
                            {
                                "entry_time": current_time,
                                "entry_price": current_price,
                                "shares": shares,
                                "leverage": self.config["leverage"],
                                "entry_zscore": current_zscore,
                                "entry_rsi": current_rsi,
                                "entry_volume_ratio": current_volume_ratio,
                            }
                        )

            # EXIT LOGIC
            elif position == 1:
                # Calculate P&L
                pnl_pct = (current_price - entry_price) / entry_price * 100

                # Exit conditions
                stop_loss_hit = pnl_pct <= self.config["stop_loss_pct"]
                target_hit = current_zscore >= self.config["z_exit"]
                time_stop = (
                    current_time - entry_time
                ).total_seconds() > 7200  # 2 hour max hold
                market_close = df["is_last_hour"].iloc[i]

                if stop_loss_hit or target_hit or time_stop or market_close:
                    # Calculate actual P&L with leverage
                    gross_pnl = shares * (current_price - entry_price)
                    net_cash = (
                        cash
                        + (shares * current_price / self.config["leverage"])
                        + gross_pnl
                    )

                    # Record trade
                    trades[-1].update(
                        {
                            "exit_time": current_time,
                            "exit_price": current_price,
                            "exit_reason": (
                                "stop_loss"
                                if stop_loss_hit
                                else (
                                    "target"
                                    if target_hit
                                    else "time_stop" if time_stop else "market_close"
                                )
                            ),
                            "pnl_pct": pnl_pct,
                            "gross_pnl": gross_pnl,
                            "hold_time": (current_time - entry_time).total_seconds()
                            / 60,  # minutes
                            "exit_zscore": current_zscore,
                            "exit_rsi": df["rsi"].iloc[i],
                        }
                    )

                    # Update cash and reset position
                    cash = max(0, net_cash)  # Prevent negative cash
                    position = 0
                    shares = 0

        # Close any open position
        if position == 1:
            final_price = df["close"].iloc[-1]
            gross_pnl = shares * (final_price - entry_price)
            cash = cash + (shares * final_price / self.config["leverage"]) + gross_pnl
            trades[-1].update(
                {
                    "exit_time": df.index[-1],
                    "exit_price": final_price,
                    "exit_reason": "end_of_data",
                    "pnl_pct": (final_price - entry_price) / entry_price * 100,
                    "gross_pnl": gross_pnl,
                }
            )

        # Calculate performance metrics
        final_equity = cash
        total_return = (final_equity - initial_capital) / initial_capital * 100

        # Create equity DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index("time", inplace=True)

        return {
            "trades": trades,
            "equity_curve": equity_df,
            "final_equity": final_equity,
            "total_return": total_return,
            "initial_capital": initial_capital,
        }

    def analyze_results(self, results, data):
        """Comprehensive analysis of backtest results."""
        trades = results["trades"]
        equity_curve = results["equity_curve"]

        print("\n" + "=" * 80)
        print("5-MINUTE VOLATILITY CRUSHER - DETAILED ANALYSIS")
        print("=" * 80)

        # Basic metrics
        print("\n📊 PERFORMANCE SUMMARY:")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")

        # Time period
        days = (data.index[-1] - data.index[0]).days
        print(f"Test Period: {days} days")

        # Annualized return (assuming 252 trading days)
        if days > 0:
            annual_multiplier = 252 / days
            projected_annual_return = results["total_return"] * annual_multiplier
            print(f"Projected Annual Return: {projected_annual_return:.1f}%")

        # Trade analysis
        print("\n📈 TRADE STATISTICS:")
        print(f"Total Trades: {len(trades)}")

        if len(trades) > 0:
            # Win rate
            winning_trades = [t for t in trades if t.get("pnl_pct", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl_pct", 0) <= 0]
            win_rate = len(winning_trades) / len(trades) * 100

            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")

            # Average P&L
            pnls = [t.get("pnl_pct", 0) for t in trades if "pnl_pct" in t]
            if pnls:
                avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades else 0
                avg_loss = np.mean([p for p in pnls if p <= 0]) if losing_trades else 0

                print(f"\nAverage Win: {avg_win:.2f}%")
                print(f"Average Loss: {avg_loss:.2f}%")
                print(
                    f"Profit Factor: {abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))):.2f}"
                    if losing_trades and avg_loss != 0
                    else "N/A"
                )

            # Hold time analysis
            hold_times = [t.get("hold_time", 0) for t in trades if "hold_time" in t]
            if hold_times:
                print(f"\nAverage Hold Time: {np.mean(hold_times):.1f} minutes")
                print(f"Max Hold Time: {max(hold_times):.1f} minutes")
                print(f"Min Hold Time: {min(hold_times):.1f} minutes")

            # Exit reason breakdown
            print("\n🎯 EXIT REASONS:")
            exit_reasons = defaultdict(int)
            for t in trades:
                if "exit_reason" in t:
                    exit_reasons[t["exit_reason"]] += 1

            for reason, count in exit_reasons.items():
                print(f"{reason}: {count} ({count/len(trades)*100:.1f}%)")

            # Hourly distribution
            print("\n🕐 TRADES BY HOUR:")
            hourly_trades = defaultdict(int)
            for t in trades:
                if "entry_time" in t:
                    hour = t["entry_time"].hour
                    hourly_trades[hour] += 1

            for hour in sorted(hourly_trades.keys()):
                print(f"{hour:02d}:00 - {hourly_trades[hour]} trades")

        # Risk metrics
        if len(equity_curve) > 0:
            print("\n⚠️  RISK METRICS:")

            # Calculate drawdown
            rolling_max = equity_curve["equity"].expanding().max()
            drawdown = (equity_curve["equity"] - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()

            print(f"Maximum Drawdown: {max_drawdown:.2f}%")
            print(
                f"Maximum Drawdown (leveraged): {max_drawdown * self.config['leverage']:.2f}%"
            )

            # Sharpe ratio (5-min returns)
            returns = equity_curve["equity"].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Annualize: ~78 5-min periods per day * 252 days
                sharpe = np.sqrt(78 * 252) * returns.mean() / returns.std()
                print(f"Sharpe Ratio: {sharpe:.2f}")

        return results

    def save_configuration(self, results, filename_prefix="5min_volatility_crusher"):
        """Save the strategy configuration and results."""
        config_data = {
            "strategy_name": "5-Minute Volatility Crusher",
            "timeframe": "5m",
            "configuration": self.config,
            "performance": {
                "total_return": results["total_return"],
                "num_trades": len(results["trades"]),
                "projected_annual_return": results["total_return"]
                * 50,  # Rough projection
            },
            "implementation_notes": {
                "data_requirements": "Real-time 5-minute bars with volume",
                "execution_requirements": "Sub-second execution capability",
                "account_requirements": "3x intraday leverage, pattern day trader status",
                "recommended_capital": "$25,000 minimum (PDT requirement)",
                "automation": "REQUIRED - Manual execution not feasible",
            },
            "risk_warnings": [
                "Uses 3x leverage - losses are magnified",
                "Requires perfect execution timing",
                "Susceptible to slippage in fast markets",
                "May hit pattern day trader limits",
                "Drawdowns can exceed 100% of initial capital",
            ],
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"\n💾 Configuration saved to: {filename}")

        return filename


def main():
    """Run the 5-Minute Volatility Crusher analysis."""

    crusher = VolatilityCrusher5Min()

    # Download data (last 30 days)
    data = crusher.download_5min_data("SPY", days=30)

    # Run backtest
    results = crusher.backtest(data)

    # Analyze results
    crusher.analyze_results(results, data)

    # Show sample trades
    if results["trades"]:
        print("\n📋 SAMPLE TRADES (Last 5):")
        print("-" * 80)

        for trade in results["trades"][-5:]:
            if "exit_time" in trade:
                print(f"\nEntry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
                print(
                    f"  Z-Score: {trade['entry_zscore']:.2f}, RSI: {trade['entry_rsi']:.1f}"
                )
                print(f"Exit: {trade['exit_time']} @ ${trade['exit_price']:.2f}")
                print(f"  Reason: {trade['exit_reason']}")
                print(
                    f"  P&L: {trade['pnl_pct']:.2f}% | Hold Time: {trade.get('hold_time', 0):.1f} min"
                )

    # Save configuration
    crusher.save_configuration(results)

    print("\n" + "=" * 80)
    print("🚀 5-MINUTE VOLATILITY CRUSHER READY FOR DEPLOYMENT")
    print("=" * 80)


if __name__ == "__main__":
    main()
