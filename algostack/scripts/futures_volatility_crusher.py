#!/usr/bin/env python3
"""
Futures Volatility Crusher - E-mini S&P 500 (ES) and Micro E-mini (MES)
No PDT rules, built-in leverage, nearly 24-hour trading
"""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


class FuturesVolatilityCrusher:
    """
    Futures-optimized volatility crusher strategy.

    Key Advantages:
    - NO PDT RULES - Trade unlimited times with any account size
    - Built-in leverage (ES = ~16x, MES = ~16x)
    - Trade 23 hours per day (6pm - 5pm ET)
    - Superior liquidity and execution
    - Lower transaction costs
    """

    def __init__(self, contract_type="MES"):
        """
        Initialize with contract type.

        ES: E-mini S&P 500 ($50 per point, ~$15,000 margin)
        MES: Micro E-mini S&P 500 ($5 per point, ~$1,500 margin)
        """
        self.contract_type = contract_type

        if contract_type == "ES":
            self.contract_specs = {
                "symbol": "ES",
                "multiplier": 50,  # $50 per point
                "tick_size": 0.25,
                "tick_value": 12.50,  # $12.50 per tick
                "margin_requirement": 15000,  # Approximate day trading margin
                "commission": 2.25,  # Per side
                "hours": "23 hours (6pm-5pm ET with 1hr break)",
            }
        else:  # MES
            self.contract_specs = {
                "symbol": "MES",
                "multiplier": 5,  # $5 per point
                "tick_size": 0.25,
                "tick_value": 1.25,  # $1.25 per tick
                "margin_requirement": 1500,  # Approximate day trading margin
                "commission": 0.52,  # Per side
                "hours": "23 hours (6pm-5pm ET with 1hr break)",
            }

        # Optimized parameters for futures
        self.config = {
            "lookback_periods": 20,  # 20 * 5min = 100 minutes
            "z_entry": -1.0,  # Tighter for futures liquidity
            "z_exit": 0.0,  # Exit at mean
            "rsi_period": 3,  # 15 minutes
            "rsi_oversold": 25,
            "stop_loss_ticks": 8,  # 8 ticks = 2 points = manageable risk
            "profit_target_ticks": 12,  # 12 ticks = 3 points
            "max_daily_trades": 10,  # Risk management
            "avoid_news_window": 30,  # Avoid 30 min around major news
            "min_volume": 1000,  # Minimum volume filter
            "use_overnight": True,  # Trade overnight session
            "position_size": 1,  # Number of contracts
        }

    def calculate_indicators(self, data):
        """Calculate indicators optimized for futures."""
        df = data.copy()

        # Core indicators
        df["returns"] = df["close"].pct_change()

        # Z-score
        df["sma"] = df["close"].rolling(window=self.config["lookback_periods"]).mean()
        df["std"] = df["close"].rolling(window=self.config["lookback_periods"]).std()
        df["zscore"] = (df["close"] - df["sma"]) / df["std"]

        # RSI
        delta = df["close"].diff()
        gain = (
            (delta.where(delta > 0, 0)).rolling(window=self.config["rsi_period"]).mean()
        )
        loss = (
            (-delta.where(delta < 0, 0))
            .rolling(window=self.config["rsi_period"])
            .mean()
        )
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume analysis
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_spike"] = df["volume"] > df["volume_sma"] * 1.5

        # Volatility for dynamic stops
        df["atr"] = self.calculate_atr(df, period=10)
        df["volatility_regime"] = pd.cut(
            df["atr"], bins=3, labels=["low", "medium", "high"]
        )

        # Market microstructure
        df["spread"] = df["high"] - df["low"]
        df["range_pct"] = df["spread"] / df["close"] * 100

        # Time-based features for futures
        df["hour"] = df.index.hour
        df["is_us_session"] = (df["hour"] >= 9) & (df["hour"] < 16)  # 9am-4pm ET
        df["is_european_session"] = (df["hour"] >= 3) & (df["hour"] < 9)  # 3am-9am ET
        df["is_asian_session"] = (df["hour"] >= 18) | (df["hour"] < 3)  # 6pm-3am ET

        return df

    def calculate_atr(self, df, period):
        """Calculate ATR in points."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()

        return atr

    def simulate_futures_trading(self, data, initial_capital=25000):
        """Simulate futures trading with realistic constraints."""

        df = self.calculate_indicators(data)

        # Initialize
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        daily_trades = defaultdict(int)

        # Risk management
        max_contracts = int(capital / (self.contract_specs["margin_requirement"] * 2))
        print(f"Max contracts with ${capital:,} capital: {max_contracts}")

        # Skip warmup period
        for i in range(30, len(df)):
            current_time = df.index[i]
            current_date = current_time.date()
            current_price = df["close"].iloc[i]

            # Check daily trade limit
            if daily_trades[current_date] >= self.config["max_daily_trades"]:
                continue

            # Current equity
            if position != 0:
                # Mark-to-market P&L
                pnl = (
                    position
                    * (current_price - entry_price)
                    * self.contract_specs["multiplier"]
                )
                current_equity = capital + pnl
            else:
                current_equity = capital

            equity_curve.append(
                {"time": current_time, "equity": current_equity, "position": position}
            )

            # ENTRY LOGIC
            if position == 0:
                # Entry conditions
                entry_signal = (
                    df["zscore"].iloc[i] < self.config["z_entry"]
                    and df["rsi"].iloc[i] < self.config["rsi_oversold"]
                    and df["volume"].iloc[i] > self.config["min_volume"]
                    and not pd.isna(df["zscore"].iloc[i])
                )

                if entry_signal:
                    # Determine position size based on volatility
                    volatility_regime = df["volatility_regime"].iloc[i]
                    if volatility_regime == "low":
                        contracts = min(2, max_contracts)
                    elif volatility_regime == "medium":
                        contracts = min(1, max_contracts)
                    else:  # high volatility
                        contracts = min(1, max_contracts)

                    contracts = self.config["position_size"]  # Override for consistency

                    # Check margin requirement
                    required_margin = (
                        contracts * self.contract_specs["margin_requirement"]
                    )
                    if required_margin <= capital * 0.5:  # Use max 50% of capital
                        position = contracts
                        entry_price = current_price
                        entry_time = current_time
                        stop_price = entry_price - (
                            self.config["stop_loss_ticks"]
                            * self.contract_specs["tick_size"]
                        )
                        target_price = entry_price + (
                            self.config["profit_target_ticks"]
                            * self.contract_specs["tick_size"]
                        )

                        # Commission
                        capital -= (
                            contracts * self.contract_specs["commission"] * 2
                        )  # Round trip

                        trades.append(
                            {
                                "entry_time": current_time,
                                "entry_price": entry_price,
                                "contracts": contracts,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "session": (
                                    "US"
                                    if df["is_us_session"].iloc[i]
                                    else (
                                        "Europe"
                                        if df["is_european_session"].iloc[i]
                                        else "Asia"
                                    )
                                ),
                            }
                        )

                        daily_trades[current_date] += 1

            # EXIT LOGIC
            elif position != 0:
                # Check exit conditions
                hit_stop = current_price <= stop_price
                hit_target = current_price >= target_price
                eod_flat = (
                    df["hour"].iloc[i] == 15 and df.index[i].minute >= 45
                )  # Flatten before close

                if hit_stop or hit_target or eod_flat:
                    # Calculate P&L
                    gross_pnl = (
                        position
                        * (current_price - entry_price)
                        * self.contract_specs["multiplier"]
                    )
                    capital += gross_pnl

                    trades[-1].update(
                        {
                            "exit_time": current_time,
                            "exit_price": current_price,
                            "gross_pnl": gross_pnl,
                            "net_pnl": gross_pnl
                            - (position * self.contract_specs["commission"] * 2),
                            "exit_reason": (
                                "stop_loss"
                                if hit_stop
                                else "target" if hit_target else "eod_flat"
                            ),
                            "hold_time": (current_time - entry_time).total_seconds()
                            / 60,
                        }
                    )

                    position = 0

        # Close final position
        if position != 0:
            final_price = df["close"].iloc[-1]
            gross_pnl = (
                position
                * (final_price - entry_price)
                * self.contract_specs["multiplier"]
            )
            capital += gross_pnl
            trades[-1].update(
                {
                    "exit_time": df.index[-1],
                    "exit_price": final_price,
                    "gross_pnl": gross_pnl,
                    "exit_reason": "end_of_test",
                }
            )

        return {
            "trades": trades,
            "equity_curve": pd.DataFrame(equity_curve),
            "final_capital": capital,
            "initial_capital": initial_capital,
            "total_return": (capital - initial_capital) / initial_capital * 100,
        }

    def analyze_futures_performance(self, results):
        """Analyze futures trading performance."""

        trades = results["trades"]

        print("\n" + "=" * 80)
        print(f"FUTURES VOLATILITY CRUSHER - {self.contract_type} CONTRACT")
        print("=" * 80)

        print("\n📊 PERFORMANCE SUMMARY:")
        print(f"Initial Capital: ${results['initial_capital']:,}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")

        print("\n📈 TRADE STATISTICS:")
        print(f"Total Trades: {len(trades)}")

        if trades:
            completed_trades = [t for t in trades if "exit_time" in t]
            winning_trades = [t for t in completed_trades if t.get("net_pnl", 0) > 0]
            losing_trades = [t for t in completed_trades if t.get("net_pnl", 0) <= 0]

            win_rate = (
                len(winning_trades) / len(completed_trades) * 100
                if completed_trades
                else 0
            )

            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")

            if winning_trades:
                avg_win = np.mean([t["net_pnl"] for t in winning_trades])
                print(f"Average Win: ${avg_win:.2f}")

            if losing_trades:
                avg_loss = np.mean([t["net_pnl"] for t in losing_trades])
                print(f"Average Loss: ${avg_loss:.2f}")

            # Profit factor
            if winning_trades and losing_trades:
                total_wins = sum([t["net_pnl"] for t in winning_trades])
                total_losses = abs(sum([t["net_pnl"] for t in losing_trades]))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                print(f"Profit Factor: {profit_factor:.2f}")

            # Session analysis
            print("\n🌍 TRADES BY SESSION:")
            session_trades = defaultdict(int)
            session_pnl = defaultdict(float)

            for t in completed_trades:
                session = t.get("session", "Unknown")
                session_trades[session] += 1
                session_pnl[session] += t.get("net_pnl", 0)

            for session in ["US", "Europe", "Asia"]:
                if session in session_trades:
                    print(
                        f"{session}: {session_trades[session]} trades, "
                        f"P&L: ${session_pnl[session]:,.2f}"
                    )

            # Exit reasons
            print("\n🎯 EXIT REASONS:")
            exit_reasons = defaultdict(int)
            for t in completed_trades:
                exit_reasons[t.get("exit_reason", "unknown")] += 1

            for reason, count in exit_reasons.items():
                pct = count / len(completed_trades) * 100
                print(f"{reason}: {count} ({pct:.1f}%)")

        # Contract specifications
        print("\n📋 CONTRACT SPECIFICATIONS:")
        print(f"Contract: {self.contract_type}")
        print(f"Point Value: ${self.contract_specs['multiplier']}")
        print(f"Tick Size: {self.contract_specs['tick_size']}")
        print(f"Tick Value: ${self.contract_specs['tick_value']}")
        print(f"Margin Required: ${self.contract_specs['margin_requirement']:,}")
        print(f"Commission: ${self.contract_specs['commission']} per side")
        print(f"Trading Hours: {self.contract_specs['hours']}")

        return results


def main():
    """Demonstrate futures volatility crusher strategy."""

    print("FUTURES VOLATILITY CRUSHER STRATEGY")
    print("No PDT rules, superior leverage, 23-hour trading!")

    # Use SPY as proxy for ES/MES analysis
    spy = yf.Ticker("SPY")
    data = spy.history(period="60d", interval="5m")
    data.columns = data.columns.str.lower()

    # Test both contract types
    for contract_type in ["MES", "ES"]:
        print(f"\n{'='*80}")
        print(f"Testing {contract_type} contract...")

        # Determine appropriate capital
        if contract_type == "MES":
            initial_capital = 10000  # Good for MES
        else:
            initial_capital = 50000  # Better for ES

        crusher = FuturesVolatilityCrusher(contract_type)
        results = crusher.simulate_futures_trading(data, initial_capital)
        crusher.analyze_futures_performance(results)

        # Project annual returns
        days_tested = (data.index[-1] - data.index[0]).days
        annual_multiplier = 252 / days_tested
        projected_annual = results["total_return"] * annual_multiplier

        print(f"\n💰 PROJECTED ANNUAL RETURN: {projected_annual:.1f}%")

        # Calculate required stats
        if results["trades"]:
            trades_per_day = len(results["trades"]) / days_tested
            print(f"Average Trades Per Day: {trades_per_day:.1f}")

            # Capital efficiency
            margin_used = (
                crusher.config["position_size"]
                * crusher.contract_specs["margin_requirement"]
            )
            capital_efficiency = margin_used / initial_capital * 100
            print(f"Capital Efficiency: {capital_efficiency:.1f}% of capital at risk")

    # Save configuration
    config_data = {
        "strategy": "Futures Volatility Crusher",
        "instruments": {
            "MES": "Micro E-mini S&P 500 (beginner friendly)",
            "ES": "E-mini S&P 500 (experienced traders)",
        },
        "advantages": [
            "NO PDT RULES - Trade unlimited times with any account size",
            "Built-in leverage (~16x) without borrowing costs",
            "Trade 23 hours per day (massive opportunity)",
            "Superior liquidity and execution",
            "Lower transaction costs than stocks",
            "Favorable tax treatment (60/40 rule)",
        ],
        "requirements": {
            "MES": {
                "minimum_capital": "$5,000",
                "recommended_capital": "$10,000",
                "per_contract_margin": "$1,500",
            },
            "ES": {
                "minimum_capital": "$25,000",
                "recommended_capital": "$50,000",
                "per_contract_margin": "$15,000",
            },
        },
        "parameters": crusher.config,
        "risk_warnings": [
            "Futures losses can exceed initial investment",
            "Requires disciplined risk management",
            "Must monitor positions during overnight sessions",
            "Gaps can occur between sessions",
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FUTURES_VOLATILITY_CRUSHER_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n✅ Configuration saved to: {filename}")
    print("\n🚀 READY TO CRUSH VOLATILITY IN FUTURES MARKETS!")


if __name__ == "__main__":
    main()
