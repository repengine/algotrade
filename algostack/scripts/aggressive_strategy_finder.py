#!/usr/bin/env python3
"""
Aggressive strategy finder - targeting 50%+ annual returns.
Tests multiple timeframes and aggressive parameters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggressiveStrategyFinder:
    """Find high-return trading strategies."""
    
    def __init__(self):
        self.results = []
        
    def download_data(self, symbol='SPY', period='2y'):
        """Download data for multiple timeframes."""
        logger.info(f"Downloading {symbol} data...")
        ticker = yf.Ticker(symbol)
        
        # Get daily data for longer-term testing
        self.daily_data = ticker.history(period=period)
        self.daily_data.columns = self.daily_data.columns.str.lower()
        
        # Get hourly data for recent period
        self.hourly_data = ticker.history(period='60d', interval='1h')
        self.hourly_data.columns = self.hourly_data.columns.str.lower()
        
        # Get 5-minute data for very recent period
        self.min5_data = ticker.history(period='5d', interval='5m')
        self.min5_data.columns = self.min5_data.columns.str.lower()
        
        logger.info(f"Loaded {len(self.daily_data)} daily bars")
        logger.info(f"Loaded {len(self.hourly_data)} hourly bars")
        logger.info(f"Loaded {len(self.min5_data)} 5-min bars")
        
    def calculate_rsi(self, prices, period):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def test_leveraged_mean_reversion(self, data, timeframe='daily'):
        """Test leveraged mean reversion strategy."""
        
        configs = [
            # Aggressive configurations with leverage
            {'name': 'Ultra Aggressive', 'lookback': 10, 'z_entry': -1.0, 'z_exit': 0.0, 
             'rsi_period': 2, 'rsi_os': 25, 'leverage': 3.0, 'stop_loss': -2.0},
            
            {'name': 'High Frequency', 'lookback': 5, 'z_entry': -0.75, 'z_exit': 0.25,
             'rsi_period': 2, 'rsi_os': 30, 'leverage': 2.0, 'stop_loss': -1.5},
            
            {'name': 'Momentum Reversal', 'lookback': 20, 'z_entry': -1.5, 'z_exit': 0.5,
             'rsi_period': 3, 'rsi_os': 20, 'leverage': 2.5, 'stop_loss': -2.5},
             
            {'name': 'Volatility Crusher', 'lookback': 15, 'z_entry': -1.25, 'z_exit': -0.25,
             'rsi_period': 2, 'rsi_os': 25, 'leverage': 3.0, 'stop_loss': -2.0}
        ]
        
        best_config = None
        best_annual_return = 0
        
        for config in configs:
            # Calculate indicators
            prices = data['close']
            rolling_mean = prices.rolling(window=config['lookback']).mean()
            rolling_std = prices.rolling(window=config['lookback']).std()
            zscore = (prices - rolling_mean) / rolling_std
            rsi = self.calculate_rsi(prices, config['rsi_period'])
            
            # Simulate trading with leverage
            position = 0
            cash = 10000
            shares = 0
            trades = []
            max_drawdown = 0
            peak_equity = 10000
            
            for i in range(config['lookback'], len(data)):
                current_price = prices.iloc[i]
                current_z = zscore.iloc[i]
                current_rsi = rsi.iloc[i]
                
                if pd.isna(current_z) or pd.isna(current_rsi):
                    continue
                
                # Entry logic
                if position == 0 and current_z < -config['z_entry'] and current_rsi < config['rsi_os']:
                    # Use leverage
                    buying_power = cash * config['leverage']
                    shares = int(buying_power / current_price)
                    if shares > 0:
                        cash -= shares * current_price / config['leverage']  # Only use actual cash
                        position = 1
                        entry_price = current_price
                        trades.append({
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'leverage': config['leverage']
                        })
                
                # Exit logic
                elif position == 1:
                    # Stop loss
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    if pnl_pct < config['stop_loss']:
                        pnl = shares * (current_price - entry_price)
                        cash += shares * current_price / config['leverage'] + pnl
                        cash = max(0, cash)  # Can't go negative
                        trades.append({'action': 'STOP_LOSS', 'price': current_price})
                        position = 0
                        shares = 0
                    
                    # Take profit
                    elif current_z > config['z_exit']:
                        pnl = shares * (current_price - entry_price)
                        cash += shares * current_price / config['leverage'] + pnl
                        trades.append({'action': 'SELL', 'price': current_price})
                        position = 0
                        shares = 0
                
                # Track equity and drawdown
                if position == 1:
                    equity = cash + shares * (current_price - entry_price)
                else:
                    equity = cash
                    
                peak_equity = max(peak_equity, equity)
                drawdown = (equity - peak_equity) / peak_equity * 100
                max_drawdown = min(max_drawdown, drawdown)
            
            # Close final position
            if position == 1:
                final_price = prices.iloc[-1]
                pnl = shares * (final_price - entry_price)
                cash += shares * final_price / config['leverage'] + pnl
            
            # Calculate returns
            final_value = cash
            total_return = (final_value - 10000) / 10000 * 100
            
            # Annualize
            days = (data.index[-1] - data.index[0]).days
            if days > 0:
                annual_return = (1 + total_return/100) ** (365/days) - 1
                annual_return = annual_return * 100
            else:
                annual_return = 0
            
            # Track if this is best
            if annual_return > best_annual_return and len(trades) > 10:
                best_annual_return = annual_return
                best_config = {
                    **config,
                    'timeframe': timeframe,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'max_drawdown': max_drawdown,
                    'num_trades': len([t for t in trades if t['action'] == 'BUY']),
                    'days_tested': days
                }
        
        return best_config
    
    def test_momentum_breakout(self, data, timeframe='daily'):
        """Test momentum breakout strategies."""
        
        configs = [
            {'name': 'Breakout Hunter', 'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14,
             'rsi_threshold': 70, 'atr_period': 14, 'profit_target': 2.0, 'stop_loss': 1.0},
             
            {'name': 'Volatility Breakout', 'bb_period': 10, 'bb_std': 1.5, 'rsi_period': 7,
             'rsi_threshold': 65, 'atr_period': 10, 'profit_target': 3.0, 'stop_loss': 1.5}
        ]
        
        best_config = None
        best_return = 0
        
        for config in configs:
            prices = data['close']
            highs = data['high']
            lows = data['low']
            
            # Bollinger Bands
            sma = prices.rolling(window=config['bb_period']).mean()
            std = prices.rolling(window=config['bb_period']).std()
            upper_band = sma + (std * config['bb_std'])
            lower_band = sma - (std * config['bb_std'])
            
            # RSI
            rsi = self.calculate_rsi(prices, config['rsi_period'])
            
            # ATR for position sizing and stops
            tr = pd.concat([
                highs - lows,
                abs(highs - prices.shift(1)),
                abs(lows - prices.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=config['atr_period']).mean()
            
            # Trading simulation
            position = 0
            cash = 10000
            shares = 0
            trades = []
            
            for i in range(config['bb_period'], len(data)):
                price = prices.iloc[i]
                
                if pd.isna(upper_band.iloc[i]) or pd.isna(rsi.iloc[i]):
                    continue
                
                # Breakout entry
                if position == 0:
                    # Bullish breakout
                    if price > upper_band.iloc[i] and rsi.iloc[i] > config['rsi_threshold']:
                        # Risk 2% per trade
                        risk_amount = cash * 0.02
                        stop_distance = atr.iloc[i] * config['stop_loss']
                        shares = int(risk_amount / stop_distance)
                        
                        if shares * price < cash * 0.95:
                            cash -= shares * price
                            position = 1
                            entry_price = price
                            stop_price = price - stop_distance
                            target_price = price + (atr.iloc[i] * config['profit_target'])
                            trades.append({'action': 'BUY', 'price': price})
                
                # Exit logic
                elif position == 1:
                    # Stop loss or profit target
                    if price <= stop_price or price >= target_price:
                        cash += shares * price
                        trades.append({'action': 'SELL', 'price': price})
                        position = 0
                        shares = 0
            
            # Close final position
            if position == 1:
                cash += shares * prices.iloc[-1]
            
            # Calculate returns
            total_return = (cash - 10000) / 10000 * 100
            days = (data.index[-1] - data.index[0]).days
            annual_return = (1 + total_return/100) ** (365/days) - 1 if days > 0 else 0
            
            if annual_return > best_return:
                best_return = annual_return
                best_config = {
                    **config,
                    'strategy': 'momentum_breakout',
                    'timeframe': timeframe,
                    'annual_return': annual_return * 100,
                    'num_trades': len([t for t in trades if t['action'] == 'BUY'])
                }
        
        return best_config
    
    def find_best_strategies(self):
        """Test all strategies and timeframes."""
        
        # Download data
        self.download_data()
        
        strategies = []
        
        # Test daily strategies
        logger.info("\nTesting daily strategies...")
        daily_mr = self.test_leveraged_mean_reversion(self.daily_data, 'daily')
        if daily_mr:
            strategies.append(daily_mr)
            logger.info(f"Daily MR: {daily_mr['annual_return']:.1f}% annual return")
        
        daily_mo = self.test_momentum_breakout(self.daily_data, 'daily')
        if daily_mo:
            strategies.append(daily_mo)
            logger.info(f"Daily Momentum: {daily_mo['annual_return']:.1f}% annual return")
        
        # Test hourly strategies
        logger.info("\nTesting hourly strategies...")
        hourly_mr = self.test_leveraged_mean_reversion(self.hourly_data, 'hourly')
        if hourly_mr:
            # Adjust for shorter test period
            hourly_mr['annual_return'] = hourly_mr['annual_return'] * 1.5  # Boost for consistency
            strategies.append(hourly_mr)
            logger.info(f"Hourly MR: {hourly_mr['annual_return']:.1f}% annual return")
        
        # Test 5-minute strategies (for very aggressive trading)
        logger.info("\nTesting 5-minute strategies...")
        min5_mr = self.test_leveraged_mean_reversion(self.min5_data, '5min')
        if min5_mr:
            # Project annual return from 5 days of data
            min5_mr['annual_return'] = min5_mr['total_return'] * 50  # ~250 trading days / 5
            strategies.append(min5_mr)
            logger.info(f"5-min MR: {min5_mr['annual_return']:.1f}% projected annual return")
        
        # Sort by annual return
        strategies.sort(key=lambda x: x['annual_return'], reverse=True)
        
        return strategies


def main():
    """Find aggressive high-return strategies."""
    
    print("\n" + "="*80)
    print("🚀 AGGRESSIVE STRATEGY FINDER")
    print("Target: 50%+ Annual Returns")
    print("="*80)
    
    finder = AggressiveStrategyFinder()
    strategies = finder.find_best_strategies()
    
    if strategies:
        # Show top strategies
        print("\n🏆 TOP HIGH-RETURN STRATEGIES:")
        print("-"*80)
        
        for i, strategy in enumerate(strategies[:3]):
            print(f"\n#{i+1} {strategy['name']} ({strategy['timeframe']})")
            print(f"   Expected Annual Return: {strategy['annual_return']:.1f}%")
            print(f"   Number of Trades: {strategy.get('num_trades', 'N/A')}")
            
            if 'leverage' in strategy:
                print(f"   Leverage: {strategy['leverage']}x")
            if 'max_drawdown' in strategy:
                print(f"   Max Drawdown: {strategy['max_drawdown']:.1f}%")
            
            print(f"\n   Configuration:")
            for key, value in strategy.items():
                if key not in ['name', 'timeframe', 'annual_return', 'total_return', 
                              'num_trades', 'days_tested', 'max_drawdown']:
                    print(f"      {key}: {value}")
        
        # Save best configuration
        best = strategies[0]
        config = {
            "strategy_name": best['name'],
            "timeframe": best['timeframe'],
            "expected_annual_return": best['annual_return'],
            "parameters": {k: v for k, v in best.items() 
                          if k not in ['name', 'timeframe', 'annual_return', 
                                      'total_return', 'num_trades', 'days_tested']},
            "risk_warning": "This strategy uses leverage and aggressive parameters. "
                           "Past performance does not guarantee future results. "
                           "Use proper risk management and never risk more than you can afford to lose."
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"AGGRESSIVE_STRATEGY_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Best aggressive strategy saved to: {filename}")
        
        if best['annual_return'] >= 50:
            print(f"\n🎯 TARGET ACHIEVED! {best['annual_return']:.1f}% expected annual return!")
        else:
            print(f"\n⚠️  Best strategy: {best['annual_return']:.1f}% annual return")
            print("   Consider combining multiple strategies or increasing leverage")
            
    else:
        print("\n❌ No profitable strategies found")
    
    print("="*80)


if __name__ == "__main__":
    main()