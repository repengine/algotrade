#!/usr/bin/env python3
"""
Backtest the Futures Momentum Strategy using existing infrastructure.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from core.data_handler import DataHandler
# from adapters.yf_fetcher import YFinanceFetcher
from strategies.futures_momentum import FuturesMomentum
# from utils.logging import setup_logging

# setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simple_backtest(strategy, data):
    """Run a simple backtest without the full backtrader framework."""
    
    # Initialize tracking
    cash = 10000  # Starting capital
    position = 0
    shares = 0
    trades = []
    equity_curve = []
    
    # Initialize strategy
    strategy.init()
    
    # Process each bar
    for i in range(strategy.config['lookback_period'], len(data)):
        # Get data window
        window = data.iloc[:i+1].copy()
        window.attrs['symbol'] = 'SPY'
        
        # Get signal
        signal = strategy.next(window)
        
        current_price = data['close'].iloc[i]
        current_time = data.index[i]
        
        # Track equity
        if position > 0:
            current_equity = cash + (shares * current_price)
        else:
            current_equity = cash
            
        equity_curve.append({
            'time': current_time,
            'equity': current_equity,
            'price': current_price
        })
        
        # Process signal
        if signal:
            if signal.direction == 'LONG' and position == 0:
                # Buy
                shares = int(cash * 0.95 / current_price)
                cash -= shares * current_price
                cash -= 0.52  # Commission
                position = 1
                
                trades.append({
                    'time': current_time,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'signal': signal.reason,
                    'metadata': signal.metadata
                })
                
            elif signal.direction == 'FLAT' and position > 0:
                # Sell
                cash += shares * current_price
                cash -= 0.52  # Commission
                
                trades.append({
                    'time': current_time,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'signal': signal.reason,
                    'pnl': signal.metadata.get('pnl_pct', 0)
                })
                
                position = 0
                shares = 0
    
    # Close final position
    if position > 0:
        final_price = data['close'].iloc[-1]
        cash += shares * final_price
        cash -= 0.52
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    total_return = (cash - 10000) / 10000 * 100
    
    # Calculate additional metrics
    num_trades = len([t for t in trades if t['action'] == 'BUY'])
    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
    win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0
    
    # Calculate Sharpe ratio
    if len(equity_df) > 1:
        returns = equity_df['equity'].pct_change().dropna()
        # Annualize based on 5-min bars (78 per day * 252 days)
        sharpe = np.sqrt(78 * 252) * returns.mean() / (returns.std() + 1e-6)
    else:
        sharpe = 0
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'trades': trades,
        'equity_curve': equity_df,
        'final_value': cash
    }


def main():
    """Run the backtest."""
    
    logger.info("="*80)
    logger.info("BACKTESTING FUTURES MOMENTUM STRATEGY")
    logger.info("="*80)
    
    # Initialize data handler
    # data_handler = DataHandler(YFinanceFetcher())
    
    # Fetch 5-minute data
    logger.info("Fetching 5-minute SPY data...")
    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    # Download using yfinance directly for 5-min data
    import yfinance as yf
    spy = yf.Ticker('SPY')
    data = spy.history(start=start_date, end=end_date, interval='5m')
    data.columns = data.columns.str.lower()
    
    logger.info(f"Loaded {len(data)} 5-minute bars")
    
    if len(data) == 0:
        logger.error("No data loaded. Exiting.")
        return
        
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create strategy config
    strategy_config = {
        'symbols': ['SPY'],
        'lookback_period': 20,
        'breakout_threshold': 0.5,
        'rsi_period': 14,
        'rsi_threshold': 60,
        'atr_period': 14,
        'stop_loss_atr': 2.0,
        'profit_target_atr': 3.0,
        'volume_multiplier': 1.2,
        'position_size': 0.95,
        'trade_hours': {
            'start': 9.5,
            'end': 15.5
        }
    }
    
    # Initialize strategy
    strategy = FuturesMomentum(strategy_config)
    
    # Run backtest
    logger.info("Running backtest...")
    results = run_simple_backtest(strategy, data)
    
    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"Initial Capital: $10,000")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"Total Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    
    # Calculate annualized return
    days = (data.index[-1] - data.index[0]).days
    if days > 0:
        annual_return = (1 + results['total_return']/100) ** (365/days) - 1
        print(f"Annualized Return: {annual_return*100:.1f}%")
    
    # Show sample trades
    if results['trades']:
        print(f"\n📋 RECENT TRADES:")
        for trade in results['trades'][-10:]:
            action = trade['action']
            time = trade['time']
            price = trade['price']
            signal = trade['signal']
            print(f"{action} at {time}: ${price:.2f} ({signal})")
            if action == 'SELL' and 'pnl' in trade:
                print(f"  P&L: {trade['pnl']:.2f}%")
    
    # Save results
    output = {
        'strategy': 'FuturesMomentum',
        'config': strategy_config,
        'performance': {
            'total_return': results['total_return'],
            'annualized_return': annual_return * 100 if days > 0 else 0,
            'sharpe_ratio': results['sharpe_ratio'],
            'num_trades': results['num_trades'],
            'win_rate': results['win_rate']
        },
        'test_period': {
            'start': str(data.index[0]),
            'end': str(data.index[-1]),
            'bars': len(data)
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"futures_momentum_backtest_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to: {filename}")
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()