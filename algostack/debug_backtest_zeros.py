#!/usr/bin/env python3
"""
Debug script to trace why backtesting returns all zeros.
Tests with both yfinance and synthetic data to isolate the issue.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.mean_reversion_equity import MeanReversionEquity
from core.data_handler import DataHandler
from core.portfolio import Portfolio
from core.executor import Executor
from adapters.paper_executor import PaperExecutor
from adapters.yf_fetcher import YFinanceFetcher

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_backtest_zeros.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def create_synthetic_data(days=100):
    """Create synthetic price data for testing."""
    logger.info("Creating synthetic price data")
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create realistic price movement with mean reversion opportunities
    np.random.seed(42)
    price = 100.0
    prices = []
    
    for i in range(days):
        # Add some mean-reverting behavior
        if i > 0:
            # Mean reversion factor
            mean_reversion = (100 - price) * 0.1
            # Random walk
            random_walk = np.random.randn() * 2
            # Combine
            price += mean_reversion + random_walk
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': [p * (1 + np.random.randn() * 0.01) for p in prices],
        'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(days)]
    })
    
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Created synthetic data with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"First few rows:\n{df.head()}")
    
    return df

def test_strategy_directly(data, symbol='TEST'):
    """Test strategy directly without the full backtest framework."""
    logger.info("\n=== Testing Strategy Directly ===")
    
    # Initialize strategy with proper config (not params)
    config = {
        'name': 'MeanReversionEquity',
        'symbols': [symbol],
        'lookback_period': 20,
        'rsi_period': 2,
        'rsi_oversold': 10,
        'rsi_overbought': 90,
        'atr_period': 14,
        'atr_band_mult': 2.5,
        'ma_exit_period': 10,
        'stop_loss_atr': 3.0,
        'max_positions': 5,
        'position_size': 0.95
    }
    
    strategy = MeanReversionEquity(config)
    logger.info(f"Strategy initialized with config: {config}")
    
    # Initialize the strategy
    strategy.init()
    
    # Test next() method which is what the actual backtest uses
    signals = []
    lookback = config['lookback_period']
    
    logger.info(f"Processing {len(data) - lookback} bars starting from index {lookback}")
    
    for i in range(lookback, len(data)):
        try:
            # Get data up to current point
            current_data = data.iloc[:i+1].copy()
            
            # Add symbol attribute which strategies expect
            current_data.attrs['symbol'] = symbol
            
            # Call next() method
            signal = strategy.next(current_data)
            
            if signal is not None:
                signals.append({
                    'timestamp': data.index[i],
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'strength': signal.strength,
                    'price': signal.price,
                    'strategy_id': signal.strategy_id
                })
                logger.info(f"Signal at {data.index[i]}: {signal.direction} with strength {signal.strength}")
                
        except Exception as e:
            logger.error(f"Error at index {i}: {e}", exc_info=True)
    
    logger.info(f"Generated {len(signals)} signals total")
    
    if signals:
        signals_df = pd.DataFrame(signals)
        logger.info(f"Signal summary:\n{signals_df['direction'].value_counts()}")
        return signals_df
    else:
        logger.warning("No signals generated!")
        return pd.DataFrame()
        
    return pd.DataFrame(signals)

def test_with_synthetic_data():
    """Test with synthetic data to isolate data issues."""
    logger.info("\n=== Testing with Synthetic Data ===")
    
    # Create synthetic data
    data = create_synthetic_data(days=100)
    
    # Test strategy directly
    signals = test_strategy_directly(data, 'SYNTHETIC')
    
    if signals is not None:
        # Create a simple backtest
        logger.info("\n--- Running simple backtest ---")
        
        # Initialize components
        portfolio = Portfolio(initial_cash=100000)
        executor = PaperExecutor()
        
        # Process each signal
        positions = []
        returns = []
        
        for i in range(len(signals)):
            date = signals.index[i]
            signal_row = signals.iloc[i]
            price_row = data.iloc[i]
            
            logger.debug(f"Processing date {date}: signal={signal_row.get('signal', 0)}, price={price_row['close']}")
            
            # Track returns
            if i > 0:
                ret = (price_row['close'] - data.iloc[i-1]['close']) / data.iloc[i-1]['close']
                returns.append(ret)
            
        logger.info(f"Processed {len(signals)} signals")
        logger.info(f"Returns statistics: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")

def test_with_yfinance():
    """Test with real yfinance data."""
    logger.info("\n=== Testing with YFinance Data ===")
    
    # Initialize data fetcher
    fetcher = YFinanceFetcher()
    
    # Fetch data for a liquid stock
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    try:
        data = fetcher.fetch_historical(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        if data is None or data.empty:
            logger.error("No data returned from yfinance")
            return
            
        logger.info(f"Fetched data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Data index type: {type(data.index)}")
        logger.info(f"First few rows:\n{data.head()}")
        
        # Test strategy directly
        signals = test_strategy_directly(data, symbol)
        
    except Exception as e:
        logger.error(f"Error fetching yfinance data: {e}", exc_info=True)

def test_full_backtest_pipeline():
    """Test the full backtest pipeline with detailed logging."""
    logger.info("\n=== Testing Full Backtest Pipeline ===")
    
    # Create synthetic data
    data = create_synthetic_data(days=100)
    
    # Initialize components with proper config
    config = {
        'name': 'MeanReversionEquity',
        'symbols': ['TEST'],
        'lookback_period': 20,
        'rsi_period': 2,
        'rsi_oversold': 10,
        'rsi_overbought': 90,
        'atr_period': 14,
        'atr_band_mult': 2.5,
        'ma_exit_period': 10,
        'stop_loss_atr': 3.0,
        'max_positions': 5,
        'position_size': 0.95
    }
    
    strategy = MeanReversionEquity(config)
    strategy.init()
    
    portfolio = Portfolio(initial_cash=100000)
    
    # Run through the data
    logger.info("Running through backtest data...")
    
    signals_generated = 0
    orders_placed = 0
    capital = 100000
    positions = {}
    
    for i in range(config['lookback_period'], len(data)):  # Start after lookback period
        current_data = data.iloc[:i+1].copy()
        current_data.attrs['symbol'] = 'TEST'
        
        timestamp = data.index[i]
        current_price = float(data.iloc[i]['close'])
        
        # Get signal using next() method
        signal = strategy.next(current_data)
        
        if signal is not None:
            signals_generated += 1
            logger.info(f"Signal at {timestamp}: {signal.direction} @ ${current_price:.2f}")
            
            # Process signal
            if signal.symbol not in positions and signal.direction in ['LONG', 'SHORT']:
                # Open position
                shares = int((capital * 0.95) / current_price)
                if shares > 0:
                    positions[signal.symbol] = {
                        'direction': signal.direction,
                        'shares': shares,
                        'entry_price': current_price,
                        'entry_time': timestamp
                    }
                    capital -= shares * current_price
                    orders_placed += 1
                    logger.info(f"  Opened {signal.direction} position: {shares} shares @ ${current_price:.2f}")
                    
            elif signal.symbol in positions and signal.direction == 'FLAT':
                # Close position
                pos = positions[signal.symbol]
                exit_price = current_price
                
                # Calculate P&L
                if pos['direction'] == 'LONG':
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['shares']
                
                capital += pos['shares'] * exit_price
                logger.info(f"  Closed position: PnL = ${pnl:.2f}")
                
                del positions[signal.symbol]
                orders_placed += 1
    
    # Calculate final equity
    final_equity = capital
    for sym, pos in positions.items():
        final_price = float(data.iloc[-1]['close'])
        if pos['direction'] == 'LONG':
            final_equity += pos['shares'] * final_price
        else:
            final_equity += pos['shares'] * (2 * pos['entry_price'] - final_price)
    
    logger.info(f"\nBacktest summary:")
    logger.info(f"Signals generated: {signals_generated}")
    logger.info(f"Orders placed: {orders_placed}")
    logger.info(f"Open positions: {len(positions)}")
    logger.info(f"Final equity: ${final_equity:,.2f}")
    logger.info(f"Return: {(final_equity / 100000 - 1) * 100:.2f}%")

def test_indicator_calculations():
    """Test indicator calculations directly."""
    logger.info("\n=== Testing Indicator Calculations ===")
    
    # Create synthetic data
    data = create_synthetic_data(days=50)
    
    # Check if we're using pandas indicators or talib
    try:
        import talib
        logger.info("Using talib for indicators")
        using_talib = True
    except ImportError:
        logger.info("Using pandas indicators (talib not available)")
        using_talib = False
    
    # Calculate indicators manually
    try:
        # RSI
        if using_talib:
            rsi = talib.RSI(data['close'], timeperiod=2)
        else:
            # Simple RSI calculation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        logger.info(f"RSI calculated: {len(rsi)} values")
        logger.info(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
        logger.info(f"RSI < 10 count: {(rsi < 10).sum()}")
        logger.info(f"RSI > 90 count: {(rsi > 90).sum()}")
        
        # ATR
        if using_talib:
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        else:
            # Simple ATR calculation
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            close_low = np.abs(data['close'].shift() - data['low'])
            tr = pd.concat([high_low, high_close, close_low], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
        
        logger.info(f"ATR calculated: {len(atr)} values")
        logger.info(f"ATR range: {atr.min():.2f} to {atr.max():.2f}")
        
        # SMA
        if using_talib:
            sma_20 = talib.SMA(data['close'], timeperiod=20)
        else:
            sma_20 = data['close'].rolling(window=20).mean()
            
        logger.info(f"SMA(20) calculated: {len(sma_20)} values")
        
        # Create bands
        upper_band = sma_20 + (atr * 2.5)
        lower_band = sma_20 - (atr * 2.5)
        
        # Check for entry conditions
        oversold_count = 0
        below_band_count = 0
        entry_signal_count = 0
        
        for i in range(20, len(data)):  # Start after enough data for indicators
            if not np.isnan(rsi.iloc[i]) and not np.isnan(lower_band.iloc[i]):
                if rsi.iloc[i] < 10:
                    oversold_count += 1
                if data['close'].iloc[i] < lower_band.iloc[i]:
                    below_band_count += 1
                if rsi.iloc[i] < 10 and data['close'].iloc[i] < lower_band.iloc[i]:
                    entry_signal_count += 1
                    logger.info(f"  Entry condition met at index {i}: RSI={rsi.iloc[i]:.2f}, Close={data['close'].iloc[i]:.2f}, Lower Band={lower_band.iloc[i]:.2f}")
        
        logger.info(f"\nCondition summary:")
        logger.info(f"  RSI < 10: {oversold_count} times")
        logger.info(f"  Price < Lower Band: {below_band_count} times")
        logger.info(f"  Both conditions (entry signals): {entry_signal_count} times")
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)

def check_data_format_compatibility():
    """Check if data format is compatible with strategy expectations."""
    logger.info("\n=== Checking Data Format Compatibility ===")
    
    # Test different column name formats
    test_formats = [
        # Format 1: lowercase
        ['open', 'high', 'low', 'close', 'volume'],
        # Format 2: capitalized
        ['Open', 'High', 'Low', 'Close', 'Volume'],
        # Format 3: uppercase
        ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],
        # Format 4: with prefixes
        ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
    ]
    
    for i, columns in enumerate(test_formats):
        logger.info(f"\nTesting format {i+1}: {columns}")
        
        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        data = pd.DataFrame({
            col: np.random.randn(50) + 100 for col in columns
        }, index=dates)
        
        # Try to create strategy and calculate signals
        try:
            strategy = MeanReversionEquity()
            signals = strategy.calculate_signals(data, 'TEST')
            
            if signals is not None:
                logger.info(f"Format {i+1} SUCCESS: Generated {len(signals)} signals")
                non_zero = (signals != 0).sum().sum()
                logger.info(f"Non-zero values: {non_zero}")
            else:
                logger.warning(f"Format {i+1} FAILED: No signals generated")
                
        except Exception as e:
            logger.error(f"Format {i+1} ERROR: {e}")

if __name__ == "__main__":
    logger.info("Starting backtest debugging script...")
    
    # Run all tests
    check_data_format_compatibility()
    test_indicator_calculations()
    test_with_synthetic_data()
    test_with_yfinance()
    test_full_backtest_pipeline()
    
    logger.info("\nDebugging complete. Check debug_backtest_zeros.log for full details.")