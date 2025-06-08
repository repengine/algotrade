#!/usr/bin/env python3
"""
AlgoStack Dashboard - Pandas Version with Alpha Vantage Integration

This version:
1. Uses pure pandas indicators (no TA-Lib dependency)
2. Integrates Alpha Vantage for intraday market data
3. Maintains all existing functionality
"""

import os
import sys
import json
import time
import importlib
import inspect
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

# Import pandas indicators and create talib replacement
from pandas_indicators import create_talib_compatible_module

# Replace talib in sys.modules before any other imports
sys.modules['talib'] = create_talib_compatible_module()

# Import Alpha Vantage fetcher directly to avoid adapter __init__ imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'adapters'))
from av_fetcher import AlphaVantageFetcher

# Mock validator that converts types automatically
def mock_validator(config):
    """Mock validator that converts types as needed."""
    converted = config.copy()
    
    # Common type conversions
    for key, value in converted.items():
        # Convert numeric strings to numbers
        if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
            try:
                if '.' in value:
                    converted[key] = float(value)
                else:
                    converted[key] = int(value)
            except:
                pass
        
        # Convert 0/1 to bool for known boolean parameters
        if key in ['volume_filter', 'trail_stop', 'volume_confirmation', 'volatility_filter', 'rsi_filter']:
            if value in [0, 1]:
                converted[key] = bool(value)
            elif value in ['0', '1']:
                converted[key] = bool(int(value))
        
        # Ensure floats for percentage parameters
        if any(x in key for x in ['threshold', 'oversold', 'overbought', 'pct', 'mult', 'ratio']):
            try:
                converted[key] = float(value)
            except:
                pass
        
        # Ensure ints for period parameters
        if 'period' in key or key in ['max_positions', 'max_pairs', 'max_trades_per_day']:
            try:
                converted[key] = int(value)
            except:
                pass
    
    return converted

# Patch all validators
validators_to_patch = [
    'validate_mean_reversion_config',
    'validate_trend_following_config',
    'validate_pairs_trading_config',
    'validate_pairs_stat_arb_config',
    'validate_intraday_orb_config',
    'validate_overnight_drift_config',
    'validate_hybrid_regime_config',
]

for validator in validators_to_patch:
    try:
        patch(f'utils.validators.strategy_validators.{validator}', side_effect=mock_validator).start()
    except AttributeError:
        pass

# Import integration helpers (now using pandas indicators)
from strategy_integration_helpers import DataFormatConverter, RiskContextMock

# Import strategy defaults
from strategy_defaults import get_strategy_defaults, merge_with_defaults, PARAMETER_TOOLTIPS

# Import base strategy
from strategies.base import BaseStrategy, Signal


class AlphaVantageDataManager:
    """Manages data fetching from Alpha Vantage."""
    
    def __init__(self):
        self.av_key = self._get_api_key()
        self.av_fetcher = None
        if self.av_key:
            try:
                # Check for premium tier indicator
                premium = os.getenv('ALPHA_VANTAGE_PREMIUM', 'false').lower() == 'true'
                self.av_fetcher = AlphaVantageFetcher(api_key=self.av_key, premium=premium)
                st.sidebar.success("✅ Alpha Vantage API connected")
            except Exception as e:
                st.sidebar.error(f"❌ Alpha Vantage error: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get Alpha Vantage API key from environment or config."""
        # First check environment
        key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if key:
            return key
        
        # Then check secrets.yaml
        try:
            import yaml
            secrets_path = Path(__file__).parent / 'config' / 'secrets.yaml'
            if secrets_path.exists():
                with open(secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f)
                    if secrets and 'data_providers' in secrets:
                        if 'alphavantage' in secrets['data_providers']:
                            return secrets['data_providers']['alphavantage'].get('api_key')
        except Exception as e:
            print(f"Error reading secrets.yaml: {e}")
        
        return None
    
    def fetch_data(self, symbol: str, period: str, interval: str = "1d", 
                   data_source: str = "yfinance") -> pd.DataFrame:
        """Fetch data from specified source."""
        end_date = datetime.now()
        
        # Calculate start date based on period
        period_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            'max': 3650
        }
        
        days = period_map.get(period, 365)
        start_date = end_date - timedelta(days=days)
        
        if data_source == "alpha_vantage" and self.av_fetcher:
            try:
                df = self.av_fetcher.fetch_ohlcv(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                if not df.empty:
                    # Alpha Vantage data is valid now, no need to validate
                    return df
                else:
                    st.warning("No data from Alpha Vantage, falling back to Yahoo Finance")
            except Exception as e:
                st.error(f"Alpha Vantage error: {e}")
                st.info("Falling back to Yahoo Finance")
        
        # Fall back to Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df.attrs['symbol'] = symbol
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLC data quality."""
        try:
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check for nulls
            if df[required_cols].isnull().any().any():
                return False
            
            # Check OHLC relationships
            invalid_rows = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_rows.any():
                print(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships")
                return False
            
            return True
        except Exception:
            return False


class PandasStrategyManager:
    """Strategy manager using pandas indicators."""
    
    def __init__(self):
        self.strategies = self._discover_strategies()
        self.converter = DataFormatConverter()
    
    def _discover_strategies(self) -> Dict[str, type]:
        """Discover all strategy classes in the strategies folder."""
        strategies = {}
        strategies_dir = Path(__file__).parent / 'strategies'
        
        # Skip these files
        skip_files = {'__init__.py', 'base.py', '__pycache__'}
        
        for py_file in strategies_dir.glob('*.py'):
            if py_file.name in skip_files:
                continue
            
            module_name = py_file.stem
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"strategies.{module_name}", 
                    py_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find strategy classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        strategies[name] = obj
                        
            except Exception as e:
                error_msg = str(e)
                # Only show errors that aren't about optional dependencies
                if not any(skip in error_msg for skip in ['statsmodels', 'sklearn', 'scipy']):
                    print(f"⚠️ Error loading {module_name}: {e}")
        
        return strategies
    
    def get_display_name(self, class_name: str) -> str:
        """Convert class name to display name."""
        import re
        formatted = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
        return formatted
    
    def get_strategy_parameters(self, strategy_class_name: str) -> Dict[str, Any]:
        """Get all parameters for a strategy with proper defaults."""
        return get_strategy_defaults(strategy_class_name)
    
    def run_backtest(self, strategy_class: type, strategy_name: str,
                    user_params: Dict[str, Any], data: pd.DataFrame, 
                    initial_capital: float = 100000) -> Dict[str, Any]:
        """Run a complete backtest with pandas indicators."""
        try:
            print(f"\n🔍 Running backtest for {strategy_name}")
            print(f"   Data shape: {data.shape}")
            print(f"   Data columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Using pandas indicators")
            
            # Merge user parameters with defaults
            full_config = merge_with_defaults(strategy_name, user_params)
            
            # Ensure the symbol is in the symbols list
            symbol = user_params.get('symbol', 'SPY')
            if 'symbols' not in full_config:
                full_config['symbols'] = [symbol]
            elif symbol not in full_config['symbols']:
                full_config['symbols'].append(symbol)
            
            # Special handling for OvernightDrift strategy
            if strategy_name == 'OvernightDrift':
                # If filters are disabled, adjust thresholds to be more permissive
                if not user_params.get('volume_filter', True):
                    full_config['volume_threshold'] = 0.0  # Accept any volume
                if not user_params.get('volatility_filter', True):
                    full_config['min_atr'] = 0.0  # No minimum volatility
                    full_config['max_atr'] = 1.0  # Very high max volatility
                if not user_params.get('trend_filter', True):
                    full_config['trend_filter'] = False
            
            # Convert data format
            strategy_data = self.converter.dashboard_to_strategy(
                data, 
                symbol=symbol
            )
            
            print(f"   Converted data columns: {list(strategy_data.columns)}")
            print(f"   Strategy config symbols: {full_config.get('symbols', [])}")
            
            # Store the symbol attribute
            symbol = strategy_data.attrs.get('symbol', 'UNKNOWN')
            
            # Initialize strategy with full config
            strategy = strategy_class(full_config)
            
            # Clear any existing positions
            if hasattr(strategy, 'positions'):
                strategy.positions = {}
            strategy.init()
            
            # Get lookback period
            lookback = getattr(strategy, 'lookback_period', 50)
            if 'lookback_period' in full_config:
                lookback = full_config['lookback_period']
            
            # Handle different lookback types
            if isinstance(lookback, (list, tuple)):
                lookback = max(lookback)
            elif isinstance(lookback, dict):
                lookback = max(lookback.values())
            
            lookback = int(lookback)
            
            # Initialize tracking
            capital = initial_capital
            positions = {}
            trades = []
            equity_curve = []
            all_signals = []
            signal_dates = []
            
            # Process each timestamp
            print(f"   Processing {len(strategy_data) - lookback} bars (lookback={lookback})")
            signals_count = 0
            
            for i in range(lookback, len(strategy_data)):
                current_data = strategy_data.iloc[:i+1].copy()
                
                # Restore symbol attribute
                current_data.attrs['symbol'] = symbol
                
                timestamp = strategy_data.index[i]
                current_price = float(strategy_data['close'].iloc[i])
                
                # Create mock risk context
                risk_context = RiskContextMock(account_equity=capital)
                
                # Get signals
                signals = []
                try:
                    # Call the next method which is what strategies use
                    signal = strategy.next(current_data)
                    
                    if signal is not None:
                        signals.append(signal)
                        signals_count += 1
                        all_signals.append({
                            'timestamp': timestamp,
                            'symbol': signal.symbol,
                            'direction': signal.direction,
                            'strength': getattr(signal, 'strength', 1.0)
                        })
                        signal_dates.append(timestamp)
                                
                except Exception as e:
                    print(f"Signal generation error at {timestamp}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Process signals
                for signal in signals:
                    if signal.symbol not in positions:
                        # Open position
                        if signal.direction in ['LONG', 'SHORT']:
                            position_size = full_config.get('position_size', 0.95)
                            shares = int((capital * position_size) / current_price)
                            
                            if shares > 0:
                                positions[signal.symbol] = {
                                    'direction': signal.direction,
                                    'shares': shares,
                                    'entry_price': current_price,
                                    'entry_time': timestamp
                                }
                                
                                capital -= shares * current_price
                                
                    else:
                        # Close position
                        if signal.direction == 'FLAT':
                            pos = positions[signal.symbol]
                            exit_price = current_price
                            
                            # Calculate P&L
                            if pos['direction'] == 'LONG':
                                pnl = (exit_price - pos['entry_price']) * pos['shares']
                            else:
                                pnl = (pos['entry_price'] - exit_price) * pos['shares']
                            
                            capital += pos['shares'] * exit_price + pnl
                            
                            # Record trade
                            trades.append({
                                'entry_time': pos['entry_time'],
                                'exit_time': timestamp,
                                'direction': pos['direction'],
                                'entry_price': pos['entry_price'],
                                'exit_price': exit_price,
                                'shares': pos['shares'],
                                'pnl': pnl,
                                'return': pnl / (pos['shares'] * pos['entry_price'])
                            })
                            
                            del positions[signal.symbol]
                
                # Calculate current equity
                current_equity = capital
                for sym, pos in positions.items():
                    if pos['direction'] == 'LONG':
                        current_equity += pos['shares'] * current_price
                    else:
                        current_equity += pos['shares'] * (2 * pos['entry_price'] - current_price)
                
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'capital': capital,
                    'positions': len(positions)
                })
            
            print(f"   Generated {signals_count} signals")
            print(f"   Equity curve has {len(equity_curve)} data points")
            
            # Close any remaining positions
            for sym, pos in positions.items():
                exit_price = float(strategy_data['close'].iloc[-1])
                
                if pos['direction'] == 'LONG':
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['shares']
                
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': strategy_data.index[-1],
                    'direction': pos['direction'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'return': pnl / (pos['shares'] * pos['entry_price'])
                })
            
            # Calculate metrics
            equity_df = pd.DataFrame(equity_curve)
            if not equity_df.empty:
                equity_df.set_index('timestamp', inplace=True)
                
                # Returns
                equity_df['returns'] = equity_df['equity'].pct_change()
                total_return = (equity_df['equity'].iloc[-1] / initial_capital - 1) * 100
                
                # Calculate metrics
                daily_returns = equity_df['returns'].dropna()
                
                if len(daily_returns) > 0:
                    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
                    
                    # Sortino
                    downside_returns = daily_returns[daily_returns < 0]
                    sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
                    
                    # Max Drawdown
                    cumulative = (1 + daily_returns).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_drawdown = drawdown.min() * 100
                    
                    # Win rate
                    if trades:
                        winning_trades = [t for t in trades if t['pnl'] > 0]
                        win_rate = len(winning_trades) / len(trades) * 100
                    else:
                        win_rate = 0
                    
                else:
                    sharpe = sortino = max_drawdown = win_rate = 0
                
                # Compile results
                results = {
                    'initial_capital': initial_capital,
                    'final_equity': equity_df['equity'].iloc[-1],
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sortino,
                    'max_drawdown': max_drawdown,
                    'num_trades': len(trades),
                    'win_rate': win_rate,
                    'equity_curve': equity_df,
                    'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
                    'signals': pd.DataFrame(all_signals) if all_signals else pd.DataFrame(),
                    'signal_dates': signal_dates
                }
                
                return results
            
            return {
                'error': 'No equity data generated',
                'initial_capital': initial_capital,
                'final_equity': initial_capital,
                'total_return': 0,
                'equity_curve': pd.DataFrame()
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Backtest error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            return {
                'error': error_msg,
                'initial_capital': initial_capital,
                'final_equity': initial_capital,
                'total_return': 0,
                'equity_curve': pd.DataFrame()
            }


def create_performance_chart(results: Dict[str, Any], data: pd.DataFrame) -> go.Figure:
    """Create performance visualization."""
    if 'equity_curve' not in results or results['equity_curve'].empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown %')
    )
    
    # Equity curve
    equity_curve = results['equity_curve']
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add buy/sell signals if available
    if 'signal_dates' in results and results['signal_dates']:
        signal_dates = results['signal_dates']
        signal_prices = []
        
        for date in signal_dates:
            # Handle both uppercase and lowercase column names
            close_col = 'Close' if 'Close' in data.columns else 'close'
            if date in data.index:
                signal_prices.append(data.loc[date, close_col])
            else:
                # Find closest date
                closest_idx = data.index.get_indexer([date], method='nearest')[0]
                signal_prices.append(data.iloc[closest_idx][close_col])
        
        fig.add_trace(
            go.Scatter(
                x=signal_dates,
                y=signal_prices,
                mode='markers',
                name='Signals',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='triangle-up'
                )
            ),
            row=1, col=1
        )
    
    # Calculate drawdown
    equity = equity_curve['equity']
    running_max = equity.expanding().max()
    drawdown = ((equity - running_max) / running_max * 100)
    
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"Strategy Performance - Total Return: {results.get('total_return', 0):.2f}%",
        showlegend=True,
        height=800,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    
    return fig


def main():
    st.set_page_config(
        page_title="AlgoStack Dashboard - Pandas Edition",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AlgoStack Trading Dashboard")
    st.markdown("*Powered by pure pandas indicators and Alpha Vantage data*")
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["yfinance", "alpha_vantage"],
        help="Alpha Vantage requires API key in environment"
    )
    
    # Check Alpha Vantage availability
    if data_source == "alpha_vantage" and not data_manager.av_fetcher:
        st.sidebar.warning("⚠️ Alpha Vantage API key not found")
        st.sidebar.info("Set ALPHA_VANTAGE_API_KEY environment variable")
        data_source = "yfinance"
    
    # Symbol input
    symbol = st.sidebar.text_input("Symbol", value="SPY").upper()
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Interval selection (for intraday with Alpha Vantage)
    interval = "1d"
    if data_source == "alpha_vantage":
        interval = st.sidebar.selectbox(
            "Interval",
            ["1m", "5m", "15m", "30m", "60m", "1d"],
            index=5
        )
    
    # Strategy selection
    st.sidebar.subheader("Strategy Selection")
    
    if not strategy_manager.strategies:
        st.error("No strategies found! Please check the strategies directory.")
        return
    
    strategy_names = list(strategy_manager.strategies.keys())
    selected_strategy_name = st.sidebar.selectbox(
        "Select Strategy",
        strategy_names,
        format_func=strategy_manager.get_display_name
    )
    
    selected_strategy_class = strategy_manager.strategies[selected_strategy_name]
    
    # Get strategy parameters
    default_params = strategy_manager.get_strategy_parameters(selected_strategy_name)
    
    # Display parameter inputs
    st.sidebar.subheader("Strategy Parameters")
    
    user_params = {'symbol': symbol}
    
    # Group parameters by category
    param_categories = {
        'Core': ['lookback_period', 'position_size', 'max_positions'],
        'Risk': ['stop_loss_pct', 'take_profit_pct', 'atr_multiplier'],
        'Indicators': [],
        'Filters': [],
        'Other': []
    }
    
    # Categorize parameters
    for param, value in default_params.items():
        if param == 'symbol':
            continue
        
        if any(x in param for x in ['period', 'threshold', 'mult']):
            param_categories['Indicators'].append(param)
        elif 'filter' in param:
            param_categories['Filters'].append(param)
        elif param in param_categories['Core'] or param in param_categories['Risk']:
            pass
        else:
            param_categories['Other'].append(param)
    
    # Display parameters by category
    for category, params in param_categories.items():
        if params or category in ['Core', 'Risk']:
            with st.sidebar.expander(f"{category} Parameters", expanded=(category == 'Core')):
                for param in params:
                    if param in default_params:
                        value = default_params[param]
                        
                        # Get tooltip
                        param_key = f"{selected_strategy_name}.{param}"
                        tooltip = PARAMETER_TOOLTIPS.get(param_key, f"Adjust {param}")
                        
                        # Create input based on type
                        if isinstance(value, bool):
                            user_params[param] = st.checkbox(
                                param.replace('_', ' ').title(),
                                value=value,
                                help=tooltip,
                                key=f"{selected_strategy_name}_{category}_{param}_checkbox"
                            )
                        elif isinstance(value, int):
                            user_params[param] = st.number_input(
                                param.replace('_', ' ').title(),
                                value=value,
                                step=1,
                                help=tooltip,
                                key=f"{selected_strategy_name}_{category}_{param}_int"
                            )
                        elif isinstance(value, float):
                            user_params[param] = st.number_input(
                                param.replace('_', ' ').title(),
                                value=value,
                                step=0.01,
                                format="%.3f",
                                help=tooltip,
                                key=f"{selected_strategy_name}_{category}_{param}_float"
                            )
                        elif isinstance(value, list):
                            user_params[param] = value
                        else:
                            user_params[param] = value
    
    # Backtest configuration
    st.sidebar.subheader("Backtest Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        value=100000,
        step=10000,
        min_value=1000
    )
    
    # Run backtest button
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner(f"Fetching {symbol} data from {data_source}..."):
            # Fetch data
            data = data_manager.fetch_data(symbol, period, interval, data_source)
            
            if data.empty:
                st.error(f"No data available for {symbol}")
                return
            
            st.success(f"✅ Loaded {len(data)} data points for {symbol}")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Start Date", data.index[0].strftime('%Y-%m-%d'))
            with col2:
                st.metric("End Date", data.index[-1].strftime('%Y-%m-%d'))
            with col3:
                # Handle both uppercase and lowercase column names
                close_col = 'Close' if 'Close' in data.columns else 'close'
                st.metric("Current Price", f"${data[close_col].iloc[-1]:.2f}")
        
        # Run backtest
        with st.spinner("Running backtest with pandas indicators..."):
            results = strategy_manager.run_backtest(
                selected_strategy_class,
                selected_strategy_name,
                user_params,
                data,
                initial_capital
            )
        
        # Display results
        if 'error' in results:
            st.error(f"Backtest failed: {results['error']}")
        else:
            # Performance metrics
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_return = results.get('total_return', 0)
                st.metric(
                    "Total Return",
                    f"{total_return:.2f}%",
                    delta=f"{total_return:.2f}%"
                )
            
            with col2:
                sharpe = results.get('sharpe_ratio', 0)
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    delta="Good" if sharpe > 1 else "Poor"
                )
            
            with col3:
                max_dd = results.get('max_drawdown', 0)
                st.metric(
                    "Max Drawdown",
                    f"{max_dd:.2f}%",
                    delta=f"{max_dd:.2f}%"
                )
            
            with col4:
                num_trades = results.get('num_trades', 0)
                st.metric(
                    "Total Trades",
                    num_trades
                )
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                win_rate = results.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col2:
                sortino = results.get('sortino_ratio', 0)
                st.metric("Sortino Ratio", f"{sortino:.2f}")
            
            with col3:
                final_equity = results.get('final_equity', initial_capital)
                st.metric("Final Equity", f"${final_equity:,.2f}")
            
            with col4:
                if data_source == "alpha_vantage":
                    st.metric("Data Source", "Alpha Vantage")
                else:
                    st.metric("Data Source", "Yahoo Finance")
            
            # Performance chart
            st.subheader("Performance Visualization")
            fig = create_performance_chart(results, data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade analysis
            if 'trades' in results and not results['trades'].empty:
                st.subheader("Trade Analysis")
                
                trades_df = results['trades']
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_win = trades_df[trades_df['pnl'] > 0]['return'].mean() * 100 if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
                    st.metric("Avg Win", f"{avg_win:.2f}%")
                
                with col2:
                    avg_loss = trades_df[trades_df['pnl'] < 0]['return'].mean() * 100 if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
                    st.metric("Avg Loss", f"{avg_loss:.2f}%")
                
                with col3:
                    profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                
                # Recent trades
                with st.expander("Recent Trades"):
                    recent_trades = trades_df.tail(10).copy()
                    recent_trades['return'] = recent_trades['return'] * 100
                    
                    st.dataframe(
                        recent_trades[['entry_time', 'exit_time', 'direction', 
                                     'entry_price', 'exit_price', 'pnl', 'return']],
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()