#!/usr/bin/env python3
"""
Enhanced AlgoStack Dashboard with Professional Backtesting

This version includes:
1. Transaction cost modeling
2. In-sample/Out-of-sample validation
3. Statistical significance testing
4. Walk-forward analysis
5. Parameter optimization
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced backtesting components
from core.backtest_engine import (
    TransactionCostModel, DataSplitter, WalkForwardAnalyzer,
    MonteCarloValidator, RegimeAnalyzer, create_backtest_report
)
from core.optimization import (
    BayesianOptimizer, CoarseToFineOptimizer, EnsembleOptimizer,
    define_param_space, create_optuna_objective
)

# Import existing dashboard components
from dashboard_pandas import (
    AlphaVantageDataManager, PandasStrategyManager,
    create_performance_chart
)

# Import pandas indicators
from pandas_indicators import create_talib_compatible_module
sys.modules['talib'] = create_talib_compatible_module()


def enhanced_backtest(strategy_manager, strategy_class, strategy_name, user_params, 
                     data, initial_capital, cost_config, split_config, validation_config):
    """Run enhanced backtest with all professional features."""
    
    results = {}
    
    # 1. Split data into IS/OOS
    splitter = DataSplitter(
        method=split_config['method'],
        oos_ratio=split_config['oos_ratio']
    )
    
    is_data, oos_data = splitter.split(data)
    st.info(f"Data split: IS {len(is_data)} bars, OOS {len(oos_data)} bars")
    
    # 2. Initialize transaction cost model
    cost_model = TransactionCostModel(cost_config) if cost_config['enabled'] else None
    
    # 3. Run IS backtest (for parameter selection)
    is_results = strategy_manager.run_backtest(
        strategy_class, strategy_name, user_params, is_data, initial_capital
    )
    
    # 4. Run OOS backtest (true performance)
    oos_results = strategy_manager.run_backtest(
        strategy_class, strategy_name, user_params, oos_data, initial_capital
    )
    
    # Add cost analysis if enabled
    if cost_model:
        oos_results = add_transaction_costs(oos_results, cost_model)
    
    # 5. Statistical validation
    if validation_config['monte_carlo']:
        validator = MonteCarloValidator(n_simulations=1000)
        # Need to pass returns series for validation
        if 'equity_curve' in oos_results and not oos_results['equity_curve'].empty:
            # Extract returns series, handling NaN values
            returns_series = oos_results['equity_curve']['returns'].dropna()
            if len(returns_series) > 0:
                oos_results['returns_series'] = returns_series
                validation_results = validator.validate_strategy(oos_results)
                results['validation'] = validation_results
            else:
                results['validation'] = {
                    'error': 'No valid returns for Monte Carlo validation',
                    'significant': False,
                    'p_value': 1.0,
                    'confidence_interval': (0, 0),
                    'effect_size': 0,
                    'interpretation': 'Insufficient data for statistical validation'
                }
        
    # 6. Regime analysis
    if validation_config['regime_analysis']:
        regime_analyzer = RegimeAnalyzer()
        regime_results = regime_analyzer.analyze_regime_performance(
            strategy_class(user_params), data,
            lambda s, d: strategy_manager.run_backtest(
                type(s), strategy_name, s.config, d, initial_capital
            )
        )
        results['regime_analysis'] = regime_results
    
    # 7. Walk-forward analysis
    if validation_config['walk_forward']:
        param_ranges = get_optimization_ranges(strategy_name)
        if param_ranges:  # Only run if we have parameters to optimize
            wf_analyzer = WalkForwardAnalyzer()
            
            # Create a proper backtest function that matches the expected signature
            def wf_backtest_func(strategy_instance, data_slice, cost_model=None):
                # Get the config from the strategy instance
                if hasattr(strategy_instance, 'config'):
                    config = strategy_instance.config
                else:
                    config = {}
                
                # Run backtest with the config
                return strategy_manager.run_backtest(
                    type(strategy_instance), 
                    strategy_name, 
                    config, 
                    data_slice, 
                    initial_capital
                )
            
            # Add progress message
            st.info(f"Running walk-forward analysis with {sum(1 for _ in [1 for v in param_ranges.values() for _ in range(len(v))])} parameter combinations...")
            
            # Run with timeout protection
            import concurrent.futures
            import threading
            
            wf_results = None
            error_msg = None
            
            def run_wf():
                nonlocal wf_results, error_msg
                try:
                    wf_results = wf_analyzer.run_analysis(
                        strategy_class, data, 
                        param_ranges,
                        wf_backtest_func,
                        cost_model
                    )
                except Exception as e:
                    error_msg = str(e)
            
            # Run in thread with timeout
            thread = threading.Thread(target=run_wf)
            thread.start()
            thread.join(timeout=300)  # 5 minute timeout
            
            if thread.is_alive():
                st.error("Walk-forward analysis timed out after 5 minutes. Try reducing parameter combinations.")
                # Create empty results
                wf_results = pd.DataFrame({
                    'window_num': [1],
                    'oos_sharpe': [0],
                    'is_sharpe': [0],
                    'sharpe_decay': [0],
                    'message': ['Walk-forward timed out']
                })
            elif error_msg:
                st.error(f"Walk-forward analysis failed: {error_msg}")
                wf_results = pd.DataFrame({
                    'window_num': [1],
                    'oos_sharpe': [0],
                    'is_sharpe': [0],
                    'sharpe_decay': [0],
                    'message': [f'Error: {error_msg}']
                })
            elif wf_results is None:
                wf_results = pd.DataFrame({
                    'window_num': [1],
                    'oos_sharpe': [0],
                    'is_sharpe': [0],
                    'sharpe_decay': [0],
                    'message': ['No results returned']
                })
            results['walk_forward'] = wf_results
        else:
            results['walk_forward'] = pd.DataFrame({
                'window_num': [1],
                'oos_sharpe': [oos_results.get('sharpe_ratio', 0)],
                'is_sharpe': [is_results.get('sharpe_ratio', 0)],
                'sharpe_decay': [calculate_performance_decay(is_results, oos_results)['sharpe_decay']],
                'message': ['No parameters to optimize for this strategy']
            })
    
    # Combine results
    results.update({
        'is_results': is_results,
        'oos_results': oos_results,
        'performance_decay': calculate_performance_decay(is_results, oos_results)
    })
    
    return results


def add_transaction_costs(results, cost_model):
    """Add transaction costs to backtest results."""
    
    if 'trades' not in results or results['trades'].empty:
        return results
    
    trades_df = results['trades']
    total_costs = 0
    
    # Calculate costs for each trade
    cost_details = []
    for idx, trade in trades_df.iterrows():
        # Estimate volatility (would need actual historical vol)
        volatility = 0.02  # Default 2% daily vol
        
        # Entry costs
        entry_costs = cost_model.calculate_costs(
            price=trade['entry_price'],
            shares=trade['shares'],
            side='BUY' if trade['direction'] == 'LONG' else 'SELL',
            volatility=volatility
        )
        
        # Exit costs
        exit_costs = cost_model.calculate_costs(
            price=trade['exit_price'],
            shares=trade['shares'],
            side='SELL' if trade['direction'] == 'LONG' else 'BUY',
            volatility=volatility
        )
        
        total_trade_cost = entry_costs.total + exit_costs.total
        total_costs += total_trade_cost
        
        cost_details.append({
            'trade_idx': idx,
            'entry_costs': entry_costs.total,
            'exit_costs': exit_costs.total,
            'total_costs': total_trade_cost
        })
    
    # Adjust results for costs
    results['transaction_costs'] = {
        'total_costs': total_costs,
        'cost_percentage': total_costs / (results['initial_capital'] * results['total_return'] / 100),
        'avg_cost_per_trade': total_costs / len(trades_df) if len(trades_df) > 0 else 0,
        'details': pd.DataFrame(cost_details)
    }
    
    # Adjust net returns
    gross_return = results['total_return']
    cost_impact = (total_costs / results['initial_capital']) * 100
    results['net_return'] = gross_return - cost_impact
    results['gross_return'] = gross_return
    
    # Adjust Sharpe ratio
    if results.get('sharpe_ratio', 0) > 0:
        results['net_sharpe'] = results['sharpe_ratio'] * (results['net_return'] / gross_return)
        results['gross_sharpe'] = results['sharpe_ratio']
    
    return results


def calculate_performance_decay(is_results, oos_results):
    """Calculate performance decay from IS to OOS."""
    
    is_sharpe = is_results.get('sharpe_ratio', 0)
    oos_sharpe = oos_results.get('sharpe_ratio', 0)
    
    if is_sharpe > 0:
        decay = (is_sharpe - oos_sharpe) / is_sharpe
    else:
        decay = 0
        
    return {
        'sharpe_decay': decay,
        'return_decay': (is_results.get('total_return', 0) - oos_results.get('total_return', 0)) / 
                       (is_results.get('total_return', 1) + 1e-6),
        'acceptable': decay < 0.3  # Less than 30% decay is acceptable
    }


def get_optimization_ranges(strategy_name):
    """Define parameter ranges for optimization."""
    
    # These would be customized per strategy
    # For walk-forward, use fewer values to avoid combinatorial explosion
    ranges = {
        'MeanReversionEquity': {
            'lookback_period': [20, 30, 40],  # 3 values
            'zscore_threshold': [2.0, 2.5, 3.0],  # 3 values
            'exit_zscore': [0.25, 0.5, 0.75],  # 3 values
            # Total: 3×3×3 = 27 combinations
        },
        'MeanReversionIntraday': {
            'lookback_period': [10, 15, 20],  # 50-100 minutes
            'zscore_threshold': [1.0, 1.5, 2.0],  # Entry thresholds
            'exit_zscore': [-0.5, 0.0, 0.5],  # Exit levels
            'rsi_period': [2, 3, 5],  # Ultra-short RSI
            'rsi_oversold': [20, 25, 30],  # Oversold levels
            'stop_loss_atr': [2.0, 2.5, 3.0],  # Stop loss distance
            # Total: 3×3×3×3×3×3 = 729 combinations - might need reduction
        },
        'TrendFollowingMulti': {
            'channel_period': [20, 30, 40],  # Reduced from 5 to 3
            'atr_period': [14, 20],  # Reduced from 3 to 2
            'stop_multiplier': [1.5, 2.0, 2.5],  # Reduced from 5 to 3
            # Total: 3×2×3 = 18 combinations (was 375)
        },
        'HybridRegime': {
            'regime_window': [10, 20, 30],  # 3 values
            'regime_threshold': [0.5, 0.6, 0.7],  # 3 values
            'trend_weight': [0.3, 0.5, 0.7],  # 3 values
            # Total: 3×3×3 = 27 combinations
        },
        'IntradayOrb': {
            'opening_range_minutes': [15, 30, 45],  # Reduced from 4 to 3
            'breakout_threshold': [0.002, 0.003],  # Reduced from 4 to 2
            'stop_loss_percent': [0.01, 0.015],  # Reduced from 4 to 2
            # Total: 3×2×2 = 12 combinations (was 64)
        },
        'OvernightDrift': {
            'lookback_days': [30, 60, 90],  # Reduced from 4 to 3
            'min_edge': [0.001, 0.0015],  # Reduced from 4 to 2
            'min_win_rate': [0.52, 0.54],  # Reduced from 4 to 2
            # Total: 3×2×2 = 12 combinations (was 64)
        },
        'PairsStatArb': {
            'zscore_entry': [2.0, 2.5],  # Reduced from 4 to 2
            'zscore_exit': [0.25, 0.5, 0.75],  # Reduced from 5 to 3
            'lookback_window': [60, 90],  # Reduced from 4 to 2
            # Total: 2×3×2 = 12 combinations (was 80)
        }
    }
    
    # Log the total combinations for debugging
    if strategy_name in ranges:
        total = 1
        for param, values in ranges[strategy_name].items():
            total *= len(values)
        print(f"Walk-forward optimization for {strategy_name}: {total} parameter combinations")
    
    return ranges.get(strategy_name, {})


def render_enhanced_results(results):
    """Render enhanced backtest results."""
    
    st.header("📊 Enhanced Backtest Results")
    
    # Main performance comparison
    col1, col2, col3, col4 = st.columns(4)
    
    is_results = results['is_results']
    oos_results = results['oos_results']
    decay = results['performance_decay']
    
    with col1:
        st.metric(
            "IS Sharpe",
            f"{is_results.get('sharpe_ratio', 0):.2f}",
            help="In-sample Sharpe ratio (training data)"
        )
        
    with col2:
        st.metric(
            "OOS Sharpe",
            f"{oos_results.get('sharpe_ratio', 0):.2f}",
            delta=f"{-decay['sharpe_decay']*100:.1f}%",
            help="Out-of-sample Sharpe ratio (test data)"
        )
        
    with col3:
        if 'net_sharpe' in oos_results:
            st.metric(
                "Net Sharpe (after costs)",
                f"{oos_results['net_sharpe']:.2f}",
                delta=f"{(oos_results['net_sharpe'] - oos_results['gross_sharpe'])/oos_results['gross_sharpe']*100:.1f}%"
            )
        else:
            st.metric("Max Drawdown", f"{oos_results.get('max_drawdown', 0):.1f}%")
            
    with col4:
        decay_color = "🟢" if decay['acceptable'] else "🔴"
        st.metric(
            "Performance Decay",
            f"{decay['sharpe_decay']*100:.1f}%",
            help=f"{decay_color} Decay should be < 30%"
        )
    
    # Statistical validation results
    if 'validation' in results:
        st.subheader("📈 Statistical Validation")
        val = results['validation']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sig_icon = "✅" if val['significant'] else "❌"
            st.metric(
                "Statistical Significance",
                f"{sig_icon} p={val['p_value']:.3f}",
                help="p < 0.05 indicates significant alpha"
            )
        with col2:
            st.metric(
                "Effect Size",
                f"{val['effect_size']:.2f}",
                help="Cohen's d: 0.2=small, 0.5=medium, 0.8=large"
            )
        with col3:
            ci_low, ci_high = val['confidence_interval']
            st.metric(
                "95% CI for Sharpe",
                f"[{ci_low:.2f}, {ci_high:.2f}]",
                help="Bootstrap confidence interval"
            )
            
        st.info(val['interpretation'])
    
    # Regime analysis results
    if 'regime_analysis' in results:
        st.subheader("🌤️ Regime Analysis")
        regime = results['regime_analysis']
        
        # Check if we have any regime results
        if regime.get('regime_results') and len(regime['regime_results']) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Regime Consistency",
                    f"{regime['consistency_score']*100:.1f}%",
                    help="How consistent is performance across regimes"
                )
            with col2:
                icon = "✅" if regime['all_positive'] else "⚠️"
                st.metric(
                    "Positive in All Regimes",
                    icon,
                    help="Whether strategy is profitable in all market conditions"
                )
                
            # Regime performance table
            regime_df = pd.DataFrame(regime['regime_results']).T
            
            # Check if required columns exist
            required_cols = ['sharpe', 'return', 'max_drawdown', 'num_trades']
            available_cols = [col for col in required_cols if col in regime_df.columns]
            
            if available_cols:
                st.dataframe(
                    regime_df[available_cols].round(2),
                    use_container_width=True
                )
            else:
                # Display whatever columns are available
                st.dataframe(regime_df.round(2), use_container_width=True)
        else:
            st.warning("Not enough data for regime analysis. Each regime needs at least 100 days of data.")
    
    # Walk-forward results
    if 'walk_forward' in results:
        st.subheader("🚶 Walk-Forward Analysis")
        wf_df = results['walk_forward']
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        # Check if wf_df has the required columns
        if 'oos_sharpe' in wf_df.columns:
            with col1:
                st.metric(
                    "Avg OOS Sharpe",
                    f"{wf_df['oos_sharpe'].mean():.2f}",
                    help="Average out-of-sample Sharpe across all windows"
                )
            with col2:
                st.metric(
                    "Consistency",
                    f"{(wf_df['oos_sharpe'] > 0).mean()*100:.0f}%",
                    help="Percentage of profitable OOS windows"
                )
            with col3:
                if 'sharpe_decay' in wf_df.columns:
                    st.metric(
                        "Avg Decay",
                        f"{wf_df['sharpe_decay'].mean()*100:.1f}%",
                        help="Average performance decay IS→OOS"
                    )
                else:
                    st.metric("Avg Decay", "N/A")
        else:
            # Show error message if walk-forward failed
            if 'message' in wf_df.columns:
                st.error(f"Walk-forward analysis issue: {wf_df['message'].iloc[0]}")
            else:
                st.error("Walk-forward analysis failed to produce valid results")
            
        # Walk-forward chart
        fig = go.Figure()
        
        # Check if required columns exist
        if 'is_sharpe' in wf_df.columns and 'oos_sharpe' in wf_df.columns:
            fig.add_trace(go.Scatter(
                x=wf_df.index,
                y=wf_df['is_sharpe'],
                name='IS Sharpe',
                line=dict(color='blue', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=wf_df.index,
                y=wf_df['oos_sharpe'],
                name='OOS Sharpe',
                line=dict(color='green')
            ))
        else:
            # Handle missing data
            st.warning("Walk-forward data incomplete. Check error messages above.")
        fig.update_layout(
            title="Walk-Forward Sharpe Ratios",
            xaxis_title="Window",
            yaxis_title="Sharpe Ratio",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction cost analysis
    if 'transaction_costs' in oos_results:
        st.subheader("💰 Transaction Cost Analysis")
        tc = oos_results['transaction_costs']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Costs",
                f"${tc['total_costs']:,.2f}",
                help="Total transaction costs incurred"
            )
        with col2:
            st.metric(
                "Cost Impact",
                f"{tc['cost_percentage']*100:.1f}%",
                help="Costs as % of gross returns"
            )
        with col3:
            st.metric(
                "Avg Cost/Trade",
                f"${tc['avg_cost_per_trade']:.2f}",
                help="Average cost per round-trip trade"
            )
    
    # Generate downloadable report
    if st.button("📄 Generate Full Report"):
        report = create_backtest_report(results)
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


def main():
    st.set_page_config(
        page_title="AlgoStack Enhanced Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AlgoStack Trading Dashboard - Professional Edition")
    st.markdown("*Enhanced with transaction costs, statistical validation, and walk-forward analysis*")
    
    # Initialize managers
    strategy_manager = PandasStrategyManager()
    data_manager = AlphaVantageDataManager()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Basic settings (same as original)
    symbol = st.sidebar.text_input("Symbol", value="SPY").upper()
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    data_source = st.sidebar.radio("Data Source", ["yfinance", "alpha_vantage"])
    
    # Strategy selection
    st.sidebar.subheader("Strategy Selection")
    strategy_names = list(strategy_manager.strategies.keys())
    selected_strategy_name = st.sidebar.selectbox(
        "Select Strategy",
        strategy_names,
        format_func=strategy_manager.get_display_name
    )
    selected_strategy_class = strategy_manager.strategies[selected_strategy_name]
    
    # Enhanced backtest settings
    st.sidebar.subheader("🔬 Enhanced Backtest Settings")
    
    # Data split configuration
    with st.sidebar.expander("Data Split Configuration", expanded=True):
        split_method = st.selectbox(
            "Split Method",
            ["sequential", "embargo", "purged_kfold"],
            help="Sequential: simple split. Embargo: gap between train/test. Purged: advanced CV"
        )
        oos_ratio = st.slider(
            "Out-of-Sample %",
            min_value=10,
            max_value=40,
            value=20,
            help="Reserve this % of data for out-of-sample testing"
        )
        
    # Transaction cost configuration
    with st.sidebar.expander("Transaction Costs", expanded=True):
        enable_costs = st.checkbox("Enable Transaction Costs", value=True)
        
        commission_per_share = st.number_input(
            "Commission per share ($)",
            min_value=0.0,
            max_value=0.1,
            value=0.005,
            format="%.3f"
        )
        
        spread_model = st.selectbox(
            "Spread Model",
            ["fixed", "dynamic", "vix_based"],
            help="How to model bid-ask spreads"
        )
        
        base_spread_bps = st.number_input(
            "Base Spread (bps)",
            min_value=1,
            max_value=20,
            value=5,
            help="Base bid-ask spread in basis points"
        )
        
        slippage_model = st.selectbox(
            "Slippage Model",
            ["linear", "square_root"],
            help="Market impact model"
        )
    
    # Validation configuration
    with st.sidebar.expander("Statistical Validation", expanded=False):
        run_monte_carlo = st.checkbox(
            "Monte Carlo Validation",
            value=True,
            help="Test if results are statistically significant"
        )
        
        run_regime_analysis = st.checkbox(
            "Regime Analysis",
            value=True,
            help="Test across different market conditions"
        )
        
        run_walk_forward = st.checkbox(
            "Walk-Forward Analysis",
            value=False,
            help="Rolling window optimization (slow - runs many backtests)"
        )
    
    # Parameter configuration - show all relevant parameters for each strategy
    st.sidebar.subheader("Strategy Parameters")
    default_params = strategy_manager.get_strategy_parameters(selected_strategy_name)
    user_params = {'symbol': symbol}
    
    # Define key parameters for each strategy type
    strategy_key_params = {
        'MeanReversionIntraday': [
            'lookback_period', 'zscore_threshold', 'exit_zscore', 
            'rsi_period', 'rsi_oversold', 'stop_loss_atr', 'position_size'
        ],
        'MeanReversionEquity': [
            'lookback_period', 'zscore_threshold', 'exit_zscore',
            'rsi_period', 'rsi_oversold', 'atr_band_mult', 'stop_loss_atr'
        ],
        'TrendFollowingMulti': [
            'channel_period', 'atr_period', 'adx_threshold',
            'stop_multiplier', 'position_size'
        ],
        'IntradayOrb': [
            'opening_range_minutes', 'breakout_threshold', 
            'stop_loss_percent', 'max_trades_per_day'
        ],
        'OvernightDrift': [
            'lookback_days', 'min_edge', 'min_win_rate',
            'momentum_period', 'position_size'
        ],
        'HybridRegime': [
            'regime_window', 'regime_threshold', 'trend_weight',
            'zscore_threshold', 'exit_zscore'
        ],
        'PairsStatArb': [
            'zscore_entry', 'zscore_exit', 'lookback_window',
            'correlation_threshold', 'max_pairs'
        ]
    }
    
    # Get parameters for current strategy or use defaults
    key_params = strategy_key_params.get(selected_strategy_name, 
                                         ['lookback_period', 'position_size', 'stop_loss_pct'])
    
    # Create input widgets for each parameter
    for param in key_params:
        if param in default_params:
            param_value = default_params[param]
            
            # Create appropriate input widget based on parameter type
            if isinstance(param_value, bool):
                user_params[param] = st.sidebar.checkbox(
                    param.replace('_', ' ').title(),
                    value=param_value
                )
            elif isinstance(param_value, int):
                # Set appropriate ranges for different parameters
                min_val = 1
                max_val = 1000
                step = 1
                
                if 'period' in param:
                    max_val = 200
                elif 'threshold' in param:
                    min_val = 0
                    max_val = 100
                elif param == 'max_positions' or param == 'max_pairs':
                    max_val = 20
                    
                user_params[param] = st.sidebar.number_input(
                    param.replace('_', ' ').title(),
                    value=param_value,
                    min_value=min_val,
                    max_value=max_val,
                    step=step
                )
            elif isinstance(param_value, float):
                # Set appropriate ranges for float parameters
                min_val = 0.0
                max_val = 10.0
                step = 0.1
                format_str = "%.3f"
                
                if 'zscore' in param:
                    min_val = -3.0
                    max_val = 3.0
                    step = 0.1
                elif 'position_size' in param:
                    min_val = 0.1
                    max_val = 1.0
                    step = 0.05
                    format_str = "%.2f"
                elif 'oversold' in param or 'overbought' in param:
                    min_val = 0.0
                    max_val = 100.0
                    step = 5.0
                    format_str = "%.1f"
                elif 'atr' in param or 'multiplier' in param:
                    min_val = 0.5
                    max_val = 5.0
                    step = 0.5
                    format_str = "%.1f"
                elif 'threshold' in param:
                    min_val = 0.0
                    max_val = 1.0
                    step = 0.01
                    format_str = "%.3f"
                    
                user_params[param] = st.sidebar.number_input(
                    param.replace('_', ' ').title(),
                    value=float(param_value),
                    min_value=min_val,
                    max_value=max_val,
                    step=step,
                    format=format_str
                )
    
    # Add interval selection for certain strategies
    if selected_strategy_name in ["MeanReversionIntraday", "IntradayOrb"]:
        st.sidebar.info("📊 This strategy uses intraday data")
        
    # Initial capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        value=100000,
        step=10000,
        min_value=1000
    )
    
    # Run enhanced backtest
    if st.sidebar.button("🚀 Run Enhanced Backtest", type="primary"):
        with st.spinner("Loading data..."):
            # Determine interval based on strategy type
            if selected_strategy_name == "MeanReversionIntraday":
                # Use 5-minute data for intraday strategy
                interval = "5m"
                # Adjust period for intraday - max 60 days for 5m data
                if period in ["1y", "2y", "5y", "max"]:
                    period = "60d"  # yfinance limit for 5m data
                    st.info("Using last 60 days of 5-minute data (yfinance limit)")
            else:
                interval = "1d"
                
            data = data_manager.fetch_data(symbol, period, interval, data_source)
            
            if data.empty:
                st.error(f"No data available for {symbol}")
                return
                
        with st.spinner("Running enhanced backtest..."):
            # Prepare configurations
            cost_config = {
                'enabled': enable_costs,
                'commission_per_share': commission_per_share,
                'commission_type': 'per_share',
                'spread_model': spread_model,
                'base_spread_bps': base_spread_bps,
                'slippage_model': slippage_model,
                'market_impact_factor': 0.1
            }
            
            split_config = {
                'method': split_method,
                'oos_ratio': oos_ratio / 100
            }
            
            validation_config = {
                'monte_carlo': run_monte_carlo,
                'regime_analysis': run_regime_analysis,
                'walk_forward': run_walk_forward
            }
            
            # Run enhanced backtest
            results = enhanced_backtest(
                strategy_manager,
                selected_strategy_class,
                selected_strategy_name,
                user_params,
                data,
                initial_capital,
                cost_config,
                split_config,
                validation_config
            )
            
            # Display results
            render_enhanced_results(results)
            
            # Show performance chart
            st.subheader("Performance Visualization")
            oos_results = results['oos_results']
            fig = create_performance_chart(oos_results, data, initial_capital)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()