"""
Default configurations for all AlgoStack strategies (Fixed).

This module provides complete default parameters for each strategy type,
ensuring all required parameters are included.
"""

from datetime import time


STRATEGY_DEFAULTS = {
    "MeanReversionEquity": {
        # Required parameters
        "lookback_period": 20,
        "zscore_threshold": 2.5,
        "exit_zscore": 0.5,
        "rsi_period": 14,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        # Optional parameters
        "atr_period": 14,
        "atr_band_mult": 2.5,
        "volume_filter": True,
        "max_positions": 5,
        "stop_loss_atr": 3.0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "position_size": 0.1,
        "ma_exit_period": 10  # Added missing parameter
    },
    
    "TrendFollowingMulti": {
        # Required parameters
        "symbols": ["SPY"],  # Will be overridden by dashboard
        "channel_period": 20,
        "atr_period": 14,
        "adx_period": 14,
        "adx_threshold": 25.0,  # Must be float
        # Optional parameters
        "volume_filter": True,
        "volume_threshold": 1.2,
        "max_positions": 10,
        "stop_multiplier": 1.5,
        "trail_stop": True,
        "position_size": 0.1,
        "risk_per_trade": 0.02,
        "lookback_period": 252  # Added for consistency
    },
    
    "PairsStatArb": {
        # Required parameters
        "zscore_entry": 2.0,
        "zscore_exit": 0.5,
        "lookback_window": 60,
        "correlation_threshold": 0.7,
        "max_pairs": 5,
        # Optional parameters
        "coint_pvalue": 0.05,
        "half_life_threshold": 30,
        "min_spread_std": 0.01,
        "rebalance_frequency": 5,
        "position_size": 0.1,
        "max_positions": 10,
        "lookback_period": 60  # Added for consistency
    },
    
    "IntradayOrb": {
        # Required parameters
        "symbols": ["SPY"],  # Will be overridden by dashboard
        "opening_range_minutes": 30,
        "trade_start_time": "09:30:00",  # Must be string
        "trade_end_time": "15:30:00",   # Must be string
        "exit_time": "15:45:00",         # Must be string
        # Optional parameters
        "breakout_threshold": 0.002,
        "volume_confirmation": True,
        "volume_multiplier": 1.5,
        "stop_loss_percent": 0.01,
        "max_trades_per_day": 2,
        "atr_period": 14,
        "position_size": 0.1,
        "lookback_period": 50  # Added for consistency
    },
    
    "OvernightDrift": {
        # Required parameters
        "symbols": ["SPY"],  # Will be overridden by dashboard
        "lookback_days": 60,
        "min_edge": 0.001,
        "min_win_rate": 0.52,
        "entry_time": "15:45:00",  # Must be string
        # Optional parameters
        "momentum_period": 10,
        "volatility_filter": True,
        "max_volatility": 0.02,
        "volume_filter": True,
        "min_volume_ratio": 0.8,
        "atr_period": 14,
        "rsi_filter": False,
        "rsi_threshold": 50.0,
        "position_size": 0.1,
        "stop_loss_pct": 0.02,
        "lookback_period": 60  # Added for consistency
    },
    
    "HybridRegime": {
        # Required parameters - Fixed to include all needed params
        "regime_window": 20,
        "regime_threshold": 0.6,
        "zscore_threshold": 2.5,  # Added missing parameter
        "exit_zscore": 0.5,       # Added missing parameter
        # Optional parameters
        "trend_weight": 0.5,
        "reversion_weight": 0.5,  # Must sum to 1.0 with trend_weight
        "volatility_bands": [25, 75],
        "trend_strength_threshold": 0.02,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30.0,     # Fixed to be float
        "rsi_overbought": 70.0,   # Fixed to be float
        "adx_period": 14,
        "adx_threshold": 25.0,
        "position_size": 0.1,
        "lookback_period": 252,   # Changed from 20 to 252 for consistency
        "atr_period": 14,         # Added missing parameter
        "atr_band_mult": 2.5,     # Added missing parameter
        "ma_exit_period": 10,     # Added missing parameter
        "stop_loss_atr": 3.0,     # Added missing parameter
        "max_positions": 5,       # Added missing parameter
        "channel_period": 20,     # Added for trend following component
        "symbols": ["SPY"]        # Added missing parameter
    }
}


# Parameter tooltips for better user understanding
PARAMETER_TOOLTIPS = {
    # Basic parameters
    "lookback_period": "Number of historical periods to analyze for generating signals",
    "position_size": "Fraction of capital to allocate per position (0.1 = 10%)",
    "max_positions": "Maximum number of concurrent positions allowed",
    
    # Mean reversion parameters
    "zscore_threshold": "Z-score level to trigger entry signals (higher = more extreme)",
    "exit_zscore": "Z-score level to exit positions (closer to 0 = earlier exit)",
    "rsi_period": "Period for RSI calculation (lower = more sensitive)",
    "rsi_oversold": "RSI level considered oversold (buy signal threshold)",
    "rsi_overbought": "RSI level considered overbought (sell signal threshold)",
    "ma_exit_period": "Moving average period for exit signals",
    
    # Trend following parameters
    "channel_period": "Period for calculating price channels (Donchian)",
    "adx_period": "Period for ADX (trend strength) calculation",
    "adx_threshold": "Minimum ADX value to confirm trend (higher = stronger trend required)",
    "stop_multiplier": "ATR multiplier for stop loss placement",
    "trail_stop": "Enable trailing stop loss",
    
    # Risk parameters
    "atr_period": "Period for Average True Range calculation",
    "atr_band_mult": "ATR multiplier for band calculation",
    "stop_loss_atr": "Stop loss distance in ATR units",
    "stop_loss_pct": "Stop loss as percentage of entry price",
    "take_profit_pct": "Take profit as percentage of entry price",
    "risk_per_trade": "Maximum risk per trade as fraction of capital",
    
    # Volume parameters
    "volume_filter": "Enable volume confirmation for signals",
    "volume_threshold": "Minimum volume ratio vs average (1.2 = 20% above average)",
    "volume_multiplier": "Required volume increase for signal confirmation",
    
    # Pairs trading parameters
    "correlation_threshold": "Minimum correlation required between pairs",
    "zscore_entry": "Z-score threshold for entering pairs trade",
    "zscore_exit": "Z-score threshold for exiting pairs trade",
    "lookback_window": "Period for calculating pair statistics",
    "max_pairs": "Maximum number of pairs to trade simultaneously",
    
    # Intraday parameters
    "opening_range_minutes": "Minutes after open to define opening range",
    "breakout_threshold": "Minimum move required to confirm breakout",
    "max_trades_per_day": "Maximum trades allowed per day",
    
    # Overnight parameters
    "lookback_days": "Days to analyze for overnight edge calculation",
    "min_edge": "Minimum expected edge to take overnight position",
    "min_win_rate": "Minimum historical win rate required",
    "momentum_period": "Period for momentum calculation",
    
    # Regime parameters
    "regime_window": "Period for regime detection",
    "regime_threshold": "Threshold for regime classification",
    "trend_weight": "Weight given to trend following signals",
    "reversion_weight": "Weight given to mean reversion signals"
}


def get_strategy_defaults(strategy_class_name: str) -> dict:
    """
    Get default configuration for a strategy class.
    
    Args:
        strategy_class_name: Name of the strategy class
        
    Returns:
        Dictionary of default parameters
    """
    # Try exact match first
    if strategy_class_name in STRATEGY_DEFAULTS:
        return STRATEGY_DEFAULTS[strategy_class_name].copy()
    
    # Try without spaces
    no_spaces = strategy_class_name.replace(" ", "")
    if no_spaces in STRATEGY_DEFAULTS:
        return STRATEGY_DEFAULTS[no_spaces].copy()
    
    # Try to find partial match
    for key in STRATEGY_DEFAULTS:
        if key.lower() in strategy_class_name.lower() or strategy_class_name.lower() in key.lower():
            return STRATEGY_DEFAULTS[key].copy()
    
    # Return generic defaults if not found
    return {
        "lookback_period": 20,
        "position_size": 0.1,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "symbols": ["SPY"]
    }


def merge_with_defaults(strategy_class_name: str, user_params: dict) -> dict:
    """
    Merge user parameters with strategy defaults.
    
    Args:
        strategy_class_name: Name of the strategy class
        user_params: User-provided parameters
        
    Returns:
        Merged configuration with all required parameters
    """
    defaults = get_strategy_defaults(strategy_class_name)
    
    # Create merged config
    merged = defaults.copy()
    
    # Override with user params
    for key, value in user_params.items():
        # Special handling for symbols - always use user's symbol
        if key == "symbol" and "symbols" in merged:
            merged["symbols"] = [value]
        else:
            merged[key] = value
    
    # Ensure symbols list includes user's symbol if provided
    if "symbol" in user_params and "symbols" in merged:
        if user_params["symbol"] not in merged["symbols"]:
            merged["symbols"] = [user_params["symbol"]]
    
    # Ensure correct types for common parameters
    type_conversions = {
        'rsi_oversold': float,
        'rsi_overbought': float,
        'adx_threshold': float,
        'zscore_threshold': float,
        'exit_zscore': float,
        'atr_band_mult': float,
        'stop_loss_atr': float,
        'stop_loss_pct': float,
        'take_profit_pct': float,
        'position_size': float,
        'lookback_period': int,
        'rsi_period': int,
        'atr_period': int,
        'adx_period': int,
        'max_positions': int
    }
    
    for param, param_type in type_conversions.items():
        if param in merged:
            try:
                merged[param] = param_type(merged[param])
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                pass
    
    return merged