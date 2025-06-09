#!/usr/bin/env python3
"""Strategy parameter validation utilities."""

from typing import Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StrategyParameterValidator:
    """Validates strategy configuration parameters."""
    
    @staticmethod
    def validate_config(config: dict[str, Any], required_params: dict[str, type], 
                       optional_params: Optional[dict[str, tuple]] = None) -> dict[str, Any]:
        """
        Validate strategy configuration parameters.
        
        Args:
            config: Strategy configuration dictionary
            required_params: Dict of required parameter names and types
            optional_params: Dict of optional params with (type, default_value) tuples
            
        Returns:
            Validated configuration with defaults applied
            
        Raises:
            ValueError: If validation fails
        """
        validated_config = config.copy()
        errors = []
        
        # Check required parameters
        for param, expected_type in required_params.items():
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
                continue
                
            value = config[param]
            if not isinstance(value, expected_type):
                errors.append(f"Parameter '{param}' must be of type {expected_type.__name__}, got {type(value).__name__}")
        
        # Apply optional parameters with defaults
        if optional_params:
            for param, (expected_type, default_value) in optional_params.items():
                if param in config:
                    value = config[param]
                    if not isinstance(value, expected_type):
                        errors.append(f"Parameter '{param}' must be of type {expected_type.__name__}, got {type(value).__name__}")
                else:
                    validated_config[param] = default_value
                    
        if errors:
            raise ValueError("Strategy configuration validation failed:\n" + "\n".join(errors))
            
        return validated_config
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None,
                              max_val: Optional[Union[int, float]] = None, param_name: str = "value") -> None:
        """Validate numeric value is within range."""
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
    
    @staticmethod
    def validate_positive(value: Union[int, float], param_name: str = "value") -> None:
        """Validate value is positive."""
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")
    
    @staticmethod
    def validate_percentage(value: float, param_name: str = "value") -> None:
        """Validate value is a valid percentage (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError(f"{param_name} must be between 0 and 1, got {value}")
    
    @staticmethod
    def validate_time_string(time_str: str, param_name: str = "time") -> None:
        """Validate time string format (HH:MM:SS)."""
        import re
        pattern = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$'
        if not re.match(pattern, time_str):
            raise ValueError(f"{param_name} must be in HH:MM:SS format, got {time_str}")
    
    @staticmethod
    def validate_symbol_list(symbols: list[str], param_name: str = "symbols") -> None:
        """Validate list of trading symbols."""
        if not symbols:
            raise ValueError(f"{param_name} cannot be empty")
        if not all(isinstance(s, str) and s.strip() for s in symbols):
            raise ValueError(f"All items in {param_name} must be non-empty strings")


def validate_mean_reversion_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate mean reversion strategy configuration."""
    validator = StrategyParameterValidator()
    
    # Define required and optional parameters
    required_params = {
        'lookback_period': int,
        'zscore_threshold': float,
        'exit_zscore': float,
        'rsi_period': int,
        'rsi_oversold': float,
        'rsi_overbought': float,
    }
    
    optional_params = {
        'atr_period': (int, 14),
        'volume_filter': (bool, True),
        'max_positions': (int, 5),
        'stop_loss_atr': (float, 2.0),
    }
    
    # Validate basic structure
    validated = validator.validate_config(config, required_params, optional_params)
    
    # Validate specific ranges
    validator.validate_positive(validated['lookback_period'], 'lookback_period')
    validator.validate_positive(validated['zscore_threshold'], 'zscore_threshold')
    validator.validate_numeric_range(validated['exit_zscore'], 0, validated['zscore_threshold'], 'exit_zscore')
    validator.validate_positive(validated['rsi_period'], 'rsi_period')
    validator.validate_numeric_range(validated['rsi_oversold'], 0, 100, 'rsi_oversold')
    validator.validate_numeric_range(validated['rsi_overbought'], 0, 100, 'rsi_overbought')
    
    if validated['rsi_oversold'] >= validated['rsi_overbought']:
        raise ValueError("rsi_oversold must be less than rsi_overbought")
    
    return validated


def validate_trend_following_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate trend following strategy configuration."""
    validator = StrategyParameterValidator()
    
    required_params = {
        'symbols': list,
        'channel_period': int,
        'atr_period': int,
        'adx_period': int,
        'adx_threshold': float,
    }
    
    optional_params = {
        'volume_filter': (bool, True),
        'volume_threshold': (float, 1.2),
        'max_positions': (int, 10),
        'stop_multiplier': (float, 1.5),
        'trail_stop': (bool, True),
    }
    
    validated = validator.validate_config(config, required_params, optional_params)
    
    # Validate specific constraints
    validator.validate_symbol_list(validated['symbols'], 'symbols')
    validator.validate_positive(validated['channel_period'], 'channel_period')
    validator.validate_positive(validated['atr_period'], 'atr_period')
    validator.validate_positive(validated['adx_period'], 'adx_period')
    validator.validate_numeric_range(validated['adx_threshold'], 0, 100, 'adx_threshold')
    
    return validated


def validate_pairs_trading_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate pairs trading strategy configuration."""
    validator = StrategyParameterValidator()
    
    required_params = {
        'zscore_entry': float,
        'zscore_exit': float,
        'lookback_window': int,
        'correlation_threshold': float,
        'max_pairs': int,
    }
    
    optional_params = {
        'coint_pvalue': (float, 0.05),
        'half_life_threshold': (int, 30),
        'min_spread_std': (float, 0.01),
        'rebalance_frequency': (int, 5),
    }
    
    validated = validator.validate_config(config, required_params, optional_params)
    
    # Validate ranges
    validator.validate_positive(validated['zscore_entry'], 'zscore_entry')
    validator.validate_numeric_range(validated['zscore_exit'], 0, validated['zscore_entry'], 'zscore_exit')
    validator.validate_positive(validated['lookback_window'], 'lookback_window')
    validator.validate_percentage(validated['correlation_threshold'], 'correlation_threshold')
    validator.validate_positive(validated['max_pairs'], 'max_pairs')
    
    return validated


def validate_intraday_orb_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate intraday ORB strategy configuration."""
    validator = StrategyParameterValidator()
    
    required_params = {
        'symbols': list,
        'opening_range_minutes': int,
        'trade_start_time': str,
        'trade_end_time': str,
        'exit_time': str,
    }
    
    optional_params = {
        'breakout_threshold': (float, 0.002),
        'volume_confirmation': (bool, True),
        'volume_multiplier': (float, 1.5),
        'stop_loss_percent': (float, 0.01),
        'max_trades_per_day': (int, 2),
        'atr_period': (int, 14),
    }
    
    validated = validator.validate_config(config, required_params, optional_params)
    
    # Validate specific constraints
    validator.validate_symbol_list(validated['symbols'], 'symbols')
    validator.validate_positive(validated['opening_range_minutes'], 'opening_range_minutes')
    validator.validate_time_string(validated['trade_start_time'], 'trade_start_time')
    validator.validate_time_string(validated['trade_end_time'], 'trade_end_time')
    validator.validate_time_string(validated['exit_time'], 'exit_time')
    
    return validated


def validate_overnight_drift_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate overnight drift strategy configuration."""
    validator = StrategyParameterValidator()
    
    required_params = {
        'symbols': list,
        'lookback_days': int,
        'min_edge': float,
        'min_win_rate': float,
        'entry_time': str,
    }
    
    optional_params = {
        'momentum_period': (int, 10),
        'volatility_filter': (bool, True),
        'max_volatility': (float, 0.02),
        'volume_filter': (bool, True),
        'min_volume_ratio': (float, 0.8),
        'atr_period': (int, 14),
        'rsi_filter': (bool, False),
        'rsi_threshold': (float, 50),
    }
    
    validated = validator.validate_config(config, required_params, optional_params)
    
    # Validate ranges
    validator.validate_symbol_list(validated['symbols'], 'symbols')
    validator.validate_positive(validated['lookback_days'], 'lookback_days')
    validator.validate_percentage(validated['min_edge'], 'min_edge')
    validator.validate_percentage(validated['min_win_rate'], 'min_win_rate')
    validator.validate_time_string(validated['entry_time'], 'entry_time')
    
    return validated


def validate_hybrid_regime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate hybrid regime strategy configuration."""
    validator = StrategyParameterValidator()
    
    required_params = {
        'regime_window': int,
        'regime_threshold': float,
    }
    
    optional_params = {
        'trend_weight': (float, 0.5),
        'reversion_weight': (float, 0.5),
        'volatility_bands': (list, [25, 75]),
        'trend_strength_threshold': (float, 0.02),
        'bb_period': (int, 20),
        'bb_std': (float, 2.0),
    }
    
    validated = validator.validate_config(config, required_params, optional_params)
    
    # Validate ranges
    validator.validate_positive(validated['regime_window'], 'regime_window')
    validator.validate_percentage(validated['regime_threshold'], 'regime_threshold')
    validator.validate_percentage(validated['trend_weight'], 'trend_weight')
    validator.validate_percentage(validated['reversion_weight'], 'reversion_weight')
    
    if abs(validated['trend_weight'] + validated['reversion_weight'] - 1.0) > 0.001:
        raise ValueError("trend_weight + reversion_weight must equal 1.0")
    
    return validated