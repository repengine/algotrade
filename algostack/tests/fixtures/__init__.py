"""
Test fixtures package for AlgoStack.

This package provides reusable test fixtures for:
- Market data generation
- Portfolio configurations and states
- Strategy mocks and configurations
- Risk management settings
- Execution mocks

All fixtures follow the FIRST principles:
- Fast: Minimal setup time
- Independent: No shared state
- Repeatable: Deterministic results
- Self-validating: Clear assertions
- Thorough: Cover edge cases
"""

# Import commonly used fixtures for convenience
from .market_data import (
    market_data_generator,
    spy_data,
    tech_stocks_data,
    earnings_event_data,
    intraday_data,
)

from .portfolios import (
    empty_portfolio,
    small_portfolio,
    diversified_portfolio,
    leveraged_portfolio,
    conservative_portfolio_config,
    aggressive_portfolio_config,
)

from .strategies import (
    always_buy_strategy,
    mean_reversion_strategy,
    trend_following_strategy,
    random_strategy,
    sample_signals,
)

__all__ = [
    # Market data
    'market_data_generator',
    'spy_data',
    'tech_stocks_data',
    'earnings_event_data',
    'intraday_data',
    
    # Portfolios
    'empty_portfolio',
    'small_portfolio',
    'diversified_portfolio',
    'leveraged_portfolio',
    'conservative_portfolio_config',
    'aggressive_portfolio_config',
    
    # Strategies
    'always_buy_strategy',
    'mean_reversion_strategy',
    'trend_following_strategy',
    'random_strategy',
    'sample_signals',
]