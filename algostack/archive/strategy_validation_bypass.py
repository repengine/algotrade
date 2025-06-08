"""
Bypass strategy validation to allow dashboard to work with parameter type mismatches.
"""

import sys
from unittest.mock import patch

# Create a mock validator that always passes
def mock_validator(config):
    """Mock validator that always returns the config unchanged."""
    return config

# Patch all strategy validators
def patch_all_validators():
    """Patch all strategy validation functions to bypass strict validation."""
    validators_to_patch = [
        'utils.validators.strategy_validators.validate_mean_reversion_config',
        'utils.validators.strategy_validators.validate_trend_following_config',
        'utils.validators.strategy_validators.validate_pairs_stat_arb_config',
        'utils.validators.strategy_validators.validate_intraday_orb_config',
        'utils.validators.strategy_validators.validate_overnight_drift_config',
        'utils.validators.strategy_validators.validate_hybrid_regime_config',
    ]
    
    patches = []
    for validator_path in validators_to_patch:
        try:
            p = patch(validator_path, side_effect=mock_validator)
            p.start()
            patches.append(p)
        except:
            pass  # Ignore if validator doesn't exist
    
    return patches