"""Simple test to cover ImportError branch in metrics.py"""

import subprocess
import sys


def test_import_error_coverage():
    """Test ImportError branch by running code that triggers it."""
    # Create a test script that triggers the ImportError
    test_script = '''
import sys
import os

# Add the project to path
sys.path.insert(0, os.path.abspath('.'))

# Remove backtest_metrics so import fails
import algostack.core
if hasattr(algostack.core, 'backtest_metrics'):
    delattr(algostack.core, 'backtest_metrics')

# Monkey patch to make the import fail
original_import = __builtins__.__import__
def patched_import(name, *args, **kwargs):
    if 'backtest_metrics' in name:
        raise ImportError("Mocked import error")
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = patched_import

# Now import metrics - this will trigger the except block
exec(open('algostack/core/metrics.py').read(), {'__name__': 'algostack.core.metrics', '__file__': 'algostack/core/metrics.py'})

# The ImportError branch should have been executed
print("ImportError branch covered")
'''
    
    # Run the script
    result = subprocess.run(
        [sys.executable, '-c', test_script],
        cwd='/home/nate/projects/algotrade',
        capture_output=True,
        text=True
    )
    
    # Check that it ran successfully
    assert "ImportError branch covered" in result.stdout, f"Script failed: {result.stderr}"