#!/usr/bin/env python3
"""Check all imports and dependencies."""

import sys
import importlib
import subprocess

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    # List of required modules
    modules_to_check = [
        'pandas',
        'numpy',
        'streamlit',
        'plotly',
        'yfinance',
        'requests',
        'yaml',
        'pydantic',
    ]
    
    print("Checking required modules...")
    print("="*60)
    
    missing = []
    errors = []
    
    for module in modules_to_check:
        success, error = check_module(module)
        if success:
            print(f"✅ {module}")
        else:
            print(f"❌ {module}: {error}")
            if "No module named" in error:
                missing.append(module)
            else:
                errors.append((module, error))
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"In virtual environment: {in_venv}")
    
    if missing:
        print(f"\n⚠️  Missing modules: {', '.join(missing)}")
        print("\nTo install missing modules:")
        print(f"pip install {' '.join(missing)}")
    
    if errors:
        print(f"\n⚠️  Modules with errors:")
        for module, error in errors:
            print(f"  {module}: {error}")
    
    # Check for pandas version
    try:
        import pandas as pd
        print(f"\nPandas version: {pd.__version__}")
    except:
        pass
    
    # Check for numpy version
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except:
        pass
    
    return len(missing) == 0 and len(errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)