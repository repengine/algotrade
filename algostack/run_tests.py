#!/usr/bin/env python3
"""Test runner script for AlgoStack."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests with coverage."""
    print("🧪 Running AlgoStack Test Suite")
    print("=" * 60)
    
    # Check if in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider activating a virtual environment first")
        print()
    
    # Test commands to run
    test_commands = [
        {
            'name': 'Code Formatting Check',
            'cmd': ['python', '-m', 'black', '--check', '.'],
            'optional': True
        },
        {
            'name': 'Import Sorting Check', 
            'cmd': ['python', '-m', 'isort', '--check-only', '.'],
            'optional': True
        },
        {
            'name': 'Linting with Ruff',
            'cmd': ['python', '-m', 'ruff', 'check', '.'],
            'optional': True
        },
        {
            'name': 'Type Checking with MyPy',
            'cmd': ['python', '-m', 'mypy', 'algostack/'],
            'optional': True
        },
        {
            'name': 'Security Check with Bandit',
            'cmd': ['python', '-m', 'bandit', '-r', 'algostack/'],
            'optional': True
        },
        {
            'name': 'Unit Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/', '-v', '-m', 'unit'],
            'optional': False
        },
        {
            'name': 'Integration Tests',
            'cmd': ['python', '-m', 'pytest', 'tests/', '-v', '-m', 'integration'],
            'optional': False
        },
        {
            'name': 'All Tests with Coverage',
            'cmd': ['python', '-m', 'pytest', 'tests/', '-v', '--cov=.', '--cov-report=term-missing'],
            'optional': False
        }
    ]
    
    results = []
    
    for test in test_commands:
        print(f"\n🔍 Running: {test['name']}")
        print("-" * 40)
        
        try:
            result = subprocess.run(test['cmd'], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test['name']} passed")
                results.append((test['name'], True))
            else:
                print(f"❌ {test['name']} failed")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                results.append((test['name'], False))
                
                if not test['optional']:
                    print("\n⛔ Critical test failed. Stopping test run.")
                    break
                    
        except FileNotFoundError:
            if test['optional']:
                print(f"⚠️  {test['name']} skipped (tool not installed)")
                results.append((test['name'], None))
            else:
                print(f"❌ {test['name']} failed (tool not found)")
                results.append((test['name'], False))
                break
        except Exception as e:
            print(f"❌ {test['name']} failed with error: {e}")
            results.append((test['name'], False))
            if not test['optional']:
                break
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{test_name:<30} {status}")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


def setup_test_environment():
    """Set up test environment."""
    print("🔧 Setting up test environment...")
    
    # Create necessary directories
    dirs_to_create = [
        Path("data/cache"),
        Path("logs"),
        Path("backtest_results"),
        Path("htmlcov")
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {dir_path}")
    
    print("✅ Test environment ready")


if __name__ == "__main__":
    # Set up environment
    setup_test_environment()
    
    # Run tests
    exit_code = run_tests()
    
    # Exit with appropriate code
    sys.exit(exit_code)