
# Initialize error counts
error_summary = {
    "Type Errors": {
        "Missing return type annotations": 254,
        "Missing parameter type annotations": 0,
        "Use of Any type": 6,
        "Non-generic dict usage": 4,
        "Non-generic list usage": 8,
        "Functions without type annotations": 254,
    },
    "Code Style": {
        "Lines over 120 characters": 49,
        "Bare except clauses": 5,
        "TODO/FIXME comments": 26,
        "Missing docstrings": 0,  # Not counted in initial analysis
    },
    "Import Issues": {
        "Circular imports": 0,
        "Unused imports": 0,  # Would need AST analysis
        "Wildcard imports": 0,
    },
}

# Module breakdown
module_errors = {
    "tests": 115,
    "scripts": 93,
    "test_files": 26,
    "core": 20,
    "strategies": 16,
    "visualization_files": 14,
    "api": 11,
    "dashboard.py": 9,
    "cli": 6,
    "backtests": 6,
    "examples": 3,
    "production_configs": 3,
    "test_scripts": 3,
    "utils": 2,
    "adapters": 2,
}

# Files with most errors
top_error_files = [
    ("tests/test_risk_manager.py", 16),
    ("tests/test_pandas_indicators.py", 14),
    ("scripts/strategy_integration_helpers.py", 13),
    ("tests/test_mean_reversion_strategy.py", 12),
    ("scripts/dashboard_pandas.py", 12),
    ("api/app.py", 11),
    ("tests/test_integration.py", 10),
    ("tests/test_data_handler.py", 10),
    ("dashboard.py", 9),
    ("tests/test_alpha_vantage_integration.py", 9),
]

# Calculate totals
total_errors = sum(sum(category.values()) for category in error_summary.values())
total_files_with_errors = sum(1 for count in module_errors.values() if count > 0)

print("=== ALGOSTACK LINTING & TYPE ERROR ANALYSIS ===\n")

print(f"Total Errors Found: {total_errors}")
print("Total Python Files: 164")
print(f"Files with Errors: {total_files_with_errors}\n")

print("=== ERROR BREAKDOWN BY TYPE ===")
for category, errors in error_summary.items():
    category_total = sum(errors.values())
    print(f"\n{category} ({category_total} total):")
    for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  - {error_type}: {count}")

print("\n=== ERROR COUNT BY MODULE ===")
for module, count in sorted(module_errors.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"{module}: {count}")

print("\n=== TOP 10 FILES WITH MOST ERRORS ===")
for filepath, count in top_error_files:
    print(f"{filepath}: {count}")

print("\n=== TYPE ANNOTATION COVERAGE ===")
print("Key modules type coverage:")
print("  - core/portfolio.py: 100.0%")
print("  - strategies/base.py: 90.9%")
print("  - core/live_engine.py: 90.0%")
print("  - core/backtest_engine.py: 74.1%")
print("  - api/app.py: 12.5%")
print("  - dashboard.py: 0.0%")
print("\nOverall type annotation coverage: 70.4%")

print("\n=== PRIORITY FIXES ===")
print("1. Add return type annotations (254 functions)")
print("2. Fix long lines (49 occurrences)")
print("3. Address TODO/FIXME comments (26 items)")
print("4. Replace bare except clauses (5 occurrences)")
print("5. Use generic types instead of dict/list (12 occurrences)")
