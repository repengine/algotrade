# AlgoStack Test Suite

## Directory Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── adapters/      # Tests for data adapters (IBKR, YFinance, etc.)
│   ├── api/           # Tests for API endpoints and models
│   ├── core/          # Tests for core components
│   │   └── engine/    # Tests for trading engine components
│   ├── strategies/    # Tests for trading strategies
│   └── utils/         # Tests for utility functions
├── integration/       # Integration tests for component interactions
├── e2e/              # End-to-end tests for complete workflows
├── benchmarks/       # Performance benchmark tests
├── fixtures/         # Shared test fixtures and data
└── conftest.py       # Pytest configuration and shared fixtures
```

## Running Tests

### Run all tests
```bash
poetry run pytest
```

### Run specific test categories
```bash
# Unit tests only
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# E2E tests
poetry run pytest tests/e2e/

# Performance benchmarks
poetry run pytest tests/benchmarks/
```

### Run with coverage
```bash
poetry run pytest --cov=. --cov-report=html
```

### Run specific test file
```bash
poetry run pytest tests/unit/core/test_portfolio.py
```

### Run with markers
```bash
# Fast tests only
poetry run pytest -m "not slow"

# Specific component
poetry run pytest -m "portfolio"
```

## Test Guidelines

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **E2E Tests**: Test complete workflows from start to finish
4. **Benchmarks**: Test performance and resource usage

## Writing Tests

Follow the FIRST principles:
- **F**ast: Tests should run quickly
- **I**ndependent: Tests should not depend on each other
- **R**epeatable: Tests should give same results every time
- **S**elf-validating: Tests should have clear pass/fail result
- **T**horough: Tests should cover edge cases
