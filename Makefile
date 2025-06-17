# Makefile for AlgoStack development

.PHONY: help install install-dev test test-unit test-integration test-coverage lint format type-check security clean run-paper run-backtest

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: install ## Install development dependencies
	pip install -r requirements-dev.txt

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/ -v -m integration

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-specific: ## Run specific test file (use TEST=path/to/test.py)
	pytest $(TEST) -v

lint: ## Run all linters
	@echo "Running ruff..."
	ruff check .
	@echo "Running flake8..."
	flake8 .
	@echo "Running pylint..."
	pylint algostack/

format: ## Format code with black and isort
	black .
	isort .

format-check: ## Check code formatting without changes
	black --check .
	isort --check-only .

type-check: ## Run mypy type checking
	mypy algostack/

security: ## Run security checks
	bandit -r algostack/
	safety check

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.coverage' -delete
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

run-paper: ## Run paper trading mode
	python main.py run --mode paper

run-backtest: ## Run backtest for mean reversion strategy
	python main.py backtest --strategy mean_reversion --start 2020-01-01 --end 2023-12-31

# Development shortcuts
dev-test: format lint type-check test ## Run all development checks

ci: ## Run CI pipeline checks
	make format-check
	make lint
	make type-check
	make test-coverage

# Docker commands
docker-build: ## Build Docker image
	docker build -t algostack:latest .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD)/config:/app/config algostack:latest

docker-test: ## Run tests in Docker
	docker run --rm algostack:latest pytest tests/ -v

# Database commands
db-migrate: ## Run database migrations
	python scripts/migrate_db.py

# Documentation
docs: ## Build documentation
	cd docs && sphinx-build -b html . _build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Performance profiling
profile-memory: ## Profile memory usage
	python -m memory_profiler main.py run --mode paper

profile-time: ## Profile execution time
	python -m cProfile -o profile.stats main.py backtest --strategy mean_reversion --start 2023-01-01 --end 2023-12-31
	python -m pstats profile.stats

# Git hooks
install-hooks: ## Install git pre-commit hooks
	pre-commit install

run-hooks: ## Run pre-commit hooks on all files
	pre-commit run --all-files