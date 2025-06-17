#!/bin/bash
# Run tests with coverage from the project root

export PYTHONPATH=/home/nate/projects/algotrade:$PYTHONPATH
poetry run pytest --cov=. --cov-report=term --cov-report=html -v