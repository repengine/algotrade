#!/usr/bin/env python3
"""Monitor paper trading performance."""

import json
from datetime import datetime

# Load config
with open("dashboard_config.json") as f:
    config = json.load(f)

print(f"Monitoring strategies: {list(config.keys())}")
print(f"Started at: {datetime.now()}")

# TODO: Connect to live trading results
# TODO: Compare with backtest expectations
# TODO: Alert if performance deviates significantly
