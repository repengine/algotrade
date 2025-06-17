"""
Safe logging utilities for tests to prevent output buffer overflow.

Prevents Claude Code crashes from excessive output by:
- Limiting string lengths
- Truncating large data structures
- Using proper logging instead of print statements
"""

import logging
from typing import Any


class SafeTestLogger:
    """Logger that prevents excessive output in tests."""

    MAX_STRING_LENGTH = 1000
    MAX_LINES = 10

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Only add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(handler)

    def _truncate(self, msg: Any) -> str:
        """Truncate message to safe length."""
        str_msg = str(msg)

        # Truncate by lines first
        lines = str_msg.split('\n')
        if len(lines) > self.MAX_LINES:
            str_msg = '\n'.join(lines[:self.MAX_LINES]) + f'\n... ({len(lines) - self.MAX_LINES} more lines)'

        # Then truncate by total length
        if len(str_msg) > self.MAX_STRING_LENGTH:
            str_msg = str_msg[:self.MAX_STRING_LENGTH] + f'... (truncated {len(str_msg) - self.MAX_STRING_LENGTH} chars)'

        return str_msg

    def info(self, msg: Any):
        """Log info message safely."""
        self.logger.info(self._truncate(msg))

    def debug(self, msg: Any):
        """Log debug message safely."""
        self.logger.debug(self._truncate(msg))

    def warning(self, msg: Any):
        """Log warning message safely."""
        self.logger.warning(self._truncate(msg))

    def error(self, msg: Any):
        """Log error message safely."""
        self.logger.error(self._truncate(msg))

    def metrics(self, label: str, metrics: dict):
        """Log metrics dictionary safely."""
        safe_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                safe_metrics[key] = round(value, 4) if isinstance(value, float) else value
            else:
                safe_metrics[key] = self._truncate(value)

        self.info(f"{label}: {safe_metrics}")

    def dataframe_summary(self, label: str, df):
        """Log dataframe summary safely."""
        if df is None:
            self.info(f"{label}: None")
            return

        try:
            summary = f"{label}: shape={df.shape}, columns={list(df.columns)[:5]}"
            if len(df.columns) > 5:
                summary += f"... ({len(df.columns) - 5} more)"
            self.info(summary)
        except Exception as e:
            self.error(f"{label}: Error summarizing dataframe - {e}")


def get_test_logger(name: str) -> SafeTestLogger:
    """Get a safe test logger instance."""
    return SafeTestLogger(name)


# Suppress verbose output in test mode
def suppress_test_output():
    """Configure logging to minimize output in tests."""
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
