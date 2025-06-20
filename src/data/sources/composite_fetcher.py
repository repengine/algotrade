"""Composite data fetcher with failover - PILLAR 3: OPERATIONAL STABILITY."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .base import DataFetcher
from .health_monitor import SourceHealthMonitor

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when all data sources fail."""
    pass


class CompositeFetcher(DataFetcher):
    """
    Fetcher that tries multiple sources with failover.

    PILLAR 3: OPERATIONAL STABILITY - Ensures data availability
    """

    def __init__(self, sources: List[DataFetcher]):
        """
        Initialize with list of data sources in priority order.

        Args:
            sources: List of DataFetcher instances
        """
        if not sources:
            raise ValueError("At least one data source required")

        self.sources = sources
        self.health_monitor = SourceHealthMonitor()
        self.source_names = {source: source.__class__.__name__ for source in sources}

    async def fetch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data with automatic failover.

        Tries each source in priority order until success.
        """
        errors = []

        # Get healthy sources in priority order
        healthy_sources = self._get_healthy_sources()

        if not healthy_sources:
            logger.error("No healthy data sources available")
            raise DataFetchError("All data sources are unhealthy")

        for source in healthy_sources:
            source_name = self.source_names[source]

            try:
                logger.info(f"Attempting to fetch data from {source_name}")
                start_time = time.time()

                # Try to fetch data
                data = await source.fetch(symbols, start_date, end_date, interval)

                # Validate the data
                validated_data = {}
                for symbol, df in data.items():
                    validated_df = source.validate_data(df, symbol)
                    if not validated_df.empty:
                        validated_data[symbol] = validated_df

                if validated_data:
                    # Success!
                    response_time = time.time() - start_time
                    self.health_monitor.record_success(source_name, response_time)
                    logger.info(
                        f"Successfully fetched data from {source_name} "
                        f"({len(validated_data)} symbols in {response_time:.2f}s)"
                    )
                    return validated_data
                else:
                    # No valid data returned
                    raise ValueError("No valid data returned after validation")

            except Exception as e:
                self.health_monitor.record_failure(source_name, e)
                errors.append(f"{source_name}: {str(e)}")
                logger.warning(f"Failed to fetch from {source_name}: {e}")
                continue

        # All sources failed
        error_msg = f"All data sources failed: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataFetchError(error_msg)

    async def fetch_realtime(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict]:
        """
        Fetch real-time quotes with failover.

        Real-time data is more critical, so we try harder.
        """
        errors = []
        healthy_sources = self._get_healthy_sources()

        if not healthy_sources:
            logger.error("No healthy data sources available for real-time quotes")
            raise DataFetchError("All data sources are unhealthy")

        for source in healthy_sources:
            source_name = self.source_names[source]

            try:
                logger.debug(f"Fetching real-time quotes from {source_name}")
                start_time = time.time()

                quotes = await source.fetch_realtime(symbols)

                # Validate quotes
                valid_quotes = {}
                for symbol, quote in quotes.items():
                    if quote.get("price", 0) > 0 and "error" not in quote:
                        valid_quotes[symbol] = quote

                if valid_quotes:
                    response_time = time.time() - start_time
                    self.health_monitor.record_success(source_name, response_time)
                    logger.debug(
                        f"Got {len(valid_quotes)} quotes from {source_name} "
                        f"in {response_time:.2f}s"
                    )
                    return valid_quotes

            except Exception as e:
                self.health_monitor.record_failure(source_name, e)
                errors.append(f"{source_name}: {str(e)}")
                continue

        # All sources failed
        error_msg = f"All real-time sources failed: {'; '.join(errors)}"
        logger.error(error_msg)
        raise DataFetchError(error_msg)

    def validate_data(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Validate data using the first source's validation.

        This is called internally during fetch operations.
        """
        if self.sources:
            return self.sources[0].validate_data(data, symbol)
        return data

    def _get_healthy_sources(self) -> List[DataFetcher]:
        """Get healthy sources in priority order."""
        source_names = list(self.source_names.keys())
        healthy_names = self.health_monitor.get_healthy_sources(
            [self.source_names[s] for s in source_names]
        )

        # Map back to source objects
        name_to_source = {self.source_names[s]: s for s in source_names}
        return [name_to_source[name] for name in healthy_names if name in name_to_source]

    def get_health_status(self) -> Dict:
        """Get health status of all sources."""
        return self.health_monitor.get_status()

    def add_source(self, source: DataFetcher):
        """Add a new data source."""
        self.sources.append(source)
        self.source_names[source] = source.__class__.__name__
        logger.info(f"Added data source: {source.__class__.__name__}")

    def remove_source(self, source_class_name: str) -> bool:
        """Remove a data source by class name."""
        for source in self.sources:
            if source.__class__.__name__ == source_class_name:
                self.sources.remove(source)
                del self.source_names[source]
                logger.info(f"Removed data source: {source_class_name}")
                return True
        return False
