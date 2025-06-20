"""Data caching layer - PILLAR 3: OPERATIONAL STABILITY & PILLAR 2: PROFIT GENERATION."""

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache(ABC):
    """Abstract base class for data caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str):
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self):
        """Clear all cache entries."""
        pass

    def make_key(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        prefix: str = "ohlcv"
    ) -> str:
        """Generate cache key for market data."""
        # Create a unique key based on parameters
        key_parts = [
            prefix,
            symbol,
            interval,
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d")
        ]
        key_str = ":".join(key_parts)

        # For very long keys, use hash
        if len(key_str) > 250:
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"{prefix}:{symbol}:{key_hash}"

        return key_str


class InMemoryCache(DataCache):
    """
    Simple in-memory cache implementation.

    PILLAR 3: OPERATIONAL STABILITY - Reduces data source load
    PILLAR 2: PROFIT GENERATION - Faster data access = faster decisions
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of cache entries
        """
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]

            if datetime.now() < expiry:
                self.hits += 1
                logger.debug(f"Cache hit for {key} (hit rate: {self.hit_rate:.1%})")
                return value
            else:
                # Expired, remove it
                del self.cache[key]
                logger.debug(f"Cache expired for {key}")

        self.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL."""
        # Enforce size limit with simple LRU
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"Cache evicted {oldest_key} (size limit)")

        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)
        logger.debug(f"Cache set for {key} (TTL: {ttl}s)")

    async def delete(self, key: str):
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache deleted {key}")

    async def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "entries": list(self.cache.keys())[:10]  # First 10 keys
        }


class CachedDataFetcher:
    """
    Wrapper that adds caching to any DataFetcher.

    PILLAR 3: OPERATIONAL STABILITY - Caching reduces API load
    """

    def __init__(self, fetcher, cache: DataCache, ttl: int = 300):
        """
        Initialize cached fetcher.

        Args:
            fetcher: The underlying DataFetcher
            cache: Cache implementation
            ttl: Default TTL in seconds
        """
        self.fetcher = fetcher
        self.cache = cache
        self.ttl = ttl

    async def fetch(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data with caching."""
        result = {}
        symbols_to_fetch = []

        # Check cache for each symbol
        for symbol in symbols:
            cache_key = self.cache.make_key(
                symbol, interval, start_date, end_date
            )

            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                # Deserialize DataFrame
                try:
                    df = pd.read_json(cached_data)
                    result[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to deserialize cached data for {symbol}: {e}")
                    symbols_to_fetch.append(symbol)
            else:
                symbols_to_fetch.append(symbol)

        # Fetch missing data
        if symbols_to_fetch:
            fresh_data = await self.fetcher.fetch(
                symbols_to_fetch, start_date, end_date, interval
            )

            # Cache the fresh data
            for symbol, df in fresh_data.items():
                cache_key = self.cache.make_key(
                    symbol, interval, start_date, end_date
                )

                try:
                    # Serialize DataFrame
                    serialized = df.to_json()
                    await self.cache.set(cache_key, serialized, self.ttl)
                except Exception as e:
                    logger.error(f"Failed to cache data for {symbol}: {e}")

                result[symbol] = df

        return result

    async def fetch_realtime(self, symbols: list[str]) -> Dict[str, Dict]:
        """
        Fetch real-time quotes (not cached).

        Real-time data should always be fresh.
        """
        return await self.fetcher.fetch_realtime(symbols)

    def validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Pass through to underlying fetcher."""
        return self.fetcher.validate_data(data, symbol)
