"""Data management with caching and multi-source support."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataHandler:
    """Unified data handler for multiple sources with parquet caching."""

    def __init__(
        self,
        providers: list[str],
        cache_dir: str = "data/cache",
        api_keys: Optional[dict[str, str]] = None,
        premium_av: bool = False,
    ) -> None:
        self.providers = providers
        self.cache_dir = Path(cache_dir)
        self.adapters: dict[str, Any] = {}
        self._cache: dict[str, pd.DataFrame] = {}
        self.api_keys = api_keys or {}
        self.premium_av = premium_av

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize adapters immediately (not async)
        self._init_adapters()
        
        # Cache configuration
        self._cache_enabled = False
        self._memory_cache = None

    def _init_adapters(self) -> None:
        """Initialize data adapters (synchronous)."""
        for provider in self.providers:
            if provider == "yfinance":
                from adapters.yf_fetcher import YFinanceFetcher

                self.adapters[provider] = YFinanceFetcher()
            elif provider == "alpha_vantage" or provider == "alphavantage":
                from adapters.av_fetcher import AlphaVantageFetcher

                api_key = self.api_keys.get("alpha_vantage") or self.api_keys.get(
                    "alphavantage"
                )
                if not api_key:
                    # Try to load from environment or config
                    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
                    if not api_key:
                        # Try to load from secrets.yaml
                        try:
                            import yaml

                            config_path = (
                                Path(__file__).parent.parent / "config" / "secrets.yaml"
                            )
                            if config_path.exists():
                                with open(config_path) as f:
                                    secrets = yaml.safe_load(f)
                                    api_key = (
                                        secrets.get("data_providers", {})
                                        .get("alphavantage", {})
                                        .get("api_key")
                                    )
                        except Exception as e:
                            logger.debug(f"Could not load API key from secrets.yaml: {e}")
                            pass

                if not api_key:
                    logger.warning("No Alpha Vantage API key provided")
                    continue

                self.adapters[provider] = AlphaVantageFetcher(
                    api_key, premium=self.premium_av
                )

    async def initialize(self) -> None:
        """Async initialization if needed."""
        pass  # Adapters are already initialized in __init__

    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical data with caching."""
        cache_key = f"{symbol}_{interval}_{start.date()}_{end.date()}"
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        # Check cache
        pickle_path = Path(str(cache_path).replace(".parquet", ".pkl"))
        if cache_path.exists() or pickle_path.exists():
            if cache_path.exists():
                logger.debug(f"Loading {symbol} from parquet cache")
                try:
                    df = pd.read_parquet(cache_path)
                except (ImportError, AttributeError):
                    logger.debug("Parquet failed, trying pickle cache")
                    df = pd.read_pickle(pickle_path) if pickle_path.exists() else None
            else:
                logger.debug(f"Loading {symbol} from pickle cache")
                df = pd.read_pickle(pickle_path)

            if df is not None:
                # Validate cache freshness (re-fetch if data is incomplete)
                if end.date() > df.index[-1].date():
                    logger.info(f"Cache for {symbol} is stale, fetching recent data")
                    # Fetch only the missing data
                    new_start = df.index[-1] + timedelta(days=1)
                    new_df = self._fetch_from_provider(
                        symbol, new_start, end, interval, provider
                    )
                    if not new_df.empty:
                        df = pd.concat([df, new_df])
                        try:
                            df.to_parquet(cache_path)
                        except (ImportError, AttributeError):
                            df.to_pickle(pickle_path)
                return df

        # Fetch from provider
        df = self._fetch_from_provider(symbol, start, end, interval, provider)

        # Save to cache
        if not df.empty:
            try:
                df.to_parquet(cache_path)
            except (ImportError, AttributeError):
                # Fallback to pickle if parquet is not available
                df.to_pickle(str(cache_path).replace(".parquet", ".pkl"))

        return df

    def _fetch_from_provider(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """Internal method to fetch from provider."""
        provider = provider or self.providers[0]
        adapter = self.adapters.get(provider)
        if not adapter:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"Fetching {symbol} from {provider}")
        return adapter.fetch_ohlcv(symbol, start, end, interval)

    async def get_latest(self, symbols: Optional[list[str]] = None) -> dict[str, dict]:
        """Get latest market data for symbols."""
        latest_data = {}

        for symbol in symbols or []:
            try:
                # Get last business day's data
                end = datetime.now()
                start = end - timedelta(days=10)  # Buffer for weekends/holidays

                df = self.get_historical(symbol, start, end, "1d")
                if not df.empty:
                    latest = df.iloc[-1]
                    latest_data[symbol] = {
                        "open": latest["open"],
                        "high": latest["high"],
                        "low": latest["low"],
                        "close": latest["close"],
                        "volume": latest["volume"],
                        "timestamp": df.index[-1],
                    }
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")

        return latest_data

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to dataframe."""
        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift())

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        return df

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for specific symbol or all symbols."""
        if symbol:
            for pattern in [f"{symbol}_*.parquet", f"{symbol}_*.pkl"]:
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                    logger.info(f"Cleared cache for {symbol}")
        else:
            for pattern in ["*.parquet", "*.pkl"]:
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
            logger.info("Cleared all cache")

    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            total_size += cache_file.stat().st_size
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
        return total_size
    
    def add_fetcher(self, fetcher: Any, name: str = None) -> None:
        """Add a new data fetcher - PILLAR 3: OPERATIONAL STABILITY."""
        if name is None:
            name = fetcher.__class__.__name__
        self.adapters[name] = fetcher
        logger.info(f"Added data fetcher: {name}")
    
    async def fetch_with_failover(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """Fetch data with automatic failover - PILLAR 3: OPERATIONAL STABILITY."""
        from data.sources.composite_fetcher import CompositeFetcher
        
        # Create composite fetcher with all available adapters
        fetchers = list(self.adapters.values())
        if not fetchers:
            raise ValueError("No data fetchers available")
        
        composite = CompositeFetcher(fetchers)
        return await composite.fetch(symbols, start_date, end_date, interval)
    
    def enable_cache(self, cache_type: str = "memory", ttl: int = 300) -> None:
        """Enable caching for data fetches - PILLAR 2: PROFIT GENERATION."""
        from data.cache import InMemoryCache
        
        self._cache_enabled = True
        if cache_type == "memory":
            self._memory_cache = InMemoryCache(max_size=1000)
        logger.info(f"Enabled {cache_type} cache with TTL={ttl}s")
    
    async def fetch_with_cache(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """Fetch data with caching - PILLAR 2: PROFIT GENERATION."""
        if not self._cache_enabled or not self._memory_cache:
            return await self.fetch_with_failover(symbols, start_date, end_date, interval)
        
        from data.cache.data_cache import CachedDataFetcher
        from data.sources.composite_fetcher import CompositeFetcher
        
        # Create composite fetcher
        fetchers = list(self.adapters.values())
        composite = CompositeFetcher(fetchers)
        
        # Wrap with cache
        cached_fetcher = CachedDataFetcher(composite, self._memory_cache)
        return await cached_fetcher.fetch(symbols, start_date, end_date, interval)
    
    def validate_data(self, data: pd.DataFrame, symbol: str = None) -> bool:
        """
        Validate data quality - PILLAR 1: CAPITAL PRESERVATION.
        
        Returns True if data is valid, False otherwise.
        Compatible with both old test API and new implementation.
        """
        if symbol is None:
            # Legacy API compatibility - extract symbol from data if possible
            symbol = data.attrs.get('symbol', 'UNKNOWN')
        
        # Basic validation
        if data.empty:
            logger.warning(f"Empty data for {symbol}")
            return False
        
        # Check for required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns for {symbol}: {missing_cols}")
            return False
        
        # Check for invalid values
        price_cols = ["open", "high", "low", "close"]
        
        # Check for NaN values
        if data[required_cols].isna().any().any():
            logger.error(f"NaN values found in data for {symbol}")
            return False
        
        # Check for negative prices
        for col in price_cols:
            if (data[col] <= 0).any():
                logger.error(f"Invalid negative/zero prices in {col} for {symbol}")
                return False
        
        # Check high >= low
        if (data["high"] < data["low"]).any():
            logger.error(f"Invalid high < low for {symbol}")
            return False
        
        # Check volume
        if (data["volume"] < 0).any():
            logger.error(f"Negative volumes for {symbol}")
            return False
        
        return True
