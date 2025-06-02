"""Data management with caching and multi-source support."""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class DataHandler:
    """Unified data handler for multiple sources with parquet caching."""
    
    def __init__(self, providers: List[str], cache_dir: str = "data/cache", api_keys: Optional[Dict[str, str]] = None) -> None:
        self.providers = providers
        self.cache_dir = Path(cache_dir)
        self.adapters = {}
        self._cache = {}
        self.api_keys = api_keys or {}
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize data providers."""
        for provider in self.providers:
            if provider == "yfinance":
                from ..adapters.yf_fetcher import YFinanceFetcher
                self.adapters[provider] = YFinanceFetcher()
            elif provider == "alpha_vantage":
                from ..adapters.av_fetcher import AlphaVantageFetcher
                api_key = self.api_keys.get("alpha_vantage")
                if not api_key:
                    logger.warning("No Alpha Vantage API key provided")
                    continue
                self.adapters[provider] = AlphaVantageFetcher(api_key)
                
    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch historical data with caching."""
        cache_key = f"{symbol}_{interval}_{start.date()}_{end.date()}"
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        
        # Check cache
        if cache_path.exists():
            logger.debug(f"Loading {symbol} from cache")
            df = pd.read_parquet(cache_path)
            
            # Validate cache freshness (re-fetch if data is incomplete)
            if end.date() > df.index[-1].date():
                logger.info(f"Cache for {symbol} is stale, fetching recent data")
                # Fetch only the missing data
                new_start = df.index[-1] + timedelta(days=1)
                new_df = self._fetch_from_provider(symbol, new_start, end, interval, provider)
                if not new_df.empty:
                    df = pd.concat([df, new_df])
                    df.to_parquet(cache_path)
            return df
            
        # Fetch from provider
        df = self._fetch_from_provider(symbol, start, end, interval, provider)
        
        # Save to cache
        if not df.empty:
            df.to_parquet(cache_path)
        
        return df
    
    def _fetch_from_provider(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """Internal method to fetch from provider."""
        provider = provider or self.providers[0]
        adapter = self.adapters.get(provider)
        if not adapter:
            raise ValueError(f"Unknown provider: {provider}")
            
        logger.info(f"Fetching {symbol} from {provider}")
        return adapter.fetch_ohlcv(symbol, start, end, interval)
        
    async def get_latest(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
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
                        "timestamp": df.index[-1]
                    }
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                
        return latest_data
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to dataframe."""
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
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
            pattern = f"{symbol}_*.parquet"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Cleared cache for {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info("Cleared all cache")
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            total_size += cache_file.stat().st_size
        return total_size