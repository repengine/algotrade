"""Yahoo Finance data fetcher adapter."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Fetches market data from Yahoo Finance."""
    
    def __init__(self):
        self.name = "yfinance"
        
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            # Convert interval format
            yf_interval = self._convert_interval(interval)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Add symbol as attribute
            df.attrs['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} bars for {symbol} from Yahoo Finance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()
            
    def fetch_info(self, symbol: str) -> dict:
        """Fetch stock info/metadata."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
            
    def fetch_multiple(
        self,
        symbols: list,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> dict:
        """Fetch data for multiple symbols."""
        data = {}
        
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, start, end, interval)
            if not df.empty:
                data[symbol] = df
                
        return data
        
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to yfinance format."""
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo"
        }
        return interval_map.get(interval, interval)