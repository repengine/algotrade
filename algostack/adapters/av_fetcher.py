"""Alpha Vantage data fetcher adapter."""

import logging
import os
from datetime import datetime
from typing import Optional
import time

import pandas as pd
import requests


logger = logging.getLogger(__name__)


class AlphaVantageFetcher:
    """Fetches market data from Alpha Vantage API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")
            
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Free tier: 5 calls/min
        
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage."""
        try:
            if interval == "1d":
                df = self._fetch_daily(symbol)
            elif interval in ["1m", "5m", "15m", "30m", "60m"]:
                df = self._fetch_intraday(symbol, interval)
            else:
                raise ValueError(f"Unsupported interval: {interval}")
                
            # Filter by date range
            if not df.empty:
                df = df[(df.index >= start) & (df.index <= end)]
                df.attrs['symbol'] = symbol
                
            logger.info(f"Fetched {len(df)} bars for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
            return pd.DataFrame()
            
    def _fetch_daily(self, symbol: str) -> pd.DataFrame:
        """Fetch daily data."""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(data['Error Message'])
        if 'Note' in data:  # Rate limit
            logger.warning(f"Rate limit hit: {data['Note']}")
            time.sleep(self.rate_limit_delay)
            return self._fetch_daily(symbol)
            
        # Parse data
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        
        return df
        
    def _fetch_intraday(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch intraday data."""
        # Convert interval format
        av_interval = interval.replace('m', 'min')
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': av_interval,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(data['Error Message'])
        if 'Note' in data:  # Rate limit
            logger.warning(f"Rate limit hit: {data['Note']}")
            time.sleep(self.rate_limit_delay)
            return self._fetch_intraday(symbol, interval)
            
        # Parse data
        time_series_key = f'Time Series ({av_interval})'
        time_series = data.get(time_series_key, {})
        if not time_series:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        
        return df
        
    def fetch_fundamental_data(self, symbol: str) -> dict:
        """Fetch fundamental data for a symbol."""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        return response.json()