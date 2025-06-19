"""Base interface for data fetchers - PILLAR 3: OPERATIONAL STABILITY."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class DataFetcher(ABC):
    """Abstract base class for all data fetchers."""
    
    @abstractmethod
    async def fetch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        pass
    
    @abstractmethod
    async def fetch_realtime(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict]:
        """
        Fetch real-time quotes for symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        pass
    
    @abstractmethod
    def validate_data(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Validate fetched data for completeness and sanity.
        
        Args:
            data: DataFrame to validate
            symbol: Symbol for logging purposes
            
        Returns:
            Validated DataFrame (may have fewer rows)
        """
        pass