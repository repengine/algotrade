"""
Test coverage for DataHandler methods missing from existing tests.
Focuses on cache fallback logic, API key loading errors, and get_latest method.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import tempfile
import shutil

import pandas as pd
import numpy as np
import pytest

from algostack.core.data_handler import DataHandler


class TestDataHandlerMissingCoverage:
    """Tests for methods with missing coverage in DataHandler."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_api_key_loading_errors(self, temp_cache_dir):
        """Test API key loading with various error scenarios."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage']
        }
        
        # Test with exception during secrets loading
        with patch('builtins.open', side_effect=Exception("Failed to open secrets")):
            # Should create handler without raising exception
            handler = DataHandler(config)
            assert 'alpha_vantage' not in handler.adapters
    
    def test_no_api_key_warning(self, temp_cache_dir):
        """Test warning when no API key is provided."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage']
        }
        
        # Test with no API key in environment or config
        with patch('os.getenv', return_value=None):
            with patch('pathlib.Path.exists', return_value=False):
                handler = DataHandler(config)
                assert 'alpha_vantage' not in handler.adapters
    
    @pytest.mark.asyncio
    async def test_initialize_method(self, temp_cache_dir):
        """Test the initialize method."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        # Should complete without error
        await handler.initialize()
    
    def test_parquet_cache_fallback(self, temp_cache_dir):
        """Test parquet read failure with pickle fallback."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        
        # Create test data
        symbol = "AAPL"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        
        # Create mock data
        test_data = pd.DataFrame({
            'open': [150.0, 151.0, 152.0],
            'high': [151.0, 152.0, 153.0],
            'low': [149.0, 150.0, 151.0],
            'close': [150.5, 151.5, 152.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start, periods=3, freq='D'))
        
        # Create cache key
        cache_key = f"{symbol}_1d_{start.date()}_{end.date()}"
        parquet_path = Path(temp_cache_dir) / f"{cache_key}.parquet"
        pickle_path = Path(temp_cache_dir) / f"{cache_key}.pkl"
        
        # Save as pickle
        test_data.to_pickle(pickle_path)
        
        # Create corrupted parquet file
        parquet_path.write_text("corrupted data")
        
        # Mock the provider fetch to avoid actual API calls
        with patch.object(handler, '_fetch_from_provider', return_value=test_data):
            # Test parquet read failure with pickle fallback
            with patch('pandas.read_parquet', side_effect=ImportError("No pyarrow")):
                result = handler.get_historical(symbol, start, end, "1d")
                assert result is not None
                assert len(result) == 3
    
    def test_parquet_write_fallback(self, temp_cache_dir):
        """Test parquet write failure with pickle fallback."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        
        # Create test data
        symbol = "GOOGL"
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        
        test_data = pd.DataFrame({
            'open': [2800.0, 2810.0, 2820.0],
            'high': [2810.0, 2820.0, 2830.0],
            'low': [2790.0, 2800.0, 2810.0],
            'close': [2805.0, 2815.0, 2825.0],
            'volume': [500000, 510000, 520000]
        }, index=pd.date_range(start, periods=3, freq='D'))
        
        # Mock the provider fetch
        with patch.object(handler, '_fetch_from_provider', return_value=test_data):
            # Mock parquet write to fail
            with patch('pandas.DataFrame.to_parquet', side_effect=ImportError("No pyarrow")):
                result = handler.get_historical(symbol, start, end, "1d")
                
                # Check that pickle file was created as fallback
                cache_key = f"{symbol}_1d_{start.date()}_{end.date()}"
                pickle_path = Path(temp_cache_dir) / f"{cache_key}.pkl"
                assert pickle_path.exists()
    
    def test_stale_cache_update(self, temp_cache_dir):
        """Test updating stale cache with new data."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        
        symbol = "MSFT"
        # Old data
        old_start = datetime(2024, 1, 1)
        old_end = datetime(2024, 1, 5)
        
        old_data = pd.DataFrame({
            'open': [370.0, 371.0, 372.0],
            'high': [371.0, 372.0, 373.0],
            'low': [369.0, 370.0, 371.0],
            'close': [370.5, 371.5, 372.5],
            'volume': [2000000, 2100000, 2200000]
        }, index=pd.date_range(old_start, periods=3, freq='D'))
        
        # Save old data to cache
        # Use the same cache key that will be used when fetching
        new_end = datetime(2024, 1, 10)
        cache_key = f"{symbol}_1d_{old_start.date()}_{new_end.date()}"
        pickle_path = Path(temp_cache_dir) / f"{cache_key}.pkl"
        old_data.to_pickle(pickle_path)
        
        # New data for the update
        new_data = pd.DataFrame({
            'open': [373.0, 374.0],
            'high': [374.0, 375.0],
            'low': [372.0, 373.0],
            'close': [373.5, 374.5],
            'volume': [2300000, 2400000]
        }, index=pd.date_range(datetime(2024, 1, 6), periods=2, freq='D'))
        
        # Mock fetch to return new data
        with patch.object(handler, '_fetch_from_provider', return_value=new_data):
            # Request data including newer dates
            result = handler.get_historical(symbol, old_start, datetime(2024, 1, 10), "1d")
            
            # Should have combined old and new data
            assert len(result) == 5  # 3 old + 2 new
            assert result.index[0].date() == old_start.date()
            assert result.index[-1].date() == datetime(2024, 1, 7).date()
    
    @pytest.mark.asyncio
    async def test_get_latest_method(self, temp_cache_dir):
        """Test the get_latest method."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        
        # Test data
        test_symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Mock historical data for each symbol
        def mock_get_historical(symbol, start, end, interval):
            if symbol == "AAPL":
                data = pd.DataFrame({
                    'open': [150.0, 151.0],
                    'high': [151.0, 152.0],
                    'low': [149.0, 150.0],
                    'close': [150.5, 151.5],
                    'volume': [1000000, 1100000]
                }, index=pd.date_range(end - timedelta(days=2), periods=2, freq='D'))
            elif symbol == "GOOGL":
                data = pd.DataFrame({
                    'open': [2800.0, 2810.0],
                    'high': [2810.0, 2820.0],
                    'low': [2790.0, 2800.0],
                    'close': [2805.0, 2815.0],
                    'volume': [500000, 510000]
                }, index=pd.date_range(end - timedelta(days=2), periods=2, freq='D'))
            elif symbol == "MSFT":
                # This one will fail
                raise Exception("API error for MSFT")
            else:
                data = pd.DataFrame()
            
            return data
        
        with patch.object(handler, 'get_historical', side_effect=mock_get_historical):
            result = await handler.get_latest(test_symbols)
            
            # Check results
            assert "AAPL" in result
            assert result["AAPL"]["close"] == 151.5
            assert result["AAPL"]["volume"] == 1100000
            
            assert "GOOGL" in result
            assert result["GOOGL"]["close"] == 2815.0
            
            # MSFT should not be in results due to error
            assert "MSFT" not in result
    
    @pytest.mark.asyncio
    async def test_get_latest_empty_symbols(self, temp_cache_dir):
        """Test get_latest with empty or None symbols."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        
        # Test with None
        result = await handler.get_latest(None)
        assert result == {}
        
        # Test with empty list
        result = await handler.get_latest([])
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_latest_empty_dataframe(self, temp_cache_dir):
        """Test get_latest when historical data returns empty DataFrame."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['yahoo']
        }
        
        handler = DataHandler(config)
        
        # Mock to return empty DataFrame
        with patch.object(handler, 'get_historical', return_value=pd.DataFrame()):
            result = await handler.get_latest(["AAPL"])
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])