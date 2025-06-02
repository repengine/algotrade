"""Tests for data handler module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from core.data_handler import DataHandler


class TestDataHandler:
    """Test data handler functionality."""
    
    @pytest.fixture
    def data_handler(self, tmp_path):
        """Create data handler with temporary cache directory."""
        cache_dir = tmp_path / "cache"
        return DataHandler(providers=['yfinance'], cache_dir=str(cache_dir))
    
    @pytest.fixture
    def mock_adapter(self, mocker, sample_ohlcv_data):
        """Mock data adapter."""
        mock = mocker.MagicMock()
        mock.fetch_ohlcv.return_value = sample_ohlcv_data
        return mock
    
    def test_initialization(self, data_handler):
        """Test data handler initialization."""
        assert len(data_handler.providers) == 1
        assert 'yfinance' in data_handler.providers
        assert Path(data_handler.cache_dir).exists()
    
    @pytest.mark.unit
    def test_cache_path_generation(self, data_handler):
        """Test cache file path generation."""
        symbol = 'SPY'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        interval = '1d'
        
        cache_key = f"{symbol}_{interval}_{start.date()}_{end.date()}"
        cache_path = data_handler.cache_dir / f"{cache_key}.parquet"
        
        assert cache_path.name == f"SPY_1d_2023-01-01_2023-12-31.parquet"
    
    @pytest.mark.unit
    def test_calculate_indicators(self, data_handler, sample_ohlcv_data):
        """Test technical indicator calculation."""
        df = data_handler.calculate_indicators(sample_ohlcv_data)
        
        # Check indicators exist
        assert 'atr' in df.columns
        assert 'returns' in df.columns
        assert 'log_returns' in df.columns
        assert 'volume_sma' in df.columns
        assert 'volume_ratio' in df.columns
        
        # Check values are reasonable
        assert df['atr'].notna().sum() > 0
        assert (df['volume_ratio'] > 0).all()
    
    def test_cache_functionality(self, data_handler, tmp_path, sample_ohlcv_data):
        """Test caching mechanism."""
        # Mock the adapter
        data_handler.adapters['yfinance'] = type('MockAdapter', (), {
            'fetch_ohlcv': lambda *args: sample_ohlcv_data
        })
        
        symbol = 'TEST'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        # First call should fetch from adapter
        df1 = data_handler.get_historical(symbol, start, end)
        assert not df1.empty
        
        # Check cache file was created
        cache_files = list(Path(data_handler.cache_dir).glob("*.parquet"))
        assert len(cache_files) == 1
        
        # Second call should load from cache
        # Mock adapter to return empty to verify cache is used
        data_handler.adapters['yfinance'].fetch_ohlcv = lambda *args: pd.DataFrame()
        
        df2 = data_handler.get_historical(symbol, start, end)
        assert not df2.empty
        assert df1.equals(df2)
    
    def test_cache_size_calculation(self, data_handler, tmp_path):
        """Test cache size calculation."""
        # Create dummy cache files
        cache_file = Path(data_handler.cache_dir) / "test.parquet"
        pd.DataFrame({'test': [1, 2, 3]}).to_parquet(cache_file)
        
        cache_size = data_handler.get_cache_size()
        assert cache_size > 0
    
    def test_clear_cache(self, data_handler, tmp_path):
        """Test cache clearing functionality."""
        # Create dummy cache files
        cache_dir = Path(data_handler.cache_dir)
        for i in range(3):
            cache_file = cache_dir / f"TEST_{i}.parquet"
            pd.DataFrame({'test': [i]}).to_parquet(cache_file)
        
        assert len(list(cache_dir.glob("*.parquet"))) == 3
        
        # Clear specific symbol
        data_handler.clear_cache('TEST')
        assert len(list(cache_dir.glob("*.parquet"))) == 0
    
    @pytest.mark.integration
    def test_multi_provider_fallback(self, data_handler):
        """Test fallback between multiple providers."""
        # Add a second provider
        data_handler.providers.append('alpha_vantage')
        
        # Mock both adapters
        mock_yf = type('MockYF', (), {
            'fetch_ohlcv': lambda *args: pd.DataFrame()  # Empty = failure
        })
        mock_av = type('MockAV', (), {
            'fetch_ohlcv': lambda *args: pd.DataFrame({'close': [100, 101, 102]})
        })
        
        data_handler.adapters['yfinance'] = mock_yf
        data_handler.adapters['alpha_vantage'] = mock_av
        
        # Should fallback to alpha_vantage
        df = data_handler.get_historical('TEST', datetime.now(), datetime.now(), provider='alpha_vantage')
        assert not df.empty
        assert len(df) == 3