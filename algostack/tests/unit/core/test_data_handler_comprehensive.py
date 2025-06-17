"""Comprehensive test suite for data handler."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from algostack.core.data_handler import DataHandler


class TestDataHandler:
    """Test suite for DataHandler class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        return pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'high': 102 + np.random.randn(len(dates)).cumsum() * 0.5,
            'low': 98 + np.random.randn(len(dates)).cumsum() * 0.5,
            'close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)

    @pytest.fixture
    def data_handler(self, tmp_path):
        """Create DataHandler instance with temp cache dir."""
        return DataHandler(
            providers=['yfinance'],
            cache_dir=str(tmp_path / 'cache')
        )

    def test_initialization(self, tmp_path):
        """Test DataHandler initialization."""
        cache_dir = tmp_path / 'test_cache'
        handler = DataHandler(
            providers=['yfinance', 'alpha_vantage'],
            cache_dir=str(cache_dir),
            api_keys={'alpha_vantage': 'test_key'},
            premium_av=True
        )

        assert handler.providers == ['yfinance', 'alpha_vantage']
        assert handler.cache_dir == cache_dir
        assert cache_dir.exists()
        assert handler.api_keys == {'alpha_vantage': 'test_key'}
        assert handler.premium_av is True
        assert 'yfinance' in handler.adapters

    def test_init_adapters_yfinance(self):
        """Test YFinance adapter initialization."""
        handler = DataHandler(providers=['yfinance'])

        assert 'yfinance' in handler.adapters
        assert handler.adapters['yfinance'] is not None

    @patch.dict('os.environ', {'ALPHA_VANTAGE_API_KEY': 'env_test_key'})
    def test_init_adapters_alpha_vantage_env(self):
        """Test Alpha Vantage adapter initialization with env var."""
        handler = DataHandler(providers=['alpha_vantage'])

        assert 'alpha_vantage' in handler.adapters
        assert handler.adapters['alpha_vantage'] is not None

    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_init_adapters_alpha_vantage_secrets(self, mock_exists, mock_open):
        """Test Alpha Vantage adapter initialization from secrets file."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = '''
        data_providers:
          alphavantage:
            api_key: secrets_test_key
        '''

        with patch('yaml.safe_load', return_value={
            'data_providers': {'alphavantage': {'api_key': 'secrets_test_key'}}
        }):
            handler = DataHandler(providers=['alphavantage'])

            assert 'alphavantage' in handler.adapters

    def test_get_historical_cache_miss(self, data_handler, sample_data, tmp_path):
        """Test fetching historical data with cache miss."""
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.fetch_ohlcv.return_value = sample_data
        data_handler.adapters['yfinance'] = mock_adapter

        # Fetch data
        result = data_handler.get_historical(symbol, start, end)

        # Verify adapter was called
        mock_adapter.fetch_ohlcv.assert_called_once_with(symbol, start, end, '1d')

        # Verify data matches
        pd.testing.assert_frame_equal(result, sample_data)

        # Verify cache was created
        cache_files = list(data_handler.cache_dir.glob('*.parquet'))
        assert len(cache_files) == 1

    def test_get_historical_cache_hit(self, data_handler, sample_data):
        """Test fetching historical data with cache hit."""
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        # Pre-populate cache
        cache_key = f"{symbol}_1d_{start.date()}_{end.date()}"
        cache_path = data_handler.cache_dir / f"{cache_key}.parquet"
        sample_data.to_parquet(cache_path)

        # Mock adapter to verify it's not called
        mock_adapter = Mock()
        data_handler.adapters['yfinance'] = mock_adapter

        # Fetch data
        result = data_handler.get_historical(symbol, start, end)

        # Verify adapter was NOT called
        mock_adapter.fetch_ohlcv.assert_not_called()

        # Verify cached data returned
        # Compare values and column names (ignore index frequency metadata)
        pd.testing.assert_frame_equal(result, sample_data, check_freq=False)

    def test_get_historical_stale_cache(self, data_handler, sample_data):
        """Test cache refresh when data is stale."""
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        # Create stale cache (missing last 5 days)
        stale_data = sample_data.iloc[:-5]
        cache_key = f"{symbol}_1d_{start.date()}_{end.date()}"
        cache_path = data_handler.cache_dir / f"{cache_key}.parquet"
        stale_data.to_parquet(cache_path)

        # Mock adapter to return new data
        new_data = sample_data.iloc[-5:]
        mock_adapter = Mock()
        mock_adapter.fetch_ohlcv.return_value = new_data
        data_handler.adapters['yfinance'] = mock_adapter

        # Fetch data
        result = data_handler.get_historical(symbol, start, end)

        # Verify adapter was called for missing data
        assert mock_adapter.fetch_ohlcv.called
        call_args = mock_adapter.fetch_ohlcv.call_args[0]
        assert call_args[1] > stale_data.index[-1]  # Start after cached data

        # Verify complete data returned
        assert len(result) == len(sample_data)

    def test_get_historical_with_interval(self, data_handler):
        """Test fetching data with different intervals."""
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.fetch_ohlcv.return_value = pd.DataFrame()
        data_handler.adapters['yfinance'] = mock_adapter

        # Test different intervals
        for interval in ['5m', '15m', '1h', '1d', '1w']:
            data_handler.get_historical(symbol, start, end, interval)

            # Verify interval passed correctly
            _, _, _, called_interval = mock_adapter.fetch_ohlcv.call_args[0]
            assert called_interval == interval

    def test_get_historical_with_provider(self, data_handler):
        """Test fetching data with specific provider."""
        # Mock the existing yfinance adapter
        mock_yf_adapter = Mock()
        mock_yf_adapter.fetch_ohlcv.return_value = pd.DataFrame()
        data_handler.adapters['yfinance'] = mock_yf_adapter
        
        # Add another provider
        mock_av_adapter = Mock()
        mock_av_adapter.fetch_ohlcv.return_value = pd.DataFrame()
        data_handler.adapters['alpha_vantage'] = mock_av_adapter
        data_handler.providers.append('alpha_vantage')

        # Fetch with specific provider
        data_handler.get_historical(
            'AAPL',
            datetime(2023, 1, 1),
            datetime(2023, 1, 31),
            provider='alpha_vantage'
        )

        # Verify correct adapter used
        mock_av_adapter.fetch_ohlcv.assert_called_once()
        mock_yf_adapter.fetch_ohlcv.assert_not_called()

    def test_get_historical_unknown_provider(self, data_handler):
        """Test error handling for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            data_handler.get_historical(
                'AAPL',
                datetime(2023, 1, 1),
                datetime(2023, 1, 31),
                provider='unknown_provider'
            )

    async def test_get_latest(self, data_handler, sample_data):
        """Test getting latest market data."""
        symbols = ['AAPL', 'GOOGL']

        # Mock get_historical to return sample data
        def mock_get_historical(symbol, start, end, interval):
            return sample_data

        data_handler.get_historical = mock_get_historical

        # Get latest data
        latest = await data_handler.get_latest(symbols)

        assert len(latest) == 2
        assert 'AAPL' in latest
        assert 'GOOGL' in latest

        for symbol in symbols:
            assert 'open' in latest[symbol]
            assert 'high' in latest[symbol]
            assert 'low' in latest[symbol]
            assert 'close' in latest[symbol]
            assert 'volume' in latest[symbol]
            assert 'timestamp' in latest[symbol]

    async def test_get_latest_error_handling(self, data_handler):
        """Test error handling in get_latest."""
        # Mock get_historical to raise exception
        def mock_get_historical(symbol, start, end, interval):
            raise Exception("Network error")

        data_handler.get_historical = mock_get_historical

        # Should handle errors gracefully
        latest = await data_handler.get_latest(['BAD_SYMBOL'])

        assert latest == {}

    def test_calculate_indicators(self, data_handler, sample_data):
        """Test technical indicator calculation."""
        result = data_handler.calculate_indicators(sample_data.copy())

        # Check ATR calculation
        assert 'atr' in result.columns
        assert not result['atr'][:13].notna().any()  # First 13 should be NaN
        assert result['atr'][14:].notna().all()  # Rest should have values

        # Check returns
        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert result['returns'].iloc[0] != result['returns'].iloc[0]  # First is NaN

        # Check volume indicators
        assert 'volume_sma' in result.columns
        assert 'volume_ratio' in result.columns
        assert result['volume_ratio'].iloc[20:].notna().all()

    def test_clear_cache_specific_symbol(self, data_handler):
        """Test clearing cache for specific symbol."""
        # Create cache files
        cache_files = [
            'AAPL_1d_2023-01-01_2023-01-31.parquet',
            'AAPL_1h_2023-01-01_2023-01-31.parquet',
            'GOOGL_1d_2023-01-01_2023-01-31.parquet'
        ]

        for filename in cache_files:
            (data_handler.cache_dir / filename).touch()

        # Clear AAPL cache
        data_handler.clear_cache('AAPL')

        # Verify only AAPL files removed
        remaining_files = list(data_handler.cache_dir.glob('*.parquet'))
        assert len(remaining_files) == 1
        assert 'GOOGL' in remaining_files[0].name

    def test_clear_cache_all(self, data_handler):
        """Test clearing all cache."""
        # Create cache files
        cache_files = ['AAPL.parquet', 'GOOGL.parquet', 'MSFT.pkl']
        for filename in cache_files:
            (data_handler.cache_dir / filename).touch()

        # Clear all cache
        data_handler.clear_cache()

        # Verify all files removed
        assert len(list(data_handler.cache_dir.glob('*'))) == 0

    def test_get_cache_size(self, data_handler):
        """Test cache size calculation."""
        # Create test files with known sizes
        test_data = b'x' * 1000

        (data_handler.cache_dir / 'test1.parquet').write_bytes(test_data)
        (data_handler.cache_dir / 'test2.pkl').write_bytes(test_data * 2)

        size = data_handler.get_cache_size()

        assert size == 3000

    def test_pickle_fallback(self, data_handler, sample_data):
        """Test pickle fallback when parquet fails."""
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.fetch_ohlcv.return_value = sample_data
        data_handler.adapters['yfinance'] = mock_adapter

        # Mock parquet to fail
        with patch('pandas.DataFrame.to_parquet', side_effect=ImportError):
            result = data_handler.get_historical(symbol, start, end)

        # Verify data returned
        pd.testing.assert_frame_equal(result, sample_data)

        # Verify pickle file created
        pickle_files = list(data_handler.cache_dir.glob('*.pkl'))
        assert len(pickle_files) == 1

    def test_concurrent_cache_access(self, data_handler, sample_data):
        """Test concurrent access to cache."""
        import threading

        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.fetch_ohlcv.return_value = sample_data
        data_handler.adapters['yfinance'] = mock_adapter

        results = []
        errors = []

        def fetch_data():
            try:
                result = data_handler.get_historical(symbol, start, end)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=fetch_data) for _ in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have at least one successful result
        assert len(results) >= 1, f"All threads failed with errors: {errors}"
        
        # All successful results should have the same data
        for result in results:
            pd.testing.assert_frame_equal(result, sample_data, check_freq=False)
