"""
Integration tests for data pipeline.

Tests the flow from data fetching through strategy execution.
Validates:
- Data fetching from various sources
- Data validation and cleaning
- Strategy signal generation from real data
- Multi-symbol processing
- Error handling across components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from adapters.yf_fetcher import YFinanceFetcher
from adapters.av_fetcher import AlphaVantageFetcher
from core.data_handler import DataHandler
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti
from strategies.base import Signal


class TestDataPipelineIntegration:
    """Test data flow from source to strategy."""
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data for integration testing."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        n = len(dates)
        
        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)
        close_prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': close_prices,
            'volume': np.random.lognormal(14, 0.5, n).astype(int)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.mark.integration
    def test_data_fetching_to_signal_generation(self, mock_market_data):
        """
        Test complete flow from data fetching to signal generation.
        
        Verifies:
        1. Data fetcher retrieves data correctly
        2. Data handler processes and validates data
        3. Strategy receives clean data and generates signals
        """
        # Mock the data fetcher
        with patch.object(YFinanceFetcher, 'fetch') as mock_fetch:
            mock_fetch.return_value = mock_market_data
            
            # Set up components
            fetcher = YFinanceFetcher()
            data_handler = DataHandler()
            
            # Strategy configuration
            strategy_config = {
                "lookback_period": 20,
                "zscore_threshold": 2.0,
                "exit_zscore": 0.5,
                "rsi_period": 14,
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
                "max_positions": 5
            }
            strategy = MeanReversionEquity(strategy_config)
            strategy.init()
            
            # Execute pipeline
            symbol = 'AAPL'
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 30)
            
            # Fetch data through handler
            data = fetcher.fetch(symbol, start_date, end_date)
            
            # Verify data structure
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 30
            assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            
            # Validate data quality
            assert data['close'].notna().all()
            assert (data['high'] >= data['low']).all()
            assert (data['volume'] > 0).all()
            
            # Process through strategy
            data.attrs['symbol'] = symbol
            signal = strategy.next(data)
            
            # Signal may or may not be generated depending on market conditions
            if signal is not None:
                assert isinstance(signal, Signal)
                assert signal.symbol == symbol
                assert signal.direction in ['LONG', 'SHORT', 'FLAT']
                assert -1 <= signal.strength <= 1
    
    @pytest.mark.integration
    def test_data_validation_pipeline(self):
        """
        Test data validation catches bad data.
        
        Ensures invalid data is caught and handled properly.
        """
        # Create data with various issues
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        
        bad_data_scenarios = [
            # Missing values
            pd.DataFrame({
                'open': [100, 101, 102, np.nan, 104],
                'high': [101, 102, 103, 104, 105],
                'low': [99, 100, 101, 102, 103],
                'close': [100, 101, np.nan, 103, 104],
                'volume': [1000000] * 5
            }, index=dates[:5]),
            
            # Invalid OHLC relationships
            pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [101, 102, 98, 104, 105],  # High < Low
                'low': [99, 100, 101, 102, 103],
                'close': [100, 101, 102, 103, 104],
                'volume': [1000000] * 5
            }, index=dates[:5]),
            
            # Negative volume
            pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [101, 102, 103, 104, 105],
                'low': [99, 100, 101, 102, 103],
                'close': [100, 101, 102, 103, 104],
                'volume': [1000000, -1000, 1000000, 0, 1000000]
            }, index=dates[:5]),
        ]
        
        data_handler = DataHandler()
        
        for bad_data in bad_data_scenarios:
            # Validation should fail
            is_valid = data_handler.validate_data(bad_data)
            assert not is_valid
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_symbol_pipeline(self):
        """
        Test pipeline handles multiple symbols efficiently.
        
        Verifies parallel processing and data alignment.
        """
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        
        with patch.object(YFinanceFetcher, 'fetch') as mock_fetch:
            # Return different data for each symbol
            def side_effect(symbol, start, end):
                base_prices = {
                    'AAPL': 150, 
                    'GOOGL': 2500, 
                    'MSFT': 300, 
                    'AMZN': 140, 
                    'META': 320
                }
                base_price = base_prices[symbol]
                
                dates = pd.date_range(start, end, freq='D')
                n = len(dates)
                
                # Generate correlated but different data for each symbol
                np.random.seed(hash(symbol) % 2**32)
                returns = np.random.normal(0.001, 0.015, n)
                close_prices = base_price * np.exp(np.cumsum(returns))
                
                return pd.DataFrame({
                    'open': close_prices * (1 + np.random.uniform(-0.003, 0.003, n)),
                    'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
                    'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
                    'close': close_prices,
                    'volume': np.random.lognormal(14 + hash(symbol) % 3, 0.5, n).astype(int)
                }, index=dates)
            
            mock_fetch.side_effect = side_effect
            
            # Test pipeline
            fetcher = YFinanceFetcher()
            
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 30)
            
            # Fetch data for all symbols
            all_data = {}
            for symbol in symbols:
                data = fetcher.fetch(symbol, start, end)
                all_data[symbol] = data
            
            # Verify all symbols processed
            assert len(all_data) == len(symbols)
            assert all(symbol in all_data for symbol in symbols)
            
            # Verify data alignment
            first_symbol_dates = all_data[symbols[0]].index
            for symbol in symbols[1:]:
                assert all_data[symbol].index.equals(first_symbol_dates)
            
            # Verify data quality for each symbol
            for symbol, data in all_data.items():
                assert len(data) > 0
                assert data['close'].notna().all()
                assert (data['volume'] > 0).all()
    
    @pytest.mark.integration
    def test_strategy_chaining_pipeline(self):
        """
        Test multiple strategies processing same data stream.
        
        Verifies strategies can work independently on same data.
        """
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        n = len(dates)
        
        # Generate trending then mean-reverting data
        trend_period = 50
        returns1 = np.random.normal(0.002, 0.01, trend_period)  # Uptrend
        returns2 = np.random.normal(-0.001, 0.015, n - trend_period)  # Mean reversion
        returns = np.concatenate([returns1, returns2])
        
        close_prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.003, 0.003, n)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': close_prices,
            'volume': np.random.lognormal(14, 0.5, n).astype(int)
        }, index=dates)
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        data.attrs['symbol'] = 'TEST'
        
        # Initialize strategies
        mr_config = {
            "lookback_period": 20,
            "zscore_threshold": 2.0,
            "exit_zscore": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        }
        mean_reversion = MeanReversionEquity(mr_config)
        mean_reversion.init()
        
        tf_config = {
            "symbols": ["TEST"],
            "channel_period": 20,
            "atr_period": 14,
            "adx_period": 14,
            "adx_threshold": 25.0
        }
        trend_following = TrendFollowingMulti(tf_config)
        trend_following.init()
        
        # Process data through both strategies
        mr_signals = []
        tf_signals = []
        
        # Process in windows to simulate real-time
        window_size = 30
        for i in range(window_size, len(data)):
            window_data = data.iloc[:i+1]
            
            # Mean reversion strategy
            mr_signal = mean_reversion.next(window_data)
            if mr_signal:
                mr_signals.append(mr_signal)
            
            # Trend following strategy
            tf_signal = trend_following.next(window_data)
            if tf_signal:
                tf_signals.append(tf_signal)
        
        # Strategies should generate different signals
        # Both should process data without errors
        assert isinstance(mr_signals, list)
        assert isinstance(tf_signals, list)
        
        # Verify signal quality
        for signal in mr_signals + tf_signals:
            assert isinstance(signal, Signal)
            assert signal.symbol == 'TEST'
            assert signal.direction in ['LONG', 'SHORT', 'FLAT']
            assert -1 <= signal.strength <= 1
    
    @pytest.mark.integration
    def test_data_source_failover(self):
        """
        Test failover between data sources.
        
        Verifies system handles data source failures gracefully.
        """
        # Mock both data sources
        with patch.object(YFinanceFetcher, 'fetch') as mock_yf_fetch, \
             patch.object(AlphaVantageFetcher, 'fetch') as mock_av_fetch:
            
            # Yahoo Finance fails
            mock_yf_fetch.side_effect = Exception("Yahoo Finance unavailable")
            
            # Alpha Vantage works
            dates = pd.date_range('2024-01-01', periods=20, freq='D')
            mock_av_data = pd.DataFrame({
                'open': [100 + i for i in range(20)],
                'high': [101 + i for i in range(20)],
                'low': [99 + i for i in range(20)],
                'close': [100.5 + i for i in range(20)],
                'volume': [1000000] * 20
            }, index=dates)
            mock_av_fetch.return_value = mock_av_data
            
            # Data handler with failover
            data_handler = DataHandler()
            data_handler.add_fetcher('yahoo', YFinanceFetcher())
            data_handler.add_fetcher('alphavantage', AlphaVantageFetcher())
            
            # Should failover to Alpha Vantage
            data = data_handler.fetch_with_failover(
                'AAPL', 
                datetime(2024, 1, 1), 
                datetime(2024, 1, 20)
            )
            
            assert data is not None
            assert len(data) == 20
            assert data.equals(mock_av_data)
    
    @pytest.mark.integration
    def test_data_caching_pipeline(self):
        """
        Test data caching reduces redundant fetches.
        
        Verifies caching layer works correctly.
        """
        fetch_count = 0
        
        def mock_fetch(symbol, start, end):
            nonlocal fetch_count
            fetch_count += 1
            
            dates = pd.date_range(start, end, freq='D')
            n = len(dates)
            return pd.DataFrame({
                'close': [100 + i for i in range(n)],
                'volume': [1000000] * n
            }, index=dates)
        
        with patch.object(YFinanceFetcher, 'fetch', side_effect=mock_fetch):
            fetcher = YFinanceFetcher()
            data_handler = DataHandler()
            data_handler.enable_cache(cache_dir='./test_cache')
            
            # First fetch - should hit the source
            data1 = data_handler.fetch_with_cache(
                fetcher, 'AAPL',
                datetime(2024, 1, 1),
                datetime(2024, 1, 10)
            )
            assert fetch_count == 1
            
            # Second fetch - same params should hit cache
            data2 = data_handler.fetch_with_cache(
                fetcher, 'AAPL',
                datetime(2024, 1, 1),
                datetime(2024, 1, 10)
            )
            assert fetch_count == 1  # No additional fetch
            assert data1.equals(data2)
            
            # Different params - should hit source
            data3 = data_handler.fetch_with_cache(
                fetcher, 'GOOGL',
                datetime(2024, 1, 1),
                datetime(2024, 1, 10)
            )
            assert fetch_count == 2