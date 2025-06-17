"""Tests for Alpha Vantage integration."""

import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from adapters.av_fetcher import AlphaVantageFetcher


class TestAlphaVantageIntegration:
    """Test Alpha Vantage data fetching and integration."""

    @pytest.fixture
    def mock_av_response(self):
        """Mock Alpha Vantage API response."""
        return {
            "Meta Data": {
                "1. Information": "Daily Time Series with Splits and Dividend Events",
                "2. Symbol": "SPY",
                "3. Last Refreshed": "2025-06-06",
            },
            "Time Series (Daily)": {
                "2025-06-06": {
                    "1. open": "598.66",
                    "2. high": "600.83",
                    "3. low": "596.86",
                    "4. close": "599.14",
                    "5. adjusted close": "599.14",
                    "6. volume": "66588743",
                    "7. dividend amount": "0.0000",
                    "8. split coefficient": "1.0",
                },
                "2025-06-05": {
                    "1. open": "597.63",
                    "2. high": "599.00",
                    "3. low": "591.05",
                    "4. close": "593.05",
                    "5. adjusted close": "593.05",
                    "6. volume": "92436397",
                    "7. dividend amount": "0.0000",
                    "8. split coefficient": "1.0",
                },
            },
        }

    @pytest.fixture
    def mock_av_intraday_response(self):
        """Mock Alpha Vantage intraday API response."""
        return {
            "Meta Data": {
                "1. Information": "Intraday (5min) open, high, low, close prices and volume",
                "2. Symbol": "SPY",
            },
            "Time Series (5min)": {
                "2025-06-06 16:00:00": {
                    "1. open": "599.10",
                    "2. high": "599.20",
                    "3. low": "599.00",
                    "4. close": "599.14",
                    "5. volume": "1234567",
                },
                "2025-06-06 15:55:00": {
                    "1. open": "598.90",
                    "2. high": "599.10",
                    "3. low": "598.85",
                    "4. close": "599.10",
                    "5. volume": "987654",
                },
            },
        }

    def test_av_fetcher_initialization(self):
        """Test AlphaVantageFetcher initialization."""
        # With API key
        fetcher = AlphaVantageFetcher(api_key="test_key")
        assert fetcher.api_key == "test_key"
        assert fetcher.premium is False

        # Premium account
        fetcher_premium = AlphaVantageFetcher(api_key="test_key", premium=True)
        assert fetcher_premium.premium is True
        assert fetcher_premium.rate_limit_delay == 0.8

    def test_av_fetcher_no_api_key(self):
        """Test that fetcher raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Alpha Vantage API key required"):
                AlphaVantageFetcher()

    @patch("requests.get")
    def test_fetch_daily_data(self, mock_get, mock_av_response):
        """Test fetching daily data from Alpha Vantage."""
        mock_get.return_value.json.return_value = mock_av_response

        fetcher = AlphaVantageFetcher(api_key="test_key")

        end_date = datetime(2025, 6, 6)
        start_date = end_date - timedelta(days=30)

        df = fetcher.fetch_ohlcv("SPY", start_date, end_date, "1d")

        # Check the request was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == fetcher.base_url
        assert kwargs["params"]["function"] == "TIME_SERIES_DAILY_ADJUSTED"
        assert kwargs["params"]["symbol"] == "SPY"

        # Check the returned dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

        # Check OHLC relationships are valid after adjustment
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    @patch("requests.get")
    def test_fetch_intraday_data(self, mock_get, mock_av_intraday_response):
        """Test fetching intraday data from Alpha Vantage."""
        mock_get.return_value.json.return_value = mock_av_intraday_response

        fetcher = AlphaVantageFetcher(api_key="test_key")

        # Use exact date range that matches mock data
        end_date = datetime(2025, 6, 6, 16, 0, 0)
        start_date = datetime(2025, 6, 6, 15, 0, 0)

        df = fetcher.fetch_ohlcv("SPY", start_date, end_date, "5m")

        # Check the request
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["function"] == "TIME_SERIES_INTRADAY"
        assert kwargs["params"]["interval"] == "5min"

        # Check the dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    @patch("requests.get")
    def test_error_handling(self, mock_get):
        """Test error handling for API errors."""
        # API error response
        mock_get.return_value.json.return_value = {"Error Message": "Invalid API call"}

        fetcher = AlphaVantageFetcher(api_key="test_key")

        # Should return empty dataframe on error
        df = fetcher.fetch_ohlcv(
            "INVALID", datetime.now() - timedelta(days=30), datetime.now(), "1d"
        )
        assert df.empty

    @patch("requests.get")
    def test_rate_limit_handling(self, mock_get):
        """Test rate limit handling."""
        # First call returns rate limit message
        rate_limit_response = {
            "Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."
        }

        # Second call returns actual data
        success_response = {
            "Time Series (Daily)": {
                "2025-06-06": {
                    "1. open": "100",
                    "2. high": "101",
                    "3. low": "99",
                    "4. close": "100",
                    "5. volume": "1000000",
                }
            }
        }

        mock_get.return_value.json.side_effect = [rate_limit_response, success_response]

        fetcher = AlphaVantageFetcher(api_key="test_key")

        with patch("time.sleep") as mock_sleep:
            fetcher.fetch_ohlcv(
                "SPY", datetime.now() - timedelta(days=30), datetime.now(), "1d"
            )

            # Should have slept due to rate limit
            mock_sleep.assert_called_once_with(12)  # Free tier delay

            # Should have made two requests
            assert mock_get.call_count == 2

    def test_ohlc_adjustment(self):
        """Test that OHLC data is properly adjusted for splits/dividends."""
        # Create test data with adjustment needed
        test_data = pd.DataFrame(
            {
                "1. open": ["100.00", "101.00"],
                "2. high": ["102.00", "103.00"],
                "3. low": ["99.00", "100.00"],
                "4. close": ["101.00", "102.00"],
                "5. adjusted close": ["50.50", "102.00"],  # 2:1 split on first day
                "6. volume": ["1000000", "2000000"],
                "7. dividend amount": ["0", "0"],
                "8. split coefficient": ["2.0", "1.0"],
            },
            index=["2025-06-05", "2025-06-06"],
        )

        # Apply the adjustment logic from av_fetcher
        df = test_data.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. adjusted close": "adj_close",
                "6. volume": "volume",
            }
        )

        for col in ["open", "high", "low", "close", "adj_close"]:
            df[col] = df[col].astype(float)

        df["adj_factor"] = df["adj_close"] / df["close"]
        df["open"] = df["open"] * df["adj_factor"]
        df["high"] = df["high"] * df["adj_factor"]
        df["low"] = df["low"] * df["adj_factor"]
        df["close"] = df["adj_close"]

        # Check adjustments
        assert df.loc["2025-06-05", "open"] == 50.0  # 100 * 0.5
        assert df.loc["2025-06-05", "high"] == 51.0  # 102 * 0.5
        assert df.loc["2025-06-05", "low"] == 49.5  # 99 * 0.5
        assert df.loc["2025-06-05", "close"] == 50.5

        # Second day should be unchanged (no adjustment)
        assert df.loc["2025-06-06", "open"] == 101.0
        assert df.loc["2025-06-06", "close"] == 102.0
