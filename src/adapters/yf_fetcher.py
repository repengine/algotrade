"""Yahoo Finance data fetcher adapter."""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Fetches market data from Yahoo Finance."""

    def __init__(self):
        self.name = "yfinance"

    def fetch_ohlcv(
        self, symbol: str, start: datetime, end: datetime, interval: str = "1d"
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
                auto_adjust=True,  # Adjust for splits/dividends
            )

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Add symbol as attribute
            df.attrs["symbol"] = symbol

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
        self, symbols: list, start: datetime, end: datetime, interval: str = "1d"
    ) -> dict:
        """Fetch data for multiple symbols."""
        data = {}

        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, start, end, interval)
            if not df.empty:
                data[symbol] = df

        return data

    async def fetch(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols - PILLAR 3: OPERATIONAL STABILITY.

        This method is required by tests and provides async compatibility.
        """
        if start_date is None:
            start_date = datetime.now() - pd.Timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Convert async to sync call
        return self.fetch_multiple(symbols, start_date, end_date, interval)

    async def fetch_realtime(
        self,
        symbols: list[str]
    ) -> dict[str, dict]:
        """Fetch real-time quotes - PILLAR 2: PROFIT GENERATION."""
        quotes = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Extract real-time quote data
                quotes[symbol] = {
                    "symbol": symbol,
                    "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                    "bid": info.get("bid", 0),
                    "ask": info.get("ask", 0),
                    "volume": info.get("volume", 0),
                    "timestamp": datetime.now()
                }

            except Exception as e:
                logger.error(f"Error fetching realtime quote for {symbol}: {e}")
                quotes[symbol] = {
                    "symbol": symbol,
                    "price": 0,
                    "error": str(e),
                    "timestamp": datetime.now()
                }

        return quotes

    def validate_data(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Validate fetched data - PILLAR 1: CAPITAL PRESERVATION.

        Bad data = bad trades = lost money.
        """
        if data.empty:
            logger.warning(f"Empty data for {symbol}")
            return data

        # Check for required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()

        # Sanity checks
        # 1. Prices must be positive
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if (data[col] <= 0).any():
                logger.error(f"Invalid negative/zero prices in {col} for {symbol}")
                data = data[data[col] > 0]

        # 2. High >= Low
        invalid_hl = data["high"] < data["low"]
        if invalid_hl.any():
            logger.warning(f"Found {invalid_hl.sum()} rows where high < low for {symbol}")
            data = data[~invalid_hl]

        # 3. High >= Open, Close
        invalid_high = (data["high"] < data["open"]) | (data["high"] < data["close"])
        if invalid_high.any():
            logger.warning(f"Found {invalid_high.sum()} rows with invalid highs for {symbol}")
            data = data[~invalid_high]

        # 4. Low <= Open, Close
        invalid_low = (data["low"] > data["open"]) | (data["low"] > data["close"])
        if invalid_low.any():
            logger.warning(f"Found {invalid_low.sum()} rows with invalid lows for {symbol}")
            data = data[~invalid_low]

        # 5. Volume must be non-negative
        if (data["volume"] < 0).any():
            logger.warning(f"Negative volumes found for {symbol}")
            data = data[data["volume"] >= 0]

        # 6. Check for extreme price movements (>50% in one period)
        if len(data) > 1:
            returns = data["close"].pct_change()
            extreme_moves = abs(returns) > 0.5
            if extreme_moves.any():
                logger.warning(
                    f"Found {extreme_moves.sum()} extreme price movements (>50%) for {symbol}"
                )
                # Flag but don't remove - could be legitimate (e.g., stock split)

        # 7. Check for stale data (no price changes)
        if len(data) > 5:
            price_changes = data[price_cols].diff().abs().sum(axis=1)
            stale_data = price_changes == 0
            consecutive_stale = stale_data.rolling(5).sum() >= 5
            if consecutive_stale.any():
                logger.warning(f"Found stale data (no price changes) for {symbol}")

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
            "1M": "1mo",
        }
        return interval_map.get(interval, interval)
