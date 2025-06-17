"""
Market data fixtures for testing.

This module provides reusable market data fixtures that ensure
consistent test data across the test suite.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest


class MarketDataGenerator:
    """Generate various types of market data for testing."""

    @staticmethod
    def create_realistic_ohlcv(
        symbol: str = "TEST",
        start_date: datetime = None,
        periods: int = 100,
        base_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0001,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create realistic OHLCV data with proper relationships.

        Args:
            symbol: Stock symbol
            start_date: Starting date (default: 100 days ago)
            periods: Number of periods
            base_price: Starting price
            volatility: Daily volatility (default: 2%)
            trend: Daily trend/drift (default: 0.01%)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with OHLCV data
        """
        if seed is not None:
            np.random.seed(seed)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=periods)

        dates = pd.date_range(start=start_date, periods=periods, freq='D')

        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, periods)
        close_prices = base_price * np.exp(np.cumsum(returns))

        # Create realistic OHLC relationships
        data = pd.DataFrame(index=dates)

        # Open is close from previous day plus overnight gap
        overnight_gaps = np.random.normal(0, volatility * 0.5, periods)
        data['open'] = close_prices * (1 + overnight_gaps)
        data.iloc[0, data.columns.get_loc('open')] = base_price

        # Intraday range
        daily_range = np.abs(np.random.normal(0, volatility, periods))

        # High and low based on daily range
        data['high'] = np.maximum(data['open'], close_prices) * (1 + daily_range * 0.5)
        data['low'] = np.minimum(data['open'], close_prices) * (1 - daily_range * 0.5)
        data['close'] = close_prices

        # Volume with some correlation to price movement
        base_volume = 1000000
        volume_volatility = np.abs(returns) * 10 + 1  # Higher volume on big moves
        data['volume'] = (base_volume * volume_volatility * np.random.lognormal(0, 0.3, periods)).astype(int)

        # Add symbol as attribute
        data.attrs['symbol'] = symbol

        return data

    @staticmethod
    def create_trending_market(
        direction: str = "up",
        strength: float = 0.02,
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Create trending market data.

        Args:
            direction: "up" or "down"
            strength: Trend strength (daily return)
            periods: Number of periods

        Returns:
            DataFrame with trending OHLCV data
        """
        trend = strength if direction == "up" else -strength
        return MarketDataGenerator.create_realistic_ohlcv(
            periods=periods,
            trend=trend,
            volatility=0.01,  # Lower volatility in trends
            seed=42
        )

    @staticmethod
    def create_ranging_market(
        range_width: float = 0.10,
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Create ranging/sideways market data.

        Args:
            range_width: Width of the range (as percentage)
            periods: Number of periods

        Returns:
            DataFrame with ranging OHLCV data
        """
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')

        # Oscillating price within a range
        base_price = 100
        t = np.linspace(0, 4 * np.pi, periods)
        prices = base_price + (base_price * range_width / 2) * np.sin(t)

        # Add some noise
        noise = np.random.normal(0, base_price * 0.005, periods)
        prices = prices + noise

        # Create OHLCV
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = np.roll(prices, 1)
        data.iloc[0, data.columns.get_loc('open')] = base_price

        # Daily ranges
        daily_range = np.random.uniform(0.002, 0.01, periods)
        data['high'] = np.maximum(data['open'], data['close']) * (1 + daily_range)
        data['low'] = np.minimum(data['open'], data['close']) * (1 - daily_range)

        # Volume - higher at range boundaries
        distance_from_center = np.abs(prices - base_price)
        volume_multiplier = 1 + distance_from_center / (base_price * range_width / 2)
        data['volume'] = (1000000 * volume_multiplier * np.random.uniform(0.8, 1.2, periods)).astype(int)

        return data

    @staticmethod
    def create_gappy_market(
        gap_probability: float = 0.1,
        gap_size: float = 0.02,
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Create market data with price gaps.

        Args:
            gap_probability: Probability of a gap on any day
            gap_size: Average gap size (as percentage)
            periods: Number of periods

        Returns:
            DataFrame with gappy OHLCV data
        """
        data = MarketDataGenerator.create_realistic_ohlcv(periods=periods, seed=42)

        # Add random gaps
        gap_mask = np.random.random(periods) < gap_probability
        gap_directions = np.random.choice([-1, 1], periods)
        gap_sizes = np.random.exponential(gap_size, periods) * gap_directions

        # Apply gaps to open prices
        for i in range(1, periods):
            if gap_mask[i]:
                gap = data.iloc[i-1]['close'] * gap_sizes[i]
                data.iloc[i, data.columns.get_loc('open')] += gap

                # Adjust high/low if necessary
                if gap > 0:
                    data.iloc[i, data.columns.get_loc('high')] = max(
                        data.iloc[i]['high'],
                        data.iloc[i]['open']
                    )
                else:
                    data.iloc[i, data.columns.get_loc('low')] = min(
                        data.iloc[i]['low'],
                        data.iloc[i]['open']
                    )

        return data

    @staticmethod
    def create_correlated_assets(
        symbols: List[str] = None,
        correlation_matrix: np.ndarray = None,
        periods: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """
        Create multiple assets with specified correlations.

        Args:
            symbols: List of symbols (default: ['SPY', 'QQQ', 'IWM', 'TLT'])
            correlation_matrix: Correlation matrix (default: realistic correlations)
            periods: Number of periods

        Returns:
            Dictionary of symbol -> DataFrame
        """
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'TLT']

        if correlation_matrix is None:
            # Realistic correlation matrix
            correlation_matrix = np.array([
                [1.00, 0.95, 0.85, -0.30],  # SPY
                [0.95, 1.00, 0.80, -0.35],  # QQQ
                [0.85, 0.80, 1.00, -0.25],  # IWM
                [-0.30, -0.35, -0.25, 1.00]  # TLT (bonds)
            ])

        n_assets = len(symbols)

        # Generate correlated returns using Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)

        # Generate uncorrelated random returns
        np.random.seed(42)
        uncorrelated_returns = np.random.normal(0, 1, (periods, n_assets))

        # Create correlated returns
        correlated_returns = uncorrelated_returns @ L.T

        # Asset-specific parameters
        base_prices = {'SPY': 400, 'QQQ': 350, 'IWM': 200, 'TLT': 120}
        volatilities = {'SPY': 0.015, 'QQQ': 0.020, 'IWM': 0.025, 'TLT': 0.010}

        market_data = {}

        for i, symbol in enumerate(symbols):
            # Scale returns by asset volatility
            vol = volatilities.get(symbol, 0.02)
            returns = correlated_returns[:, i] * vol + 0.0002  # Small drift

            # Create price series
            base_price = base_prices.get(symbol, 100)
            prices = base_price * np.exp(np.cumsum(returns))

            # Create OHLCV data
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            data = pd.DataFrame(index=dates)

            data['close'] = prices
            data['open'] = np.roll(prices, 1) * (1 + np.random.normal(0, 0.002, periods))
            data.iloc[0, data.columns.get_loc('open')] = base_price

            # Realistic high/low
            daily_range = np.random.uniform(0.002, 0.008, periods) * vol / 0.02
            data['high'] = np.maximum(data['open'], data['close']) * (1 + daily_range)
            data['low'] = np.minimum(data['open'], data['close']) * (1 - daily_range)

            # Volume
            data['volume'] = np.random.lognormal(14 + i * 0.5, 0.5, periods).astype(int)

            # Add symbol attribute
            data.attrs['symbol'] = symbol

            market_data[symbol] = data

        return market_data


@pytest.fixture
def market_data_generator():
    """Provide MarketDataGenerator instance."""
    return MarketDataGenerator()


@pytest.fixture
def spy_data():
    """Generate SPY data for testing."""
    return MarketDataGenerator.create_realistic_ohlcv(
        symbol="SPY",
        base_price=400.0,
        volatility=0.015,
        periods=252,
        seed=42
    )


@pytest.fixture
def tech_stocks_data():
    """Generate correlated tech stock data."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META']

    # High correlation matrix for tech stocks
    correlation_matrix = np.array([
        [1.00, 0.85, 0.82, 0.88],
        [0.85, 1.00, 0.80, 0.83],
        [0.82, 0.80, 1.00, 0.85],
        [0.88, 0.83, 0.85, 1.00]
    ])

    return MarketDataGenerator.create_correlated_assets(
        symbols=symbols,
        correlation_matrix=correlation_matrix,
        periods=252
    )


@pytest.fixture
def earnings_event_data():
    """Generate data with earnings-like events (gaps and volume spikes)."""
    data = MarketDataGenerator.create_realistic_ohlcv(periods=90, seed=42)

    # Add earnings events every 20-25 days
    earnings_days = [20, 45, 70]

    for day in earnings_days:
        if day < len(data):
            # Earnings gap (can be positive or negative)
            gap_size = np.random.choice([-0.08, -0.05, 0.05, 0.10])
            data.iloc[day, data.columns.get_loc('open')] *= (1 + gap_size)

            # Adjust high/low
            if gap_size > 0:
                data.iloc[day, data.columns.get_loc('high')] *= (1 + gap_size * 1.2)
            else:
                data.iloc[day, data.columns.get_loc('low')] *= (1 + gap_size * 1.2)

            # Volume spike (3-5x normal)
            data.iloc[day, data.columns.get_loc('volume')] *= np.random.uniform(3, 5)

            # Increased volatility for next few days
            for i in range(1, min(5, len(data) - day)):
                vol_multiplier = 1.5 - i * 0.1  # Decay over 5 days
                daily_range = np.abs(data.iloc[day + i]['high'] - data.iloc[day + i]['low'])
                data.iloc[day + i, data.columns.get_loc('high')] += daily_range * (vol_multiplier - 1) * 0.5
                data.iloc[day + i, data.columns.get_loc('low')] -= daily_range * (vol_multiplier - 1) * 0.5

    return data


@pytest.fixture
def intraday_data():
    """Generate intraday (5-minute) data."""
    # One trading day of 5-minute bars (9:30 AM - 4:00 PM = 390 minutes = 78 bars)
    trading_date = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    times = pd.date_range(start=trading_date, periods=78, freq='5min')

    # Intraday patterns
    # Morning volatility, lunch lull, afternoon activity
    time_of_day = np.array([(t.hour - 9.5) / 6.5 for t in times])  # Normalize to [0, 1]

    # U-shaped volume curve
    volume_pattern = 1.5 - np.abs(time_of_day - 0.5) * 2

    # Higher volatility at open and close
    volatility_pattern = 1 + 0.5 * (np.exp(-time_of_day * 10) + np.exp((time_of_day - 1) * 10))

    # Generate prices with intraday patterns
    base_price = 100
    returns = np.random.normal(0, 0.0002, len(times)) * volatility_pattern
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame(index=times)
    data['close'] = prices
    data['open'] = np.roll(prices, 1)
    data.iloc[0, data.columns.get_loc('open')] = base_price

    # Intraday ranges
    tick_range = 0.0001 * volatility_pattern
    data['high'] = np.maximum(data['open'], data['close']) * (1 + tick_range)
    data['low'] = np.minimum(data['open'], data['close']) * (1 - tick_range)

    # Volume with U-shaped pattern
    data['volume'] = (50000 * volume_pattern * np.random.uniform(0.8, 1.2, len(times))).astype(int)

    return data
