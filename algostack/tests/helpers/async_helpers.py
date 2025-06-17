"""
Helper utilities for async testing.

Provides utilities to handle async/await patterns in tests.
"""

import asyncio
import functools
from typing import Any, Callable, Coroutine

import pytest


def async_test(func: Callable) -> Callable:
    """
    Decorator to run async tests with proper event loop handling.

    Example:
        @async_test
        async def test_async_function():
            result = await some_async_function()
            assert result == expected
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper


class AsyncContextManager:
    """
    Helper for testing async context managers.

    Example:
        async with AsyncContextManager(resource) as mgr:
            result = await mgr.do_something()
            assert result is not None
    """

    def __init__(self, resource: Any):
        self.resource = resource

    async def __aenter__(self):
        if hasattr(self.resource, '__aenter__'):
            return await self.resource.__aenter__()
        return self.resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.resource, '__aexit__'):
            return await self.resource.__aexit__(exc_type, exc_val, exc_tb)
        return False


async def await_or_return(value: Any) -> Any:
    """
    Helper to handle both sync and async values.

    Args:
        value: Value that might be a coroutine

    Returns:
        The value or awaited result

    Example:
        result = await await_or_return(maybe_async_func())
    """
    if asyncio.iscoroutine(value):
        return await value
    return value


async def gather_with_timeout(
    *coroutines: Coroutine,
    timeout: float = 5.0,
    return_exceptions: bool = True
) -> list:
    """
    Gather multiple coroutines with timeout.

    Args:
        *coroutines: Coroutines to run
        timeout: Maximum time to wait
        return_exceptions: Whether to return exceptions

    Returns:
        List of results

    Example:
        results = await gather_with_timeout(
            fetch_data('AAPL'),
            fetch_data('GOOGL'),
            timeout=10.0
        )
    """
    try:
        return await asyncio.wait_for(
            asyncio.gather(*coroutines, return_exceptions=return_exceptions),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return [TimeoutError(f"Operation timed out after {timeout}s")] * len(coroutines)


class MockAsyncIterator:
    """
    Mock async iterator for testing.

    Example:
        async_iter = MockAsyncIterator([1, 2, 3])
        async for value in async_iter:
            print(value)
    """

    def __init__(self, items: list):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        value = self.items[self.index]
        self.index += 1
        await asyncio.sleep(0)  # Yield control
        return value


def run_async_test(coro: Coroutine) -> Any:
    """
    Run an async test in a new event loop.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine

    Example:
        result = run_async_test(async_function())
        assert result == expected
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class AsyncMockManager:
    """
    Manager for async mocks in tests.

    Example:
        async with AsyncMockManager() as mgr:
            mock = mgr.create_mock('order_manager')
            mock.create_order.return_value = test_order

            result = await mock.create_order(...)
            mgr.assert_called_once(mock.create_order)
    """

    def __init__(self):
        self.mocks = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up any resources
        self.mocks.clear()
        return False

    def create_mock(self, name: str, **attrs) -> Any:
        """Create an async mock with given attributes."""
        from unittest.mock import AsyncMock

        mock = AsyncMock(**attrs)
        self.mocks[name] = mock
        return mock

    def assert_called_once(self, mock_method):
        """Assert mock was called exactly once."""
        mock_method.assert_called_once()

    def assert_called_with(self, mock_method, *args, **kwargs):
        """Assert mock was called with specific arguments."""
        mock_method.assert_called_with(*args, **kwargs)

    def reset_all(self):
        """Reset all mocks."""
        for mock in self.mocks.values():
            mock.reset_mock()


# Fixtures for async testing

@pytest.fixture
async def async_client():
    """
    Async HTTP client for testing.

    Example:
        async def test_api(async_client):
            response = await async_client.get('/api/v1/status')
            assert response.status_code == 200
    """
    import httpx

    async with httpx.AsyncClient(base_url="http://test") as client:
        yield client


@pytest.fixture
def async_mock_factory():
    """
    Factory for creating async mocks.

    Example:
        def test_with_async_mock(async_mock_factory):
            mock = async_mock_factory('return_value', 42)
            result = await mock()
            assert result == 42
    """
    from unittest.mock import AsyncMock

    def factory(name: str = None, **kwargs):
        mock = AsyncMock(**kwargs)
        if name:
            mock.name = name
        return mock

    return factory


@pytest.fixture
async def async_event_loop():
    """
    Provide an event loop for async tests.

    Example:
        async def test_with_loop(async_event_loop):
            task = async_event_loop.create_task(some_async_func())
            result = await task
            assert result is not None
    """
    loop = asyncio.get_event_loop()
    yield loop
    # Cleanup pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
