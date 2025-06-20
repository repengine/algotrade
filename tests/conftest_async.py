"""
Async test fixtures for proper resource cleanup.
"""
import asyncio
from typing import List

import pytest


@pytest.fixture
async def async_task_tracker():
    """Track and cleanup all tasks created during test."""
    tasks: List[asyncio.Task] = []

    def track_task(task: asyncio.Task) -> asyncio.Task:
        """Track a task for cleanup."""
        tasks.append(task)
        return task

    yield track_task

    # Cleanup all tasks
    for task in tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.fixture
async def connected_paper_executor(paper_executor):
    """Paper executor fixture with automatic cleanup."""
    await paper_executor.connect()
    try:
        yield paper_executor
    finally:
        await paper_executor.disconnect()


@pytest.fixture
async def connected_engine(engine_config):
    """LiveTradingEngine fixture with automatic cleanup."""
    from core.live_engine import LiveTradingEngine

    engine = LiveTradingEngine(engine_config)
    # Don't actually start the engine, just initialize
    yield engine

    # Cleanup
    if hasattr(engine, 'is_running') and engine.is_running:
        await engine.stop()


@pytest.fixture
async def websocket_client_connected(ws_client):
    """Connected WebSocket client with automatic cleanup."""
    # Mock the connection since we're testing
    import ssl
    from unittest.mock import AsyncMock, MagicMock, patch

    with patch('adapters.ibkr_adapter.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_ssl_context = MagicMock(spec=ssl.SSLContext)
        with patch('ssl.create_default_context', return_value=mock_ssl_context):
            await ws_client.connect()
            try:
                yield ws_client
            finally:
                await ws_client.disconnect()
