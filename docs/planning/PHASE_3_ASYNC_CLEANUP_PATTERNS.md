# Phase 3: Async Test Cleanup Patterns

## Common Async Test Issues Found

### 1. Task Creation Without Cleanup
**Problem**: Tests create tasks with `asyncio.create_task()` but don't properly cancel or await them
**Example**: 
```python
# BAD - Task not cleaned up
heartbeat_task = asyncio.create_task(ws_client._heartbeat_loop())
await asyncio.sleep(0.1)
heartbeat_task.cancel()  # Cancel without awaiting

# GOOD - Proper cleanup
heartbeat_task = asyncio.create_task(ws_client._heartbeat_loop())
try:
    await asyncio.sleep(0.1)
finally:
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
```

### 2. Missing Connection Cleanup
**Problem**: Tests connect to resources but don't disconnect in finally blocks
**Example**:
```python
# BAD - No cleanup on failure
await executor.connect()
# test code that might fail
await executor.disconnect()

# GOOD - Guaranteed cleanup
await executor.connect()
try:
    # test code
finally:
    await executor.disconnect()
```

### 3. Event Loop Warnings
**Problem**: Tests don't properly close event loops or clean up pending tasks
**Solution**: Use pytest-asyncio fixtures properly and ensure all tasks complete

### 4. WebSocket Cleanup
**Problem**: WebSocket connections not properly closed
**Solution**: Always close WebSocket connections in finally blocks

## Implementation Strategy

### Step 1: Create Async Test Fixtures
```python
@pytest.fixture
async def connected_executor(paper_executor):
    """Fixture that ensures executor cleanup."""
    await paper_executor.connect()
    try:
        yield paper_executor
    finally:
        await paper_executor.disconnect()
```

### Step 2: Add Task Tracking
```python
@pytest.fixture
async def task_tracker():
    """Track and cleanup all tasks created during test."""
    tasks = []
    
    def track_task(task):
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
```

### Step 3: Fix Individual Test Files
1. test_ibkr_websocket_reconnection.py - Fix task cleanup
2. test_live_engine.py - Add connection cleanup
3. test_executor.py - Use async fixtures
4. test_data_handler*.py - Fix event loop handling
5. test_order_manager*.py - Add proper task cancellation

## Files to Update

### High Priority (Critical Trading Paths)
- tests/unit/adapters/test_ibkr_websocket_reconnection.py
- tests/unit/core/test_live_engine.py
- tests/unit/core/test_executor.py
- tests/unit/core/engine/test_order_manager.py
- tests/unit/core/engine/test_execution_handler*.py

### Medium Priority (Integration Tests)
- tests/integration/test_component_interactions.py
- tests/e2e/test_live_trading_simulation.py

### Low Priority (Already Working)
- tests/unit/api/test_api.py (uses proper test client cleanup)
- tests/unit/core/test_portfolio*.py (synchronous tests)