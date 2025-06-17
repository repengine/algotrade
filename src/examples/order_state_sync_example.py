"""
Example: Order State Synchronization

Demonstrates how the order state synchronization prevents duplicate orders
and detects missed fills, ensuring consistent state between local and broker systems.
"""

import asyncio
import logging

from adapters.paper_executor import PaperExecutor
from core.engine.enhanced_order_manager import EnhancedOrderManager
from core.executor import OrderSide, OrderType
from core.risk import EnhancedRiskManager
from utils.logging import setup_logger

# Set up logging
setup_logger(level=logging.INFO)
logger = logging.getLogger(__name__)


async def order_sync_demo():
    """Demonstrate order state synchronization."""

    # Initialize components
    risk_manager = EnhancedRiskManager({
        "max_position_size": 0.1,
        "max_portfolio_risk": 0.02,
        "max_correlation": 0.7,
    })

    # Create order manager with synchronization config
    sync_config = {
        "sync_interval": 2.0,  # Sync every 2 seconds
        "duplicate_window": 3.0,  # 3 second duplicate detection window
        "enable_auto_resolve": True,
        "stale_order_hours": 24,
    }

    order_manager = EnhancedOrderManager(
        risk_manager=risk_manager,
        sync_config=sync_config
    )

    # Add paper executor
    paper_executor = PaperExecutor({
        "initial_capital": 100000,
        "commission": 1.0,
        "slippage": 0.0001,
    })

    await paper_executor.connect()
    order_manager.add_executor("paper", paper_executor)

    # Initialize synchronization
    await order_manager.initialize_synchronization()

    logger.info("=" * 60)
    logger.info("Order State Synchronization Demo")
    logger.info("=" * 60)

    # Demo 1: Duplicate order prevention
    logger.info("\n1. Testing duplicate order prevention...")

    # Create first order
    order1 = await order_manager.create_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy"
    )

    # Submit first order
    success = await order_manager.submit_order(order1)
    logger.info(f"First order submitted: {success}")

    # Try to submit duplicate order immediately
    order2 = await order_manager.create_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy"
    )

    success = await order_manager.submit_order(order2)
    logger.info(f"Duplicate order submitted: {success} (should be False)")

    # Wait for duplicate window to pass
    await asyncio.sleep(3.5)

    # Now it should work
    order3 = await order_manager.create_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy"
    )

    success = await order_manager.submit_order(order3)
    logger.info(f"Order after window submitted: {success} (should be True)")

    # Demo 2: Synchronization metrics
    logger.info("\n2. Checking synchronization metrics...")

    # Wait for a sync cycle
    await asyncio.sleep(2.5)

    metrics = order_manager.get_sync_metrics()
    if metrics:
        logger.info("Synchronization Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    # Demo 3: Simulate missed fill (manually update executor state)
    logger.info("\n3. Simulating missed fill...")

    # Create and submit an order
    order4 = await order_manager.create_order(
        symbol="TSLA",
        side=OrderSide.BUY,
        quantity=50,
        order_type=OrderType.LIMIT,
        limit_price=200.0,
        strategy_id="test_strategy"
    )

    await order_manager.submit_order(order4)

    # Manually fill the order in the executor (simulating broker fill)
    # that our local system doesn't know about
    if hasattr(paper_executor, '_orders'):
        broker_order = paper_executor._orders.get(order4.order_id)
        if broker_order:
            # Simulate a partial fill that wasn't reported
            broker_order.filled_quantity = 25
            broker_order.average_fill_price = 199.95
            broker_order.status = paper_executor.OrderStatus.PARTIALLY_FILLED
            logger.info("Simulated broker-side fill without notification")

    # Wait for sync to detect the missed fill
    await asyncio.sleep(2.5)

    # Check if the fill was detected
    local_order = order_manager.get_order(order4.order_id)
    if local_order:
        logger.info(f"Local order filled quantity: {local_order.filled_quantity}")
        logger.info(f"Fill detection working: {local_order.filled_quantity > 0}")

    # Demo 4: Order statistics
    logger.info("\n4. Order Statistics:")
    stats = order_manager.get_order_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Final sync metrics
    await asyncio.sleep(2.5)
    logger.info("\n5. Final Synchronization Metrics:")
    final_metrics = order_manager.get_sync_metrics()
    if final_metrics:
        logger.info(f"  Total syncs: {final_metrics['total_syncs']}")
        logger.info(f"  Success rate: {final_metrics['success_rate']:.2%}")
        logger.info(f"  Duplicates prevented: {final_metrics['duplicate_orders_prevented']}")
        logger.info(f"  Missed fills detected: {final_metrics['missed_fills_detected']}")
        logger.info(f"  Average sync duration: {final_metrics['average_sync_duration_ms']:.2f}ms")

    # Cleanup
    await order_manager.stop_synchronization()
    await paper_executor.disconnect()

    logger.info("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(order_sync_demo())
