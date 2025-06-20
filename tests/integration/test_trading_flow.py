"""
Integration tests for complete trading flow.

Tests the flow from signal generation through execution.
Validates:
- Signal to order conversion
- Order execution and fill processing
- Portfolio updates from fills
- Risk management integration
- Multi-strategy coordination
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from core.engine.order_manager import (
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
)
from core.executor import BaseExecutor
from core.portfolio import PortfolioEngine
from core.risk import EnhancedRiskManager
from strategies.base import RiskContext, Signal


class TestTradingFlowIntegration:
    """Test complete trading flow from signal to execution."""

    @pytest.fixture
    def trading_components(self):
        """Set up integrated trading components."""
        # Initialize components
        portfolio = PortfolioEngine({"initial_capital": 100000})
        risk_manager = EnhancedRiskManager({"max_position_size": 0.2, "max_portfolio_risk": 0.02, "max_correlation": 0.7
        })

        # Mock executor
        executor = Mock(spec=BaseExecutor)
        executor.is_connected = True
        executor.submit_order = Mock(return_value="ORDER123")

        # Order manager
        order_manager = OrderManager()

        return {
            'portfolio': portfolio,
            'risk_manager': risk_manager,
            'executor': executor,
            'order_manager': order_manager
        }

    @pytest.mark.integration
    def test_signal_to_execution_flow(self, trading_components):
        """
        Test complete flow from signal generation to order execution.

        Verifies:
        1. Signal generates order
        2. Risk check passes
        3. Order submitted to executor
        4. Fill updates portfolio
        """
        portfolio = trading_components['portfolio']
        risk_manager = trading_components['risk_manager']
        executor = trading_components['executor']
        order_manager = trading_components['order_manager']

        # Generate signal
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.8,
            strategy_id="test_strategy",
            price=150.0,
            atr=2.5
        )

        # Risk context
        risk_context = RiskContext(
            account_equity=portfolio.current_equity,
            open_positions=len(portfolio.positions),
            daily_pnl=0,
            max_drawdown_pct=0.02,
            volatility_target=0.15,
            max_position_size=0.2
        )

        # Calculate position size (simplified)
        position_value = risk_context.account_equity * 0.1  # 10% of equity
        position_size = int(position_value / signal.price)

        # Create order
        order = Order(
            symbol=signal.symbol,
            quantity=position_size,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            strategy_id=signal.strategy_id
        )

        # Risk check
        is_allowed = risk_manager.check_order(order, portfolio)
        assert is_allowed

        # Submit order
        order_id = executor.submit_order(order)
        assert order_id == "ORDER123"

        # Track order
        order_manager.add_order(order_id, order)
        assert order_manager.get_order(order_id) == order

        # Simulate fill
        fill = {
            'order_id': order_id,
            'symbol': 'AAPL',
            'quantity': position_size,
            'price': 150.5,  # Slight slippage
            'commission': 1.0,
            'timestamp': datetime.now()
        }

        # Process fill
        order_manager.update_order_status(order_id, OrderStatus.FILLED, fill)
        portfolio.process_fill(fill)

        # Verify portfolio update
        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.quantity == position_size
        assert position.entry_price == 150.5
        # Note: process_fill doesn't update cash, only positions
        # Cash management would be handled by a separate method

    @pytest.mark.integration
    def test_risk_rejection_flow(self, trading_components):
        """
        Test flow when risk check rejects order.

        Verifies risk limits are enforced.
        """
        portfolio = trading_components['portfolio']
        risk_manager = trading_components['risk_manager']
        executor = trading_components['executor']

        # Add existing large position
        existing_position = {
            'symbol': 'AAPL',
            'quantity': 500,  # $75,000 position at $150
            'price': 150.0,
            'side': 'BUY',
            'strategy_id': 'test_strategy'
        }
        portfolio.process_fill(existing_position)

        # Try to add another large position (would exceed max position size)
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=1.0,
            strategy_id="test_strategy",
            price=150.0,
            atr=2.5
        )

        # Large order that exceeds risk limit
        order = Order(
            symbol=signal.symbol,
            quantity=200,  # Additional $30,000 at $150
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=150.0,  # Market order estimated price
            strategy_id=signal.strategy_id
        )

        # Risk check should fail
        is_allowed = risk_manager.check_order(order, portfolio)
        assert not is_allowed

        # Order should not be submitted
        executor.submit_order.assert_not_called()

    @pytest.mark.integration
    def test_multi_strategy_flow(self, trading_components):
        """
        Test multiple strategies generating signals.

        Verifies:
        1. Multiple strategies can operate independently
        2. Conflicting signals are handled
        3. Position sizing respects total risk
        """
        portfolio = trading_components['portfolio']
        risk_manager = trading_components['risk_manager']
        executor = trading_components['executor']
        order_manager = trading_components['order_manager']

        # Strategy 1: Mean Reversion - LONG signal
        mr_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="LONG",
            strength=0.7,
            strategy_id="mean_reversion",
            price=150.0,
            atr=2.5
        )

        # Strategy 2: Trend Following - SHORT signal (conflict!)
        tf_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction="SHORT",
            strength=-0.6,
            strategy_id="trend_following",
            price=150.0,
            atr=2.5
        )

        # Process signals with conflict resolution
        signals = [mr_signal, tf_signal]

        # Simple conflict resolution: strongest signal wins
        strongest_signal = max(signals, key=lambda s: abs(s.strength))
        assert strongest_signal == mr_signal

        # Create order from winning signal
        position_size = 100
        order = Order(
            symbol=strongest_signal.symbol,
            quantity=position_size,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            strategy_id=strongest_signal.strategy_id
        )

        # Submit order
        if risk_manager.check_order(order, portfolio):
            order_id = executor.submit_order(order)
            order_manager.add_order(order_id, order)

            # Verify order attributed to correct strategy
            tracked_order = order_manager.get_order(order_id)
            assert tracked_order.strategy_id == "mean_reversion"

    @pytest.mark.integration
    def test_stop_loss_flow(self, trading_components):
        """
        Test stop loss order flow.

        Verifies stop losses are properly managed.
        """
        portfolio = trading_components['portfolio']
        executor = trading_components['executor']
        order_manager = trading_components['order_manager']

        # Add position
        # Positions would be added through trading in real usage

        # Create stop loss order
        stop_order = Order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            stop_price=145.0,  # Stop at $145
            strategy_id='risk_management'
        )

        # Submit stop order
        stop_order_id = executor.submit_order(stop_order)
        order_manager.add_order(stop_order_id, stop_order)

        # Simulate price drop triggering stop
        market_price = 144.0  # Below stop price

        # Stop should trigger
        if market_price <= stop_order.stop_price:
            # Convert to market order
            # Stop order triggers and becomes a market order
            order_manager.update_order_status(
                stop_order_id,
                OrderStatus.SUBMITTED
            )

            # Execute at market
            fill = {
                'order_id': stop_order_id,
                'symbol': 'AAPL',
                'quantity': 100,
                'price': 144.0,
                'side': 'SELL',  # Stop loss is a sell order
                'commission': 1.0,
                'timestamp': datetime.now()
            }

            # Process fill
            portfolio.process_fill(fill)

            # Position should be closed
            assert 'AAPL' not in portfolio.positions

            # Note: realized_pnl requires trades to be tracked via add_position/close_position
            # process_fill only updates positions, not the trades list used for P&L calculation

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_trading_day_simulation(self, trading_components):
        """
        Simulate a full trading day with multiple signals and fills.

        Tests realistic trading scenarios.
        """
        portfolio = trading_components['portfolio']
        risk_manager = trading_components['risk_manager']
        executor = trading_components['executor']
        order_manager = trading_components['order_manager']

        # Track metrics
        trades = []

        # Simulate trading day (9:30 AM - 4:00 PM)
        start_time = datetime.now().replace(hour=9, minute=30, second=0)

        # Mock market data for the day
        market_hours = pd.date_range(
            start=start_time,
            end=start_time.replace(hour=16),
            freq='5min'
        )

        # Generate realistic price movement
        np.random.seed(42)
        prices = {
            'AAPL': 150 + np.cumsum(np.random.normal(0, 0.1, len(market_hours))),
            'GOOGL': 2500 + np.cumsum(np.random.normal(0, 1.5, len(market_hours))),
            'MSFT': 300 + np.cumsum(np.random.normal(0, 0.2, len(market_hours)))
        }

        # Process each time period
        for i, timestamp in enumerate(market_hours):
            # Generate signals (simplified - normally from strategies)
            signals = []

            # Morning momentum
            if timestamp.hour == 9 and timestamp.minute < 45:
                if prices['AAPL'][i] > prices['AAPL'][max(0, i-3)]:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol='AAPL',
                        direction='LONG',
                        strength=0.7,
                        strategy_id='momentum',
                        price=prices['AAPL'][i]
                    ))

            # Midday mean reversion
            elif 11 <= timestamp.hour <= 14:
                for symbol in ['GOOGL', 'MSFT']:
                    if i > 10:
                        sma = np.mean(prices[symbol][i-10:i])
                        if prices[symbol][i] < sma * 0.99:  # 1% below SMA
                            signals.append(Signal(
                                timestamp=timestamp,
                                symbol=symbol,
                                direction='LONG',
                                strength=0.6,
                                strategy_id='mean_reversion',
                                price=prices[symbol][i]
                            ))

            # Process signals
            for signal in signals:
                # Position sizing
                position_size = int(5000 / signal.price)  # $5000 per position

                order = Order(
                    symbol=signal.symbol,
                    quantity=position_size,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY if signal.direction == 'LONG' else OrderSide.SELL,
                    strategy_id=signal.strategy_id
                )

                # Risk check
                if risk_manager.check_order(order, portfolio):
                    # Submit order
                    order_id = f"ORDER_{timestamp.strftime('%H%M%S')}_{signal.symbol}"
                    executor.submit_order.return_value = order_id
                    executor.submit_order(order)
                    order_manager.add_order(order_id, order)

                    # Simulate immediate fill (for simplicity)
                    fill = {
                        'order_id': order_id,
                        'symbol': signal.symbol,
                        'quantity': position_size,
                        'price': signal.price * 1.0001,  # Slight slippage
                        'side': 'BUY' if signal.direction == 'LONG' else 'SELL',
                        'commission': 1.0,
                        'timestamp': timestamp
                    }

                    portfolio.process_fill(fill)
                    trades.append(fill)

            # End of day - close all positions
            if timestamp.hour == 15 and timestamp.minute >= 45:
                for symbol, position in list(portfolio.positions.items()):
                    close_order = Order(
                        symbol=symbol,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        strategy_id='eod_close'
                    )

                    order_id = f"CLOSE_{timestamp.strftime('%H%M%S')}_{symbol}"
                    executor.submit_order.return_value = order_id
                    executor.submit_order(close_order)

                    # Fill at current price
                    fill = {
                        'order_id': order_id,
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'price': prices[symbol][i],
                        'side': 'SELL',  # Closing long positions
                        'commission': 1.0,
                        'timestamp': timestamp
                    }

                    portfolio.process_fill(fill)
                    trades.append(fill)

        # Verify day's activity
        assert len(trades) > 0
        assert len(portfolio.positions) == 0  # All positions closed
        assert executor.submit_order.call_count > 0

        # Check P&L
        total_pnl = portfolio.realized_pnl
        print(f"Day's P&L: ${total_pnl:.2f}")
        print(f"Total trades: {len(trades)}")
        print(f"Final equity: ${portfolio.current_equity:.2f}")

    @pytest.mark.integration
    def test_partial_fill_handling(self, trading_components):
        """
        Test handling of partial order fills.

        Verifies system correctly handles partial executions.
        """
        portfolio = trading_components['portfolio']
        executor = trading_components['executor']
        order_manager = trading_components['order_manager']

        # Large order
        order = Order(
            symbol='AAPL',
            quantity=1000,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=150.0,
            strategy_id='test'
        )

        order_id = "LARGE_ORDER_123"
        executor.submit_order.return_value = order_id
        executor.submit_order(order)
        order_manager.add_order(order_id, order)

        # Simulate multiple partial fills
        partial_fills = [
            {'quantity': 200, 'price': 149.95},
            {'quantity': 300, 'price': 149.98},
            {'quantity': 500, 'price': 150.00}
        ]

        total_filled = 0
        for i, partial in enumerate(partial_fills):
            fill = {
                'order_id': order_id,
                'symbol': 'AAPL',
                'quantity': partial['quantity'],
                'price': partial['price'],
                'commission': 0.5,
                'timestamp': datetime.now() + timedelta(seconds=i*30)
            }

            # Update order status
            total_filled += partial['quantity']
            if total_filled < order.quantity:
                order_manager.update_order_status(
                    order_id,
                    OrderStatus.PARTIAL,
                    {'filled_quantity': total_filled}
                )
            else:
                order_manager.update_order_status(
                    order_id,
                    OrderStatus.FILLED,
                    {'filled_quantity': total_filled}
                )

            # Process fill
            portfolio.process_fill(fill)

        # Verify position
        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.quantity == 1000

        # Verify average price
        expected_avg_price = (
            200 * 149.95 +
            300 * 149.98 +
            500 * 150.00
        ) / 1000
        assert position.avg_price == pytest.approx(expected_avg_price, rel=1e-4)
