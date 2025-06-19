"""Integration tests for AlgoStack system."""

from datetime import datetime

import numpy as np
import pytest
from core.portfolio import PortfolioEngine
from core.risk import EnhancedRiskManager
from strategies.base import Signal
from strategies.mean_reversion_equity import MeanReversionEquity
from strategies.trend_following_multi import TrendFollowingMulti


@pytest.mark.integration
class TestSystemIntegration:
    """Test integration between components."""

    @pytest.fixture
    def system_config(self):
        """Complete system configuration."""
        return {
            "initial_capital": 10000.0,
            "target_vol": 0.10,
            "max_position_size": 0.20,
            "max_drawdown": 0.15,
            "strategies": {
                "mean_reversion": {
                    "symbols": ["SPY", "QQQ"],
                    "lookback_period": 20,
                    "zscore_threshold": 2.0,
                    "exit_zscore": 0.5,
                    "rsi_period": 2,
                    "rsi_oversold": 10.0,
                    "rsi_overbought": 90.0,
                },
                "trend_following": {
                    "symbols": ["SPY", "QQQ"],
                    "channel_period": 20,
                    "adx_threshold": 25.0,
                },
            },
        }

    @pytest.fixture
    def portfolio_engine(self, system_config):
        """Create portfolio engine."""
        return PortfolioEngine(system_config)

    @pytest.fixture
    def mock_signals(self):
        """Create mock signals for testing."""
        return [
            Signal(
                symbol="SPY",
                direction="LONG",
                strength=0.8,
                confidence=0.9,
                timestamp=datetime.now(),
                strategy_id="test_strategy",
                price=400.0
            ),
            Signal(
                symbol="QQQ",
                direction="SHORT",
                strength=-0.6,
                confidence=0.7,
                timestamp=datetime.now(),
                strategy_id="test_strategy",
                price=350.0
            ),
            Signal(
                symbol="SPY",
                direction="FLAT",
                strength=0.0,
                confidence=0.5,
                timestamp=datetime.now(),
                strategy_id="test_strategy",
                price=401.0
            )
        ]

    @pytest.fixture
    def risk_manager(self, system_config):
        """Create risk manager."""
        return EnhancedRiskManager(system_config)

    @pytest.fixture
    def strategies(self, system_config):
        """Create strategy instances."""
        return {
            "mean_reversion": MeanReversionEquity(
                system_config["strategies"]["mean_reversion"]
            ),
            "trend_following": TrendFollowingMulti(
                system_config["strategies"]["trend_following"]
            ),
        }

    def test_signal_generation_to_execution(
        self, portfolio_engine, strategies, mock_market_data
    ):
        """Test signal generation through execution."""
        # Initialize strategies
        for strategy in strategies.values():
            strategy.init()

        # Generate signals
        signals = []
        for symbol, data in mock_market_data.items():
            if symbol in ["SPY", "QQQ"]:
                for strategy in strategies.values():
                    signal = strategy.next(data)
                    if signal:
                        signals.append(signal)

        # Allocate capital
        if signals:
            allocations = portfolio_engine.allocate_capital(signals, mock_market_data)

            # Execute signals
            for signal in signals:
                if signal.strategy_id in allocations:
                    allocation = allocations[signal.strategy_id]
                    position_size, stop_loss = portfolio_engine.size_position(
                        signal, allocation
                    )

                    if position_size > 0:
                        position = portfolio_engine.execute_signal(
                            signal, position_size, stop_loss
                        )
                        assert position is not None

        # Verify portfolio state
        metrics = portfolio_engine.calculate_portfolio_metrics()
        assert metrics["total_equity"] == portfolio_engine.initial_capital

    def test_risk_integration(self, portfolio_engine, risk_manager, mock_signals):
        """Test risk manager integration with portfolio."""
        # Execute some signals
        for signal in mock_signals:
            position_size, stop_loss = portfolio_engine.size_position(signal, 0.1)
            portfolio_engine.execute_signal(signal, position_size, stop_loss)

        # Update risk state
        risk_manager.update_risk_state(portfolio_engine)

        # Check risk compliance
        is_compliant, violations = risk_manager.check_risk_compliance(portfolio_engine)

        # Generate risk report
        report = risk_manager.get_risk_report()
        assert "risk_state" in report
        assert "metrics" in report

    def test_stop_loss_monitoring(self, portfolio_engine):
        """Test stop loss monitoring and execution."""
        # Create a position with stop loss
        signal = Signal(
            timestamp=datetime.now(),
            symbol="SPY",
            direction="LONG",
            strength=0.8,
            strategy_id="test",
            price=100.0,
            atr=2.0,
        )

        position_size = 100
        stop_loss = 95.0

        portfolio_engine.execute_signal(signal, position_size, stop_loss)

        # Simulate price drop below stop
        current_prices = {"SPY": 94.0}
        exit_signals = portfolio_engine.check_stops_and_targets(current_prices)

        assert len(exit_signals) == 1
        assert exit_signals[0].direction == "FLAT"
        assert exit_signals[0].metadata["reason"] == "stop_loss"

    def test_global_risk_shutdown(self, portfolio_engine):
        """Test global risk shutdown on drawdown breach."""
        # Simulate large drawdown
        portfolio_engine.peak_equity = 10000
        portfolio_engine.current_equity = 8000  # 20% drawdown
        portfolio_engine.current_drawdown = 0.20

        # Add some positions
        for symbol in ["SPY", "QQQ"]:
            position = type(
                "Position",
                (),
                {
                    "symbol": symbol,
                    "strategy_id": "test",
                    "direction": "LONG",
                    "current_price": 100,
                    "market_value": 1000,
                },
            )
            portfolio_engine.positions[symbol] = position

        # Check global risk
        is_ok, exit_signals = portfolio_engine.global_risk_check()

        assert not is_ok
        assert len(exit_signals) == 2  # Exit all positions
        assert portfolio_engine.is_risk_off
        assert portfolio_engine.risk_off_until is not None

    def test_multi_strategy_coordination(
        self, portfolio_engine, strategies, mock_market_data
    ):
        """Test multiple strategies working together."""
        # Initialize all strategies
        for strategy in strategies.values():
            strategy.init()

        # Simulate 10 days of trading
        for day in range(10):
            all_signals = []

            # Collect signals from all strategies
            for symbol, data in mock_market_data.items():
                if symbol in ["SPY", "QQQ"]:
                    # Use last N days of data
                    data_slice = data.iloc[: 100 + day]

                    for strategy in strategies.values():
                        signal = strategy.next(data_slice)
                        if signal:
                            all_signals.append(signal)

            # Process signals through portfolio
            if all_signals:
                allocations = portfolio_engine.allocate_capital(
                    all_signals, mock_market_data
                )

                for signal in all_signals:
                    if signal.direction == "FLAT":
                        # Close position
                        if signal.symbol in portfolio_engine.positions:
                            portfolio_engine.close_position(signal.symbol, signal.price)
                    else:
                        # Open/adjust position
                        if signal.strategy_id in allocations:
                            allocation = allocations[signal.strategy_id]
                            position_size, stop_loss = portfolio_engine.size_position(
                                signal, allocation
                            )
                            portfolio_engine.execute_signal(
                                signal, position_size, stop_loss
                            )

            # Update Kelly fractions periodically
            if day % 5 == 0:
                portfolio_engine.update_strategy_kelly_fractions()

        # Verify final state
        summary = portfolio_engine.get_portfolio_summary()
        assert "portfolio_metrics" in summary
        assert "strategy_exposure" in summary
        assert "risk_status" in summary

    @pytest.mark.slow
    def test_full_system_stress_test(self, portfolio_engine, risk_manager, strategies):
        """Stress test the full system with many signals."""
        np.random.seed(42)

        # Generate many random signals
        symbols = ["SPY", "QQQ", "IWM", "DIA"]

        for _ in range(100):
            # Random signal
            signal = Signal(
                timestamp=datetime.now(),
                symbol=np.random.choice(symbols),
                direction=np.random.choice(["LONG", "SHORT", "FLAT"]),
                strength=np.random.uniform(-1, 1),
                strategy_id=np.random.choice(list(strategies.keys())),
                price=np.random.uniform(50, 500),
                atr=np.random.uniform(1, 10),
            )

            # Process signal
            if (
                signal.direction == "FLAT"
                and signal.symbol in portfolio_engine.positions
            ):
                portfolio_engine.close_position(signal.symbol, signal.price)
            elif signal.direction != "FLAT":
                position_size, stop_loss = portfolio_engine.size_position(signal, 0.05)
                if position_size > 0:
                    portfolio_engine.execute_signal(signal, position_size, stop_loss)

            # Random price updates
            if np.random.random() > 0.5:
                current_prices = {
                    pos.symbol: pos.current_price * np.random.uniform(0.95, 1.05)
                    for pos in portfolio_engine.positions.values()
                }
                portfolio_engine.update_market_prices(current_prices)

                # Check stops
                exit_signals = portfolio_engine.check_stops_and_targets(current_prices)
                for exit_signal in exit_signals:
                    portfolio_engine.close_position(
                        exit_signal.symbol, exit_signal.price
                    )

            # Risk checks
            is_ok, _ = portfolio_engine.global_risk_check()
            if not is_ok:
                break

        # Verify system stability
        metrics = portfolio_engine.calculate_portfolio_metrics()
        assert metrics["total_equity"] > 0
        assert abs(metrics["current_drawdown"]) < 1.0  # Not 100% loss
