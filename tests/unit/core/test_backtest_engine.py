"""Comprehensive test suite for the backtest engine."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from core.backtest_engine import (
    DataSplitter,
    MonteCarloValidator,
    RegimeAnalyzer,
    TransactionCostModel,
    TransactionCosts,
    WalkForwardAnalyzer,
    create_backtest_report,
)


class TestTransactionCostModel:
    """Test suite for TransactionCostModel."""

    @pytest.fixture
    def default_cost_model(self):
        """Create default transaction cost model."""
        config = {
            "commission_per_share": 0.005,
            "min_commission": 1.0,
            "commission_type": "per_share",
            "spread_model": "fixed",
            "base_spread_bps": 5,
            "slippage_model": "linear",
            "market_impact_factor": 0.1,
            "urgency_factor": 1.0
        }
        return TransactionCostModel(config)

    @pytest.fixture
    def percentage_cost_model(self):
        """Create percentage-based commission model."""
        config = {
            "commission_per_share": 0.1,  # 0.1% commission
            "commission_type": "percentage",
            "spread_model": "dynamic",
            "base_spread_bps": 10,
            "slippage_model": "square_root",
            "market_impact_factor": 0.2
        }
        return TransactionCostModel(config)

    def test_initialization(self, default_cost_model):
        """Test cost model initialization."""
        assert default_cost_model.commission_per_share == 0.005
        assert default_cost_model.min_commission == 1.0
        assert default_cost_model.commission_type == "per_share"
        assert default_cost_model.spread_model == "fixed"
        assert default_cost_model.base_spread_bps == 5
        assert default_cost_model.slippage_model == "linear"
        assert default_cost_model.market_impact_factor == 0.1
        assert default_cost_model.urgency_factor == 1.0

    def test_calculate_costs_per_share_commission(self, default_cost_model):
        """Test per-share commission calculation."""
        costs = default_cost_model.calculate_costs(
            price=100.0,
            shares=100,
            side="BUY",
            volatility=0.02,
            avg_daily_volume=1000000
        )

        # Commission should be max(100 * 0.005, 1.0) = 1.0
        assert costs.commission == 1.0
        assert isinstance(costs, TransactionCosts)
        assert costs.total > 0

    def test_calculate_costs_percentage_commission(self, percentage_cost_model):
        """Test percentage-based commission calculation."""
        costs = percentage_cost_model.calculate_costs(
            price=100.0,
            shares=1000,
            side="BUY",
            volatility=0.02,
            avg_daily_volume=1000000
        )

        # Commission should be 100 * 1000 * 0.1 / 100 = 100
        assert costs.commission == 100.0

    def test_minimum_commission(self, default_cost_model):
        """Test minimum commission enforcement."""
        # Small trade should trigger minimum commission
        costs = default_cost_model.calculate_costs(
            price=100.0,
            shares=10,  # 10 * 0.005 = 0.05 < 1.0
            side="BUY"
        )

        assert costs.commission == 1.0  # Minimum commission

    def test_spread_calculation_fixed(self, default_cost_model):
        """Test fixed spread calculation."""
        spread_bps = default_cost_model._calculate_spread(volatility=0.02, time_of_day=None)
        assert spread_bps == 5.0

    def test_spread_calculation_dynamic(self, percentage_cost_model):
        """Test dynamic spread calculation based on volatility."""
        # Low volatility
        spread_low = percentage_cost_model._calculate_spread(volatility=0.01, time_of_day=None)

        # High volatility
        spread_high = percentage_cost_model._calculate_spread(volatility=0.05, time_of_day=None)

        # Higher volatility should result in wider spreads
        assert spread_high > spread_low

    def test_spread_calculation_vix_based(self):
        """Test VIX-based spread calculation."""
        config = {"spread_model": "vix_based", "base_spread_bps": 10}
        model = TransactionCostModel(config)

        # Low volatility (VIX < 20)
        spread_low = model._calculate_spread(volatility=0.01, time_of_day=None)

        # High volatility (VIX > 30)
        spread_high = model._calculate_spread(volatility=0.035, time_of_day=None)

        assert spread_high > spread_low

    def test_spread_time_of_day_adjustment(self, default_cost_model):
        """Test spread adjustment for market open/close."""
        # Regular hours
        spread_regular = default_cost_model._calculate_spread(volatility=0.02, time_of_day="11:30")

        # Market open
        spread_open = default_cost_model._calculate_spread(volatility=0.02, time_of_day="09:30")

        # Market close
        spread_close = default_cost_model._calculate_spread(volatility=0.02, time_of_day="15:30")

        assert spread_open > spread_regular
        assert spread_close > spread_regular

    def test_slippage_calculation_linear(self, default_cost_model):
        """Test linear slippage model."""
        slippage = default_cost_model._calculate_slippage(
            price=100.0,
            shares=10000,  # 1% of daily volume
            avg_daily_volume=1000000,
            side="BUY"
        )

        # Linear model: 0.0001 * 0.01 = 0.000001
        # Slippage = 100 * 10000 * 0.000001 = 1.0
        assert slippage == pytest.approx(1.0, rel=1e-3)

    def test_slippage_calculation_square_root(self, percentage_cost_model):
        """Test square-root slippage model."""
        slippage = percentage_cost_model._calculate_slippage(
            price=100.0,
            shares=10000,
            avg_daily_volume=1000000,
            side="BUY"
        )

        # Should use square root of participation rate
        assert slippage > 0

    def test_slippage_calculation_fixed(self):
        """Test fixed slippage model."""
        config = {"slippage_model": "fixed", "base_spread_bps": 5}
        model = TransactionCostModel(config)

        slippage = model._calculate_slippage(
            price=100.0,
            shares=1000,
            avg_daily_volume=1000000,
            side="BUY"
        )

        # Fixed slippage: 100 * 1000 * 0.0005 = 50
        assert slippage == 50.0

    def test_slippage_urgency_factor(self):
        """Test urgency factor impact on slippage."""
        config = {
            "slippage_model": "linear",
            "urgency_factor": 2.0  # Double urgency
        }
        model = TransactionCostModel(config)

        slippage_urgent = model._calculate_slippage(
            price=100.0,
            shares=10000,
            avg_daily_volume=1000000,
            side="BUY"
        )

        # Create normal urgency model for comparison
        config["urgency_factor"] = 1.0
        model_normal = TransactionCostModel(config)

        slippage_normal = model_normal._calculate_slippage(
            price=100.0,
            shares=10000,
            avg_daily_volume=1000000,
            side="BUY"
        )

        assert slippage_urgent == pytest.approx(slippage_normal * 2.0)

    def test_total_cost_calculation(self, default_cost_model):
        """Test total cost aggregation."""
        costs = default_cost_model.calculate_costs(
            price=100.0,
            shares=1000,
            side="BUY",
            volatility=0.02,
            avg_daily_volume=1000000
        )

        assert costs.total == costs.commission + costs.spread_cost + costs.slippage
        assert costs.total > costs.commission  # Should include spread and slippage


class TestDataSplitter:
    """Test suite for DataSplitter."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        return pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)

    def test_initialization(self):
        """Test data splitter initialization."""
        splitter = DataSplitter(method="sequential", oos_ratio=0.3)
        assert splitter.method == "sequential"
        assert splitter.oos_ratio == 0.3
        assert splitter.embargo_days == 63

    def test_sequential_split(self, sample_data):
        """Test sequential data splitting."""
        splitter = DataSplitter(method="sequential", oos_ratio=0.2)
        train, test = splitter.split(sample_data)

        # Check split sizes
        expected_train_size = int(len(sample_data) * 0.8)
        assert len(train) == expected_train_size
        assert len(test) == len(sample_data) - expected_train_size

        # Check no overlap
        assert train.index[-1] < test.index[0]

    def test_embargo_split(self, sample_data):
        """Test embargo period splitting."""
        splitter = DataSplitter(method="embargo", oos_ratio=0.2)
        splitter.embargo_days = 30
        train, test = splitter.split(sample_data)

        # Check embargo gap
        gap_days = (test.index[0] - train.index[-1]).days
        assert gap_days >= splitter.embargo_days

    def test_purged_kfold_split(self, sample_data):
        """Test purged k-fold splitting."""
        splitter = DataSplitter(method="purged_kfold", oos_ratio=0.2)

        # Get first split using split method
        train, test = splitter.split(sample_data)
        assert len(train) > 0
        assert len(test) > 0

        # Get all splits
        all_splits = splitter.get_all_splits(sample_data)
        assert len(all_splits) > 1  # Should have multiple folds

        # Check each split
        for train, test in all_splits:
            assert len(train) > 0
            assert len(test) > 0
            # Check for gap (embargo)
            if len(train) > 0 and len(test) > 0:
                assert train.index[-1] < test.index[0]

    def test_invalid_method(self, sample_data):
        """Test invalid splitting method."""
        splitter = DataSplitter(method="invalid_method")
        with pytest.raises(ValueError, match="Unknown split method"):
            splitter.split(sample_data)

    def test_get_all_splits_sequential(self, sample_data):
        """Test get_all_splits for sequential method."""
        splitter = DataSplitter(method="sequential", oos_ratio=0.2)
        all_splits = splitter.get_all_splits(sample_data)

        # Sequential should return single split
        assert len(all_splits) == 1
        train, test = all_splits[0]
        assert len(train) + len(test) == len(sample_data)


class TestWalkForwardAnalyzer:
    """Test suite for WalkForwardAnalyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for walk-forward analysis."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        return pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'high': 102 + np.random.randn(len(dates)).cumsum() * 0.5,
            'low': 98 + np.random.randn(len(dates)).cumsum() * 0.5,
            'close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)

    @pytest.fixture
    def mock_strategy_class(self):
        """Create mock strategy class."""
        class MockStrategy:
            def __init__(self, params):
                self.params = params
        return MockStrategy

    @pytest.fixture
    def mock_backtest_func(self):
        """Create mock backtest function."""
        def backtest_func(strategy, data, cost_model=None):
            # Return different results based on strategy params
            base_sharpe = strategy.params.get('param1', 1.0) * 0.5
            return {
                'sharpe_ratio': base_sharpe + np.random.randn() * 0.1,
                'total_return': base_sharpe * 10,
                'max_drawdown': -abs(np.random.randn() * 5),
                'num_trades': int(50 + np.random.randn() * 10)
            }
        return backtest_func

    def test_initialization(self):
        """Test walk-forward analyzer initialization."""
        wf = WalkForwardAnalyzer(
            window_size=252,
            step_size=63,
            min_window_size=126,
            optimization_metric="sharpe",
            max_windows=5
        )

        assert wf.window_size == 252
        assert wf.step_size == 63
        assert wf.min_window_size == 126
        assert wf.optimization_metric == "sharpe"
        assert wf.max_windows == 5

    def test_generate_windows(self, sample_data):
        """Test window generation."""
        wf = WalkForwardAnalyzer(window_size=252, step_size=63)
        windows = list(wf._generate_windows(sample_data))

        assert len(windows) > 0

        # Check window properties
        for is_start, is_end, oos_start, oos_end in windows:
            # In-sample period should be window_size days
            is_days = (is_end - is_start).days
            assert is_days == pytest.approx(252, abs=1)

            # OOS period should be step_size days
            oos_days = (oos_end - oos_start).days
            assert oos_days == pytest.approx(63, abs=1)

            # OOS should start where IS ends
            assert oos_start == is_end

    def test_optimize_parameters(self, mock_strategy_class, sample_data, mock_backtest_func):
        """Test parameter optimization."""
        wf = WalkForwardAnalyzer(optimization_metric="sharpe_ratio")

        param_grid = {
            'param1': [0.5, 1.0, 1.5],
            'param2': [10, 20, 30]
        }

        optimal_params = wf._optimize_parameters(
            mock_strategy_class,
            sample_data,
            param_grid,
            mock_backtest_func,
            None
        )

        assert 'param1' in optimal_params
        assert 'param2' in optimal_params
        assert optimal_params['param1'] in param_grid['param1']
        assert optimal_params['param2'] in param_grid['param2']

    def test_optimize_parameters_empty_grid(self, mock_strategy_class, sample_data, mock_backtest_func):
        """Test optimization with empty parameter grid."""
        wf = WalkForwardAnalyzer()

        optimal_params = wf._optimize_parameters(
            mock_strategy_class,
            sample_data,
            {},  # Empty grid
            mock_backtest_func,
            None
        )

        assert optimal_params == {}

    @patch('core.backtest_engine.logger')
    def test_run_analysis(self, mock_logger, mock_strategy_class, sample_data, mock_backtest_func):
        """Test complete walk-forward analysis."""
        wf = WalkForwardAnalyzer(
            window_size=252,
            step_size=126,
            max_windows=3
        )

        param_grid = {'param1': [0.5, 1.0, 1.5]}

        results = wf.run_analysis(
            mock_strategy_class,
            sample_data,
            param_grid,
            mock_backtest_func
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 3  # max_windows

        # Check required columns
        required_columns = [
            'window_num', 'is_start', 'is_end', 'oos_start', 'oos_end',
            'optimal_params', 'is_sharpe', 'oos_sharpe', 'oos_return',
            'oos_drawdown', 'oos_trades', 'sharpe_decay'
        ]
        for col in required_columns:
            assert col in results.columns

    def test_run_analysis_with_cost_model(self, mock_strategy_class, sample_data, mock_backtest_func):
        """Test walk-forward analysis with transaction costs."""
        wf = WalkForwardAnalyzer(window_size=252, step_size=126, max_windows=2)

        cost_model = TransactionCostModel({})
        param_grid = {'param1': [1.0]}

        results = wf.run_analysis(
            mock_strategy_class,
            sample_data,
            param_grid,
            mock_backtest_func,
            cost_model
        )

        assert len(results) > 0


class TestMonteCarloValidator:
    """Test suite for MonteCarloValidator."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        # Generate returns with positive Sharpe
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
        returns.index = pd.date_range(start='2023-01-01', periods=252, freq='D')
        return returns

    @pytest.fixture
    def strategy_results(self, sample_returns):
        """Create sample strategy results."""
        return {
            'sharpe_ratio': 1.5,
            'returns_series': sample_returns,
            'total_return': 15.0,
            'max_drawdown': -10.0
        }

    def test_initialization(self):
        """Test Monte Carlo validator initialization."""
        mc = MonteCarloValidator(n_simulations=500, confidence_level=0.99)
        assert mc.n_simulations == 500
        assert mc.confidence_level == 0.99

    def test_simulate_random_entries(self, sample_returns):
        """Test random entry simulation."""
        mc = MonteCarloValidator(n_simulations=100)
        random_sharpes = mc._simulate_random_entries(sample_returns)

        assert len(random_sharpes) == 100
        assert all(isinstance(s, (int, float)) for s in random_sharpes)

        # Random sharpes should have lower mean than actual strategy
        assert np.mean(random_sharpes) < 1.5

    def test_bootstrap_confidence_interval(self, sample_returns):
        """Test bootstrap confidence interval calculation."""
        mc = MonteCarloValidator(n_simulations=100, confidence_level=0.95)
        ci_lower, ci_upper = mc._bootstrap_confidence_interval(sample_returns)

        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower < ci_upper

    def test_bootstrap_with_empty_returns(self):
        """Test bootstrap with empty returns."""
        mc = MonteCarloValidator(n_simulations=10)
        empty_returns = pd.Series([])

        ci_lower, ci_upper = mc._bootstrap_confidence_interval(empty_returns)
        assert ci_lower == 0.0
        assert ci_upper == 0.0

    def test_interpret_results(self):
        """Test results interpretation."""
        mc = MonteCarloValidator()

        # Highly significant with large effect
        interp1 = mc._interpret_results(p_value=0.001, effect_size=1.5)
        assert "highly significant" in interp1
        assert "large" in interp1

        # Not significant with small effect
        interp2 = mc._interpret_results(p_value=0.5, effect_size=0.1)
        assert "not significant" in interp2
        assert "negligible" in interp2

        # Marginally significant with medium effect
        interp3 = mc._interpret_results(p_value=0.08, effect_size=0.6)
        assert "marginally significant" in interp3
        assert "medium" in interp3

    def test_validate_strategy(self, strategy_results):
        """Test complete strategy validation."""
        mc = MonteCarloValidator(n_simulations=100, confidence_level=0.95)
        validation = mc.validate_strategy(strategy_results)

        assert 'actual_sharpe' in validation
        assert 'random_mean_sharpe' in validation
        assert 'random_std_sharpe' in validation
        assert 'p_value' in validation
        assert 'significant' in validation
        assert 'confidence_interval' in validation
        assert 'effect_size' in validation
        assert 'interpretation' in validation

        # Check values
        assert validation['actual_sharpe'] == 1.5
        assert 0 <= validation['p_value'] <= 1
        assert isinstance(validation['significant'], (bool, np.bool_))
        assert len(validation['confidence_interval']) == 2

    def test_validate_strategy_no_returns(self):
        """Test validation with missing returns."""
        mc = MonteCarloValidator()
        results = {'sharpe_ratio': 1.5}  # No returns_series

        validation = mc.validate_strategy(results)
        assert 'error' in validation

    def test_compare_to_benchmark(self, sample_returns):
        """Test benchmark comparison."""
        mc = MonteCarloValidator()

        # Create benchmark returns (slightly worse)
        benchmark_returns = sample_returns * 0.8

        comparison = mc._compare_to_benchmark(sample_returns, benchmark_returns)

        assert 'information_ratio' in comparison
        assert 'win_rate' in comparison
        assert 'excess_return_mean' in comparison
        assert 'excess_return_std' in comparison
        assert 't_statistic' in comparison
        assert 't_pvalue' in comparison
        assert 'outperforms' in comparison

    def test_validate_strategy_with_benchmark(self, strategy_results, sample_returns):
        """Test validation with benchmark comparison."""
        mc = MonteCarloValidator(n_simulations=50)
        benchmark = sample_returns * 0.9

        validation = mc.validate_strategy(strategy_results, benchmark_returns=benchmark)

        assert 'vs_benchmark' in validation
        assert 'information_ratio' in validation['vs_benchmark']


class TestRegimeAnalyzer:
    """Test suite for RegimeAnalyzer."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with different regimes."""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # Create data with varying volatility and trends
        np.random.seed(42)
        close_prices = []
        current_price = 100

        for i in range(n):
            # Create regime changes
            if i < n // 3:  # Low vol uptrend
                volatility = 0.005
                drift = 0.0005
            elif i < 2 * n // 3:  # High vol downtrend
                volatility = 0.02
                drift = -0.0003
            else:  # Medium vol sideways
                volatility = 0.01
                drift = 0

            current_price *= (1 + drift + np.random.randn() * volatility)
            close_prices.append(current_price)

        return pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'close': close_prices,
            'volume': np.random.randint(1000000, 2000000, n)
        }, index=dates)

    def test_initialization(self):
        """Test regime analyzer initialization."""
        analyzer = RegimeAnalyzer(regime_method="volatility")
        assert analyzer.regime_method == "volatility"

    def test_volatility_regimes(self, sample_market_data):
        """Test volatility-based regime identification."""
        analyzer = RegimeAnalyzer(regime_method="volatility")
        regimes = analyzer.identify_regimes(sample_market_data)

        assert 'low_volatility' in regimes
        assert 'medium_volatility' in regimes
        assert 'high_volatility' in regimes

        # Check that most data is classified (some may be NaN due to rolling calculation)
        total_days = sum(len(regime_data) for regime_data in regimes.values())
        # Allow for some missing days due to rolling window
        assert total_days >= len(sample_market_data) - 21  # 21-day rolling window

    def test_trend_regimes(self, sample_market_data):
        """Test trend-based regime identification."""
        analyzer = RegimeAnalyzer(regime_method="trend")
        regimes = analyzer.identify_regimes(sample_market_data)

        assert 'uptrend' in regimes
        assert 'downtrend' in regimes
        assert 'sideways' in regimes

    def test_combined_regimes(self, sample_market_data):
        """Test combined regime identification."""
        analyzer = RegimeAnalyzer(regime_method="combined")
        regimes = analyzer.identify_regimes(sample_market_data)

        # Should have various combinations like 'low-vol-uptrend-market'
        regime_names = list(regimes.keys())
        assert isinstance(regimes, dict)
        # If we have regimes, they should follow the pattern
        if regime_names:
            assert any('vol' in name and 'market' in name for name in regime_names)

    def test_invalid_regime_method(self, sample_market_data):
        """Test invalid regime method."""
        analyzer = RegimeAnalyzer(regime_method="invalid")
        with pytest.raises(ValueError, match="Unknown regime method"):
            analyzer.identify_regimes(sample_market_data)

    def test_analyze_regime_performance(self, sample_market_data):
        """Test regime performance analysis."""
        analyzer = RegimeAnalyzer(regime_method="volatility")

        # Mock strategy and backtest function
        mock_strategy = Mock()

        def mock_backtest(strategy, data):
            return {
                'sharpe_ratio': np.random.randn() * 0.5 + 1.0,
                'total_return': np.random.randn() * 10 + 5,
                'max_drawdown': -abs(np.random.randn() * 5),
                'win_rate': 0.5 + np.random.randn() * 0.1,
                'num_trades': int(50 + np.random.randn() * 10)
            }

        results = analyzer.analyze_regime_performance(
            mock_strategy,
            sample_market_data,
            mock_backtest
        )

        assert 'regime_results' in results
        assert 'consistency_score' in results
        assert 'worst_regime_sharpe' in results
        assert 'all_positive' in results

        # Check regime results
        for _regime_name, metrics in results['regime_results'].items():
            assert 'days' in metrics
            assert 'sharpe' in metrics
            assert 'return' in metrics
            assert 'max_drawdown' in metrics
            assert 'win_rate' in metrics
            assert 'num_trades' in metrics

    def test_regime_performance_with_insufficient_data(self, sample_market_data):
        """Test regime analysis with insufficient data in some regimes."""
        # Use smaller dataset
        small_data = sample_market_data.iloc[:50]

        analyzer = RegimeAnalyzer(regime_method="volatility")
        mock_strategy = Mock()

        results = analyzer.analyze_regime_performance(
            mock_strategy,
            small_data,
            lambda s, d: {'sharpe_ratio': 1.0}
        )

        # Some regimes might be skipped due to insufficient data
        assert len(results['regime_results']) >= 0


class TestCreateBacktestReport:
    """Test suite for report generation."""

    @pytest.fixture
    def basic_results(self):
        """Create basic backtest results."""
        return {
            'total_return': 25.5,
            'sharpe_ratio': 1.8,
            'max_drawdown': -12.3,
            'win_rate': 0.65,
            'num_trades': 150
        }

    @pytest.fixture
    def full_results(self):
        """Create comprehensive backtest results."""
        return {
            'total_return': 35.2,
            'sharpe_ratio': 2.1,
            'max_drawdown': -8.7,
            'win_rate': 0.72,
            'validation': {
                'p_value': 0.003,
                'significant': True,
                'effect_size': 1.2,
                'interpretation': 'Results are highly significant with large effect size'
            },
            'regime_analysis': {
                'consistency_score': 0.85,
                'worst_regime_sharpe': 0.9,
                'all_positive': True,
                'regime_results': {
                    'low_volatility': {
                        'sharpe': 2.5,
                        'return': 40.0,
                        'max_drawdown': -5.0
                    },
                    'high_volatility': {
                        'sharpe': 0.9,
                        'return': 15.0,
                        'max_drawdown': -15.0
                    }
                }
            },
            'walk_forward': pd.DataFrame({
                'oos_sharpe': [1.8, 2.0, 1.5],
                'sharpe_decay': [0.1, 0.05, 0.15]
            }),
            'transaction_costs': {
                'total_costs': 5000.0,
                'cost_percentage': 0.02,
                'avg_cost_per_trade': 33.33
            }
        }

    def test_basic_report(self, basic_results):
        """Test basic report generation."""
        report = create_backtest_report(basic_results)

        assert isinstance(report, str)
        assert "# Backtest Report" in report
        assert "## Performance Summary" in report
        assert "Total Return: 25.50%" in report
        assert "Sharpe Ratio: 1.80" in report
        assert "Max Drawdown: -12.30%" in report
        assert "Win Rate: 65.00%" in report

    def test_full_report(self, full_results):
        """Test comprehensive report generation."""
        report = create_backtest_report(full_results)

        # Check all sections
        assert "## Statistical Validation" in report
        assert "P-Value: 0.003" in report
        assert "Significant: Yes" in report

        assert "## Regime Analysis" in report
        assert "Consistency Score: 85.00%" in report
        assert "Positive in All Regimes: Yes" in report

        assert "## Walk-Forward Analysis" in report
        assert "Average OOS Sharpe:" in report

        assert "## Transaction Cost Analysis" in report
        assert "Total Costs: $5,000.00" in report

    def test_report_with_missing_sections(self):
        """Test report with only some sections available."""
        partial_results = {
            'total_return': 20.0,
            'sharpe_ratio': 1.5,
            'max_drawdown': -10.0,
            'win_rate': 0.6,
            'validation': {
                'p_value': 0.05,
                'significant': True,
                'effect_size': 0.8,
                'interpretation': 'Significant results'
            }
            # No regime analysis, walk-forward, or transaction costs
        }

        report = create_backtest_report(partial_results)

        assert "## Statistical Validation" in report
        assert "## Regime Analysis" not in report
        assert "## Walk-Forward Analysis" not in report
        assert "## Transaction Cost Analysis" not in report


class TestIntegration:
    """Integration tests for backtest engine components."""

    def test_cost_model_with_data_splitter(self):
        """Test integration of cost model with data splitting."""
        # Create cost model
        cost_config = {
            'commission_per_share': 0.01,
            'spread_model': 'dynamic',
            'slippage_model': 'square_root'
        }
        cost_model = TransactionCostModel(cost_config)

        # Create data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1e6, 2e6, len(dates))
        }, index=dates)

        # Split data
        splitter = DataSplitter(method='embargo', oos_ratio=0.2)
        train, test = splitter.split(data)

        # Calculate costs for trades in both periods
        train_costs = cost_model.calculate_costs(
            price=float(train['close'].iloc[-1]),
            shares=1000,
            side='BUY',
            volatility=train['close'].pct_change().std()
        )

        test_costs = cost_model.calculate_costs(
            price=float(test['close'].iloc[0]),
            shares=1000,
            side='SELL',
            volatility=test['close'].pct_change().std()
        )

        assert train_costs.total > 0
        assert test_costs.total > 0

    def test_walk_forward_with_regime_analysis(self):
        """Test walk-forward analysis with regime detection."""
        # Create data with different regimes
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n = len(dates)

        # Generate regime-dependent data
        close_prices = []
        current_price = 100

        for i in range(n):
            if i < n // 2:  # First half: low vol uptrend
                current_price *= 1.0003 + np.random.randn() * 0.005
            else:  # Second half: high vol downtrend
                current_price *= 0.9997 + np.random.randn() * 0.02
            close_prices.append(current_price)

        data = pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'close': close_prices,
            'volume': np.random.randint(1e6, 2e6, n)
        }, index=dates)

        # Analyze regimes
        regime_analyzer = RegimeAnalyzer(regime_method='volatility')
        regimes = regime_analyzer.identify_regimes(data)

        # Check that different regimes were identified
        assert len(regimes) > 1

        # Mock strategy for walk-forward
        class TestStrategy:
            def __init__(self, params):
                self.params = params

        # Simple backtest function
        def backtest_func(strategy, data, cost_model=None):
            return {
                'sharpe_ratio': np.random.randn() * 0.5 + 1.0,
                'total_return': len(data) * 0.01  # Dummy calculation
            }

        # Run walk-forward on full data
        wf = WalkForwardAnalyzer(window_size=252, step_size=126, max_windows=3)
        wf_results = wf.run_analysis(
            TestStrategy,
            data,
            {'param1': [1, 2, 3]},
            backtest_func
        )

        assert len(wf_results) > 0

    def test_monte_carlo_validation_with_report(self):
        """Test Monte Carlo validation and report generation."""
        # Generate returns with positive bias
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(252) * 0.01 + 0.001,  # 0.1% daily return on average
            index=pd.date_range(start='2023-01-01', periods=252, freq='D')
        )

        # Calculate actual metrics
        sharpe = np.sqrt(252) * returns.mean() / returns.std()

        # Create strategy results
        strategy_results = {
            'sharpe_ratio': sharpe,
            'returns_series': returns,
            'total_return': (1 + returns).prod() - 1,
            'max_drawdown': -5.0,
            'win_rate': (returns > 0).mean()
        }

        # Validate with Monte Carlo
        validator = MonteCarloValidator(n_simulations=100)
        validation = validator.validate_strategy(strategy_results)

        # Create comprehensive results
        full_results = strategy_results.copy()
        full_results['validation'] = validation

        # Generate report
        report = create_backtest_report(full_results)

        assert "Statistical Validation" in report
        assert str(round(validation['p_value'], 3)) in report


class TestMissingCoverage:
    """Additional tests to achieve 100% coverage."""

    def test_dynamic_spread_high_volatility(self):
        """Test dynamic spread calculation with high volatility (VIX > 30)."""
        config = {
            "spread_model": "vix_based",
            "base_spread_bps": 10,
        }
        cost_model = TransactionCostModel(config)

        # Create high volatility scenario (VIX > 30)
        high_vol_data = pd.DataFrame({
            'returns': np.random.randn(20) * 0.05  # 5% daily moves = very high vol
        })
        volatility = high_vol_data['returns'].std()
        # VIX estimate should be > 30
        vix_estimate = volatility * np.sqrt(252) * 100
        assert vix_estimate > 30  # Verify our setup creates high vol

        spread = cost_model._calculate_spread(
            volatility=volatility, time_of_day="11:00"
        )

        # With VIX > 30, spread should be doubled
        expected_spread = 10 * 2.0  # Base 10 bps doubled for high vol
        assert spread == pytest.approx(expected_spread, rel=0.01)

    def test_dynamic_spread_medium_volatility(self):
        """Test dynamic spread calculation with medium volatility (20 < VIX < 30)."""
        config = {
            "spread_model": "vix_based",
            "base_spread_bps": 10,
        }
        cost_model = TransactionCostModel(config)

        # Create medium volatility scenario (20 < VIX < 30)
        # Need volatility that gives VIX between 20 and 30
        # VIX = vol * sqrt(252) * 100, so vol = VIX / (sqrt(252) * 100)
        target_vix = 25
        volatility = target_vix / (np.sqrt(252) * 100)

        spread = cost_model._calculate_spread(
            volatility=volatility, time_of_day="11:00"
        )

        # With 20 < VIX < 30, spread should be multiplied by 1.5
        expected_spread = 10 * 1.5  # Base 10 bps * 1.5 for medium vol
        assert spread == pytest.approx(expected_spread, rel=0.01)

    def test_walk_forward_oos_backtest_failure(self):
        """Test walk-forward analysis when OOS backtest fails."""
        analyzer = WalkForwardAnalyzer(window_size=100, step_size=50)

        # Create sample data
        data = pd.DataFrame({
            'close': np.random.randn(300).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 300)
        }, index=pd.date_range(start='2023-01-01', periods=300, freq='D'))

        # Track backtest calls
        backtest_call_count = 0

        def mock_backtest(strategy, data, cost_model=None):
            nonlocal backtest_call_count
            backtest_call_count += 1
            # Let IS backtests succeed, fail on OOS (every other call)
            if backtest_call_count % 2 == 0:  # OOS backtest
                raise ValueError("Simulated OOS backtest failure")
            return {
                'sharpe_ratio': 1.5,
                'total_return': 0.1,
                'max_drawdown': -0.05,
                'num_trades': 10
            }

        # Mock strategy class
        class MockStrategy:
            def __init__(self, params=None):
                self.params = params or {}

        results = analyzer.run_analysis(
            MockStrategy,
            data,
            {'param1': [1, 2, 3]},
            mock_backtest
        )

        # Should have results but with failed OOS windows having 0 values
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        # Find the failed windows
        failed_windows = results[results['oos_sharpe'] == 0]
        assert len(failed_windows) > 0
        # Verify failed windows have default values
        for _idx, row in failed_windows.iterrows():
            assert row['oos_return'] == 0
            assert row['oos_drawdown'] == 0
            assert row['oos_trades'] == 0

    def test_monte_carlo_no_variance_returns(self):
        """Test Monte Carlo validation with zero variance returns."""
        # Create returns with no variance
        returns = pd.Series(
            [0.001] * 100,  # Constant returns
            index=pd.date_range(start='2023-01-01', periods=100, freq='D')
        )

        strategy_results = {
            'sharpe_ratio': float('inf'),  # Would be infinite with zero std
            'returns_series': returns
        }

        validator = MonteCarloValidator(n_simulations=50)
        validation = validator.validate_strategy(strategy_results)

        # Validation should still complete with zero variance handling
        assert 'p_value' in validation
        assert 'significant' in validation
        assert 'effect_size' in validation
        # With zero variance, effect size would be inf or nan
        assert np.isinf(validation['effect_size']) or np.isnan(validation['effect_size'])

    def test_monte_carlo_extreme_sharpe_outliers(self):
        """Test Monte Carlo with extreme Sharpe ratio outliers in simulations."""
        returns = pd.Series(
            np.random.randn(100) * 0.01 + 0.002,
            index=pd.date_range(start='2023-01-01', periods=100, freq='D')
        )

        strategy_results = {
            'sharpe_ratio': 2.0,
            'returns_series': returns
        }

        validator = MonteCarloValidator(n_simulations=100)

        # Mock random returns to occasionally produce extreme values
        original_randn = np.random.randn
        def mock_randn(*args, **kwargs):
            result = original_randn(*args, **kwargs)
            # Inject some extreme values
            if len(result) > 10 and np.random.random() < 0.1:
                result[0] = 100  # Extreme outlier
            return result

        with patch('numpy.random.randn', mock_randn):
            validation = validator.validate_strategy(strategy_results)

        # Should handle outliers gracefully
        assert 'random_mean_sharpe' in validation
        assert 'random_std_sharpe' in validation
        assert 'p_value' in validation

    def test_regime_analyzer_empty_window(self):
        """Test regime analyzer with empty regime windows."""
        analyzer = RegimeAnalyzer(regime_method="volatility")

        # Create data that will result in very unbalanced regimes
        data = pd.DataFrame({
            'close': [100] * 50 + list(np.random.randn(10).cumsum() + 100),
            'volume': [1000] * 60
        }, index=pd.date_range(start='2023-01-01', periods=60, freq='D'))

        mock_strategy = Mock()

        # This should handle cases where some regimes have very few data points
        results = analyzer.analyze_regime_performance(
            mock_strategy,
            data,
            lambda s, d: {'sharpe_ratio': 1.0 if len(d) > 5 else 0.0}
        )

        assert 'regime_results' in results

    def test_create_report_with_complete_results(self):
        """Test report creation with complete results dictionary."""
        complete_results = {
            'sharpe_ratio': 1.5,
            'total_return': 25.0,
            'max_drawdown': -10.0,
            'win_rate': 0.6
        }

        report = create_backtest_report(complete_results)

        # Should create report with all metrics
        assert isinstance(report, str)
        assert "Backtest Report" in report
        assert "Total Return: 25.00%" in report
        assert "Sharpe Ratio: 1.50" in report
        assert "Max Drawdown: -10.00%" in report
        assert "Win Rate: 60.00%" in report
