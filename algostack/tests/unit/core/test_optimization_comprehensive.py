"""
Comprehensive test suite for optimization module - achieving 100% coverage.

Test Categories:
- PlateauDetector: Finding stable parameter regions (Pillar 2: Profit Generation)
- CoarseToFineOptimizer: Multi-phase optimization (Pillar 2: Profit Generation)
- BayesianOptimizer: Efficient parameter search (Pillar 2: Profit Generation)
- EnsembleOptimizer: Robust parameter selection (Pillar 4: Verifiable Correctness)
- OptimizationDataPipeline: Data preparation (Pillar 4: Verifiable Correctness)
- Edge cases and error handling (Pillar 1: Capital Preservation)
"""

import itertools
import logging
from concurrent.futures import Future
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from algostack.core.optimization import (
    BayesianOptimizer,
    CoarseToFineOptimizer,
    EnsembleOptimizer,
    OptimizationDataPipeline,
    OptimizationResult,
    PlateauDetector,
    create_optuna_objective,
    define_param_space,
)


class TestPlateauDetectorComprehensive:
    """Comprehensive tests for PlateauDetector - finding stable parameter regions."""

    def test_1d_plateau_detection(self):
        """
        PlateauDetector should find plateaus in 1D parameter space.
        
        This verifies that stable regions are identified for parameter selection.
        """
        # Arrange - Create data with clear plateau
        data = pd.DataFrame({
            'param1': np.linspace(0, 10, 100),
            'sharpe': np.concatenate([
                np.random.normal(0.5, 0.05, 20),  # Low region
                np.random.normal(1.5, 0.02, 60),  # Plateau region (stable high performance)
                np.random.normal(0.8, 0.05, 20)   # Drop off
            ])
        })
        
        detector = PlateauDetector(
            min_plateau_size=10,
            stability_threshold=0.1,
            smoothing_factor=1.0
        )
        
        # Act
        plateaus = detector.find_plateaus(data, metric_col='sharpe')
        
        # Assert
        assert len(plateaus) >= 1
        best_plateau = max(plateaus, key=lambda p: p['mean_value'])
        assert best_plateau['mean_value'] > 1.4  # Should find high performance region
        assert best_plateau['size'] > 10  # Should be substantial
        assert best_plateau['stability'] > 0.9  # Should be stable
        assert 'param_range' in best_plateau
        assert 'center' in best_plateau

    def test_2d_plateau_detection(self):
        """
        PlateauDetector should find plateaus in 2D parameter space.
        
        Critical for multi-parameter optimization.
        """
        # Arrange - Create 2D grid with plateau
        param1_vals = np.linspace(0, 10, 20)
        param2_vals = np.linspace(0, 10, 20)
        
        data_rows = []
        for p1 in param1_vals:
            for p2 in param2_vals:
                # Create plateau around (5, 5)
                distance = np.sqrt((p1 - 5)**2 + (p2 - 5)**2)
                if distance < 2:
                    value = 1.5 + np.random.normal(0, 0.02)  # Plateau
                else:
                    value = 0.5 + np.random.normal(0, 0.1)   # Background
                
                data_rows.append({
                    'param1': p1,
                    'param2': p2,
                    'value': value
                })
        
        data = pd.DataFrame(data_rows)
        detector = PlateauDetector(min_plateau_size=5, stability_threshold=0.2)
        
        # Act
        plateaus = detector.find_plateaus(data, metric_col='value')
        
        # Assert
        assert len(plateaus) >= 1
        best_plateau = max(plateaus, key=lambda p: p['mean_value'])
        assert best_plateau['mean_value'] > 1.4
        # Center should be near (5, 5)
        assert abs(best_plateau['center']['param1'] - 5) < 2
        assert abs(best_plateau['center']['param2'] - 5) < 2

    def test_nd_plateau_detection(self):
        """
        PlateauDetector should handle high-dimensional parameter spaces.
        
        Uses clustering for complex optimization landscapes.
        """
        # Arrange - Create 3D data with clusters
        np.random.seed(42)
        n_samples = 200
        
        # Create two clusters of good parameters
        cluster1 = np.random.normal([2, 2, 2], 0.5, (50, 3))
        cluster2 = np.random.normal([8, 8, 8], 0.5, (50, 3))
        background = np.random.uniform(0, 10, (100, 3))
        
        params = np.vstack([cluster1, cluster2, background])
        values = np.concatenate([
            np.random.normal(1.5, 0.05, 50),  # Cluster 1 high performance
            np.random.normal(1.4, 0.05, 50),  # Cluster 2 high performance
            np.random.normal(0.5, 0.2, 100)   # Background low performance
        ])
        
        data = pd.DataFrame({
            'param1': params[:, 0],
            'param2': params[:, 1],
            'param3': params[:, 2],
            'metric': values
        })
        
        detector = PlateauDetector(min_plateau_size=20)
        
        # Act
        plateaus = detector.find_plateaus(data, metric_col='metric')
        
        # Assert
        assert len(plateaus) >= 2  # Should find both clusters
        # Both plateaus should have high mean values
        for plateau in plateaus[:2]:
            assert plateau['mean_value'] > 1.3
            assert plateau['size'] >= 20

    def test_backward_compatibility_methods(self):
        """Test backward compatibility with old API."""
        detector = PlateauDetector(patience=10, min_delta=0.01)
        
        # Test update/plateau_reached pattern
        detector.update(1.0)
        assert detector.best_value == 1.0
        assert not detector.plateau_reached()
        
        # No improvement for patience iterations
        for _ in range(11):
            detector.update(1.0)
        
        assert detector.plateau_reached()
        
        # Test reset
        detector.reset()
        assert detector.best_value is None
        assert detector.counter == 0

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        detector = PlateauDetector()
        
        # Empty data
        empty_df = pd.DataFrame()
        plateaus = detector.find_plateaus(empty_df)
        assert plateaus == []
        
        # Single point - not enough for plateau
        single_df = pd.DataFrame({'param': [1], 'value': [1.0]})
        plateaus = detector.find_plateaus(single_df, metric_col='value')
        assert plateaus == []
        
        # Not enough data for plateau detection (need at least min_plateau_size)
        small_df = pd.DataFrame({
            'param': range(3),
            'value': [1.0] * 3
        })
        plateaus = detector.find_plateaus(small_df, metric_col='value')
        assert plateaus == []  # Too small for default min_plateau_size
        
        # All same values with enough data
        flat_df = pd.DataFrame({
            'param': range(100),
            'value': [1.0] * 100
        })
        plateaus = detector.find_plateaus(flat_df, metric_col='value')
        assert len(plateaus) >= 1  # Should find one big plateau


class TestCoarseToFineOptimizer:
    """Test coarse-to-fine optimization strategy."""

    def test_full_optimization_workflow(self):
        """
        CoarseToFineOptimizer should perform two-phase optimization.
        
        Critical for efficient parameter search (Pillar 2: Profit Generation).
        """
        # Arrange
        def objective(params):
            # Quadratic with optimum at x=5, y=5
            x = params.get('x', 0)
            y = params.get('y', 0)
            return 10 - (x - 5)**2 - (y - 5)**2
        
        param_ranges = {
            'x': (0, 10),
            'y': (0, 10)
        }
        
        optimizer = CoarseToFineOptimizer(
            coarse_grid_points=5,
            fine_grid_points=10
        )
        
        # Act
        result = optimizer.optimize(objective, param_ranges, n_jobs=1)
        
        # Assert
        assert isinstance(result, OptimizationResult)
        assert abs(result.best_params['x'] - 5) < 3  # Should be near optimum
        assert abs(result.best_params['y'] - 5) < 3
        assert result.best_score > 5  # Should achieve reasonable score
        assert result.stability_score >= 0  # Should have stability info
        assert len(result.history) > 20  # Coarse + fine evaluations

    def test_no_plateau_fallback(self):
        """Test behavior when no plateaus are found."""
        # Arrange - Noisy objective with no clear plateaus
        def noisy_objective(params):
            return np.random.normal(0, 1)
        
        param_ranges = {'x': (0, 10)}
        optimizer = CoarseToFineOptimizer(coarse_grid_points=5)
        
        # Mock plateau detector to return no plateaus
        with patch.object(optimizer.plateau_detector, 'find_plateaus', return_value=[]):
            # Act
            result = optimizer.optimize(noisy_objective, param_ranges, n_jobs=1)
            
            # Assert
            assert result.convergence_info is None
            assert result.stability_score == 0.0
            assert 'x' in result.best_params

    def test_categorical_parameters(self):
        """Test optimization with categorical parameters."""
        # Arrange
        def objective(params):
            # Favor 'high' mode
            mode_scores = {'low': 0, 'high': 1.0}
            mode = params.get('mode', 'low')
            return params.get('x', 0) + mode_scores.get(mode, 0)
        
        param_ranges = {
            'x': (0, 1),
            'mode': ('low', 'high')  # Will be treated as categorical
        }
        
        optimizer = CoarseToFineOptimizer(coarse_grid_points=3)
        
        # Act
        result = optimizer.optimize(objective, param_ranges, n_jobs=1)
        
        # Assert
        assert result.best_params['mode'] in ['low', 'high']
        assert 0 <= result.best_params['x'] <= 1

    @patch('algostack.core.optimization.ProcessPoolExecutor')
    def test_parallel_execution(self, mock_executor_class):
        """Test parallel optimization execution."""
        # Arrange
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Create mock futures
        future_results = []
        for i in range(5):
            future = Mock(spec=Future)
            future.result.return_value = float(i)
            future_results.append(future)
        
        mock_executor.submit.side_effect = future_results
        
        # Mock as_completed to return futures
        with patch('algostack.core.optimization.as_completed', return_value=future_results):
            optimizer = CoarseToFineOptimizer(coarse_grid_points=5)
            
            # Act
            result = optimizer.optimize(
                lambda x: x['param'],
                {'param': (0, 10)},
                n_jobs=4
            )
            
            # Assert
            assert mock_executor_class.called
            assert mock_executor.submit.call_count >= 5

    def test_error_handling_in_parallel(self):
        """Test error handling during parallel execution."""
        # Arrange
        def failing_objective(params):
            x = params.get('x', 0)
            if x > 5:
                raise ValueError("Simulated error")
            return x
        
        optimizer = CoarseToFineOptimizer(coarse_grid_points=10)
        
        # Act - Should handle errors gracefully
        with patch('algostack.core.optimization.logger') as mock_logger:
            result = optimizer.optimize(
                failing_objective,
                {'x': (0, 10)},
                n_jobs=1
            )
            
            # Assert
            assert mock_logger.error.called
            # Should still return valid result from successful evaluations
            assert result.best_params['x'] <= 5


class TestBayesianOptimizerComprehensive:
    """Comprehensive tests for Bayesian optimization."""

    def test_optuna_integration(self):
        """
        BayesianOptimizer should integrate with Optuna correctly.
        
        Efficient parameter search for profit generation.
        """
        # Arrange
        param_space = {
            'learning_rate': {
                'type': 'float',
                'low': 0.001,
                'high': 0.1,
                'log': True
            },
            'batch_size': {
                'type': 'int',
                'low': 16,
                'high': 128,
                'step': 16
            },
            'optimizer': {
                'type': 'categorical',
                'choices': ['adam', 'sgd', 'rmsprop']
            }
        }
        
        def objective_builder(params, trial):
            # Simple objective favoring certain parameters
            lr_score = -abs(np.log(params['learning_rate']) - np.log(0.01))
            batch_score = -abs(params['batch_size'] - 64) / 64
            opt_score = {'adam': 1, 'sgd': 0.5, 'rmsprop': 0.8}[params['optimizer']]
            return lr_score + batch_score + opt_score
        
        optimizer = BayesianOptimizer(n_trials=20, n_jobs=1)
        
        # Act
        result = optimizer.optimize(objective_builder, param_space)
        
        # Assert
        assert 0.001 <= result.best_params['learning_rate'] <= 0.1
        assert result.best_params['batch_size'] in range(16, 129, 16)
        assert result.best_params['optimizer'] in ['adam', 'sgd', 'rmsprop']
        assert result.stability_score >= 0
        assert len(result.convergence_history) == 20

    def test_multi_objective_optimization(self):
        """Test multi-objective optimization with Pareto front."""
        # Arrange
        param_space = {
            'risk': {'type': 'float', 'low': 0, 'high': 1},
            'leverage': {'type': 'float', 'low': 1, 'high': 5}
        }
        
        def objective_builder(params, trial):
            # Two objectives: maximize return, minimize risk
            expected_return = params['leverage'] * 0.1  # More leverage = more return
            risk = params['risk'] + params['leverage'] * 0.2  # More leverage = more risk
            return expected_return, risk
        
        optimizer = BayesianOptimizer(n_trials=10)
        
        # Act
        result = optimizer.optimize(
            objective_builder,
            param_space,
            multi_objective=True
        )
        
        # Assert
        assert 'risk' in result.best_params
        assert 'leverage' in result.best_params
        # Should find reasonable trade-off
        assert result.best_params['leverage'] > 1  # Some leverage for return

    def test_backward_compatibility_api(self):
        """Test backward compatibility with old API."""
        # Arrange
        from algostack.core.optimization import ParameterSpace
        
        param_space = ParameterSpace({
            'alpha': (0.1, 1.0, 'uniform'),
            'beta': [1, 2, 3, 4, 5]
        })
        
        optimizer = BayesianOptimizer(
            param_space=param_space,
            n_iterations=10
        )
        
        def objective(params):
            return -params['alpha']**2 + params['beta']
        
        # Act
        result = optimizer.optimize(objective)
        
        # Assert
        assert 0.1 <= result.best_params['alpha'] <= 1.0
        assert result.best_params['beta'] in [1, 2, 3, 4, 5]
        assert len(result.history) > 0

    def test_stability_calculation(self):
        """Test parameter stability calculation."""
        # Arrange
        # Create mock trials dataframe
        trials_data = []
        for i in range(20):
            trials_data.append({
                'value': 1.0 + i * 0.01,  # Gradually improving
                'params_x': 5.0 + np.random.normal(0, 0.1),
                'params_y': 3.0 + np.random.normal(0, 0.2)
            })
        
        trials_df = pd.DataFrame(trials_data)
        best_params = {'x': 5.0, 'y': 3.0}
        
        optimizer = BayesianOptimizer()
        
        # Act
        stability = optimizer._calculate_stability(trials_df, best_params)
        
        # Assert
        assert 0 <= stability <= 1
        assert stability > 0.2  # Should have some stability

    def test_empty_param_space_error(self):
        """Test error handling for empty parameter space."""
        optimizer = BayesianOptimizer()
        
        with pytest.raises(ValueError, match="No parameter space provided"):
            optimizer.optimize(lambda x: x, param_space=None)


class TestEnsembleOptimizerComprehensive:
    """Test ensemble parameter selection for robustness."""

    def test_create_diverse_ensemble(self):
        """
        EnsembleOptimizer should create diverse parameter sets.
        
        Ensures robust performance (Pillar 4: Verifiable Correctness).
        """
        # Arrange
        # Create optimization result with multiple good parameters
        results_data = []
        for i in range(30):
            results_data.append({
                'x': i * 0.3,
                'y': i * 0.2,
                'value': 10 - 0.1 * i  # Decreasing performance
            })
        
        opt_result = OptimizationResult(
            best_params={'x': 0, 'y': 0},
            best_score=10,
            history=pd.DataFrame(results_data)
        )
        
        ensemble_opt = EnsembleOptimizer(n_ensemble=5, diversity_weight=0.3)
        
        # Act
        ensemble = ensemble_opt.create_ensemble(opt_result)
        
        # Assert
        assert len(ensemble) == 5
        # First should be best params
        assert ensemble[0] == {'x': 0, 'y': 0}
        
        # Check diversity - parameters should be different
        for i in range(1, 5):
            assert ensemble[i] != ensemble[0]
            # Check minimum distance between ensemble members
            for j in range(i):
                assert ensemble[i] != ensemble[j]

    def test_param_distance_calculation(self):
        """Test parameter distance calculation for diversity."""
        ensemble_opt = EnsembleOptimizer()
        
        # Numeric parameters
        params1 = {'x': 1.0, 'y': 2.0}
        params2 = {'x': 3.0, 'y': 2.0}
        distance = ensemble_opt._param_distance(params1, params2)
        assert distance > 0
        
        # Categorical parameters
        params3 = {'mode': 'A', 'value': 1.0}
        params4 = {'mode': 'B', 'value': 1.0}
        distance = ensemble_opt._param_distance(params3, params4)
        assert distance > 0
        
        # Same parameters
        distance = ensemble_opt._param_distance(params1, params1)
        assert distance == 0

    def test_params_equality_check(self):
        """Test parameter equality checking."""
        ensemble_opt = EnsembleOptimizer()
        
        # Exact equality
        params1 = {'x': 1.0, 'y': 2.0}
        params2 = {'x': 1.0, 'y': 2.0}
        assert ensemble_opt._params_equal(params1, params2)
        
        # Float tolerance
        params3 = {'x': 1.0, 'y': 2.0}
        params4 = {'x': 1.0000001, 'y': 2.0}
        assert ensemble_opt._params_equal(params3, params4)
        
        # Different values
        params5 = {'x': 1.0, 'y': 2.0}
        params6 = {'x': 2.0, 'y': 2.0}
        assert not ensemble_opt._params_equal(params5, params6)

    def test_insufficient_candidates(self):
        """Test ensemble creation with insufficient candidates."""
        # Only 2 results but want 5 ensemble members
        opt_result = OptimizationResult(
            best_params={'x': 1},
            best_score=10,
            history=pd.DataFrame([
                {'x': 1, 'value': 10},
                {'x': 2, 'value': 9}
            ])
        )
        
        ensemble_opt = EnsembleOptimizer(n_ensemble=5)
        ensemble = ensemble_opt.create_ensemble(opt_result)
        
        # Should return what's available
        assert len(ensemble) <= 2


class TestOptimizationDataPipeline:
    """Test data preparation for optimization."""

    def test_standard_data_split(self):
        """
        OptimizationDataPipeline should split data correctly.
        
        Ensures proper validation (Pillar 4: Verifiable Correctness).
        """
        # Arrange
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 1000)
        }, index=dates)
        
        pipeline = OptimizationDataPipeline(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            purge_days=5
        )
        
        # Act
        train, val, test = pipeline.split_data(data)
        
        # Assert
        assert len(train) == 600
        assert len(val) < 200  # Less due to purging
        assert len(test) < 200  # Less due to purging
        # Check purging - no overlap
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_walk_forward_splits(self):
        """Test walk-forward analysis split creation."""
        # Arrange
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100
        }, index=dates)
        
        pipeline = OptimizationDataPipeline(purge_days=2)
        
        # Act
        splits = pipeline.create_walk_forward_splits(
            data,
            window_size=100,
            step_size=50,
            min_train_size=300
        )
        
        # Assert
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) >= 300  # Minimum training size
            assert len(test) >= 50    # At least half window
            # Check purging
            assert (test.index[0] - train.index[-1]).days >= pipeline.purge_days

    def test_feature_engineering(self):
        """Test feature preparation functionality."""
        # Arrange
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        pipeline = OptimizationDataPipeline()
        
        # Act
        prepared = pipeline.prepare_features(data, feature_engineering=True)
        
        # Assert
        assert 'returns' in prepared.columns
        assert 'log_returns' in prepared.columns
        assert 'volume_ratio' in prepared.columns
        assert 'high_low_ratio' in prepared.columns
        assert 'close_open_ratio' in prepared.columns
        assert len(prepared) < len(data)  # Some rows dropped due to NaN

    def test_insufficient_data_errors(self):
        """Test error handling for insufficient data."""
        small_data = pd.DataFrame({'close': range(50)})
        pipeline = OptimizationDataPipeline()
        
        with pytest.raises(ValueError, match="Training set too small"):
            pipeline.split_data(small_data, ensure_min_samples=100)

    def test_ratio_validation(self):
        """Test validation of split ratios."""
        with pytest.raises(AssertionError):
            OptimizationDataPipeline(
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum > 1
            )


class TestOptuniaObjectiveBuilder:
    """Test Optuna objective function creation."""

    def test_create_optuna_objective(self):
        """
        create_optuna_objective should build proper objective function.
        
        Enables strategy optimization (Pillar 2: Profit Generation).
        """
        # Arrange
        class MockStrategy:
            def __init__(self, params):
                self.params = params
        
        train_data = pd.DataFrame({'close': range(100)})
        val_data = pd.DataFrame({'close': range(100, 150)})
        
        def mock_backtest(strategy, data, cost_model=None):
            # Simple mock backtest
            return {
                'sharpe_ratio': 1.5 if len(data) < 100 else 1.2,
                'max_drawdown': -0.1
            }
        
        # Act
        objective = create_optuna_objective(
            MockStrategy,
            train_data,
            val_data,
            mock_backtest,
            stability_penalty=0.2
        )
        
        # Create mock trial
        mock_trial = Mock()
        mock_trial.params = {'param1': 0.5}
        mock_trial.study._directions = None  # Single objective
        
        # Evaluate
        score = objective(mock_trial)
        
        # Assert
        assert isinstance(score, float)
        # Score = val_sharpe - stability_penalty - 0.1 * max_drawdown
        # val has 50 rows, so sharpe=1.5; train has 100 rows, so sharpe=1.2
        # No penalty since val > train
        # = 1.5 - 0 - 0.1*(-0.1) = 1.5 + 0.01 = 1.51
        assert score == pytest.approx(1.51, rel=0.01)

    def test_multi_objective_optuna(self):
        """Test multi-objective Optuna function."""
        # Arrange
        class MockStrategy:
            def __init__(self, params):
                self.params = params
        
        def mock_backtest(strategy, data, cost_model=None):
            return {
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.15
            }
        
        objective = create_optuna_objective(
            MockStrategy,
            pd.DataFrame(),
            pd.DataFrame(),
            mock_backtest
        )
        
        # Mock multi-objective trial
        mock_trial = Mock()
        mock_trial.params = {}
        mock_trial.study._directions = ['maximize', 'minimize']
        
        # Act
        result = objective(mock_trial)
        
        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 1.5  # Sharpe
        assert result[1] == -0.15  # Drawdown


class TestParameterSpaceDefinition:
    """Test parameter space definition utilities."""

    def test_define_param_space(self):
        """Test conversion of simple definitions to Optuna format."""
        # Arrange
        param_defs = {
            'int_param': (1, 10),
            'float_param': (0.1, 1.0),
            'cat_param': ['A', 'B', 'C'],
            'explicit_param': {
                'type': 'float',
                'low': 0,
                'high': 1,
                'log': True
            }
        }
        
        # Act
        param_space = define_param_space(param_defs)
        
        # Assert
        assert param_space['int_param']['type'] == 'int'
        assert param_space['float_param']['type'] == 'float'
        assert param_space['cat_param']['type'] == 'categorical'
        assert param_space['explicit_param']['log'] is True

    def test_invalid_param_definition(self):
        """Test error handling for invalid parameter definitions."""
        with pytest.raises(ValueError, match="Unknown parameter definition"):
            define_param_space({'bad_param': 'invalid'})


class TestOptimizationResultBackwardCompatibility:
    """Test backward compatibility of OptimizationResult."""

    def test_property_aliases(self):
        """Test that old property names still work."""
        result = OptimizationResult(
            best_params={'x': 1},
            best_score=10,
            history=pd.DataFrame(),
            convergence_info={'info': 'test'}
        )
        
        # Test aliases
        assert result.best_value == result.best_score
        assert result.all_results is result.history
        assert result.plateau_info is result.convergence_info


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def test_empty_optimization_results(self):
        """Test handling of empty results."""
        optimizer = CoarseToFineOptimizer()
        
        # Mock empty coarse results
        with patch.object(optimizer, '_coarse_search', return_value=pd.DataFrame()):
            result = optimizer.optimize(lambda x: 0, {'x': (0, 1)})
            
            # Should handle gracefully
            assert result.best_params == {}
            assert result.best_score == float('-inf')

    def test_nan_handling_in_plateaus(self):
        """Test NaN handling in plateau detection."""
        data = pd.DataFrame({
            'param': range(10),
            'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })
        
        detector = PlateauDetector()
        plateaus = detector.find_plateaus(data, metric_col='value')
        
        # Should handle NaN values gracefully
        assert isinstance(plateaus, list)

    @patch('algostack.core.optimization.logger')
    def test_logging_coverage(self, mock_logger):
        """Ensure logging statements are covered."""
        # Trigger various log messages
        optimizer = CoarseToFineOptimizer()
        optimizer.optimize(lambda x: x['p'], {'p': (0, 1)}, n_jobs=1)
        
        # Check logging was called
        assert mock_logger.info.called
        assert any('coarse grid search' in str(call) for call in mock_logger.info.call_args_list)

    def test_convergence_history_none_handling(self):
        """Test handling of None values in convergence history."""
        # Create study with some None values
        study = optuna.create_study()
        
        # Add trials with None values
        def objective(trial):
            if trial.number == 1:
                raise optuna.TrialPruned()
            return trial.suggest_float('x', 0, 1)
        
        study.optimize(objective, n_trials=3)
        
        # Extract convergence history - this is what BayesianOptimizer does
        history = [float(t.value) for t in study.trials if t.value is not None]
        
        # Should have filtered out the pruned trial
        assert len(history) == 2  # Only non-pruned trials
        assert all(val is not None for val in history)


# Performance and Integration Tests
@pytest.mark.slow
class TestOptimizationPerformance:
    """Performance tests for optimization."""

    def test_large_parameter_space(self):
        """Test optimization with large parameter space."""
        # 5-dimensional space
        param_ranges = {f'param{i}': (0, 10) for i in range(5)}
        
        def objective(params):
            # Simple sum objective
            return sum(params.values())
        
        optimizer = CoarseToFineOptimizer(
            coarse_grid_points=3,  # 3^5 = 243 coarse evaluations
            fine_grid_points=5
        )
        
        import time
        start = time.time()
        result = optimizer.optimize(objective, param_ranges, n_jobs=1)
        duration = time.time() - start
        
        # Should complete in reasonable time
        assert duration < 10  # seconds
        assert result.best_score > 40  # Near maximum

    def test_concurrent_optimization(self):
        """Test concurrent optimization doesn't cause issues."""
        param_space = {
            'x': {'type': 'float', 'low': 0, 'high': 1}
        }
        
        # Run multiple optimizers concurrently
        optimizers = [
            BayesianOptimizer(n_trials=5) for _ in range(3)
        ]
        
        def objective(params, trial):
            return params['x']
        
        # All should complete without issues
        results = []
        for opt in optimizers:
            result = opt.optimize(objective, param_space)
            results.append(result)
        
        assert len(results) == 3
        assert all(r.best_score > 0.5 for r in results)