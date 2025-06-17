"""Comprehensive test suite for optimization module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from algostack.core.optimization import (
    BacktestObjective,
    BayesianOptimizer,
    EnsembleOptimizer,
    GeneticOptimizer,
    OptimizationResult,
    ParameterSpace,
    PlateauDetector,
    RandomSearchOptimizer,
)


class TestPlateauDetector:
    """Test suite for PlateauDetector class."""

    @pytest.fixture
    def detector(self):
        """Create PlateauDetector instance."""
        return PlateauDetector(
            patience=5,
            min_delta=0.001,
            mode='max'
        )

    def test_initialization(self):
        """Test PlateauDetector initialization."""
        detector = PlateauDetector(patience=10, min_delta=0.01, mode='min')

        assert detector.patience == 10
        assert detector.min_delta == 0.01
        assert detector.mode == 'min'
        assert detector.best_value is None
        assert detector.counter == 0

    def test_update_improvement_max_mode(self, detector):
        """Test update with improvement in max mode."""
        # First update
        improved = detector.update(1.0)
        assert improved is True
        assert detector.best_value == 1.0
        assert detector.counter == 0

        # Better value
        improved = detector.update(1.1)
        assert improved is True
        assert detector.best_value == 1.1
        assert detector.counter == 0

        # Slightly worse (within min_delta)
        improved = detector.update(1.0995)
        assert improved is False
        assert detector.counter == 1

    def test_update_improvement_min_mode(self):
        """Test update with improvement in min mode."""
        detector = PlateauDetector(patience=5, min_delta=0.001, mode='min')

        # First update
        improved = detector.update(1.0)
        assert improved is True

        # Better value (lower)
        improved = detector.update(0.9)
        assert improved is True
        assert detector.best_value == 0.9

    def test_plateau_detection(self, detector):
        """Test plateau detection."""
        detector.update(1.0)

        # No improvement for patience iterations
        for i in range(detector.patience):
            improved = detector.update(1.0)
            assert improved is False
            assert detector.counter == i + 1
            assert not detector.plateau_reached()

        # One more should trigger plateau
        detector.update(1.0)
        assert detector.plateau_reached()

    def test_reset(self, detector):
        """Test reset functionality."""
        detector.update(1.0)
        detector.counter = 3

        detector.reset()

        assert detector.best_value is None
        assert detector.counter == 0


class TestParameterSpace:
    """Test suite for ParameterSpace class."""

    def test_parameter_space_creation(self):
        """Test creating parameter space."""
        space = ParameterSpace({
            'learning_rate': (0.001, 0.1, 'log'),
            'batch_size': [16, 32, 64, 128],
            'dropout': (0.0, 0.5, 'uniform'),
            'optimizer': ['adam', 'sgd', 'rmsprop']
        })

        assert 'learning_rate' in space.parameters
        assert space.parameters['learning_rate']['type'] == 'continuous'
        assert space.parameters['learning_rate']['distribution'] == 'log'

        assert space.parameters['batch_size']['type'] == 'discrete'
        assert space.parameters['optimizer']['type'] == 'categorical'

    def test_sample_parameters(self):
        """Test sampling from parameter space."""
        space = ParameterSpace({
            'alpha': (0.1, 1.0, 'uniform'),
            'beta': [1, 2, 3, 4, 5],
            'gamma': ['low', 'medium', 'high']
        })

        sample = space.sample()

        assert 0.1 <= sample['alpha'] <= 1.0
        assert sample['beta'] in [1, 2, 3, 4, 5]
        assert sample['gamma'] in ['low', 'medium', 'high']

    def test_get_bounds(self):
        """Test getting parameter bounds."""
        space = ParameterSpace({
            'x': (0, 10, 'uniform'),
            'y': (-5, 5, 'uniform')
        })

        bounds = space.get_bounds()

        assert bounds['x'] == (0, 10)
        assert bounds['y'] == (-5, 5)


class TestRandomSearchOptimizer:
    """Test suite for RandomSearchOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create RandomSearchOptimizer instance."""
        param_space = ParameterSpace({
            'param1': (0, 1, 'uniform'),
            'param2': [1, 2, 3, 4, 5]
        })

        return RandomSearchOptimizer(
            param_space=param_space,
            n_iterations=10,
            n_jobs=1,
            random_state=42
        )

    def test_initialization(self):
        """Test optimizer initialization."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})
        opt = RandomSearchOptimizer(param_space, n_iterations=20, n_jobs=2)

        assert opt.n_iterations == 20
        assert opt.n_jobs == 2
        assert opt.best_params is None
        assert opt.best_score == float('-inf')

    def test_optimize(self, optimizer):
        """Test optimization process."""
        # Simple quadratic objective
        def objective(params):
            x = params['param1']
            y = params['param2']
            return -(x - 0.5)**2 - (y - 3)**2

        result = optimizer.optimize(objective)

        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score > float('-inf')
        assert len(result.history) == optimizer.n_iterations
        assert result.convergence_info is not None

    def test_parallel_optimization(self):
        """Test parallel optimization."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})
        optimizer = RandomSearchOptimizer(param_space, n_iterations=10, n_jobs=2)

        def objective(params):
            return -params['x']**2

        result = optimizer.optimize(objective)

        assert result.best_params['x'] < 0.5  # Should find something in lower half


class TestBayesianOptimizer:
    """Test suite for BayesianOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create BayesianOptimizer instance."""
        param_space = ParameterSpace({
            'x': (-5, 5, 'uniform'),
            'y': (-5, 5, 'uniform')
        })

        return BayesianOptimizer(
            param_space=param_space,
            n_iterations=10,
            n_initial_points=5,
            acq_func='ei'
        )

    def test_initialization(self):
        """Test Bayesian optimizer initialization."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})
        opt = BayesianOptimizer(
            param_space,
            n_iterations=20,
            n_initial_points=10,
            acq_func='ucb'
        )

        assert opt.n_iterations == 20
        assert opt.n_initial_points == 10
        assert opt.acq_func == 'ucb'

    def test_optimize(self, optimizer):
        """Test Bayesian optimization."""
        def objective(params):
            # Simple quadratic function with minimum at (5, 5)
            # We want to MINIMIZE this, so return negative for maximization
            return -((params['x'] - 5)**2 + (params['y'] - 5)**2)

        # The optimizer maximizes by default in the old API
        result = optimizer.optimize(objective)

        # Check that optimization improved (found values close to minimum)
        assert result.best_score > -50  # Should find something better than random
        assert len(result.history) > 0
        assert 'x' in result.best_params
        assert 'y' in result.best_params
        # Best params should be somewhat close to (5, 5)
        assert 0 <= result.best_params['x'] <= 10
        assert 0 <= result.best_params['y'] <= 10

    def test_acquisition_functions(self):
        """Test different acquisition functions."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})

        for acq_func in ['ei', 'ucb', 'poi']:
            opt = BayesianOptimizer(
                param_space,
                n_iterations=5,
                acq_func=acq_func
            )
            assert opt.acq_func == acq_func


class TestGeneticOptimizer:
    """Test suite for GeneticOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create GeneticOptimizer instance."""
        param_space = ParameterSpace({
            'x': (0, 10, 'uniform'),
            'y': (0, 10, 'uniform'),
            'z': [1, 2, 3, 4, 5]
        })

        return GeneticOptimizer(
            param_space=param_space,
            population_size=20,
            n_generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

    def test_initialization(self):
        """Test genetic optimizer initialization."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})
        opt = GeneticOptimizer(
            param_space,
            population_size=50,
            n_generations=100,
            mutation_rate=0.2
        )

        assert opt.population_size == 50
        assert opt.n_generations == 100
        assert opt.mutation_rate == 0.2

    def test_create_population(self, optimizer):
        """Test population creation."""
        population = optimizer._create_population()

        assert len(population) == optimizer.population_size
        for individual in population:
            assert 0 <= individual['x'] <= 10
            assert 0 <= individual['y'] <= 10
            assert individual['z'] in [1, 2, 3, 4, 5]

    def test_selection(self, optimizer):
        """Test selection mechanism."""
        # Create population with fitness scores
        population = [
            {'params': {'x': 1}, 'fitness': 10},
            {'params': {'x': 2}, 'fitness': 20},
            {'params': {'x': 3}, 'fitness': 5},
            {'params': {'x': 4}, 'fitness': 15}
        ]

        selected = optimizer._selection(population, 2)

        assert len(selected) == 2
        # Should prefer higher fitness
        assert selected[0]['fitness'] >= selected[1]['fitness']

    def test_crossover(self, optimizer):
        """Test crossover operation."""
        parent1 = {'x': 1.0, 'y': 2.0, 'z': 1}
        parent2 = {'x': 5.0, 'y': 8.0, 'z': 3}

        child1, child2 = optimizer._crossover(parent1, parent2)

        # Children should have mixed parameters
        assert child1 != parent1 and child1 != parent2
        assert child2 != parent1 and child2 != parent2

    def test_mutation(self, optimizer):
        """Test mutation operation."""
        individual = {'x': 5.0, 'y': 5.0, 'z': 3}

        # Test mutation multiple times to account for randomness
        mutations_found = False
        for _ in range(20):  # Try multiple times
            mutated = optimizer._mutate(individual.copy())
            if mutated != individual:
                mutations_found = True
                break

        # Should find at least one mutation in 20 attempts
        assert mutations_found, "No mutations found in 20 attempts"

    def test_optimize(self, optimizer):
        """Test genetic optimization."""
        # Simple objective: maximize -(x-5)^2 - (y-5)^2
        def objective(params):
            x = params['x']
            y = params['y']
            return -(x - 5)**2 - (y - 5)**2

        result = optimizer.optimize(objective)

        assert result.best_params is not None
        # History contains all evaluations (population_size * n_generations)
        assert len(result.history) == optimizer.population_size * optimizer.n_generations
        # Should find values close to (5, 5)
        assert abs(result.best_params['x'] - 5) < 2
        assert abs(result.best_params['y'] - 5) < 2
        # Check convergence
        assert result.convergence_info['generations'] == optimizer.n_generations


class TestEnsembleOptimizer:
    """Test suite for EnsembleOptimizer."""

    @pytest.fixture
    def ensemble_optimizer(self):
        """Create EnsembleOptimizer instance."""
        param_space = ParameterSpace({
            'alpha': (0.1, 1.0, 'uniform'),
            'beta': [1, 2, 3, 4, 5]
        })

        optimizers = [
            RandomSearchOptimizer(param_space, n_iterations=5),
            BayesianOptimizer(param_space=param_space, n_iterations=5),
            GeneticOptimizer(param_space, population_size=10, n_generations=5)
        ]

        return EnsembleOptimizer(
            optimizers=optimizers,
            voting='soft',
            weights=[0.3, 0.4, 0.3]
        )

    def test_initialization(self):
        """Test ensemble optimizer initialization."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})
        opt1 = RandomSearchOptimizer(param_space, n_iterations=5)
        opt2 = RandomSearchOptimizer(param_space, n_iterations=5)

        ensemble = EnsembleOptimizer(
            optimizers=[opt1, opt2],
            voting='hard'
        )

        assert len(ensemble.optimizers) == 2
        assert ensemble.voting == 'hard'
        assert ensemble.weights == [0.5, 0.5]  # Default equal weights

    def test_optimize_soft_voting(self, ensemble_optimizer):
        """Test ensemble optimization with soft voting."""
        def objective(params):
            return -params['alpha']**2

        result = ensemble_optimizer.optimize(objective)

        assert result.best_params is not None
        assert 'ensemble_scores' in result.metadata
        assert len(result.metadata['ensemble_scores']) == 3

    def test_optimize_hard_voting(self):
        """Test ensemble optimization with hard voting."""
        param_space = ParameterSpace({'x': (0, 1, 'uniform')})
        optimizers = [
            RandomSearchOptimizer(param_space, n_iterations=5),
            RandomSearchOptimizer(param_space, n_iterations=5)
        ]

        ensemble = EnsembleOptimizer(optimizers=optimizers, voting='hard')

        def objective(params):
            return -params['x']**2

        result = ensemble.optimize(objective)

        assert result.best_params is not None


class TestBacktestObjective:
    """Test suite for BacktestObjective."""

    @pytest.fixture
    def mock_backtest_engine(self):
        """Create mock backtest engine."""
        engine = Mock()
        engine.run_backtest = Mock(return_value={
            'sharpe_ratio': 1.5,
            'total_return': 0.15,
            'max_drawdown': -0.10,
            'profit_factor': 1.8
        })
        return engine

    def test_single_objective(self, mock_backtest_engine):
        """Test single objective optimization."""
        objective = BacktestObjective(
            backtest_engine=mock_backtest_engine,
            strategy_class=Mock,
            data=pd.DataFrame(),
            metric='sharpe_ratio'
        )

        params = {'param1': 0.5, 'param2': 10}
        score = objective(params)

        assert score == 1.5
        mock_backtest_engine.run_backtest.assert_called_once()

    def test_multi_objective(self, mock_backtest_engine):
        """Test multi-objective optimization."""
        objective = BacktestObjective(
            backtest_engine=mock_backtest_engine,
            strategy_class=Mock,
            data=pd.DataFrame(),
            metric=['sharpe_ratio', 'profit_factor'],
            weights=[0.6, 0.4]
        )

        params = {'param1': 0.5}
        score = objective(params)

        # Weighted sum: 0.6 * 1.5 + 0.4 * 1.8 = 1.62
        assert score == pytest.approx(1.62)

    def test_constraint_handling(self, mock_backtest_engine):
        """Test optimization with constraints."""
        # Add constraint: max drawdown must be > -15%
        objective = BacktestObjective(
            backtest_engine=mock_backtest_engine,
            strategy_class=Mock,
            data=pd.DataFrame(),
            metric='sharpe_ratio',
            constraints={'max_drawdown': (-0.15, None)}
        )

        params = {'param1': 0.5}
        score = objective(params)

        # Should pass constraint (-0.10 > -0.15)
        assert score == 1.5

        # Test failing constraint
        mock_backtest_engine.run_backtest.return_value['max_drawdown'] = -0.20
        score = objective(params)

        # Should return penalty
        assert score < 0
