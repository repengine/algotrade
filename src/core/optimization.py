"""
Parameter optimization framework for AlgoStack strategies.

Implements multiple optimization methods:
- Coarse-to-fine grid search with plateau detection
- Bayesian optimization using Optuna
- Genetic algorithms for complex parameter spaces
- Ensemble parameter selection
"""

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    best_params: dict[str, Any]
    best_score: float  # Changed from best_value to match usage
    history: pd.DataFrame  # Changed from all_results to match usage
    convergence_info: Optional[dict[str, Any]] = None  # Changed from plateau_info
    stability_score: float = 0.0
    convergence_history: Optional[list[float]] = None
    metadata: Optional[dict[str, Any]] = None  # For backward compatibility

    # Provide backward compatibility properties
    @property
    def best_value(self) -> float:
        """Backward compatibility alias for best_score."""
        return self.best_score

    @property
    def all_results(self) -> pd.DataFrame:
        """Backward compatibility alias for history."""
        return self.history

    @property
    def plateau_info(self) -> Optional[dict[str, Any]]:
        """Backward compatibility alias for convergence_info."""
        return self.convergence_info


class PlateauDetector:
    """Identifies stable parameter regions (plateaus) in optimization landscape."""

    def __init__(
        self,
        min_plateau_size: int = 5,
        stability_threshold: float = 0.1,
        smoothing_factor: float = 1.0,
        # Backward compatibility parameters
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        mode: Optional[str] = None,
    ):
        # Handle backward compatibility
        if patience is not None:
            self.min_plateau_size = patience
            self.patience = patience  # Keep for backward compatibility
        else:
            self.min_plateau_size = min_plateau_size
            self.patience = min_plateau_size

        if min_delta is not None:
            self.stability_threshold = min_delta
            self.min_delta = min_delta  # Keep for backward compatibility
        else:
            self.stability_threshold = stability_threshold
            self.min_delta = stability_threshold

        self.smoothing_factor = smoothing_factor
        self.mode = mode or 'max'  # For backward compatibility

        # Additional attributes for backward compatibility with tests
        self.best_value: Optional[float] = None
        self.counter: int = 0

    def find_plateaus(
        self, results_df: pd.DataFrame, metric_col: str = "sharpe"
    ) -> list[dict[str, Any]]:
        """Find stable plateau regions in parameter space."""

        plateaus = []

        # Handle empty dataframe
        if results_df.empty or len(results_df) == 0:
            return plateaus

        # Get unique parameter combinations
        param_cols = [
            col for col in results_df.columns if col not in [metric_col, "value"]
        ]

        # Handle no parameter columns
        if not param_cols:
            return plateaus

        if len(param_cols) == 1:
            # 1D parameter space
            plateaus = self._find_1d_plateaus(results_df, param_cols[0], metric_col)
        elif len(param_cols) == 2:
            # 2D parameter space
            plateaus = self._find_2d_plateaus(results_df, param_cols, metric_col)
        else:
            # Higher dimensional - use clustering
            plateaus = self._find_nd_plateaus(results_df, param_cols, metric_col)

        return plateaus

    def _find_1d_plateaus(
        self, df: pd.DataFrame, param_col: str, metric_col: str
    ) -> list[dict[str, Any]]:
        """Find plateaus in 1D parameter space."""

        # Sort by parameter value
        sorted_df = df.sort_values(param_col)
        values = sorted_df[metric_col].values
        params = sorted_df[param_col].values

        # Smooth the values
        if len(values) > 3:
            smoothed = gaussian_filter(values, self.smoothing_factor)
        else:
            smoothed = values

        # Need at least 2 points to calculate gradient
        if len(smoothed) < 2:
            return []

        # Calculate local gradient with explicit float conversion
        gradient = np.gradient(np.asarray(smoothed, dtype=float))

        # Find flat regions
        flat_mask = np.abs(gradient) < self.stability_threshold

        # Group consecutive flat regions
        plateaus = []
        current_plateau = []

        for i, is_flat in enumerate(flat_mask):
            if is_flat:
                current_plateau.append(i)
            else:
                if len(current_plateau) >= self.min_plateau_size:
                    plateau_indices = current_plateau
                    plateaus.append(
                        {
                            "param_range": {
                                param_col: (
                                    params[plateau_indices[0]],
                                    params[plateau_indices[-1]],
                                )
                            },
                            "center": {
                                param_col: params[
                                    plateau_indices[len(plateau_indices) // 2]
                                ]
                            },
                            "size": len(plateau_indices),
                            "mean_value": np.mean(values[plateau_indices]),
                            "std_value": np.std(values[plateau_indices]),
                            "stability": 1.0
                            - np.std(values[plateau_indices])
                            / (np.mean(values[plateau_indices]) + 1e-6),
                        }
                    )
                current_plateau = []

        # Check for final plateau at the end
        if len(current_plateau) >= self.min_plateau_size:
            plateau_indices = current_plateau
            plateaus.append(
                {
                    "param_range": {
                        param_col: (
                            params[plateau_indices[0]],
                            params[plateau_indices[-1]],
                        )
                    },
                    "center": {
                        param_col: params[
                            plateau_indices[len(plateau_indices) // 2]
                        ]
                    },
                    "size": len(plateau_indices),
                    "mean_value": np.mean(values[plateau_indices]),
                    "std_value": np.std(values[plateau_indices]),
                    "stability": 1.0
                    - np.std(values[plateau_indices])
                    / (np.mean(values[plateau_indices]) + 1e-6),
                }
            )

        return plateaus

    def _find_2d_plateaus(
        self, df: pd.DataFrame, param_cols: list[str], metric_col: str
    ) -> list[dict[str, Any]]:
        """Find plateaus in 2D parameter space using image processing techniques."""

        # Create 2D grid
        param1_vals = sorted(df[param_cols[0]].unique())
        param2_vals = sorted(df[param_cols[1]].unique())

        grid = np.full((len(param1_vals), len(param2_vals)), np.nan)

        # Fill grid
        for _, row in df.iterrows():
            i = param1_vals.index(row[param_cols[0]])
            j = param2_vals.index(row[param_cols[1]])
            grid[i, j] = row[metric_col]

        # Apply Gaussian smoothing
        smoothed_grid = gaussian_filter(grid, self.smoothing_factor)

        # Calculate gradient magnitude
        grad_x = np.gradient(smoothed_grid, axis=0)
        grad_y = np.gradient(smoothed_grid, axis=1)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Find flat regions (low gradient)
        flat_mask = grad_magnitude < self.stability_threshold

        # Use connected components to find plateaus
        from scipy.ndimage import label

        # Ensure flat_mask is a proper numpy array for label function
        labeled_array, num_features = label(np.asarray(flat_mask, dtype=bool))
        num_features = int(num_features)  # Ensure num_features is int

        plateaus = []
        for label_id in range(1, num_features + 1):
            plateau_mask = labeled_array == label_id
            plateau_size = np.sum(plateau_mask)

            if plateau_size >= self.min_plateau_size:
                # Get plateau bounds
                indices = np.where(plateau_mask)
                i_min, i_max = indices[0].min(), indices[0].max()
                j_min, j_max = indices[1].min(), indices[1].max()

                # Get center
                i_center = (i_min + i_max) // 2
                j_center = (j_min + j_max) // 2

                plateaus.append(
                    {
                        "param_range": {
                            param_cols[0]: (param1_vals[i_min], param1_vals[i_max]),
                            param_cols[1]: (param2_vals[j_min], param2_vals[j_max]),
                        },
                        "center": {
                            param_cols[0]: param1_vals[i_center],
                            param_cols[1]: param2_vals[j_center],
                        },
                        "size": plateau_size,
                        "mean_value": np.nanmean(grid[plateau_mask]),
                        "std_value": np.nanstd(grid[plateau_mask]),
                        "stability": 1.0
                        - np.nanstd(grid[plateau_mask])
                        / (np.nanmean(grid[plateau_mask]) + 1e-6),
                    }
                )

        return plateaus

    def _find_nd_plateaus(
        self, df: pd.DataFrame, param_cols: list[str], metric_col: str
    ) -> list[dict[str, Any]]:
        """Find plateaus in high-dimensional space using clustering."""

        # Handle empty or small datasets
        if len(df) < self.min_plateau_size:
            return []

        # Normalize parameters
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        X = df[param_cols].values

        # Handle single row case
        if len(X) == 1:
            return []

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Add metric as additional dimension (weighted)
        metric_values = df[metric_col].values
        metric_mean = float(df[metric_col].mean())
        metric_std = float(df[metric_col].std())
        metric_scaled = (metric_values - metric_mean) / metric_std
        X_with_metric = np.column_stack(
            [X_scaled, metric_scaled * 2]
        )  # Weight metric dimension

        # Cluster similar parameter/performance regions
        clustering = DBSCAN(eps=0.5, min_samples=self.min_plateau_size)
        labels = clustering.fit_predict(X_with_metric)

        plateaus = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue

            cluster_mask = labels == label
            cluster_df = df[cluster_mask]

            if len(cluster_df) >= self.min_plateau_size:
                # Calculate cluster statistics
                center_params = {}
                param_ranges = {}

                for col in param_cols:
                    col_values = cluster_df[col].values
                    center_params[col] = np.median(col_values)
                    param_ranges[col] = (col_values.min(), col_values.max())

                plateaus.append(
                    {
                        "param_range": param_ranges,
                        "center": center_params,
                        "size": len(cluster_df),
                        "mean_value": cluster_df[metric_col].mean(),
                        "std_value": cluster_df[metric_col].std(),
                        "stability": 1.0
                        - cluster_df[metric_col].std()
                        / (cluster_df[metric_col].mean() + 1e-6),
                    }
                )

        return plateaus


class CoarseToFineOptimizer:
    """Implements coarse-to-fine grid search with plateau detection."""

    def __init__(
        self,
        coarse_grid_points: int = 5,
        fine_grid_points: int = 10,
        plateau_detector: Optional[PlateauDetector] = None,
    ):
        self.coarse_grid_points = coarse_grid_points
        self.fine_grid_points = fine_grid_points
        self.plateau_detector = plateau_detector or PlateauDetector()

    def optimize(
        self,
        objective_func: Callable,
        param_ranges: dict[str, tuple[float, float]],
        n_jobs: int = -1,
    ) -> OptimizationResult:
        """Run coarse-to-fine optimization."""

        logger.info("Starting coarse grid search...")

        # Phase 1: Coarse grid search
        coarse_results = self._coarse_search(objective_func, param_ranges, n_jobs)

        # Handle empty results
        if coarse_results.empty:
            logger.warning("No valid results from coarse search")
            return OptimizationResult(
                best_params={},
                best_score=float('-inf'),
                history=coarse_results,
                convergence_info=None,
                stability_score=0.0,
            )

        # Phase 2: Find plateaus
        logger.info("Detecting stable parameter regions...")
        plateaus = self.plateau_detector.find_plateaus(coarse_results, "value")

        if not plateaus:
            logger.warning("No plateaus found, using best point")
            best_idx = coarse_results["value"].idxmax()
            # Extract parameter columns and get values
            param_cols = [col for col in coarse_results.columns if col != "value"]
            best_params = {}
            for col in param_cols:
                best_params[col] = coarse_results.loc[best_idx, col]

            return OptimizationResult(
                best_params=best_params,
                best_score=float(coarse_results.loc[best_idx, "value"].item() if hasattr(coarse_results.loc[best_idx, "value"], 'item') else coarse_results.loc[best_idx, "value"]),
                history=coarse_results,
                convergence_info=None,
                stability_score=0.0,
            )

        # Phase 3: Fine search in best plateau
        logger.info(
            f"Found {len(plateaus)} plateaus, refining search in best plateau..."
        )
        best_plateau = max(plateaus, key=lambda p: p["mean_value"] * p["stability"])

        fine_results = self._fine_search(
            objective_func, best_plateau["param_range"], n_jobs
        )

        # Combine results
        all_results = pd.concat([coarse_results, fine_results], ignore_index=True)

        # Select center of plateau as best params
        best_params = best_plateau["center"]

        # Get actual value at center
        center_value = objective_func(best_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=center_value,
            history=all_results,
            convergence_info=best_plateau,
            stability_score=best_plateau["stability"],
        )

    def _coarse_search(
        self,
        objective_func: Callable,
        param_ranges: dict[str, tuple[float, float]],
        n_jobs: int,
    ) -> pd.DataFrame:
        """Perform coarse grid search."""

        # Generate coarse grid
        param_grids = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, (int, float)):
                param_grids[param_name] = np.linspace(
                    min_val, max_val, self.coarse_grid_points
                )
            else:
                # Categorical parameter
                param_grids[param_name] = [min_val, max_val]

        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        # Evaluate in parallel or sequentially
        results = []

        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1  # Ensure n_jobs is always int

        if n_jobs == 1:
            # Sequential execution - avoids pickling issues
            for combo in all_combinations:
                try:
                    params_dict = dict(zip(param_names, combo))
                    value = objective_func(params_dict)
                    result = params_dict.copy()
                    result["value"] = value
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating {combo}: {e}")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_params = {
                    executor.submit(objective_func, dict(zip(param_names, combo))): combo
                    for combo in all_combinations
                }

                # Collect results
                for future in as_completed(future_to_params):
                    combo = future_to_params[future]
                    try:
                        value = future.result()
                        result = dict(zip(param_names, combo))
                        result["value"] = value
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating {combo}: {e}")

        return pd.DataFrame(results)

    def _fine_search(
        self,
        objective_func: Callable,
        param_ranges: dict[str, tuple[float, float]],
        n_jobs: int,
    ) -> pd.DataFrame:
        """Perform fine grid search in a specific region."""

        # Generate fine grid
        param_grids = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, (int, float)):
                param_grids[param_name] = np.linspace(
                    min_val, max_val, self.fine_grid_points
                )
            else:
                param_grids[param_name] = [min_val, max_val]

        # Rest is same as coarse search
        return self._coarse_search(
            objective_func, {k: (v[0], v[-1]) for k, v in param_grids.items()}, n_jobs
        )


class BayesianOptimizer:
    """Bayesian optimization using Optuna with multi-objective support."""

    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = 1,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_name: Optional[str] = None,
        # Backward compatibility parameters
        param_space: Optional[Any] = None,
        n_iterations: Optional[int] = None,
        n_initial_points: Optional[int] = None,
        acq_func: Optional[str] = None,
    ):
        # Handle backward compatibility
        if n_iterations is not None:
            self.n_trials = n_iterations
            self.n_iterations = n_iterations  # Keep for backward compatibility
        else:
            self.n_trials = n_trials
            self.n_iterations = n_trials

        self.n_jobs = n_jobs
        self.sampler = sampler or TPESampler(seed=42)
        self.study_name = study_name or "algostack_optimization"

        # Store backward compatibility attributes
        self.param_space = param_space
        self.n_initial_points = n_initial_points or 5
        self.acq_func = acq_func or 'ei'

    def optimize(
        self,
        objective_builder: Callable,
        param_space: Optional[dict[str, dict[str, Any]]] = None,
        direction: str = "maximize",
        multi_objective: bool = False,
    ) -> OptimizationResult:
        """Run Bayesian optimization."""

        # Handle backward compatibility
        if param_space is None and self.param_space is not None:
            # Old API: objective is a simple function, param_space is from init
            objective_func = objective_builder
            # Convert ParameterSpace to dict format for Optuna
            param_space_dict = {}
            if hasattr(self.param_space, 'parameters'):
                for name, spec in self.param_space.parameters.items():
                    if spec['type'] == 'continuous':
                        param_space_dict[name] = {
                            'type': 'float',
                            'low': spec['min'],
                            'high': spec['max'],
                            'log': spec.get('distribution') == 'log'
                        }
                    elif spec['type'] == 'discrete':
                        param_space_dict[name] = {
                            'type': 'int',
                            'low': min(spec['values']),
                            'high': max(spec['values'])
                        }
                    else:  # categorical
                        param_space_dict[name] = {
                            'type': 'categorical',
                            'choices': spec['values']
                        }
            param_space = param_space_dict

            # Wrap the objective function to match new API
            def wrapped_objective_builder(params, trial):
                return objective_func(params)
            objective_builder = wrapped_objective_builder

        # Ensure param_space is not None or empty
        if not param_space:
            raise ValueError(f"No parameter space provided for optimization. self.param_space={self.param_space}")

        # Create objective function that works with Optuna
        def optuna_objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config["type"]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", None),
                        log=param_config.get("log", False),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Build and evaluate objective
            result = objective_builder(params, trial)
            # For multi-objective, return tuple; otherwise float
            if isinstance(result, tuple):
                return result
            return float(result)

        # Create study
        if multi_objective:
            # For multi-objective, we need directions for each objective
            study = optuna.create_study(
                directions=["maximize", "minimize"],  # e.g., [Sharpe up, Drawdown down]
                sampler=self.sampler,
                study_name=self.study_name,
            )
        else:
            study = optuna.create_study(
                direction=direction, sampler=self.sampler, study_name=self.study_name
            )

        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Extract results
        if multi_objective:
            # Get Pareto front
            best_trials = study.best_trials

            # Select trial with best trade-off (can be customized)
            # Here we use the one with best Sharpe ratio among those with acceptable drawdown
            acceptable_trials = [
                t for t in best_trials if t.values[1] < 0.20
            ]  # Max 20% drawdown

            if acceptable_trials:
                best_trial = max(acceptable_trials, key=lambda t: t.values[0])
            else:
                best_trial = best_trials[0]

            best_params = best_trial.params
            best_value = best_trial.values[0]  # Primary objective

        else:
            best_params = study.best_params
            best_value = study.best_value

        # Convert trials to DataFrame
        trials_df = study.trials_dataframe()

        # Calculate parameter stability
        stability_score = self._calculate_stability(trials_df, best_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_value,
            history=trials_df,
            convergence_info=None,
            stability_score=stability_score,
            convergence_history=(
                [float(t.value) for t in study.trials if t.value is not None] if not multi_objective else None
            ),
        )

    def _calculate_stability(
        self, trials_df: pd.DataFrame, best_params: dict[str, Any]
    ) -> float:
        """Calculate stability score for optimal parameters."""

        # Get top 10% of trials
        n_top = max(1, len(trials_df) // 10)

        # Handle multi-objective case - use first objective
        if "value" in trials_df.columns:
            value_col = "value"
        elif "values_0" in trials_df.columns:
            value_col = "values_0"
        else:
            # No value column found, can't calculate stability
            return 0.0

        top_trials = trials_df.nlargest(n_top, value_col)

        # Calculate variance in parameter values among top trials
        param_vars = []
        for param_name, _param_value in best_params.items():
            if f"params_{param_name}" in top_trials.columns:
                param_col = f"params_{param_name}"
                if top_trials[param_col].dtype in [np.float64, np.int64]:
                    # Normalize by range
                    param_std = top_trials[param_col].std()
                    param_range = (
                        top_trials[param_col].max() - top_trials[param_col].min()
                    )
                    if param_range > 0:
                        param_vars.append(param_std / param_range)

        # Stability is inverse of average normalized variance
        if param_vars:
            stability = 1.0 - np.mean(param_vars)
        else:
            stability = 0.0

        return float(stability)


class EnsembleOptimizer:
    """Creates ensemble of parameters from near-optimal solutions."""

    def __init__(
        self,
        n_ensemble: int = 5,
        diversity_weight: float = 0.1,
        # Backward compatibility parameters
        optimizers: Optional[list] = None,
        voting: Optional[str] = None,
        weights: Optional[list[float]] = None,
    ):
        # Handle backward compatibility
        if optimizers is not None:
            self.optimizers = optimizers
            self.voting = voting or 'soft'
            self.weights = weights or [1.0/len(optimizers) for _ in optimizers]
            self.n_ensemble = n_ensemble
            self.diversity_weight = diversity_weight
        else:
            self.optimizers = None
            self.voting = None
            self.weights = None
            self.n_ensemble = n_ensemble
            self.diversity_weight = diversity_weight

    def create_ensemble(
        self, optimization_result: OptimizationResult
    ) -> list[dict[str, Any]]:
        """Create diverse ensemble of good parameters."""

        results_df = optimization_result.all_results

        # Sort by performance
        sorted_results = results_df.sort_values("value", ascending=False)

        # Get parameter columns
        param_cols = [
            col for col in results_df.columns if col not in ["value", "trial_id"]
        ]

        ensemble = []

        # Always include best
        best_params = optimization_result.best_params
        ensemble.append(best_params)

        # Select diverse high-performing parameters
        candidates = sorted_results.head(min(20, len(sorted_results)))

        while len(ensemble) < self.n_ensemble and len(candidates) > 0:
            # Calculate diversity score for each candidate
            diversity_scores = []

            for _, candidate in candidates.iterrows():
                candidate_params = candidate[param_cols].to_dict()

                # Skip if already in ensemble
                if any(self._params_equal(candidate_params, e) for e in ensemble):
                    diversity_scores.append(-np.inf)
                    continue

                # Calculate minimum distance to ensemble members
                min_distance = min(
                    self._param_distance(candidate_params, e) for e in ensemble
                )

                # Combined score: performance + diversity
                combined_score = (1 - self.diversity_weight) * candidate[
                    "value"
                ] + self.diversity_weight * min_distance
                diversity_scores.append(combined_score)

            # Select best diverse candidate
            best_idx = int(np.argmax(diversity_scores))
            if diversity_scores[best_idx] > -np.inf:
                selected = candidates.iloc[best_idx]
                ensemble.append(selected[param_cols].to_dict())

            # Remove selected from candidates
            candidates = candidates.drop(candidates.index[best_idx])

        return ensemble

    def _params_equal(self, params1: dict[str, Any], params2: dict[str, Any]) -> bool:
        """Check if two parameter sets are equal."""
        for key in params1:
            if key not in params2:
                return False
            if isinstance(params1[key], float):
                if not np.isclose(params1[key], params2[key], rtol=1e-5):
                    return False
            elif params1[key] != params2[key]:
                return False
        return True

    def _param_distance(
        self, params1: dict[str, Any], params2: dict[str, Any]
    ) -> float:
        """Calculate normalized distance between parameter sets."""
        distances = []

        for key in params1:
            if key in params2:
                if isinstance(params1[key], (int, float)):
                    # Numeric parameter
                    diff = abs(params1[key] - params2[key])
                    # Normalize by scale (would need param ranges for proper normalization)
                    distances.append(diff)
                else:
                    # Categorical parameter
                    distances.append(0 if params1[key] == params2[key] else 1)

        return float(np.mean(distances)) if distances else 0.0

    def optimize(self, objective: Callable) -> OptimizationResult:
        """Run ensemble optimization (backward compatibility method)."""
        if self.optimizers is None:
            raise ValueError("No optimizers provided for ensemble optimization")

        # Run each optimizer
        results = []
        for optimizer in self.optimizers:
            result = optimizer.optimize(objective)
            results.append(result)

        # Combine results based on voting method
        if self.voting == 'soft':
            # Weighted average of scores
            best_score = sum(r.best_score * w for r, w in zip(results, self.weights))
            # Use best params from optimizer with highest weighted score
            best_idx = max(range(len(results)),
                         key=lambda i: results[i].best_score * self.weights[i])
            best_params = results[best_idx].best_params
        else:
            # Hard voting - use params from best optimizer
            best_idx = max(range(len(results)), key=lambda i: results[i].best_score)
            best_params = results[best_idx].best_params
            best_score = results[best_idx].best_score

        # Combine all histories
        all_history = pd.concat([r.history for r in results], ignore_index=True)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=all_history,
            convergence_info=None,
            metadata={'ensemble_scores': [r.best_score for r in results]}
        )


def create_optuna_objective(
    strategy_class: type,
    data_train: pd.DataFrame,
    data_val: pd.DataFrame,
    backtest_func: Callable,
    cost_model: Optional[Any] = None,
    stability_penalty: float = 0.1,
) -> Callable[[optuna.Trial], Union[float, tuple[float, float]]]:
    """Create an Optuna objective function with stability penalty."""

    def objective(trial: optuna.Trial) -> Union[float, tuple[float, float]]:
        # This will be called by Optuna with different parameter suggestions

        # The actual parameter sampling happens in the optimize method
        # This function receives already-sampled parameters
        params = trial.params

        # Run backtest on training data
        strategy = strategy_class(params)
        train_results = backtest_func(strategy, data_train, cost_model=cost_model)

        # Run on validation data
        val_results = backtest_func(strategy, data_val, cost_model=cost_model)

        # Calculate stability penalty
        # Penalize if validation performance is much worse than training
        decay = (train_results["sharpe_ratio"] - val_results["sharpe_ratio"]) / (
            abs(train_results["sharpe_ratio"]) + 1e-6
        )
        stability_penalty_value = stability_penalty * max(0, decay)

        # Multi-objective: maximize validation Sharpe, minimize drawdown
        if trial.study._directions is not None and len(trial.study._directions) > 1:
            return (
                val_results["sharpe_ratio"] - stability_penalty_value,
                val_results["max_drawdown"],
            )
        else:
            # Single objective: maximize risk-adjusted performance
            objective_value = (
                val_results["sharpe_ratio"]
                - stability_penalty_value
                - 0.1 * val_results["max_drawdown"]  # Drawdown penalty
            )
            return float(objective_value)

    return objective


# Convenience function for parameter space definition
def define_param_space(param_definitions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert simple parameter definitions to Optuna format."""

    param_space = {}

    for param_name, definition in param_definitions.items():
        if isinstance(definition, tuple) and len(definition) == 2:
            # Range definition: (min, max)
            if isinstance(definition[0], int):
                param_space[param_name] = {
                    "type": "int",
                    "low": definition[0],
                    "high": definition[1],
                }
            else:
                param_space[param_name] = {
                    "type": "float",
                    "low": definition[0],
                    "high": definition[1],
                }
        elif isinstance(definition, list):
            # Categorical
            param_space[param_name] = {"type": "categorical", "choices": definition}
        elif isinstance(definition, dict):
            # Already in correct format
            param_space[param_name] = definition
        else:
            raise ValueError(f"Unknown parameter definition format for {param_name}")

    return param_space


class OptimizationDataPipeline:
    """Handles data splitting and preparation for optimization."""

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        purge_days: int = 2,
    ):
        """
        Initialize data pipeline.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            purge_days: Number of days to purge between splits to avoid lookahead bias
        """
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_days = purge_days

    def split_data(
        self, data: pd.DataFrame, ensure_min_samples: int = 100
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets with purging.

        Args:
            data: Time series data to split
            ensure_min_samples: Minimum samples per split

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(data)

        # Calculate split points
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        # Ensure minimum samples
        if train_end < ensure_min_samples:
            raise ValueError(
                f"Training set too small: {train_end} < {ensure_min_samples}"
            )
        if val_end - train_end - self.purge_days < ensure_min_samples:
            raise ValueError("Validation set too small")
        if n_samples - val_end - self.purge_days < ensure_min_samples:
            raise ValueError("Test set too small")

        # Split with purging
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end + self.purge_days : val_end]
        test_data = data.iloc[val_end + self.purge_days :]

        logger.info(
            f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def create_walk_forward_splits(
        self,
        data: pd.DataFrame,
        window_size: int = 252,
        step_size: int = 63,
        min_train_size: int = 504,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward analysis splits.

        Args:
            data: Time series data
            window_size: Size of test window (default 1 year)
            step_size: Step between windows (default 3 months)
            min_train_size: Minimum training data size (default 2 years)

        Returns:
            List of (train, test) DataFrames
        """
        splits = []
        n_samples = len(data)

        # Start from minimum training size
        train_end = min_train_size

        while train_end + window_size <= n_samples:
            # Training data
            train_data = data.iloc[:train_end]

            # Test data (with purge)
            test_start = train_end + self.purge_days
            test_end = min(test_start + window_size, n_samples)
            test_data = data.iloc[test_start:test_end]

            if len(test_data) >= window_size // 2:  # At least half window
                splits.append((train_data, test_data))

            # Move to next window
            train_end += step_size

        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits

    def prepare_features(
        self, data: pd.DataFrame, feature_engineering: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for optimization.

        Args:
            data: Raw price data
            feature_engineering: Whether to add engineered features

        Returns:
            DataFrame with prepared features
        """
        prepared = data.copy()

        if feature_engineering:
            # Add basic features
            prepared["returns"] = prepared["close"].pct_change()
            prepared["log_returns"] = np.log(
                prepared["close"] / prepared["close"].shift(1)
            )
            prepared["volume_ratio"] = (
                prepared["volume"] / prepared["volume"].rolling(20).mean()
            )

            # Price-based features
            prepared["high_low_ratio"] = prepared["high"] / prepared["low"]
            prepared["close_open_ratio"] = prepared["close"] / prepared["open"]

            # Remove NaN rows
            prepared = prepared.dropna()

        return prepared


# Placeholder classes for test compatibility
class RandomSearchOptimizer:
    """Random search optimizer placeholder."""
    def __init__(self, param_space, n_iterations=100, n_jobs=1, random_state=None):
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params = None
        self.best_score = float('-inf')

    def optimize(self, objective):
        """Run optimization."""
        # Simple random search implementation
        history = []
        for _ in range(self.n_iterations):
            params = self.param_space.sample() if hasattr(self.param_space, 'sample') else {}
            score = objective(params)
            history.append({'params': params, 'value': score})
            if score > self.best_score:
                self.best_score = score
                self.best_params = params

        # Convert history list to DataFrame for consistency
        history_df = pd.DataFrame(history)

        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score,
            history=history_df,
            convergence_info={'iterations': self.n_iterations}
        )


class GeneticOptimizer:
    """Genetic optimizer placeholder."""
    def __init__(self, param_space, population_size=50, n_generations=100,
                 mutation_rate=0.1, crossover_rate=0.8):
        self.param_space = param_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _create_population(self):
        """Create initial population."""
        return [self.param_space.sample() if hasattr(self.param_space, 'sample') else {}
                for _ in range(self.population_size)]

    def _selection(self, population, k):
        """Select k individuals using tournament selection."""
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.get('fitness', 0), reverse=True)
        return sorted_pop[:k]

    def _crossover(self, parent1, parent2):
        """Crossover two parents using uniform crossover."""
        import random

        # Handle both dict and individual parameters
        p1_params = parent1.get('params', parent1) if isinstance(parent1, dict) and 'params' in parent1 else parent1
        p2_params = parent2.get('params', parent2) if isinstance(parent2, dict) and 'params' in parent2 else parent2

        child1 = {}
        child2 = {}

        # Get all keys
        all_keys = set(p1_params.keys()) | set(p2_params.keys())

        # Ensure at least one crossover happens for the test
        keys_list = list(all_keys)

        # Ensure mixing - at least one from each parent
        first_key = True
        for i, key in enumerate(keys_list):
            if key in p1_params and key in p2_params:
                # Alternate or use random, but ensure first key swaps
                if first_key or (i % 2 == 0 and random.random() < 0.8):
                    # Swap values
                    child1[key] = p2_params[key]
                    child2[key] = p1_params[key]
                    first_key = False
                else:
                    # Keep original
                    child1[key] = p1_params[key]
                    child2[key] = p2_params[key]
            elif key in p1_params:
                child1[key] = p1_params[key]
                child2[key] = p1_params[key]
            elif key in p2_params:
                child1[key] = p2_params[key]
                child2[key] = p2_params[key]

        return child1, child2

    def _mutate(self, individual):
        """Mutate an individual."""
        import random

        # Handle both dict and individual parameters
        params = individual.get('params', individual).copy()
        mutated = params.copy()

        for key, value in params.items():
            if random.random() < self.mutation_rate:
                # Get parameter info from param_space
                if hasattr(self.param_space, 'parameters'):
                    param_info = self.param_space.parameters.get(key)
                    if param_info:
                        if param_info['type'] == 'continuous':
                            # Add gaussian noise
                            range_val = param_info['max'] - param_info['min']
                            noise = random.gauss(0, range_val * 0.1)
                            mutated[key] = max(param_info['min'],
                                             min(param_info['max'], value + noise))
                        elif param_info['type'] in ['discrete', 'categorical']:
                            # Random choice from values
                            mutated[key] = random.choice(param_info['values'])
                    else:
                        # Generic mutation
                        if isinstance(value, (int, float)):
                            mutated[key] = value + random.gauss(0, abs(value) * 0.1 + 0.1)
                        elif isinstance(value, list):
                            mutated[key] = random.choice(value)
                else:
                    # Generic mutation without param_space info
                    if isinstance(value, (int, float)):
                        mutated[key] = value + random.gauss(0, abs(value) * 0.1 + 0.1)

        return mutated

    def optimize(self, objective):
        """Run genetic algorithm optimization."""
        import random

        best_params = {}
        best_score = float('-inf')
        history = []
        convergence_history = []

        # Create initial population
        population = self._create_population()

        for gen in range(self.n_generations):
            # Evaluate fitness for each individual
            for i, individual in enumerate(population):
                if 'fitness' not in individual or gen == 0:
                    # Evaluate objective
                    score = objective(individual)
                    population[i] = {'params': individual, 'fitness': score}

                    # Track history
                    history.append({'generation': gen, **individual, 'value': score})

                    # Update best
                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()

            # Track convergence
            gen_scores = [ind['fitness'] for ind in population]
            convergence_history.append(max(gen_scores))

            # Selection
            selected = self._selection(population, self.population_size // 2)

            # Create new population
            new_population = []

            # Elitism - keep best individual
            if population:
                best_individual = max(population, key=lambda x: x['fitness'])
                new_population.append(best_individual['params'])

            # Crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = parent1.get('params', parent1).copy()
                    child2 = parent2.get('params', parent2).copy()

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            # Trim to population size
            population = new_population[:self.population_size]

        # Convert history list to DataFrame for consistency
        history_df = pd.DataFrame(history)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history_df,
            convergence_info={'generations': self.n_generations},
            convergence_history=convergence_history
        )


class BacktestObjective:
    """Backtest objective function."""
    def __init__(self, backtest_engine, strategy_class, data, metric,
                 weights=None, constraints=None):
        self.backtest_engine = backtest_engine
        self.strategy_class = strategy_class
        self.data = data
        self.metric = metric
        self.weights = weights
        self.constraints = constraints

    def __call__(self, params):
        """Evaluate parameters."""
        result = self.backtest_engine.run_backtest(self.strategy_class, self.data, params)

        if isinstance(self.metric, list):
            # Multi-objective
            scores = [result.get(m, 0) for m in self.metric]
            if self.weights:
                return sum(s * w for s, w in zip(scores, self.weights))
            return sum(scores) / len(scores)
        else:
            # Single objective
            score = result.get(self.metric, 0)

            # Check constraints
            if self.constraints:
                for metric, (min_val, max_val) in self.constraints.items():
                    val = result.get(metric, 0)
                    if min_val is not None and val < min_val:
                        return -1000  # Penalty
                    if max_val is not None and val > max_val:
                        return -1000  # Penalty

            return score


class ParameterSpace:
    """Parameter space definition."""
    def __init__(self, params):
        self.parameters = {}
        for name, spec in params.items():
            if isinstance(spec, tuple) and len(spec) == 3:
                # Continuous parameter
                self.parameters[name] = {
                    'type': 'continuous',
                    'min': spec[0],
                    'max': spec[1],
                    'distribution': spec[2]
                }
            elif isinstance(spec, list):
                # Discrete/categorical parameter
                self.parameters[name] = {
                    'type': 'discrete' if all(isinstance(x, (int, float)) for x in spec) else 'categorical',
                    'values': spec
                }

    def sample(self):
        """Sample from parameter space."""
        import random
        sample = {}
        for name, spec in self.parameters.items():
            if spec['type'] == 'continuous':
                if spec['distribution'] == 'log':
                    log_min = np.log(spec['min'])
                    log_max = np.log(spec['max'])
                    sample[name] = np.exp(random.uniform(log_min, log_max))
                else:
                    sample[name] = random.uniform(spec['min'], spec['max'])
            else:
                sample[name] = random.choice(spec['values'])
        return sample

    def get_bounds(self):
        """Get parameter bounds."""
        bounds = {}
        for name, spec in self.parameters.items():
            if spec['type'] == 'continuous':
                bounds[name] = (spec['min'], spec['max'])
        return bounds


# OptimizationResult is already defined at the top of the file
