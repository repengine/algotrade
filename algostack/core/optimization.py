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
from typing import Any, Callable, Optional

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
    best_value: float
    all_results: pd.DataFrame
    plateau_info: Optional[dict[str, Any]] = None
    stability_score: float = 0.0
    convergence_history: Optional[list[float]] = None


class PlateauDetector:
    """Identifies stable parameter regions (plateaus) in optimization landscape."""

    def __init__(
        self,
        min_plateau_size: int = 5,
        stability_threshold: float = 0.1,
        smoothing_factor: float = 1.0,
    ):
        self.min_plateau_size = min_plateau_size
        self.stability_threshold = stability_threshold
        self.smoothing_factor = smoothing_factor

    def find_plateaus(
        self, results_df: pd.DataFrame, metric_col: str = "sharpe"
    ) -> list[dict[str, Any]]:
        """Find stable plateau regions in parameter space."""

        plateaus = []

        # Get unique parameter combinations
        param_cols = [
            col for col in results_df.columns if col not in [metric_col, "value"]
        ]

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

        # Calculate local gradient
        gradient = np.gradient(smoothed)

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

        labeled_array, num_features = label(flat_mask)

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

        # Normalize parameters
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        X = df[param_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Add metric as additional dimension (weighted)
        metric_scaled = (df[metric_col].values - df[metric_col].mean()) / df[
            metric_col
        ].std()
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

        # Phase 2: Find plateaus
        logger.info("Detecting stable parameter regions...")
        plateaus = self.plateau_detector.find_plateaus(coarse_results, "value")

        if not plateaus:
            logger.warning("No plateaus found, using best point")
            best_idx = coarse_results["value"].idxmax()
            best_params = coarse_results.loc[
                best_idx, [col for col in coarse_results.columns if col != "value"]
            ].to_dict()

            return OptimizationResult(
                best_params=best_params,
                best_value=coarse_results.loc[best_idx, "value"],
                all_results=coarse_results,
                plateau_info=None,
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
            best_value=center_value,
            all_results=all_results,
            plateau_info=best_plateau,
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

        # Evaluate in parallel
        results = []

        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count()

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
    ):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler or TPESampler(seed=42)
        self.study_name = study_name or "algostack_optimization"

    def optimize(
        self,
        objective_builder: Callable,
        param_space: dict[str, dict[str, Any]],
        direction: str = "maximize",
        multi_objective: bool = False,
    ) -> OptimizationResult:
        """Run Bayesian optimization."""

        # Create objective function that works with Optuna
        def optuna_objective(trial):
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
            return objective_builder(params, trial)

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
            best_value=best_value,
            all_results=trials_df,
            stability_score=stability_score,
            convergence_history=(
                [t.value for t in study.trials] if not multi_objective else None
            ),
        )

    def _calculate_stability(
        self, trials_df: pd.DataFrame, best_params: dict[str, Any]
    ) -> float:
        """Calculate stability score for optimal parameters."""

        # Get top 10% of trials
        n_top = max(1, len(trials_df) // 10)
        top_trials = trials_df.nlargest(n_top, "value")

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

        return stability


class EnsembleOptimizer:
    """Creates ensemble of parameters from near-optimal solutions."""

    def __init__(self, n_ensemble: int = 5, diversity_weight: float = 0.1):
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
            best_idx = np.argmax(diversity_scores)
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

        return np.mean(distances) if distances else 0


def create_optuna_objective(
    strategy_class,
    data_train: pd.DataFrame,
    data_val: pd.DataFrame,
    backtest_func: Callable,
    cost_model: Optional[Any] = None,
    stability_penalty: float = 0.1,
) -> Callable:
    """Create an Optuna objective function with stability penalty."""

    def objective(trial):
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
            return objective_value

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
