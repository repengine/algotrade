"""
Enhanced backtesting engine with professional features.

This module implements:
- Transaction cost modeling (commission, spread, slippage)
- In-sample/Out-of-sample data splitting
- Walk-forward analysis
- Statistical validation (Monte Carlo, regime analysis)
- Parameter optimization with stability checks
"""

import numpy as np
import pandas as pd
from typing import Optional, Any, Callable, Dict, List
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransactionCosts:
    """Container for transaction cost components."""
    commission: float
    spread_cost: float
    slippage: float
    total: float


class TransactionCostModel:
    """Realistic transaction cost modeling."""
    
    def __init__(self, config: dict[str, Any]):
        # Commission settings
        self.commission_per_share = config.get('commission_per_share', 0.005)
        self.min_commission = config.get('min_commission', 1.0)
        self.commission_type = config.get('commission_type', 'per_share')  # per_share, percentage
        
        # Spread modeling
        self.spread_model = config.get('spread_model', 'fixed')  # fixed, dynamic, vix_based
        self.base_spread_bps = config.get('base_spread_bps', 5)  # 5 basis points
        
        # Slippage modeling
        self.slippage_model = config.get('slippage_model', 'linear')  # linear, square_root
        self.market_impact_factor = config.get('market_impact_factor', 0.1)
        self.urgency_factor = config.get('urgency_factor', 1.0)  # 1.0 = normal, >1 = urgent
        
    def calculate_costs(self, 
                       price: float, 
                       shares: int, 
                       side: str,
                       volatility: float = 0.02,
                       avg_daily_volume: int = 1000000,
                       time_of_day: Optional[str] = None) -> TransactionCosts:
        """Calculate all transaction costs for a trade."""
        
        # Commission
        if self.commission_type == 'per_share':
            commission = max(shares * self.commission_per_share, self.min_commission)
        else:  # percentage
            commission = price * shares * self.commission_per_share / 100
            
        # Spread cost
        spread_bps = self._calculate_spread(volatility, time_of_day)
        spread_cost = price * shares * (spread_bps / 10000)
        
        # Slippage (market impact)
        slippage = self._calculate_slippage(price, shares, avg_daily_volume, side)
        
        # Total cost
        total = commission + spread_cost + slippage
        
        return TransactionCosts(
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            total=total
        )
    
    def _calculate_spread(self, volatility: float, time_of_day: Optional[str]) -> float:
        """Calculate bid-ask spread in basis points."""
        base_spread = self.base_spread_bps
        
        if self.spread_model == 'dynamic':
            # Spread widens with volatility
            vol_multiplier = 1 + (volatility - 0.02) * 10  # Baseline 2% vol
            base_spread *= max(0.5, min(3.0, vol_multiplier))
            
        elif self.spread_model == 'vix_based':
            # Use VIX as proxy (simplified - would need actual VIX data)
            vix_estimate = volatility * np.sqrt(252) * 100
            if vix_estimate > 30:
                base_spread *= 2.0
            elif vix_estimate > 20:
                base_spread *= 1.5
                
        # Time of day adjustment
        if time_of_day:
            hour = int(time_of_day.split(':')[0])
            if hour < 10 or hour >= 15:  # First/last hour
                base_spread *= 1.5
                
        return base_spread
    
    def _calculate_slippage(self, price: float, shares: int, 
                           avg_daily_volume: int, side: str) -> float:
        """Calculate market impact (slippage)."""
        
        # Participation rate (what % of volume are we)
        participation = shares / avg_daily_volume
        
        if self.slippage_model == 'square_root':
            # Square-root market impact model (Almgren et al.)
            # Impact = spread * (volume_fraction)^0.5
            spread_decimal = self.base_spread_bps / 10000
            impact = spread_decimal * np.sqrt(participation) * self.market_impact_factor
            
        elif self.slippage_model == 'linear':
            # Simple linear impact
            impact = 0.0001 * participation  # 1 bp per 1% of volume
            
        else:  # fixed
            impact = 0.0005  # 5 bps
            
        # Adjust for urgency
        impact *= self.urgency_factor
        
        # Slippage cost
        slippage = price * shares * impact
        
        # Buy orders pay more, sell orders receive less
        if side == 'SELL':
            slippage *= -1  # Negative slippage for sells
            
        return abs(slippage)


class DataSplitter:
    """Handles various data splitting methods for backtesting."""
    
    def __init__(self, method: str = 'sequential', oos_ratio: float = 0.2):
        self.method = method
        self.oos_ratio = oos_ratio
        self.embargo_days = 63  # 3-month embargo for some methods
        
    def split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into in-sample and out-of-sample sets."""
        
        if self.method == 'sequential':
            return self._sequential_split(data)
        elif self.method == 'embargo':
            return self._embargo_split(data)
        elif self.method == 'purged_kfold':
            # For purged k-fold, return the first split as a default
            # Use get_all_splits() to access all k-fold splits
            generator = self._purged_kfold_split(data)
            return next(generator)
        else:
            raise ValueError(f"Unknown split method: {self.method}")
            
    def _sequential_split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simple sequential split - last X% for OOS."""
        split_idx = int(len(data) * (1 - self.oos_ratio))
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    def _embargo_split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split with embargo period to prevent lookahead bias."""
        split_idx = int(len(data) * (1 - self.oos_ratio))
        embargo_start = max(0, split_idx - self.embargo_days)
        
        is_data = data.iloc[:embargo_start]
        oos_data = data.iloc[split_idx:]
        
        return is_data, oos_data
    
    def _purged_kfold_split(self, data: pd.DataFrame, n_splits: int = 5):
        """Purged K-fold cross-validation for time series."""
        # This returns a generator of train/test splits
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=self.embargo_days)
        
        for train_idx, test_idx in tscv.split(data):
            yield data.iloc[train_idx], data.iloc[test_idx]
    
    def get_all_splits(self, data: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Get all splits for methods that support multiple splits."""
        if self.method == 'purged_kfold':
            return list(self._purged_kfold_split(data))
        else:
            # For other methods, return single split as a list
            return [self.split(data)]


class WalkForwardAnalyzer:
    """Implements walk-forward analysis for robust parameter selection."""
    
    def __init__(self, 
                 window_size: int = 504,  # 2 years
                 step_size: int = 63,     # 3 months
                 min_window_size: int = 252,  # 1 year minimum
                 optimization_metric: str = 'sharpe',
                 max_windows: int = 10):  # Limit number of windows
        
        self.window_size = window_size
        self.step_size = step_size
        self.min_window_size = min_window_size
        self.optimization_metric = optimization_metric
        self.max_windows = max_windows
        
    def run_analysis(self,
                    strategy_class: type,
                    data: pd.DataFrame,
                    param_grid: dict[str, list[Any]],
                    backtest_func: Callable,
                    cost_model: Optional[TransactionCostModel] = None) -> pd.DataFrame:
        """Run walk-forward analysis."""
        
        results = []
        windows = list(self._generate_windows(data))
        
        # Limit number of windows
        if len(windows) > self.max_windows:
            logger.warning(f"Limiting walk-forward from {len(windows)} to {self.max_windows} windows")
            windows = windows[:self.max_windows]
        
        logger.info(f"Running walk-forward analysis with {len(windows)} windows")
        
        for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}: IS {is_start.date()} to {is_end.date()}, "
                       f"OOS {oos_start.date()} to {oos_end.date()}")
            
            # Get data slices
            is_data = data[is_start:is_end]
            oos_data = data[oos_start:oos_end]
            
            # Optimize on in-sample data
            logger.info(f"  Optimizing {len(param_grid)} parameters on {len(is_data)} days of IS data...")
            optimal_params = self._optimize_parameters(
                strategy_class, is_data, param_grid, backtest_func, cost_model
            )
            logger.info(f"  Optimal params found: {optimal_params}")
            
            # Test on out-of-sample data
            logger.info(f"  Testing on {len(oos_data)} days of OOS data...")
            try:
                oos_strategy = strategy_class(optimal_params)
                oos_results = backtest_func(
                    oos_strategy,
                    oos_data,
                    cost_model=cost_model
                )
            except Exception as e:
                logger.error(f"OOS backtest failed: {e}")
                oos_results = {
                    'sharpe_ratio': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'num_trades': 0
                }
            
            # Store results
            results.append({
                'window_num': i + 1,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'optimal_params': optimal_params,
                'is_sharpe': optimal_params.get('_is_sharpe', 0) if optimal_params else 0,
                'oos_sharpe': oos_results.get('sharpe_ratio', 0),
                'oos_return': oos_results.get('total_return', 0),
                'oos_drawdown': oos_results.get('max_drawdown', 0),
                'oos_trades': oos_results.get('num_trades', 0),
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        if len(results_df) > 0:
            results_df['sharpe_decay'] = (
                (results_df['is_sharpe'] - results_df['oos_sharpe']) / 
                results_df['is_sharpe'].replace(0, 1)
            )
        else:
            results_df['sharpe_decay'] = 0
        
        return results_df
    
    def _generate_windows(self, data: pd.DataFrame):
        """Generate walk-forward windows."""
        
        data_start = data.index[0]
        data_end = data.index[-1]
        
        current_start = data_start
        
        while current_start + pd.Timedelta(days=self.window_size + self.step_size) <= data_end:
            is_end = current_start + pd.Timedelta(days=self.window_size)
            oos_start = is_end
            oos_end = oos_start + pd.Timedelta(days=self.step_size)
            
            yield current_start, is_end, oos_start, oos_end
            
            current_start += pd.Timedelta(days=self.step_size)
    
    def _optimize_parameters(self,
                           strategy_class: type,
                           data: pd.DataFrame,
                           param_grid: dict[str, list[Any]],
                           backtest_func: Callable,
                           cost_model: Optional[TransactionCostModel]) -> dict[str, Any]:
        """Find optimal parameters for a given window."""
        
        # If no parameters to optimize, return empty dict
        if not param_grid:
            return {}
        
        # Simple grid search for now - can be replaced with Bayesian optimization
        best_metric = -np.inf
        best_params = {}
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Count total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        # Log progress only at key points
        combination_count = 0
        
        for values in itertools.product(*param_values):
            combination_count += 1
            params = dict(zip(param_names, values))
            
            # Suppress output during optimization by redirecting stdout
            import io
            import contextlib
            
            # Run backtest with suppressed output
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    strategy = strategy_class(params)
                    results = backtest_func(strategy, data, cost_model=cost_model)
                except Exception as e:
                    logger.error(f"Backtest failed for params {params}: {e}")
                    results = {'sharpe_ratio': -np.inf}  # Return bad result to skip
            
            # Check if better
            metric_value = results.get(self.optimization_metric, -np.inf)
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params.copy()
                best_params['_is_sharpe'] = results.get('sharpe_ratio', 0)
            
            # Log progress every 25% of combinations
            if combination_count % max(1, total_combinations // 4) == 0:
                progress = (combination_count / total_combinations) * 100
                logger.info(f"Optimization progress: {progress:.0f}% ({combination_count}/{total_combinations} combinations)")
        
        logger.info(f"Best {self.optimization_metric}: {best_metric:.3f}")
        return best_params


class MonteCarloValidator:
    """Statistical validation using Monte Carlo methods."""
    
    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        
    def validate_strategy(self,
                         strategy_results: dict[str, Any],
                         benchmark_returns: Optional[pd.Series] = None) -> dict[str, Any]:
        """Validate strategy results using Monte Carlo simulation."""
        
        actual_sharpe = strategy_results.get('sharpe_ratio', 0)
        actual_returns = strategy_results.get('returns_series', pd.Series())
        
        if actual_returns.empty:
            return {'error': 'No returns series provided'}
        
        # Method 1: Random entry/exit simulation
        random_sharpes = self._simulate_random_entries(actual_returns)
        
        # Method 2: Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_confidence_interval(actual_returns)
        
        # Calculate p-value
        p_value = np.sum(np.array(random_sharpes) >= actual_sharpe) / len(random_sharpes)
        
        # Effect size (Cohen's d)
        effect_size = (actual_sharpe - np.mean(random_sharpes)) / np.std(random_sharpes)
        
        validation_results = {
            'actual_sharpe': actual_sharpe,
            'random_mean_sharpe': np.mean(random_sharpes),
            'random_std_sharpe': np.std(random_sharpes),
            'p_value': p_value,
            'significant': p_value < (1 - self.confidence_level),
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': effect_size,
            'interpretation': self._interpret_results(p_value, effect_size)
        }
        
        # Compare to benchmark if provided
        if benchmark_returns is not None:
            validation_results['vs_benchmark'] = self._compare_to_benchmark(
                actual_returns, benchmark_returns
            )
        
        return validation_results
    
    def _simulate_random_entries(self, returns: pd.Series) -> list[float]:
        """Simulate random entry/exit points."""
        random_sharpes = []
        
        for _ in range(self.n_simulations):
            # Randomly shuffle returns
            shuffled_returns = returns.sample(frac=1, replace=False).values
            
            # Calculate Sharpe ratio
            if len(shuffled_returns) > 1 and np.std(shuffled_returns) > 0:
                sharpe = np.sqrt(252) * float(np.mean(shuffled_returns)) / float(np.std(shuffled_returns))
                random_sharpes.append(sharpe)
            else:
                random_sharpes.append(0)
                
        return random_sharpes
    
    def _bootstrap_confidence_interval(self, returns: pd.Series) -> tuple[float, float]:
        """Calculate bootstrap confidence intervals for Sharpe ratio."""
        bootstrap_sharpes = []
        
        for _ in range(self.n_simulations):
            # Resample with replacement
            sample_returns = returns.sample(frac=1, replace=True)
            
            if len(sample_returns) > 1 and sample_returns.std() > 0:
                sharpe = np.sqrt(252) * sample_returns.mean() / sample_returns.std()
                bootstrap_sharpes.append(sharpe)
        
        # Check if we have any bootstrap samples
        if len(bootstrap_sharpes) == 0:
            # Return the actual Sharpe for both bounds if no samples
            if len(returns) > 1 and returns.std() > 0:
                actual_sharpe = np.sqrt(252) * returns.mean() / returns.std()
                return actual_sharpe, actual_sharpe
            else:
                return 0.0, 0.0
        
        # Calculate percentiles
        alpha = (1 - self.confidence_level) / 2
        ci_lower = np.percentile(bootstrap_sharpes, alpha * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha) * 100)
        
        return float(ci_lower), float(ci_upper)
    
    def _interpret_results(self, p_value: float, effect_size: float) -> str:
        """Provide interpretation of statistical results."""
        
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.10:
            significance = "marginally significant"
        else:
            significance = "not significant"
            
        if abs(effect_size) < 0.2:
            effect = "negligible"
        elif abs(effect_size) < 0.5:
            effect = "small"
        elif abs(effect_size) < 0.8:
            effect = "medium"
        else:
            effect = "large"
            
        return f"Results are {significance} (p={p_value:.3f}) with {effect} effect size (d={effect_size:.2f})"
    
    def _compare_to_benchmark(self, 
                             strategy_returns: pd.Series, 
                             benchmark_returns: pd.Series) -> dict[str, Any]:
        """Compare strategy to benchmark using various metrics."""
        
        # Align series
        aligned_strat, aligned_bench = strategy_returns.align(benchmark_returns, join='inner')
        
        # Calculate metrics
        excess_returns = aligned_strat - aligned_bench
        
        # Information ratio
        if excess_returns.std() > 0:
            info_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            info_ratio = 0
            
        # Win rate (days strategy beats benchmark)
        win_rate = (aligned_strat > aligned_bench).mean()
        
        # T-test for difference in means
        t_stat, t_pvalue = stats.ttest_rel(aligned_strat, aligned_bench)
        
        return {
            'information_ratio': info_ratio,
            'win_rate': win_rate,
            'excess_return_mean': excess_returns.mean() * 252,
            'excess_return_std': excess_returns.std() * np.sqrt(252),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'outperforms': t_pvalue < 0.05 and excess_returns.mean() > 0
        }


class RegimeAnalyzer:
    """Analyze strategy performance across different market regimes."""
    
    def __init__(self, regime_method: str = 'volatility'):
        self.regime_method = regime_method
        
    def identify_regimes(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Identify different market regimes in the data."""
        
        if self.regime_method == 'volatility':
            return self._volatility_regimes(data)
        elif self.regime_method == 'trend':
            return self._trend_regimes(data)
        elif self.regime_method == 'combined':
            return self._combined_regimes(data)
        else:
            raise ValueError(f"Unknown regime method: {self.regime_method}")
    
    def _volatility_regimes(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split data by volatility regimes."""
        
        # Handle both uppercase and lowercase column names
        close_col = 'Close' if 'Close' in data.columns else 'close'
        
        # Calculate rolling volatility
        returns = data[close_col].pct_change()
        volatility = returns.rolling(window=21).std() * np.sqrt(252)
        
        # Define regime thresholds
        vol_percentiles = volatility.quantile([0.33, 0.67])
        
        regimes = {
            'low_volatility': data[volatility <= vol_percentiles.iloc[0]],
            'medium_volatility': data[(volatility > vol_percentiles.iloc[0]) & 
                                    (volatility <= vol_percentiles.iloc[1])],
            'high_volatility': data[volatility > vol_percentiles.iloc[1]]
        }
        
        return regimes
    
    def _trend_regimes(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split data by trend regimes."""
        
        # Handle both uppercase and lowercase column names
        close_col = 'Close' if 'Close' in data.columns else 'close'
        
        # Calculate trend using moving averages
        ma_short = data[close_col].rolling(window=50).mean()
        ma_long = data[close_col].rolling(window=200).mean()
        
        # Define regimes
        uptrend = (ma_short > ma_long) & (data[close_col] > ma_short)
        downtrend = (ma_short < ma_long) & (data[close_col] < ma_short)
        
        regimes = {
            'uptrend': data[uptrend],
            'downtrend': data[downtrend],
            'sideways': data[~uptrend & ~downtrend]
        }
        
        return regimes
    
    def _combined_regimes(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Identify combined volatility and trend regimes."""
        vol_regimes = self._volatility_regimes(data)
        trend_regimes = self._trend_regimes(data)
        
        combined_regimes = {}
        
        # Get volatility regime labels
        vol_labels = pd.Series(index=data.index, dtype=str)
        for regime_name, regime_data in vol_regimes.items():
            vol_labels.loc[regime_data.index] = regime_name.split('-')[0]  # 'low' or 'high'
        
        # Get trend regime labels  
        trend_labels = pd.Series(index=data.index, dtype=str)
        for regime_name, regime_data in trend_regimes.items():
            trend_labels.loc[regime_data.index] = regime_name.split('-')[0]  # 'bull' or 'bear'
        
        # Combine labels
        for vol in ['low', 'high']:
            for trend in ['bull', 'bear']:
                mask = (vol_labels == vol) & (trend_labels == trend)
                if mask.sum() > 0:
                    regime_name = f"{vol}-vol-{trend}-market"
                    combined_regimes[regime_name] = data[mask]
        
        return combined_regimes
    
    def analyze_regime_performance(self,
                                 strategy,
                                 data: pd.DataFrame,
                                 backtest_func: Callable) -> dict[str, Any]:
        """Analyze strategy performance across regimes."""
        
        # Identify regimes
        regimes = self.identify_regimes(data)
        
        results = {}
        for regime_name, regime_data in regimes.items():
            if len(regime_data) < 100:  # Skip if too little data
                continue
                
            # Run backtest for this regime
            regime_results = backtest_func(strategy, regime_data)
            
            results[regime_name] = {
                'days': len(regime_data),
                'sharpe': regime_results.get('sharpe_ratio', 0),
                'return': regime_results.get('total_return', 0),
                'max_drawdown': regime_results.get('max_drawdown', 0),
                'win_rate': regime_results.get('win_rate', 0),
                'num_trades': regime_results.get('num_trades', 0)
            }
        
        # Calculate consistency metrics
        sharpe_values = [r['sharpe'] for r in results.values() if r['sharpe'] is not None]
        
        if len(sharpe_values) > 1:
            consistency_score = 1 - (np.std(sharpe_values) / (np.mean(sharpe_values) + 1e-6))
            worst_regime_sharpe = min(sharpe_values)
        else:
            consistency_score = 0
            worst_regime_sharpe = 0
            
        return {
            'regime_results': results,
            'consistency_score': consistency_score,
            'worst_regime_sharpe': worst_regime_sharpe,
            'all_positive': all(s > 0 for s in sharpe_values)
        }


def create_backtest_report(results: dict[str, Any]) -> str:
    """Generate a comprehensive backtest report."""
    
    report = []
    report.append("# Backtest Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Performance Summary
    report.append("## Performance Summary")
    report.append(f"- Total Return: {results['total_return']:.2f}%")
    report.append(f"- Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    report.append(f"- Max Drawdown: {results['max_drawdown']:.2f}%")
    report.append(f"- Win Rate: {results['win_rate']:.2%}")
    report.append("")
    
    # Statistical Validation
    if 'validation' in results:
        report.append("## Statistical Validation")
        val = results['validation']
        report.append(f"- P-Value: {val['p_value']:.3f}")
        report.append(f"- Significant: {'Yes' if val['significant'] else 'No'}")
        report.append(f"- Effect Size: {val['effect_size']:.2f}")
        report.append(f"- Interpretation: {val['interpretation']}")
        report.append("")
    
    # Regime Analysis
    if 'regime_analysis' in results:
        report.append("## Regime Analysis")
        regime = results['regime_analysis']
        report.append(f"- Consistency Score: {regime['consistency_score']:.2%}")
        report.append(f"- Worst Regime Sharpe: {regime['worst_regime_sharpe']:.2f}")
        report.append(f"- Positive in All Regimes: {'Yes' if regime['all_positive'] else 'No'}")
        report.append("")
        
        report.append("### Regime Performance:")
        for regime_name, metrics in regime['regime_results'].items():
            report.append(f"- {regime_name}: Sharpe={metrics['sharpe']:.2f}, "
                         f"Return={metrics['return']:.2f}%, "
                         f"MaxDD={metrics['max_drawdown']:.2f}%")
        report.append("")
    
    # Walk-Forward Results
    if 'walk_forward' in results:
        report.append("## Walk-Forward Analysis")
        wf_df = results['walk_forward']
        report.append(f"- Average OOS Sharpe: {wf_df['oos_sharpe'].mean():.2f}")
        report.append(f"- Average Decay: {wf_df['sharpe_decay'].mean():.2%}")
        report.append(f"- Consistency: {(wf_df['oos_sharpe'] > 0).mean():.2%} positive")
        report.append("")
    
    # Transaction Costs
    if 'transaction_costs' in results:
        report.append("## Transaction Cost Analysis")
        tc = results['transaction_costs']
        report.append(f"- Total Costs: ${tc['total_costs']:,.2f}")
        report.append(f"- Cost as % of Gross: {tc['cost_percentage']:.2%}")
        report.append(f"- Average Cost per Trade: ${tc['avg_cost_per_trade']:.2f}")
        report.append("")
    
    return "\n".join(report)