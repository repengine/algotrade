# AlgoStack Professional Backtesting Framework Plan

## My Assessment of Current State vs ChatGPT's Recommendations

### Current Gaps in AlgoStack:
1. **No transaction costs** - Currently assumes zero fees/slippage
2. **No IS/OOS split** - All data used for both parameter selection and testing
3. **Simple parameter selection** - No optimization framework
4. **Limited statistical validation** - Only basic Sharpe/drawdown metrics
5. **No walk-forward analysis** - Static backtests only
6. **No parameter stability checks** - Can overfit to noise

### My Additional Recommendations Beyond ChatGPT:

1. **Monte Carlo Permutation Tests**
   - Shuffle trade entry dates to test if results are due to skill vs luck
   - Bootstrap confidence intervals for Sharpe ratios
   - Test strategy robustness to different market regimes

2. **Regime-Aware Backtesting**
   - Separate performance by VIX regimes (low/medium/high vol)
   - Test across different market cycles (bull/bear/sideways)
   - Ensure strategy works in multiple environments

3. **Portfolio-Level Testing**
   - Test strategies together, not just individually
   - Account for correlation between strategies
   - Optimize for portfolio-level metrics

4. **Market Microstructure Realism**
   - Model bid-ask spreads dynamically (widen in volatility)
   - Implement realistic fill assumptions (partial fills, market impact)
   - Account for overnight gaps and weekend risk

## Detailed Implementation Plan

### Phase 1: Robust Backtesting Engine (Week 1)

#### 1.1 Transaction Cost Module
```python
class TransactionCostModel:
    """Realistic transaction cost modeling"""
    
    def __init__(self, config):
        self.commission_per_share = config.get('commission_per_share', 0.005)
        self.min_commission = config.get('min_commission', 1.0)
        self.spread_model = config.get('spread_model', 'fixed')  # fixed, dynamic, vix_based
        self.base_spread_bps = config.get('base_spread_bps', 5)  # 5 basis points
        self.slippage_model = config.get('slippage_model', 'linear')
        self.market_impact_factor = config.get('market_impact_factor', 0.1)
        
    def calculate_entry_costs(self, price, shares, volatility=None, time_of_day=None):
        # Commission
        commission = max(shares * self.commission_per_share, self.min_commission)
        
        # Spread cost
        if self.spread_model == 'dynamic':
            spread_bps = self.base_spread_bps * (1 + volatility * 2)  # Wider in high vol
        else:
            spread_bps = self.base_spread_bps
        spread_cost = price * shares * (spread_bps / 10000)
        
        # Slippage (market impact)
        if self.slippage_model == 'square_root':
            # Square-root market impact model
            slippage = price * self.market_impact_factor * np.sqrt(shares / avg_daily_volume)
        else:
            slippage = price * shares * 0.0005  # 5 bps fixed
            
        return commission + spread_cost + slippage
```

#### 1.2 IS/OOS Split Framework
```python
class DataSplitter:
    """Handles in-sample/out-of-sample data splitting"""
    
    def __init__(self, split_method='fixed', oos_ratio=0.2):
        self.split_method = split_method
        self.oos_ratio = oos_ratio
        
    def split_data(self, data, method='sequential'):
        if method == 'sequential':
            # Last X% for OOS
            split_point = int(len(data) * (1 - self.oos_ratio))
            return data[:split_point], data[split_point:]
            
        elif method == 'embargo':
            # Add gap between IS and OOS to prevent lookahead
            embargo_days = 63  # 3 months
            split_point = int(len(data) * (1 - self.oos_ratio))
            return data[:split_point-embargo_days], data[split_point:]
            
        elif method == 'purged_kfold':
            # Combinatorial purged cross-validation
            # Prevents data leakage in time series
            return self._purged_kfold_split(data)
```

### Phase 2: Parameter Optimization Framework (Week 2)

#### 2.1 Coarse-to-Fine Grid Search with Plateau Detection
```python
class PlateauOptimizer:
    """Find stable parameter regions, not just peaks"""
    
    def optimize(self, strategy_class, param_ranges, data):
        # Phase 1: Coarse grid
        coarse_results = self._coarse_grid_search(strategy_class, param_ranges, data)
        
        # Phase 2: Identify plateaus (stable regions)
        plateaus = self._find_plateaus(coarse_results)
        
        # Phase 3: Fine search only in plateau regions
        optimal_params = self._refine_plateaus(plateaus)
        
        return optimal_params
        
    def _find_plateaus(self, results_df):
        """Find regions where performance is stable"""
        # Calculate local variance of Sharpe ratio
        sharpe_gradient = np.gradient(results_df['sharpe'].values)
        stable_regions = np.where(np.abs(sharpe_gradient) < threshold)[0]
        
        # Cluster stable regions
        return self._cluster_stable_regions(stable_regions)
```

#### 2.2 Bayesian Optimization with Optuna
```python
def create_objective(strategy_class, data_is, data_oos, cost_model):
    """Multi-objective optimization function"""
    
    def objective(trial):
        # Sample parameters
        params = {
            'lookback': trial.suggest_int('lookback', 10, 100),
            'entry_zscore': trial.suggest_float('entry_zscore', 1.5, 3.5),
            'exit_zscore': trial.suggest_float('exit_zscore', 0.0, 1.0),
            'use_volume_filter': trial.suggest_categorical('use_volume', [True, False])
        }
        
        # Run IS backtest
        is_metrics = run_backtest(strategy_class, params, data_is, cost_model)
        
        # Multi-objective: maximize Sharpe, minimize drawdown
        # Add stability penalty
        neighbor_variance = calculate_neighbor_variance(trial, params)
        
        return (
            -is_metrics['sharpe'] +  # Maximize
            0.5 * is_metrics['max_drawdown'] +  # Minimize
            0.2 * neighbor_variance  # Penalize unstable regions
        )
    
    return objective
```

### Phase 3: Walk-Forward Analysis (Week 3)

#### 3.1 Anchored Walk-Forward
```python
class WalkForwardAnalyzer:
    """Professional walk-forward optimization"""
    
    def __init__(self, config):
        self.window_size = config['window_size']  # e.g., 504 days (2 years)
        self.step_size = config['step_size']      # e.g., 63 days (quarter)
        self.optimization_method = config['optimization_method']
        self.min_trades = config['min_trades']    # Minimum trades to be valid
        
    def run_analysis(self, strategy_class, data, param_ranges):
        results = []
        
        for window_start, window_end, oos_start, oos_end in self._generate_windows(data):
            # Optimize on IS data
            is_data = data[window_start:window_end]
            optimal_params = self._optimize_window(strategy_class, is_data, param_ranges)
            
            # Test on OOS data
            oos_data = data[oos_start:oos_end]
            oos_metrics = self._test_oos(strategy_class, optimal_params, oos_data)
            
            # Store results
            results.append({
                'window': (window_start, window_end),
                'oos_period': (oos_start, oos_end),
                'params': optimal_params,
                'oos_sharpe': oos_metrics['sharpe'],
                'oos_drawdown': oos_metrics['max_drawdown'],
                'param_stability': self._check_param_stability(optimal_params, prev_params)
            })
            
        return self._analyze_results(results)
```

### Phase 4: Statistical Validation (Week 4)

#### 4.1 Monte Carlo Permutation Tests
```python
class MonteCarloValidator:
    """Test if strategy alpha is statistically significant"""
    
    def validate_strategy(self, strategy, data, n_simulations=1000):
        # Get actual strategy results
        actual_results = run_backtest(strategy, data)
        actual_sharpe = actual_results['sharpe']
        
        # Run permutation tests
        random_sharpes = []
        for i in range(n_simulations):
            # Randomly shuffle entry signals
            shuffled_data = self._shuffle_signals(data)
            random_results = run_backtest(strategy, shuffled_data)
            random_sharpes.append(random_results['sharpe'])
            
        # Calculate p-value
        p_value = np.sum(random_sharpes >= actual_sharpe) / n_simulations
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper = np.percentile(random_sharpes, [2.5, 97.5])
        
        return {
            'actual_sharpe': actual_sharpe,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': (actual_sharpe - np.mean(random_sharpes)) / np.std(random_sharpes)
        }
```

#### 4.2 Regime Analysis
```python
class RegimeAnalyzer:
    """Ensure strategy works across different market regimes"""
    
    def analyze_regimes(self, strategy, data):
        # Define regimes
        regimes = self._identify_regimes(data)
        
        results = {}
        for regime_name, regime_data in regimes.items():
            if len(regime_data) < min_days:
                continue
                
            metrics = run_backtest(strategy, regime_data)
            results[regime_name] = {
                'sharpe': metrics['sharpe'],
                'drawdown': metrics['max_drawdown'],
                'num_trades': metrics['num_trades'],
                'win_rate': metrics['win_rate'],
                'days': len(regime_data)
            }
            
        # Check consistency across regimes
        sharpe_values = [r['sharpe'] for r in results.values()]
        consistency_score = 1 - (np.std(sharpe_values) / np.mean(sharpe_values))
        
        return results, consistency_score
```

### Phase 5: Implementation in Dashboard

#### 5.1 Enhanced Backtest Configuration
```python
# In dashboard_pandas.py
st.sidebar.subheader("Backtest Configuration")

# Data split
split_method = st.sidebar.selectbox(
    "Data Split Method",
    ["Sequential", "Embargo", "Purged K-Fold"],
    help="Sequential: simple train/test split. Embargo: gap between train/test. Purged: advanced CV"
)

oos_percentage = st.sidebar.slider(
    "Out-of-Sample %",
    min_value=10,
    max_value=40,
    value=20,
    help="Reserve this % of data for out-of-sample testing"
)

# Transaction costs
st.sidebar.subheader("Transaction Costs")
enable_costs = st.sidebar.checkbox("Enable Transaction Costs", value=True)

if enable_costs:
    commission_model = st.sidebar.selectbox(
        "Commission Model",
        ["Fixed Per Share", "Percentage", "Tiered"]
    )
    
    spread_model = st.sidebar.selectbox(
        "Spread Model",
        ["Fixed", "Dynamic (VIX-based)", "Time-of-day"]
    )
    
    slippage_bps = st.sidebar.number_input(
        "Slippage (bps)",
        min_value=0,
        max_value=50,
        value=5,
        help="Expected slippage in basis points"
    )

# Statistical validation
st.sidebar.subheader("Validation")
run_monte_carlo = st.sidebar.checkbox(
    "Run Monte Carlo Validation",
    help="Test if results are statistically significant"
)

run_regime_analysis = st.sidebar.checkbox(
    "Run Regime Analysis",
    help="Test strategy across different market conditions"
)
```

#### 5.2 Results Display
```python
# Enhanced results section
if backtest_results:
    # Main metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # IS/OOS comparison
    with col1:
        st.metric("IS Sharpe", f"{is_sharpe:.2f}")
        st.metric("OOS Sharpe", f"{oos_sharpe:.2f}", 
                  delta=f"{oos_sharpe - is_sharpe:.2f}")
    
    # Statistical significance
    if monte_carlo_results:
        with col2:
            st.metric("P-Value", f"{p_value:.3f}")
            st.metric("Significant", "Yes ✓" if p_value < 0.05 else "No ✗")
    
    # Regime consistency
    if regime_results:
        with col3:
            st.metric("Regime Consistency", f"{consistency_score:.1%}")
            st.metric("Worst Regime Sharpe", f"{worst_regime_sharpe:.2f}")
    
    # Parameter stability (from walk-forward)
    if walk_forward_results:
        with col4:
            st.metric("Parameter Stability", f"{param_stability:.1%}")
            st.metric("OOS Decay", f"{oos_decay:.1%}")
```

### Phase 6: AI-Driven Strategy Refinement

#### 6.1 Drawdown Control Framework
```python
class DrawdownController:
    """AI-assisted drawdown management"""
    
    def suggest_improvements(self, backtest_results, strategy_params):
        suggestions = []
        
        # Analyze drawdown patterns
        dd_analysis = self._analyze_drawdowns(backtest_results)
        
        # Suggest dynamic position sizing
        if dd_analysis['drawdowns_cluster']:
            suggestions.append({
                'type': 'dynamic_sizing',
                'description': 'Reduce position size after consecutive losses',
                'implementation': 'scale_factor = max(0.5, 1 - (consecutive_losses * 0.1))'
            })
        
        # Suggest regime filters
        if dd_analysis['regime_dependent']:
            suggestions.append({
                'type': 'regime_filter',
                'description': 'Avoid trading in high volatility regimes',
                'implementation': 'if VIX > 25: position_size *= 0.5'
            })
        
        return suggestions
```

## Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Implement TransactionCostModel
- [ ] Add IS/OOS data splitting
- [ ] Update backtesting engine with realistic fills

### Week 2: Optimization Framework  
- [ ] Implement plateau-finding optimizer
- [ ] Integrate Optuna for Bayesian optimization
- [ ] Add parameter stability metrics

### Week 3: Walk-Forward Analysis
- [ ] Build walk-forward engine
- [ ] Add anchored and rolling windows
- [ ] Implement parameter evolution tracking

### Week 4: Statistical Validation
- [ ] Add Monte Carlo permutation tests
- [ ] Implement regime analysis
- [ ] Create validation report generator

### Week 5: Dashboard Integration
- [ ] Update UI with new backtest options
- [ ] Add visualization for optimization results
- [ ] Create exportable backtest reports

### Week 6: AI Enhancement Tools
- [ ] Build suggestion engine for parameter tuning
- [ ] Add automated drawdown analysis
- [ ] Create strategy improvement recommendations

## Key Metrics to Track

1. **Performance Decay**: `(IS_Sharpe - OOS_Sharpe) / IS_Sharpe`
2. **Parameter Stability**: Standard deviation of optimal params across windows
3. **Regime Robustness**: Minimum Sharpe across all regimes
4. **Statistical Significance**: Monte Carlo p-value < 0.05
5. **Transaction Cost Impact**: `(Gross_Sharpe - Net_Sharpe) / Gross_Sharpe`

## Success Criteria

A strategy is considered "production-ready" when:
1. OOS Sharpe > 0.5 (after costs)
2. Performance decay < 30%
3. Statistically significant (p < 0.05)
4. Works in at least 3 different market regimes
5. Parameter stability > 70%
6. Maximum drawdown < 20%
7. Transaction costs < 30% of gross returns