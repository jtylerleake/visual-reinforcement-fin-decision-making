
from common.modules import np, pd, plt, sns, List, Dict, dt, json, os


def get_portfolio_factors(cross_validation_results) -> Dict:
    """Retrieve the portfolio factors lists from visual and numeric agents"""

    try:
        target_strategies = ('Visual agent', 'Numeric agent')        
        factors_data = {'Visual agent': {}, 'Numeric agent': {}}
        
        for fold_id, window_results in cross_validation_results.items():
            for window_id, strategy_results in window_results.items():
                for strategy, stock_results in strategy_results.items():
                    if strategy not in target_strategies: continue
                                        
                    # initialize nested dicts
                    factors_data[strategy].setdefault(fold_id, {}).setdefault(window_id, {})
                    
                    for stock, stock_metrics in stock_results.items():
                        if not isinstance(stock_metrics, dict): continue
                        portfolio_factors = stock_metrics.get('portfolio factors')
                        actions = stock_metrics.get('actions')
                        factors_data[strategy][fold_id][window_id].setdefault(stock, {})                        
                        if portfolio_factors is not None:
                            factors_data[strategy][fold_id][window_id][stock]['Factors'] = portfolio_factors
                        if actions is not None: 
                            factors_data[strategy][fold_id][window_id][stock]['Actions'] = actions
        
        return factors_data
        
    except Exception as e:
        return {'Visual agent': {}, 'Numeric agent': {}}


def compute_stats(values) -> Dict[str, float]:
    """Helper function to compute statistics from a list of values"""
    if not values:
        return {
        'mean': 0.0,
        'std': 0.0,
        'min': 0.0,
        'max': 0.0
        }
    # Convert to numpy array and filter out any NaN/Inf values
    values_array = np.array(values, dtype=np.float64)
    values_array = values_array[~np.isnan(values_array) & ~np.isinf(values_array)]
    
    if len(values_array) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    # For std, need at least 2 values to avoid degrees of freedom warning
    std_val = float(np.std(values_array)) if len(values_array) > 1 else 0.0
    
    return {
        'mean': float(np.mean(values_array)),
        'std': std_val,
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array))
    }


def get_aggregate_stats(cross_validation_results) -> Dict:
    """Aggregate temporal cross-validation results by fold and window and
    compute basic statistics for each"""
    
    try:

        # ignore for aggregation purposes
        ignore = {'actions', 'num steps', 'action distribution', 'portfolio factors'}
        
        # collect data by fold and window
        fold_data, window_data = {}, {}
        
        for fold_id, window_results in cross_validation_results.items():
            for window_id, strategy_results in window_results.items():
                for strategy_name, stock_results in strategy_results.items():
                    
                    # initialize nested dicts
                    fold_data.setdefault(fold_id, {}).setdefault(strategy_name, {})
                    window_data.setdefault(window_id, {}).setdefault(strategy_name, {})
                    
                    for stock, stock_metrics in stock_results.items():
                        for metric, value in stock_metrics.items():
                            if metric in ignore: continue
                            
                            # append values
                            fold_data[fold_id][strategy_name].setdefault(metric, []).append(value)
                            window_data[window_id][strategy_name].setdefault(metric, []).append(value)
        
        # helper function to aggregate grouped data
        def aggregate_across_groups(grouped_data) -> Dict:
            aggregated = {}
            for group_data in grouped_data.values():
                for strategy_name, metric_data in group_data.items():
                    if strategy_name not in aggregated:
                        aggregated[strategy_name] = {}
                    for metric_name, values in metric_data.items():
                        aggregated[strategy_name].setdefault(metric_name, []).extend(values)
            
            # compute stats for each strategy/metric
            return {
                strategy: {
                    metric: compute_stats(values)
                    for metric, values in metrics.items()
                }
                for strategy, metrics in aggregated.items()
            }
        
        # aggregate across folds and windows and compute stats
        fold_stats = aggregate_across_groups(fold_data)
        window_stats = aggregate_across_groups(window_data)
        
        return {
            'Fold-wise': fold_stats,
            'Window-wise': window_stats,
        }
        
    except Exception as e:
        return {}



def get_performance_metrics(
    portfolio_factors,
    start_date,
    end_date,
    sig_figs: int = 4,
    )-> Dict[str, float]:
    """Compute evaluation metrics for a stock test environment"""
    
    risk_free_rate = 0.02 # baseline discount rate
    
    # compute annualization factor
    start_date = dt.strptime(start_date, '%Y-%m-%d')
    end_date = dt.strptime(end_date, '%Y-%m-%d')
    annualization_factor = ((end_date - start_date).days) / 365.25

    # return figures for metric calculation
    portfolio_factors = np.array(portfolio_factors, dtype=np.float64)
    
    # Check if we have enough data points
    if len(portfolio_factors) < 2:
        # Return default values if insufficient data
        return {
            'cumulative return': 0.0,
            'annualized cumulative return': 0.0,
            'sharpe ratio': 0.0,
            'sortino ratio': 0.0,
            'max drawdown': 0.0
        }
    
    interval_returns = np.diff(portfolio_factors) / portfolio_factors[:-1]
    interval_returns = interval_returns[~np.isnan(interval_returns) & ~np.isinf(interval_returns)]
    
    # Check if we have any valid returns
    if len(interval_returns) == 0:
        return {
            'cumulative return': 0.0,
            'annualized cumulative return': 0.0,
            'sharpe ratio': 0.0,
            'sortino ratio': 0.0,
            'max drawdown': 0.0
        }
    
    excess_returns = interval_returns - risk_free_rate
    downside_returns = excess_returns[interval_returns < 0]
    
    # compute mean and std metrics with proper empty array handling
    std_interval_returns = np.std(interval_returns) if len(interval_returns) > 1 else 0.0
    std_downside_returns = np.std(downside_returns) if len(downside_returns) > 1 else 0.0
    avg_excess_return = np.mean(excess_returns) if len(excess_returns) > 0 else 0.0

    # cumulative and annualized return
    if portfolio_factors[0] != 0:
        cumulative_return = (portfolio_factors[-1] - portfolio_factors[0]) / portfolio_factors[0]
    else:
        cumulative_return = 0.0
    
    # Preserve original annualization formula, but add safety check
    if annualization_factor > 0:
        annualized_return = (1 + cumulative_return) ** annualization_factor - 1
    else:
        annualized_return = 0.0

    # sharpe ratio
    if std_interval_returns > 0:
        sharpe_ratio = avg_excess_return / std_interval_returns
    else:
        sharpe_ratio = 0.0
        
    # sortino ratio
    if len(downside_returns) > 0 and std_downside_returns > 0:
        sortino_ratio = avg_excess_return / std_downside_returns
    else:
        sortino_ratio = 0.0

    # maximum drawdown
    running_max = np.maximum.accumulate(portfolio_factors)
    drawdowns = (running_max - portfolio_factors) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return {
        'cumulative return': round(float(cumulative_return), sig_figs),
        'annualized cumulative return': round(float(annualized_return), sig_figs),
        'sharpe ratio': round(float(sharpe_ratio), sig_figs),
        'sortino ratio': round(float(sortino_ratio), sig_figs),
        'max drawdown': round(float(max_drawdown), sig_figs)
    }
