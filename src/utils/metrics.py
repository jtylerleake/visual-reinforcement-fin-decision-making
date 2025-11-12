
from common.modules import np, pd, plt, sns, List, Dict, dt
from src.utils.logging import get_logger, log_function_call



def convert_to_latex_table():
    """ """
    raise NotImplementedError

def plot_performance():
    """ """
    raise NotImplementedError

def plot_trade_overlap():
    """ """
    raise NotImplementedError





def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Helper function to compute statistics from a list of values"""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def aggregate_cv_results(cv_results: Dict) -> Dict:
    """

    """
    
    try:
        # Strategy name mapping: normalize from 'macd results' -> 'macd'
        strategy_mapping = {
            'macd results': 'macd',
            'signr results': 'signr',
            'buyandhold results': 'buyandhold',
            'random results': 'random'
        }
        
        # Metric name mapping: normalize from dict keys to standard names
        metric_mapping = {
            'episode reward': 'episode_reward',
            'cumulative return': 'cumulative_return',
            'annualized return': 'annualized_return',
            'sharpe ratio': 'sharpe_ratio',
            'sortino ratio': 'sortino_ratio',
            'max drawdown': 'max_drawdown'
        }
        


        # Initialize data collectors
        fold_data = {}  # {fold_id: {strategy: {metric: [values]}}}
        window_data = {}  # {window_id: {strategy: {metric: [values]}}}
        overall_data = {}  # {strategy: {metric: [values]}}
        
        total_stocks = 0
        
        # Iterate through fold_results structure
        for fold_id, window_results in cv_results.items():

            if fold_id not in fold_data:
                fold_data[fold_id] = {}
            
            for window_id, fold_window_results in window_results.items():
                # Windows are 1-indexed in output
                window_key = window_id + 1 if isinstance(window_id, int) else window_id
                if window_key not in window_data:
                    window_data[window_key] = {}
                


                # Process each benchmark strategy result
                for result_key, strategy_result in fold_window_results.items():
                    # Skip non-benchmark keys
                    if result_key not in strategy_mapping:
                        continue
                    
                    strategy_name = strategy_mapping[result_key]
                    
                    # Initialize strategy dicts if needed
                    if strategy_name not in fold_data[fold_id]:
                        fold_data[fold_id][strategy_name] = {}
                    if strategy_name not in window_data[window_key]:
                        window_data[window_key][strategy_name] = {}
                    if strategy_name not in overall_data:
                        overall_data[strategy_name] = {}
                    


                    # Extract stock_metrics from strategy_result
                    # strategy_result should be a dict with stock tickers as keys
                    if not isinstance(strategy_result, dict):
                        continue
                    
                    for stock, stock_metrics in strategy_result.items():
                        if not isinstance(stock_metrics, dict):
                            continue
                        


                        
                        total_stocks += 1
                        
                        # Extract each metric
                        for metric_key, metric_name in metric_mapping.items():
                            if metric_key not in stock_metrics:
                                continue
                            
                            value = stock_metrics[metric_key]
                            
                            # Initialize metric lists if needed
                            if metric_name not in fold_data[fold_id][strategy_name]:
                                fold_data[fold_id][strategy_name][metric_name] = []
                            if metric_name not in window_data[window_key][strategy_name]:
                                window_data[window_key][strategy_name][metric_name] = []
                            if metric_name not in overall_data[strategy_name]:
                                overall_data[strategy_name][metric_name] = []
                            
                            # Collect values
                            fold_data[fold_id][strategy_name][metric_name].append(value)
                            window_data[window_key][strategy_name][metric_name].append(value)
                            overall_data[strategy_name][metric_name].append(value)
        
        # Aggregate by fold
        aggregated_by_fold = {}
        for fold_id, strategy_data in fold_data.items():
            aggregated_by_fold[fold_id] = {}
            for strategy_name, metric_data in strategy_data.items():
                aggregated_by_fold[fold_id][strategy_name] = {}
                for metric_name, values in metric_data.items():
                    aggregated_by_fold[fold_id][strategy_name][metric_name] = _compute_stats(values)
        
        # Aggregate by window
        aggregated_by_window = {}
        for window_id, strategy_data in window_data.items():
            aggregated_by_window[window_id] = {}
            for strategy_name, metric_data in strategy_data.items():
                aggregated_by_window[window_id][strategy_name] = {}
                for metric_name, values in metric_data.items():
                    aggregated_by_window[window_id][strategy_name][metric_name] = _compute_stats(values)
        
        # Aggregate overall
        aggregated_overall = {}
        for strategy_name, metric_data in overall_data.items():
            aggregated_overall[strategy_name] = {}
            for metric_name, values in metric_data.items():
                aggregated_overall[strategy_name][metric_name] = _compute_stats(values)
        
        # Build final structure
        aggregated = {
            'summary': {
                'total_folds': len(cv_results),
                'total_windows': sum(len(window_results) for window_results in cv_results.values()),
                'total_stocks_evaluated': total_stocks
            },
            'aggregated_metrics': {
                'by_fold': aggregated_by_fold,
                'by_window': aggregated_by_window,
                'overall': aggregated_overall
            },
            'fold_results': cv_results  # Keep original for reference
        }
        
        return aggregated
        
    except Exception as e:
        logger = get_logger("aggregate_cv_results")
        logger.error(f"Error aggregating CV results: {e}")
        return {}




def compute_performance_metrics(
        portfolio_factors: List[float],
        start_date: str, 
        end_date: str,
        sig_figs: int = 4,
    )-> Dict[str, float]:
    """Compute experiment evaluation metrics for window/fold test environment"""
    
    risk_free_rate = 0.02 # baseline discount rate
    
    # annualization factor
    start_date = dt.strptime(start_date, '%Y-%m-%d')
    end_date = dt.strptime(end_date, '%Y-%m-%d')
    days_difference = (end_date - start_date).days
    annualization_factor = days_difference / 365.25

    # return figures for metric calculation
    portfolio_factors = np.array(portfolio_factors, dtype=np.float64)
    interval_returns = np.diff(portfolio_factors) / portfolio_factors[:-1]
    interval_returns = interval_returns[~np.isnan(interval_returns)]
    excess_returns = interval_returns - risk_free_rate
    downside_returns = excess_returns[interval_returns < 0]
    std_interval_returns = np.std(interval_returns)
    std_downside_returns = np.std(downside_returns)
    avg_excess_return = np.mean(excess_returns)

    # cumulative and annualized return
    cumulative_return = (portfolio_factors[-1] - portfolio_factors[0]) / portfolio_factors[0]
    annualized_return = (1 + cumulative_return) ** annualization_factor - 1

    # sharpe and sortino ratios
    if std_interval_returns > 0:
        sharpe_ratio = avg_excess_return / std_interval_returns
    else:
        sharpe_ratio = 0.0
    if len(downside_returns) > 0 and std_downside_returns > 0:
        sortino_ratio = avg_excess_return / std_downside_returns
    else:
        sortino_ratio = 0.0

    # maximum drawdown
    running_max = np.maximum.accumulate(portfolio_factors)
    drawdowns = (running_max - portfolio_factors) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return {
        'cumulative_return': round(float(cumulative_return), sig_figs),
        'annualized_cumulative_return': round(float(annualized_return), sig_figs),
        'sharpe_ratio': round(float(sharpe_ratio), sig_figs),
        'sortino_ratio': round(float(sortino_ratio), sig_figs),
        'max_drawdown': round(float(max_drawdown), sig_figs)
    }


    

    



    




'''


def plot_temporal_test_performance(cv_results: Dict, is_test: bool = True):
    """Plot K lines, one for the test/validation performance of 
    each k fold from temporal cross validation"""

    # gather performance per fold
    fold_data = []
    for fold_idx, fold_results in cv_results.items(): 
        # individual_window_results = []
        for window_idx, window_results in fold_results.items(): 
            metrics = fold_results[window_idx]
            metrics = metrics['test_results'] if is_test else metrics['validation_results']
            stock_results = metrics['individual_results']
            stock_results = []
            for stock, rewards in stock_results.items():
                reward_seq = np.array(rewards['cumulative_episode_rewards'])
                stock_results.append(reward_seq)
            fold_data.append(stock_results)
            # individual_window_results.append(stock_results)
        # fold_data.append(individual_window_results)
        
    # compute per fold averages
    fold_avgs = []
    for fold_list in fold_data:
        stacked_results = np.stack(fold_list)
        avgs = np.mean(stacked_results, axis=0)
        fold_avgs.append(avgs)
    
    # plot temporal fold averages
    for i in range(len(fold_avgs)):
        sns.lineplot(
            x = list(range(1, len(fold_avgs[i])+1)),
            y = fold_avgs[i],
            label = f"Fold {i}"
        )

    plt.title("Temporal Cross-Validation Test Performance (5 folds)")
    plt.xlabel("Test Window")
    plt.ylabel("Test Performance (return/reward)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    

'''