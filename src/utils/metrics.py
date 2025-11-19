
from common.modules import np, pd, plt, sns, List, Dict, dt
from src.utils.logging import get_logger, log_function_call



def plot_best_model_overlay(
    aggregated_results: Dict,
    timeseries_data: Dict[str, pd.DataFrame] = None,
    metric_for_best: str = 'cumulative_return',
    action_buy: int = 1,
    action_sell: int = 2,
    figsize: tuple = (16, 10),
    save_path: str = None
):
    """
    Plot candlestick chart for the best performing model with buy/sell action overlays.
    
    Parameters:
    -----------
    aggregated_results : Dict
        Output from aggregate_cross_validation_results function
    timeseries_data : Dict[str, pd.DataFrame], optional
        Dictionary mapping stock tickers to OHLC DataFrames. If None, will try to extract
        from fold_results structure.
    metric_for_best : str
        Metric to use for determining best model ('cumulative_return' or 'sharpe_ratio')
    action_buy : int
        Action value that represents a buy (default: 1)
    action_sell : int
        Action value that represents a sell (default: 2)
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    try:
        from matplotlib.patches import Rectangle
        
        # Extract fold_results
        fold_results = aggregated_results.get('fold_results', {})
        if not fold_results:
            raise ValueError("No fold_results found in aggregated_results")
        
        # Find the best performing model
        best_performance = -np.inf
        best_info = None  # (fold_id, window_id, stock, strategy_key)
        
        # Metric name mapping
        metric_mapping_reverse = {
            'cumulative_return': 'cumulative return',
            'annualized_return': 'annualized return',
            'sharpe_ratio': 'sharpe ratio',
            'sortino_ratio': 'sortino ratio',
            'max_drawdown': 'max drawdown'
        }
        metric_key = metric_mapping_reverse.get(metric_for_best, metric_for_best)
        
        # Search through all test results to find the best
        for fold_id, window_results in fold_results.items():
            for window_id, fold_window_results in window_results.items():
                test_results = fold_window_results.get('test results', {})
                if isinstance(test_results, dict):
                    for stock, stock_metrics in test_results.items():
                        if isinstance(stock_metrics, dict) and metric_key in stock_metrics:
                            value = stock_metrics[metric_key]
                            if value is not None and not np.isnan(value):
                                # For max_drawdown, we want the smallest (least negative)
                                if metric_for_best == 'max_drawdown':
                                    if value > best_performance:
                                        best_performance = value
                                        best_info = (fold_id, window_id, stock, 'test results')
                                else:
                                    if value > best_performance:
                                        best_performance = value
                                        best_info = (fold_id, window_id, stock, 'test results')
        
        if not best_info:
            raise ValueError(f"No valid {metric_for_best} values found in test results")
        
        fold_id, window_id, stock, strategy_key = best_info
        
        # Extract actions and get stock metrics
        best_window_results = fold_results[fold_id][window_id]
        best_stock_metrics = best_window_results[strategy_key][stock]
        actions = best_stock_metrics.get('actions', [])
        
        if not actions:
            raise ValueError(f"No actions found for best model (Fold {fold_id}, Window {window_id}, Stock {stock})")
        
        # Get OHLC data
        if timeseries_data and stock in timeseries_data:
            ohlc_data = timeseries_data[stock].copy()
        else:
            # Try to get from config or raise error
            raise ValueError(f"OHLC data not provided and stock {stock} not found in timeseries_data")
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in ohlc_data.columns]
        if missing_cols:
            # Try alternative column names
            col_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close'
            }
            for alt_col, std_col in col_mapping.items():
                if alt_col in ohlc_data.columns and std_col not in ohlc_data.columns:
                    ohlc_data[std_col] = ohlc_data[alt_col]
        
        # Get the test period data (we need to align actions with the data)
        # Actions correspond to steps in the test environment
        # We'll use the last len(actions) rows of the data
        if len(actions) > len(ohlc_data):
            logger = get_logger("plot_best_model_overlay")
            logger.warning(f"More actions ({len(actions)}) than data points ({len(ohlc_data)}). Using available data.")
            actions = actions[:len(ohlc_data)]
        
        # Get the relevant portion of OHLC data
        plot_data = ohlc_data.iloc[-len(actions):].copy()
        
        # Set up the plot style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create candlestick chart manually
        dates = plot_data.index if isinstance(plot_data.index, pd.DatetimeIndex) else range(len(plot_data))
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine color (green for up, red for down)
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw the wick (high-low line)
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.5, alpha=0.7)
            
            # Draw the body (open-close rectangle)
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle(
                (i - 0.3, body_bottom),
                0.6,
                body_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
        
        # Overlay buy/sell action lines
        buy_indices = []
        sell_indices = []
        
        for i, action in enumerate(actions):
            if action == action_buy:
                buy_indices.append(i)
            elif action == action_sell:
                sell_indices.append(i)
        
        # Plot buy lines (green, vertical)
        if buy_indices:
            for idx in buy_indices:
                if idx < len(plot_data):
                    ax.axvline(
                        x=idx,
                        color='green',
                        linestyle='--',
                        linewidth=2.5,
                        alpha=0.8,
                        label='Buy' if idx == buy_indices[0] else '',
                        zorder=10
                    )
        
        # Plot sell lines (red, vertical)
        if sell_indices:
            for idx in sell_indices:
                if idx < len(plot_data):
                    ax.axvline(
                        x=idx,
                        color='red',
                        linestyle='--',
                        linewidth=2.5,
                        alpha=0.8,
                        label='Sell' if idx == sell_indices[0] else '',
                        zorder=10
                    )
        
        # Customize the plot
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Best Model Performance: {stock} (Fold {fold_id}, Window {window_id})\n'
            f'{metric_for_best.replace("_", " ").title()}: {best_performance:.4f}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Set x-axis labels
        if isinstance(plot_data.index, pd.DatetimeIndex):
            # Use date labels if available
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in plot_data.index], rotation=45, ha='right')
        else:
            ax.set_xticks(range(0, len(plot_data), max(1, len(plot_data) // 10)))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        if buy_indices or sell_indices:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger = get_logger("plot_best_model_overlay")
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger = get_logger("plot_best_model_overlay")
        logger.error(f"Error creating best model overlay plot: {e}")
        raise



def plot_cross_sectional_violin():
    """ """
    raise NotImplementedError



def plot_normalized_test_lines(
    cv_results: Dict,
    strategy_key: str = 'test results',
    figsize: tuple = (14, 8),
    save_path: str = None
):
    """
    Plot normalized profit factors as line charts for each test window.
    Each line starts at 1.0 and shows the evolution of portfolio factors over time.
    
    Parameters:
    -----------
    cv_results : Dict
        Raw cross-validation results with structure:
        {fold_id: {window_id: {strategy_key: {stock: {'portfolio factors': [values]}}}}}
    strategy_key : str
        Key for the strategy results to plot (default: 'test results')
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    try:
        # Collect portfolio factors for each window
        window_data = {}  # {(fold_id, window_id): [all portfolio factor arrays]}
        
        for fold_id, window_results in cv_results.items():
            for window_id, fold_window_results in window_results.items():
                # Get the strategy results for this window
                strategy_results = fold_window_results.get(strategy_key, {})
                if not isinstance(strategy_results, dict):
                    continue
                
                # Collect portfolio factors from all stocks in this window
                portfolio_factor_arrays = []
                for stock, stock_metrics in strategy_results.items():
                    if isinstance(stock_metrics, dict):
                        portfolio_factors = stock_metrics.get('portfolio factors', [])
                        if portfolio_factors and isinstance(portfolio_factors, list):
                            # Convert to numpy array and ensure it's numeric
                            factors = np.array(portfolio_factors, dtype=np.float64)
                            # Remove any NaN or invalid values
                            factors = factors[~np.isnan(factors)]
                            if len(factors) > 0:
                                portfolio_factor_arrays.append(factors)
                
                if portfolio_factor_arrays:
                    window_data[(fold_id, window_id)] = portfolio_factor_arrays
        
        if not window_data:
            raise ValueError(f"No portfolio factors found for strategy '{strategy_key}'")
        
        # Set up the plot style
        sns.set_style("whitegrid")
        sns.set_palette("husl", n_colors=len(window_data))
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each window's averaged portfolio factors
        for (fold_id, window_id), factor_arrays in window_data.items():
            # Find the maximum length to align all arrays
            max_length = max(len(arr) for arr in factor_arrays)
            
            # Align and average portfolio factors across stocks
            aligned_factors = []
            for arr in factor_arrays:
                # Pad shorter arrays with the last value
                if len(arr) < max_length:
                    padded = np.pad(arr, (0, max_length - len(arr)), mode='edge')
                else:
                    padded = arr
                aligned_factors.append(padded)
            
            # Average across stocks
            if aligned_factors:
                averaged_factors = np.mean(aligned_factors, axis=0)
                
                # Normalize to start at 1.0
                if len(averaged_factors) > 0 and averaged_factors[0] != 0:
                    normalized_factors = averaged_factors / averaged_factors[0]
                else:
                    normalized_factors = np.ones_like(averaged_factors)
                
                # Create time steps (x-axis)
                time_steps = np.arange(len(normalized_factors))
                
                # Plot the line
                label = f'Fold {fold_id}, Window {window_id}'
                ax.plot(
                    time_steps,
                    normalized_factors,
                    linewidth=2,
                    alpha=0.7,
                    label=label
                )
        
        # Customize the plot
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Profit Factor', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Normalized Test Results: Portfolio Factors Over Time\n({len(window_data)} test windows)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Add horizontal line at 1.0 for reference
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0)')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend (but make it more compact if there are many lines)
        if len(window_data) <= 10:
            ax.legend(
                loc='best',
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=9,
                ncol=2
            )
        else:
            # For many lines, use a more compact legend or omit it
            ax.legend(
                loc='upper left',
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=7,
                ncol=3
            )
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger = get_logger("plot_normalized_test_lines")
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger = get_logger("plot_normalized_test_lines")
        logger.error(f"Error creating normalized test lines plot: {e}")
        raise



def plot_temporal_stability_lines(
    aggregated_results: Dict,
    strategy_name: str = 'test',
    metric_name: str = 'cumulative_return',
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """Plot temporal stability lines showing average metric values across windows for each fold"""

    try:
        # Extract fold_results from aggregated results
        fold_results = aggregated_results.get('fold_results', {})
        if not fold_results:
            raise ValueError("No fold_results found in aggregated_results")
        
        # Strategy and metric name mappings (reverse of what's in aggregate function)
        strategy_mapping_reverse = {
            'macd': 'macd results',
            'signr': 'signr results',
            'buyandhold': 'buyandhold results',
            'random': 'random results'
        }
        
        metric_mapping_reverse = {
            'episode_reward': 'episode reward',
            'cumulative_return': 'cumulative return',
            'annualized_return': 'annualized return',
            'sharpe_ratio': 'sharpe ratio',
            'sortino_ratio': 'sortino ratio',
            'max_drawdown': 'max drawdown'
        }
        
        strategy_key = strategy_mapping_reverse.get(strategy_name, f'{strategy_name} results')
        metric_key = metric_mapping_reverse.get(metric_name, metric_name)
        
        # Extract window-level averages per fold
        fold_window_data = {}  # {fold_id: {window_id: [values]}}
        
        for fold_id, window_results in fold_results.items():
            fold_window_data[fold_id] = {}
            
            for window_id, fold_window_results in window_results.items():
                # Windows are 0-indexed in fold_results, convert to 1-indexed
                window_key = window_id + 1 if isinstance(window_id, int) else window_id
                
                # Get strategy results for this window
                strategy_result = fold_window_results.get(strategy_key, {})
                if not isinstance(strategy_result, dict):
                    continue
                
                # Collect metric values for all stocks in this window
                values = []
                for stock, stock_metrics in strategy_result.items():
                    if isinstance(stock_metrics, dict) and metric_key in stock_metrics:
                        value = stock_metrics[metric_key]
                        if value is not None and not np.isnan(value):
                            values.append(float(value))
                
                # Store values for this window in this fold
                if values:
                    fold_window_data[fold_id][window_key] = values
        
        # Prepare data for plotting
        plot_data = []
        for fold_id, window_data in fold_window_data.items():
            for window_id in sorted(window_data.keys()):
                values = window_data[window_id]
                if values:
                    avg_value = np.mean(values)
                    plot_data.append({
                        'Fold': f'Fold {fold_id}',
                        'Window': window_id,
                        'Average Metric': avg_value
                    })
        
        if not plot_data:
            raise ValueError(f"No data found for strategy '{strategy_name}' and metric '{metric_name}'")
        
        # Convert to DataFrame for seaborn
        df = pd.DataFrame(plot_data)
        
        # Set up the plot style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot lines using seaborn lineplot
        sns.lineplot(
            data=df,
            x='Window',
            y='Average Metric',
            hue='Fold',
            marker='o',
            linewidth=2.5,
            markersize=8,
            ax=ax,
            legend='full'
        )
        
        # Customize the plot
        ax.set_xlabel('Window Index', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Average {metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Temporal Stability: {strategy_name.upper()} - {metric_name.replace("_", " ").title()}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Set x-axis to show integer window indices
        unique_windows = sorted(df['Window'].unique())
        ax.set_xticks(unique_windows)
        ax.set_xticklabels([int(w) for w in unique_windows])
        
        # Customize legend
        ax.legend(
            title='Fold',
            loc='best',
            frameon=True,
            fancybox=True,
            shadow=True,
            title_fontsize=11,
            fontsize=10
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger = get_logger("plot_temporal_stability")
            logger.info(f"Plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger = get_logger("plot_temporal_stability")
        logger.error(f"Error creating temporal stability plot: {e}")
        raise







def convert_to_latex_table(
    aggregated_results: Dict = None,
    cap_results: Dict[str, Dict] = None,
    decimal_places: int = 3,
    save_path: str = None
) -> str:
    """
    Convert aggregated cross-validation results to LaTeX table format.
    
    Parameters:
    -----------
    aggregated_results : Dict, optional
        Output from aggregate_cross_validation_results. If provided and cap_results is None,
        the same data will be used for all three cap sections.
    cap_results : Dict[str, Dict], optional
        Dictionary with keys 'Large Cap', 'Medium Cap', 'Small Cap' mapping to
        aggregated_results dictionaries. If provided, this takes precedence over aggregated_results.
    decimal_places : int
        Number of decimal places to display (default: 3)
    save_path : str, optional
        Path to save the LaTeX table. If None, returns the string.
    
    Returns:
    --------
    str
        LaTeX table code as a string
    """
    try:
        # Strategy name mapping for table display
        strategy_display_names = {
            'buyandhold': 'Buy and Hold',
            'random': 'Random',
            'macd': 'MACD',
            'signr': 'Sign(R)',
            'test': 'Visual A2C Agent',  # Assuming 'test' is the visual agent
            'numeric_a2c': 'Numeric A2C Agent',
            'visual_a2c': 'Visual A2C Agent',
            # Add other agent types if needed
        }
        
        # Metric column mapping
        metric_columns = {
            'cumulative_return': ('mean', 'std', 'min', 'max'),
            'annualized_return': ('ann',),
            'sharpe_ratio': ('sharpe',),
            'sortino_ratio': ('sortino',),
            'max_drawdown': ('mdd',)
        }
        
        # Determine which results to use
        if cap_results:
            results_dict = cap_results
        elif aggregated_results:
            # Use same results for all caps
            results_dict = {
                'Large Cap': aggregated_results,
                'Medium Cap': aggregated_results,
                'Small Cap': aggregated_results
            }
        else:
            raise ValueError("Either aggregated_results or cap_results must be provided")
        
        # Helper function to format a value
        def format_value(value, default=0.0):
            if value is None or np.isnan(value):
                return default
            return round(float(value), decimal_places)
        
        # Helper function to get metric value from aggregated results
        def get_metric_value(results, strategy, metric, stat='mean'):
            try:
                overall = results.get('aggregated_metrics', {}).get('overall', {})
                strategy_data = overall.get(strategy, {})
                metric_data = strategy_data.get(metric, {})
                if isinstance(metric_data, dict):
                    return metric_data.get(stat, 0.0)
                return 0.0
            except:
                return 0.0
        
        # Helper function to extract test results if not in aggregated metrics
        def extract_test_results(results, metric, stat='mean'):
            """Extract test results from fold_results if not in aggregated_metrics"""
            try:
                fold_results = results.get('fold_results', {})
                if not fold_results:
                    return 0.0
                
                # Metric name mapping (reverse)
                metric_mapping_reverse = {
                    'cumulative_return': 'cumulative return',
                    'annualized_return': 'annualized return',
                    'sharpe_ratio': 'sharpe ratio',
                    'sortino_ratio': 'sortino ratio',
                    'max_drawdown': 'max drawdown'
                }
                metric_key = metric_mapping_reverse.get(metric, metric)
                
                # Collect all values across folds and windows
                all_values = []
                for fold_id, window_results in fold_results.items():
                    for window_id, fold_window_results in window_results.items():
                        test_results = fold_window_results.get('test results', {})
                        if isinstance(test_results, dict):
                            for stock, stock_metrics in test_results.items():
                                if isinstance(stock_metrics, dict) and metric_key in stock_metrics:
                                    value = stock_metrics[metric_key]
                                    if value is not None and not np.isnan(value):
                                        all_values.append(float(value))
                
                if not all_values:
                    return 0.0
                
                # Compute requested statistic
                if stat == 'mean':
                    return np.mean(all_values)
                elif stat == 'std':
                    return np.std(all_values)
                elif stat == 'min':
                    return np.min(all_values)
                elif stat == 'max':
                    return np.max(all_values)
                else:
                    return np.mean(all_values)
            except:
                return 0.0
        
        # Build LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table*}[h]")
        latex_lines.append("    \\caption{Experiment Results by \\textbf{Cross Section}}")
        latex_lines.append("    \\vspace{5pt}")
        latex_lines.append("    \\fontsize{9}{11}\\selectfont")
        latex_lines.append("    \\label{table:cross-sectional-results}")
        latex_lines.append("    \\centering")
        latex_lines.append("    \\begin{tabular}{l|*{8}{>{\\centering\\arraybackslash}p{1cm}}}")
        latex_lines.append("    ")
        latex_lines.append("    \\toprule")
        latex_lines.append("    & \\multicolumn{5}{c}{\\textbf{Cumulative Returns}} & \\multicolumn{3}{c}{\\textbf{Mean Risk-Adjusted Returns}} \\\\")
        latex_lines.append("    \\cmidrule(lr){2-6} \\cmidrule(lr){7-9}")
        latex_lines.append("    & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} & \\textbf{Ann.} & \\textbf{Sharpe} & \\textbf{Sortino} & \\textbf{MDD} \\\\")
        latex_lines.append("    \\midrule")
        
        # Process each cap section
        cap_order = ['Large Cap', 'Medium Cap', 'Small Cap']
        strategy_order = ['buyandhold', 'random', 'macd', 'signr', 'test']
        
        for cap_name in cap_order:
            latex_lines.append(f"    & \\multicolumn{{8}}{{c}}{{\\textbf{{{cap_name}}}}} \\\\")
            latex_lines.append("    ")
            latex_lines.append("    \\midrule")
            
            results = results_dict.get(cap_name, {})
            
            for strategy in strategy_order:
                display_name = strategy_display_names.get(strategy, strategy.title())
                
                # Check if strategy exists in aggregated results
                overall = results.get('aggregated_metrics', {}).get('overall', {})
                strategy_exists = strategy in overall
                
                # For 'test' strategy, try to extract from fold_results if not in aggregated
                if strategy == 'test' and not strategy_exists:
                    # Extract from fold_results
                    cumret_mean_val = extract_test_results(results, 'cumulative_return', 'mean')
                    cumret_std_val = extract_test_results(results, 'cumulative_return', 'std')
                    cumret_min_val = extract_test_results(results, 'cumulative_return', 'min')
                    cumret_max_val = extract_test_results(results, 'cumulative_return', 'max')
                    ann_ret_val = extract_test_results(results, 'annualized_return', 'mean')
                    sharpe_val = extract_test_results(results, 'sharpe_ratio', 'mean')
                    sortino_val = extract_test_results(results, 'sortino_ratio', 'mean')
                    mdd_val = extract_test_results(results, 'max_drawdown', 'mean')
                else:
                    # Use aggregated results
                    cumret_mean_val = get_metric_value(results, strategy, 'cumulative_return', 'mean')
                    cumret_std_val = get_metric_value(results, strategy, 'cumulative_return', 'std')
                    cumret_min_val = get_metric_value(results, strategy, 'cumulative_return', 'min')
                    cumret_max_val = get_metric_value(results, strategy, 'cumulative_return', 'max')
                    ann_ret_val = get_metric_value(results, strategy, 'annualized_return', 'mean')
                    sharpe_val = get_metric_value(results, strategy, 'sharpe_ratio', 'mean')
                    sortino_val = get_metric_value(results, strategy, 'sortino_ratio', 'mean')
                    mdd_val = get_metric_value(results, strategy, 'max_drawdown', 'mean')
                
                # Format values
                cumret_mean = format_value(cumret_mean_val)
                cumret_std = format_value(cumret_std_val)
                cumret_min = format_value(cumret_min_val)
                cumret_max = format_value(cumret_max_val)
                ann_ret = format_value(ann_ret_val)
                sharpe = format_value(sharpe_val)
                sortino = format_value(sortino_val)
                mdd = format_value(mdd_val)
                
                # Format the row
                row = f"    {display_name:<20} & {cumret_mean:.{decimal_places}f} & {cumret_std:.{decimal_places}f} & {cumret_min:.{decimal_places}f} & {cumret_max:.{decimal_places}f} & {ann_ret:.{decimal_places}f} & {sharpe:.{decimal_places}f} & {sortino:.{decimal_places}f} & {mdd:.{decimal_places}f} \\\\"
                latex_lines.append(row)
            
            latex_lines.append("    \\midrule")
            latex_lines.append("    ")
        
        latex_lines.append("    \\bottomrule")
        latex_lines.append("    ")
        latex_lines.append("    \\end{tabular}")
        latex_lines.append("    \\end{table*}")
        
        # Join all lines
        latex_table = "\n".join(latex_lines)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(latex_table)
            logger = get_logger("convert_to_latex_table")
            logger.info(f"LaTeX table saved to {save_path}")
        
        return latex_table
        
    except Exception as e:
        logger = get_logger("convert_to_latex_table")
        logger.error(f"Error creating LaTeX table: {e}")
        raise


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


def aggregate_cross_validation_results(cv_results: Dict) -> Dict:
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
