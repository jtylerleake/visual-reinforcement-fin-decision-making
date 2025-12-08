
from common.modules import np, pd, List, Dict, random, os, glob, PPO
from src.utils.logging import get_logger
from src.utils.metrics import get_performance_metrics, get_aggregate_stats, get_portfolio_factors
from src.utils.results_io import save_results
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.environment_pipeline import EnvironmentPipeline
from src.models.visual_agent import VisualAgent
from src.models.numeric_agent import NumericAgent
from src.models.benchmarks import MACD, Long, Random


class TemporalCrossValidation:
    
    """
    Temporal walk-forward K-fold cross validation experiment
    
    :param experiment_name (str): name of experiment config being run
    :param config (dict): experiment configuration dictionary
    :param run_id (int): experiment run differentiator id
    
    """
    
    def __init__(
        self, 
        experiment_name, 
        config,
        run_id = None,
    ) -> None:
        
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.logger = get_logger(experiment_name, run_id=run_id)
        self.config = config
        
        # cache config values
        self.num_folds = config.get('K folds')
        self.num_time_windows = config.get('Time splits')
        self.stratification_type = config.get('Stratification type')
        self.random_seed = config.get('Random seed')
        self.walk_throughs = config.get('Walk throughs')
        self.features = config.get('Features')
        self.target = config.get('Target')
        self.gaf_periods = config.get('GAF periods')
        self.lookback_window = config.get('Lookback window')
        self.checkpoint_freq = config.get('Checkpoint frequency')
        self.deterministic = config.get('Deterministic', True)
        self.visual_save_path = config.get('Visual agent save path')
        self.numeric_save_path = config.get('Numeric agent save path')
        self.results_save_path = config.get('Results save path')
        self.portfolio_save_path = config.get('Portfolio factors save path')
        
        # cache results
        self.cross_validation_results = None
        self.aggregated_results = None
        
        # seed reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.logger.info("Temporal Cross Validation initialized")
    
    def get_folds(
        self, 
        stocks,
    ) -> Dict[int, List[str]]:
        """Partitions stocks into folds based on stratification type"""
        if not stocks or self.num_folds <= 1 or len(stocks) < self.num_folds:
            self.logger.error("Invalid fold configuration")
            return {}
        if self.stratification_type == 'random':
            shuffled = stocks.copy()
            random.shuffle(shuffled)
            fold_assignments = {}
            stocks_per_fold = len(stocks) // self.num_folds
            remainder = len(stocks) % self.num_folds
            start_idx = 0
            for fold in range(self.num_folds):
                fold_size = stocks_per_fold + (1 if fold < remainder else 0)
                end_idx = start_idx + fold_size
                fold_assignments[fold] = shuffled[start_idx:end_idx]
                start_idx = end_idx
            self.logger.info(f"Assigned {len(stocks)} stocks to {self.num_folds} folds")
            return fold_assignments
        elif self.stratification_type == 'sector balanced':
            raise NotImplementedError("Sector balanced stratification not implemented")
        else:
            self.logger.error(f"Unknown stratification: {self.stratification_type}")
            return {}
    
    def get_frame_bounds(
        self, 
        timeseries
    ) -> Dict[str, List]:
        """Creates frame bounds for temporal walk-forward windows"""
        df = next(iter(timeseries.values()))
        num_sequences = len(df) - self.gaf_periods + 1
        
        # get and normalize ratios
        train_ratio = self.config.get('Training ratio')
        val_ratio = self.config.get('Validation ratio')
        eval_ratio = self.config.get('Evaluation ratio')
        total_ratio = train_ratio + val_ratio + eval_ratio
        
        if abs(total_ratio - 1.0) > 0.01:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            eval_ratio /= total_ratio
        
        # calculate window sizes
        train_window_size = int((num_sequences * train_ratio) / self.walk_throughs) - 1
        val_window_size = int(num_sequences * val_ratio)
        test_window_size = int(num_sequences * eval_ratio)
        
        # generate bounds for each walk-through
        train_bounds, val_bounds, eval_bounds = [], [], []
        walk_start = 0
        max_index = num_sequences - 1  # Maximum valid index (0-indexed)
        
        for _ in range(self.walk_throughs):
            train_end = min(walk_start + train_window_size, max_index)
            val_start = min(train_end + 1, max_index)
            val_end = min(val_start + val_window_size, max_index)
            test_start = min(val_end + 1, max_index)
            test_end = min(test_start + test_window_size - 1, max_index)
            
            # Ensure bounds are valid (start <= end)
            train_end = max(train_end, walk_start)
            val_end = max(val_end, val_start)
            test_end = max(test_end, test_start)
            
            train_bounds.append([walk_start, train_end])
            val_bounds.append([val_start, val_end])
            eval_bounds.append([test_start, test_end])
            walk_start = train_end + 1
            
            # Break if we've reached the end of available data
            if walk_start > max_index:
                break
        
        self.logger.info(f"Created frame bounds for {self.walk_throughs} walk-throughs")
        return {'training': train_bounds, 'validation': val_bounds, 'evaluation': eval_bounds}
    
    def _set_env_frame_bounds(
        self, 
        env_pipeline, 
        bounds,
        modalities = ['image', 'numeric']
    ) -> None:
        """Helper to set frame bounds for all environments in a pipeline"""
        for modality in modalities:
            env_dict = getattr(env_pipeline, f'{modality}_environments', {})
            if not env_dict:
                continue
            for monitor in env_dict.values():
                env = monitor.env
                env.frame_bound = tuple(bounds)
                env._start_tick = bounds[0]
                env._end_tick = bounds[1]
                env._process_data()
                env.reset()
    
    def _create_env_pipeline(
        self, 
        timeseries,
    ) -> EnvironmentPipeline:
        """Create and build environment pipeline"""
        env = EnvironmentPipeline(
            experiment_name = self.experiment_name,
            timeseries = timeseries,
            features = self.features,
            target = self.target,
            gaf_periods = self.gaf_periods,
            lookback_window = self.lookback_window,
            run_id = self.run_id
        )
        if not env.exe_env_pipeline('image') or not env.exe_env_pipeline('numeric'):
            raise RuntimeError("Failed to build environments")
        return env
    
    def assemble_environments(
        self, 
        fold_idx,
        fold_assignments, 
        timeseries,
    ) -> tuple:
        """Build vectorized environments for training/testing"""
        test_stocks = fold_assignments[fold_idx]
        train_stocks = [
            s for fold_stocks in fold_assignments.values() 
            for s in fold_stocks if s not in test_stocks
        ]
        train_set = {s: timeseries[s] for s in train_stocks}
        test_set = {s: timeseries[s] for s in test_stocks}
        train_env = self._create_env_pipeline(train_set)
        test_env = self._create_env_pipeline(test_set)
        return train_env, test_env
    
    def select_best_checkpoint(
        self, 
        checkpoint_files,
        agent, 
        agent_class, 
        train_env, 
        modality
    ):
        """Evaluate checkpoints and return best model"""
        if not checkpoint_files or checkpoint_files == [None]:
            return agent
        
        best_agent = agent
        best_score = float('-inf')
        model_class = PPO
        vec_env = getattr(train_env, f'{modality}_vec_environment')
        
        for checkpoint_path in checkpoint_files:
            if not checkpoint_path:
                continue
            
            try:
                checkpoint_agent = agent_class(train_env, self.config)
                checkpoint_agent.model = model_class.load(checkpoint_path, env=vec_env)
                val_results = self.evaluate(checkpoint_agent, train_env, modality)
                
                if val_results:
                    avg_return = np.mean(
                        [m.get('cumulative return', 0) for m in val_results.values()]
                    )
                    
                    if avg_return > best_score:
                        best_score = avg_return
                        best_agent = checkpoint_agent
                        
            except Exception as e:
                self.logger.warning(f"Error loading checkpoint {checkpoint_path}: {e}")
                continue
        
        return best_agent
    
    def evaluate(
        self, 
        strategy, 
        environment, 
        modality,
    ) -> Dict:
        """Evaluate a strategy in an environment using vectorized batch processing for maximum speedup"""
        stock_metrics = {}
        
        # Check if this is a benchmark strategy (needs individual stock data)
        is_benchmark = hasattr(strategy, 'strategy_type') and strategy.strategy_type == 'Benchmark'
        is_agent = hasattr(strategy, 'strategy_type') and strategy.strategy_type == 'Agent'
        
        if is_benchmark:
            # Benchmark strategies need sequential processing (they use individual stock data)
            env_dict = getattr(environment, 'image_environments', {})
            
            for stock, monitor in env_dict.items():
                portfolio_factors = []
                episode_reward = 0
                actions = []
                env = monitor.env
                obs, info = env.reset()
                obs = obs[0] if isinstance(obs, tuple) else obs
                portfolio_factors.append(info.get('total_profit', 1.0))
                done = False
                step_count = 0
                
                while not done:
                    # for benchmark strategies
                    data = environment.timeseries[stock]
                    last_action = actions[-1] if actions else None
                    action = strategy.predict(data, last_action, step_count)
                    
                    if isinstance(action, np.ndarray):
                        action = int(action.item()) if action.ndim == 0 else int(action[0])
                    elif isinstance(action, list):
                        action = int(action[0]) if len(action) > 0 else 0
                    else:
                        action = int(action)
                    actions.append(action)
                    
                    # step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    portfolio_factors.append(info.get('total_profit', 1.0))
                    episode_reward += reward
                    step_count += 1
                
                unique_actions, action_counts = np.unique(actions, return_counts=True)
                
                # compute metrics
                metrics = get_performance_metrics(
                    portfolio_factors = portfolio_factors,
                    start_date = self.config.get('Start date'),
                    end_date = self.config.get('End date'),
                    sig_figs = 4
                )
                            
                # compile stock metrics
                stock_metrics[stock] = {
                    'portfolio factors': portfolio_factors,
                    'episode reward': episode_reward,
                    'actions': actions,
                    'action distribution': dict(zip(unique_actions.tolist(), action_counts.tolist())),
                    'num steps': step_count,
                    'cumulative return': metrics['cumulative return'],
                    'annualized return': metrics['annualized cumulative return'],
                    'sharpe ratio': metrics['sharpe ratio'],
                    'sortino ratio': metrics['sortino ratio'],
                    'max drawdown': metrics['max drawdown'],
                }
        
        elif is_agent:
            # Agent strategies: Use vectorized environment for parallel batch processing
            vec_env = getattr(environment, f'{modality}_vec_environment', None)
            env_dict = getattr(environment, f'{modality}_environments', {})
            
            if vec_env is None or not env_dict:
                raise NotImplementedError
            
            # Get stock names in the same order as vectorized environment
            stock_names = list(env_dict.keys())
            num_stocks = len(stock_names)
            
            # Initialize tracking structures for all stocks
            portfolio_factors_dict = {stock: [] for stock in stock_names}
            episode_rewards = {stock: 0.0 for stock in stock_names}
            actions_dict = {stock: [] for stock in stock_names}
            step_counts = {stock: 0 for stock in stock_names}
            
            # Reset all environments at once (vectorized)
            obs_batch = vec_env.reset()
            # Handle tuple return from reset
            if isinstance(obs_batch, tuple):
                obs_batch = obs_batch[0]
            
            # Initialize portfolio factors with 1.0 (normalized portfolios start at 1.0)
            # We'll get the actual initial value from the first step's info
            for stock in stock_names:
                portfolio_factors_dict[stock].append(1.0)
            
            # Track which environments are done
            dones = np.zeros(num_stocks, dtype=bool)
            
            # Main evaluation loop - process all stocks in parallel
            max_steps = 10000  # Safety limit to prevent infinite loops
            step = 0
            
            while not np.all(dones) and step < max_steps:
                # Batch predict actions for all active environments
                # Only predict for environments that aren't done
                active_mask = ~dones
                
                if np.any(active_mask):
                    # Get observations for active environments
                    active_obs = obs_batch[active_mask]
                    num_active = np.sum(active_mask)
                    
                    # Batch prediction - single call for all active stocks
                    actions_batch, _ = strategy.model.predict(active_obs, deterministic=self.deterministic)
                    
                    # Convert actions to proper format
                    if isinstance(actions_batch, np.ndarray):
                        if actions_batch.ndim == 0:
                            actions_batch = np.array([int(actions_batch.item())])
                        else:
                            actions_batch = actions_batch.flatten().astype(int)
                    elif isinstance(actions_batch, list):
                        actions_batch = np.array([int(a) if isinstance(a, (int, np.integer)) else int(a[0]) for a in actions_batch])
                    else:
                        actions_batch = np.array([int(actions_batch)])
                    
                    # Ensure actions_batch has the correct length
                    if len(actions_batch) != num_active:
                        # If mismatch, pad or truncate (shouldn't happen, but safety check)
                        if len(actions_batch) < num_active:
                            actions_batch = np.pad(actions_batch, (0, num_active - len(actions_batch)), mode='constant', constant_values=0)
                        else:
                            actions_batch = actions_batch[:num_active]
                    
                    # Create full action array (use 0 for done environments)
                    full_actions = np.zeros(num_stocks, dtype=int)
                    full_actions[active_mask] = actions_batch
                    
                    # Store actions for active stocks
                    for idx in np.where(active_mask)[0]:
                        stock = stock_names[idx]
                        actions_dict[stock].append(int(full_actions[idx]))
                else:
                    # All done, break
                    break
                
                # Step all environments in parallel (vectorized)
                step_results = vec_env.step(full_actions)
                obs_batch, rewards, terminated, infos = step_results
                
                # Update done states
                new_dones = terminated 
                dones = dones | new_dones
                
                # Update tracking for all stocks
                for idx, stock in enumerate(stock_names):
                    if not dones[idx]:
                        # Active environment: track portfolio and rewards
                        portfolio_factors_dict[stock].append(infos[idx].get('total_profit', 1.0))
                        episode_rewards[stock] += float(rewards[idx])
                        step_counts[stock] += 1
                    elif new_dones[idx]:
                        # Environment just finished: capture final state
                        portfolio_factors_dict[stock].append(infos[idx].get('total_profit', 1.0))
                        episode_rewards[stock] += float(rewards[idx])
                        step_counts[stock] += 1
                
                step += 1
            
            # Compute metrics for all stocks
            for stock in stock_names:
                portfolio_factors = portfolio_factors_dict[stock]
                actions = actions_dict[stock]
                
                if len(portfolio_factors) > 1:
                    unique_actions, action_counts = np.unique(actions, return_counts=True)
                    
                    metrics = get_performance_metrics(
                        portfolio_factors = portfolio_factors,
                        start_date = self.config.get('Start date'),
                        end_date = self.config.get('End date'),
                        sig_figs = 4
                    )
                    
                    stock_metrics[stock] = {
                        'portfolio factors': portfolio_factors,
                        'episode reward': episode_rewards[stock],
                        'actions': actions,
                        'action distribution': dict(zip(unique_actions.tolist(), action_counts.tolist())),
                        'num steps': step_counts[stock],
                        'cumulative return': metrics['cumulative return'],
                        'annualized return': metrics['annualized cumulative return'],
                        'sharpe ratio': metrics['sharpe ratio'],
                        'sortino ratio': metrics['sortino ratio'],
                        'max drawdown': metrics['max drawdown'],
                    }
                else:
                    # Fallback for stocks with insufficient data
                    stock_metrics[stock] = {
                        'portfolio factors': portfolio_factors,
                        'episode reward': episode_rewards[stock],
                        'actions': actions,
                        'action distribution': {},
                        'num steps': step_counts[stock],
                        'cumulative return': 0.0,
                        'annualized return': 0.0,
                        'sharpe ratio': 0.0,
                        'sortino ratio': 0.0,
                        'max drawdown': 0.0,
                    }
        
        return stock_metrics
    
    def train_validate_test(
        self, 
        train_env, 
        test_env, 
        frame_bounds,
        fold_idx, 
        window_idx, 
        mode,
    ) -> Dict:
        """Train, validate, and test for a fold/window combination"""
        
        train_bounds = frame_bounds['training'][window_idx]
        val_bounds = frame_bounds['validation'][window_idx]
        eval_bounds = frame_bounds['evaluation'][window_idx]
        
        checkpoint_dirs = {
            'image': os.path.join(
                self.visual_save_path, 
                f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints'
            ),
            'numeric': os.path.join(
                self.numeric_save_path,
                f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints'
            )
        }
        
        if mode == 'training':
            # set training bounds and train agents
            self._set_env_frame_bounds(train_env, train_bounds)
            visual_agent = VisualAgent(train_env, self.config)
            numeric_agent = NumericAgent(train_env, self.config)
            
            visual_agent.train(
                checkpoint_save_path = checkpoint_dirs['image'],
                checkpoint_freq=self.checkpoint_freq
            )
            numeric_agent.train(
                checkpoint_save_path=checkpoint_dirs['numeric'],
                checkpoint_freq=self.checkpoint_freq
            )
            return {}
        
        # inference mode; find checkpoints
        image_checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dirs['image'], 'checkpoint_*.zip'))
            ) or [None]
        numeric_checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dirs['numeric'], 'checkpoint_*.zip'))
            ) or [None]
        
        # set validation bounds and evaluate
        self._set_env_frame_bounds(train_env, val_bounds)
        visual_agent = VisualAgent(train_env, self.config)
        numeric_agent = NumericAgent(train_env, self.config)
        
        best_visual = self.select_best_checkpoint(
            image_checkpoints, visual_agent,
            VisualAgent, train_env, 'image'
        )
        best_numeric = self.select_best_checkpoint(
            numeric_checkpoints, numeric_agent,
            NumericAgent, train_env, 'numeric'
        )
        
        # set test bounds and evaluate
        self._set_env_frame_bounds(test_env, eval_bounds)
        visual_results = self.evaluate(best_visual, test_env, 'image')
        numeric_results = self.evaluate(best_numeric, test_env, 'numeric')
        
        # evaluate baselines
        baseline_results = {}
        for baseline_class in [MACD, Long, Random]:
            baseline = baseline_class()
            baseline_results[f'{baseline_class.__name__}'] = \
                self.evaluate(baseline, test_env, 'image')
        
        return {
            'Visual agent': visual_results,
            'Numeric agent': numeric_results,
            **baseline_results
        }
    
    def exe_cross_validation(
        self, 
        fold_assignments,
        frame_bounds,
        timeseries, 
        mode
    ) -> Dict:
        """Execute cross-validation across folds and windows"""
        fold_results = {}
        
        for fold_idx in range(self.num_folds):
            self.logger.info(f"Processing fold {fold_idx+1}/{self.num_folds}")
            
            train_env, test_env = self.assemble_environments(
                fold_idx, 
                fold_assignments, 
                timeseries
            )
            window_results = {}
            
            for window_idx in range(self.walk_throughs):
                self.logger.info(f"  Window {window_idx+1}/{self.walk_throughs}")
                
                window_metrics = self.train_validate_test(
                    train_env, 
                    test_env, 
                    frame_bounds, 
                    fold_idx, 
                    window_idx, mode
                )
                
                if mode == 'inference' and window_metrics:
                    window_results[window_idx+1] = window_metrics
            
            if mode == 'inference':
                fold_results[fold_idx+1] = window_results
        
        return fold_results
    
    def exe_experiment(
        self, 
        mode,
    ) -> Dict:
        """Execute the complete cross-validation experiment"""
        
        self.logger.info("Starting Temporal Cross-Validation Experiment")
        
        try:
            
            # step 1: load data
            self.logger.info("Loading timeseries data...")
            data_pipeline = DataPipeline(self.experiment_name, run_id=self.run_id)
            timeseries_data = data_pipeline.exe_data_pipeline(self.config)
            if not timeseries_data:
                raise RuntimeError("Failed to load timeseries data")
            
            # step 2: create folds
            self.logger.info("Creating fold assignments...")
            stocks = list(timeseries_data.keys())
            fold_assignments = self.get_folds(stocks)
            if not fold_assignments:
                raise RuntimeError("Failed to create fold assignments")
            
            # step 3: create frame bounds
            self.logger.info("Creating temporal frame bounds...")
            frame_bounds = self.get_frame_bounds(timeseries_data)
            if not frame_bounds:
                raise RuntimeError("Failed to create frame bounds")
            
            # step 4: execute cross-validation
            self.logger.info("Executing cross-validation...")
            self.cross_validation_results = self.exe_cross_validation(
                fold_assignments, 
                frame_bounds, 
                timeseries_data, 
                mode
            )
            
            # step 5: aggregate and results
            if mode == 'inference':

                # tabular statistics
                self.logger.info("Aggregating results...")
                self.aggregated_stats = get_aggregate_stats(self.cross_validation_results)
                self.save_experiment_results(
                    self.aggregated_stats,
                    filepath = self.results_save_path,
                    format = 'json',
                    compress = True
                )
                
                # portfolio factors
                self.logger.info("Collecting portfolio factors...")
                portfolio_factors = get_portfolio_factors(self.cross_validation_results)
                self.save_experiment_results(
                    portfolio_factors,
                    filepath = self.portfolio_save_path,
                    format = 'json',
                    compress = True
                )
                
                self.logger.info("Inference phase completed")
                return True
                
            self.logger.info("Training phase completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in experiment: {e}")
            return {}
    
    def save_experiment_results(
        self, 
        results: Dict, 
        filepath: str = None, 
        format: str = 'pickle', 
        compress: bool = True
    ) -> str:
        """Save experiment results"""
        save_results(
            results = results,
            experiment_name = self.experiment_name,
            filepath = filepath,
            format = format,
            compress = compress,
            run_id = self.run_id
        )

