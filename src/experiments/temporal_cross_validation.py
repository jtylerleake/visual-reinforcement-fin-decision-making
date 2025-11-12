
from common.modules import np, pd, List, Dict, random, os, glob
from src.utils.logging import log_function_call, get_logger
from src.utils.metrics import compute_performance_metrics, aggregate_cv_results
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.environment_pipeline import EnvironmentPipeline
from src.models.visual_learner import VisualLearner
from src.models.benchmarks import MACD, SignR, BuyAndHold, Random


import PPO, A2C, DQN


class TemporalCrossValidation:
    
    """
    Temporal walk-forward K-fold cross validation experiment. Model evaluation 
    metrics are collected and aggregated across folds/windows.
    
    :param experiment_name (str): name of experiment config being run
    :param config (dict): experiment configuration dictionary
    :param run_id (int): experiment run differentiator id
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        config: Dict, 
        run_id = None
    ) -> None:
        try:
            # basic configuration
            self.experiment_name = experiment_name
            self.run_id = run_id
            self.logger = get_logger(experiment_name, run_id=run_id)
            self.config = config
            self.num_folds = self.config.get('K folds') 
            self.num_time_windows = self.config.get('Time splits')
            self.stratification_type = self.config.get('Stratification type')
            self.random_seed = self.config.get('Random seed')
            # seed reproducibility
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        except Exception as e: 
            self.logger.error(f"""Could not initialize temporal cross validation. 
            Error with configuation file parameters: {e}""")
        self.logger.info("Experiment Manager initialized")
    
    @log_function_call
    def get_folds(
        self, 
        stocks: List[str]
    ) -> Dict[int, List[str]]:
        """Partitions stocks into folds based on stratification type"""
        
        try:
            if not stocks:
                self.logger.error("No stocks provided for fold assignment")
                return {}
            if self.num_folds <= 1:
                self.logger.error("Number of folds must be greater than 1")
                return {}
            if len(stocks) < self.num_folds:
                self.logger.error(f"""Number of stocks ({len(stocks)}) is less 
                than number of folds ({self.num_folds})""")
                return {}

            # partition using stratification type == random
            if self.stratification_type == 'random':
                shuffled_stocks = stocks.copy()
                random.shuffle(shuffled_stocks)
                fold_assignments = {}
                stocks_per_fold = len(stocks) // self.num_folds
                remainder = len(stocks) % self.num_folds
                start_idx = 0
                for fold in range(self.num_folds):
                    # add one extra stock to first 'remainder' folds
                    fold_size = stocks_per_fold + (1 if fold < remainder else 0)
                    end_idx = start_idx + fold_size
                    fold_assignments[fold] = shuffled_stocks[start_idx:end_idx]
                    start_idx = end_idx
                self.logger.info(f"""Randomly assigned {len(stocks)} stocks to 
                {self.num_folds} folds""")
            
            # partition using stratification type == sector balanced
            elif self.stratification_type == 'sector balanced':
                raise NotImplementedError
            
            else:
                self.logger.error(f"""Unknown stratification strategy: 
                {self.stratification_strategy}""")
                return {}
            
            self.fold_assignments = fold_assignments
            return fold_assignments
            
        except Exception as e:
            self.logger.error(f"Error assigning stocks to folds: {e}")
            return {}
    
    @log_function_call
    def get_frame_bounds(
        self, 
        timeseries_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List]:
        """Creates frame bounds for environments based on number of total data 
        points, temporal window size, and train/val/test ratios from config"""
        
        try:
            
            # fetch config parameters
            walk_throughs = self.config.get('Walk throughs')
            lookback_window = self.config.get('Lookback window')
            training_ratio = self.config.get('Training ratio')
            validation_ratio = self.config.get('Validation ratio')
            evaluation_ratio = self.config.get('Evaluation ratio')

            # fetch number of data points in dataframe
            any_key = next(iter(timeseries_data))
            df = timeseries_data[any_key]
            num_data_points = len(df)
            adj_points = num_data_points - lookback_window
            
            # validate ratios sum to 1.0
            total_ratio = training_ratio + validation_ratio + evaluation_ratio
            if abs(total_ratio - 1.0) > 0.01:
                self.logger.warning(f"Ratios sum to {total_ratio}, not 1.0. Normalizing...")
                training_ratio /= total_ratio
                validation_ratio /= total_ratio
                evaluation_ratio /= total_ratio
                
            # calculate window sizes in data points
            train_window_size = int(adj_points * training_ratio) // walk_throughs
            val_window_size = int(adj_points * validation_ratio)
            test_window_size = int(adj_points * evaluation_ratio)
            
            # calculate frame bounds for each walk-through
            train_bounds = []
            validation_bounds = []
            evaluation_bounds = []
            
            walk_start = lookback_window
            for walk_idx in range(walk_throughs):
                
                train_start = walk_start
                train_end = train_start + train_window_size
                val_start = train_end + 1
                val_end = val_start + val_window_size
                test_start = val_end + 1
                test_end = test_start + test_window_size
                
                train_bounds.append([train_start, train_end])
                validation_bounds.append([val_start, val_end])
                evaluation_bounds.append([test_start, test_end])
                
                walk_start = train_end + 1
            
            # aggregate bounds and return
            frame_bounds = {
                'training': train_bounds,
                'validation': validation_bounds,
                'evaluation': evaluation_bounds
            }
            
            self.logger.info(f"""Created frame bounds for {len(train_bounds)} walk-throughs""")
            return frame_bounds
            
        except Exception as e:
            self.logger.error(f"Error creating temporal window frame bounds: {e}")
            return {}
    
    @log_function_call
    def exe_experiment(self) -> Dict:
        """Execute the cross-validation with temporal walk-through experiment"""
        
        self.logger.h1("Executing Temporal Walk-Forward RL Performance Experiment")
        try:
            
            # Step 1: load and preprocess data
            self.logger.info("Step 1: Preparing timeseries data pipeline...")
            timeseries_pipeline = DataPipeline(self.experiment_name, run_id=self.run_id)
            timeseries_data = timeseries_pipeline.exe_data_pipeline(self.config)
            if not timeseries_data:
                self.logger.error("Failed to prepare stock data")
                return {}
            
            # Step 2: get fold assignments
            self.logger.info("Step 2: Creating fold assignments...")
            stocks = list(timeseries_data.keys())
            fold_assignments = self.get_folds(stocks)
            if not fold_assignments:
                self.logger.error("Failed to create fold assignments")
                return {}
            
            # Step 3: get temporal window frame bounds
            self.logger.info("Step 3: Creating temporal window frame bounds...")
            frame_bounds = self.get_frame_bounds(timeseries_data)
            if not frame_bounds:
                self.logger.error("Failed to create temporal window frame bounds")
                return {}
            
            # Step 4: execute cross-validation
            self.logger.info("Step 4: Executing cross-validation...")
            results = self.exe_cross_validation(fold_assignments, frame_bounds, timeseries_data)
            
            # Step 5: aggregate results
            self.logger.info("Step 5: Aggregating results...")
            aggregated_results = aggregate_cv_results(results)
            
            self.logger.h1("Cross-Validation Experiment Completed")
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"Error executing cross-validation: {e}")
            return {}
    
    @log_function_call
    def exe_cross_validation(
        self, 
        fold_assignments: Dict[int, List[str]], 
        frame_bounds: Dict[str, List[List]],
        timeseries_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Cross validaion overhead"""        
        try:
            
            # K-fold outer loop
            fold_results = {}
            
            for fold_idx in range(self.num_folds):
                window_results = {}
                
                train_and_val_env, test_env = self.assemble_environments(
                    fold_idx, 
                    fold_assignments, 
                    timeseries_data
                )
                
                # Temporal walk-forward inner loop
                for window_idx in range(config.get('Walk throughs')):
                    self.logger.info(f"Training window {window_idx+1} in fold {fold_idx+1}")
                    
                    window_metrics = self.train_validate_test(
                        train_and_val_env,
                        test_env,
                        frame_bounds,
                        fold_idx, 
                        window_idx
                    )
                    
                    if window_metrics:
                        window_results[window_idx] = window_metrics
                        self.logger.info(f"Completed fold {fold_idx+1}, window {window_idx +1}")
                    else:
                        self.logger.warning(f"Failed fold {fold_idx+1}, window {window_idx+1}")
                        
                fold_results[fold_idx+1] = window_results
                
            return fold_results
            
        except Exception as e:
            self.logger.error(f"Error executing fold validation: {e}")
            return {}
        
    @log_function_call
    def assemble_environments(
        self,
        fold_idx,
        fold_assignments,
        timeseries_data
    ) -> tuple: 
        """Build the vectorized GAF environments for training/validation/testing"""
        
        try:
        
            # isolate the training/vaidation data from the test data
            test_stocks = fold_assignments[fold_idx] 
            train_and_val_stocks = [
                stock for stocks in fold_assignments.values() for stock 
                in stocks if stock not in test_stocks
            ]
            train_and_val_set = {stock: timeseries_data[stock] for stock in train_and_val_stocks}
            test_set = {stock: timeseries_data[stock] for stock in test_stocks}
            
            # initialize the training/validation/test environment classes
            train_and_val_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                tickers = train_and_val_stocks,
                gaf_timeseries_periods = self.config.get('GAF timeseries periods'),
                gaf_features = self.config.get('GAF features'),
                gaf_target = self.config.get('GAF target'),
                lookback_window = self.config.get('Lookback window'),
                run_id = self.run_id
            )
            
            test_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                tickers = test_stocks,
                gaf_timeseries_periods = self.config.get('GAF timeseries periods'),
                gaf_features = self.config.get('GAF features'),
                gaf_target = self.config.get('GAF target'),
                lookback_window = self.config.get('Lookback window'),
                run_id = self.run_id
            )
            
            train_and_val_env.timeseries_data = train_and_val_set
            test_env.timeseries_data = test_set
            
            # assemble the vectorized GAF environments
            if not train_and_val_env.exe_env_pipeline():
                self.logger.error("Failed to build GAF environment for training")
                return {}
            
            self.logger.info(f"""Successfully created training environment with 
            {train_and_val_env.num_vec_environments} environments""")
            
            if not test_env.exe_env_pipeline():
                self.logger.error("Failed to build GAF environment for training")
                return {}
            self.logger.info(f"""Successfully created training environment with 
            {test_env.num_vec_environments} environments""")
    
            return train_and_val_env, test_env
        
        except Exception as e: 
            self.logger.error(f"Error creating environments: {e}")
            return {}
        
    @log_function_call
    def select_best_checkpoint(
        self,
        checkpoint_files: List[str],
        agent: VisualLearner,
        train_and_val_env: EnvironmentPipeline,
    ) -> tuple:
        """Evaluate all checkpoints on validation set and return the best one"""
        
        try:

            # evaluate all checkpoints on validation
            self.logger.info("Evaluating checkpoints on validation set...")
            best_checkpoint = None
            best_validation_score = float('-inf')
            validation_results_by_checkpoint = {}
            best_agent = None
            best_validation_results = None
            
            for checkpoint_path in checkpoint_files:
                if checkpoint_path:
                    checkpoint_num = os.path.basename(checkpoint_path).replace('checkpoint_', '').replace('.zip', '')
                    self.logger.info(f"Evaluating checkpoint {checkpoint_num}...")
                    # Load the checkpoint
                    rl_models = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}
                    model_class = rl_models[self.config.get('RL model')]
                    checkpoint_agent = VisualLearner(train_and_val_env, self.config)
                    checkpoint_agent.model = model_class.load(checkpoint_path, env=train_and_val_env.vec_environment)
                    eval_agent = checkpoint_agent
                else:
                    self.logger.info("Evaluating the final model...")
                    eval_agent = agent
                
                # evaluate on validation set
                val_results = self.evaluate(eval_agent, train_and_val_env)
                
                # Calculate validation score (using average cumulative return across stocks)
                if val_results:
                    avg_cumulative_return = np.mean([
                        stock_metrics.get('cumulative return', 0)
                        for stock_metrics in val_results.values()
                    ])
                    validation_results_by_checkpoint[checkpoint_path or 'final'] = {
                        'results': val_results,
                        'score': avg_cumulative_return
                    }
                    
                    if avg_cumulative_return > best_validation_score:
                        best_validation_score = avg_cumulative_return
                        best_checkpoint = checkpoint_path
                        best_agent = eval_agent
                        best_validation_results = val_results
            
            self.logger.info(f"Best checkpoint: {best_checkpoint or 'final model'} with validation score: {best_validation_score:.4f}")
            
            return best_checkpoint, best_agent, best_validation_results, validation_results_by_checkpoint
            
        except Exception as e:
            self.logger.error(f"Error selecting best checkpoint: {e}")
            return None, agent, None, {} # fallback to final model
    
    @log_function_call
    def train_validate_test(
        self, 
        train_and_val_env,
        test_env,
        frame_bounds: dict,
        fold_idx: int, 
        window_idx: int
    ) -> Dict:
        """Train, validate, and test a model for a fold/window combination"""
        
        try:
            
            self.logger.info(f"Training model for fold {fold_idx+1}, window {window_idx+1}")
            
            training_bounds = frame_bounds['training'][window_idx]
            validation_bounds = frame_bounds['validation'][window_idx]
            evaluation_bounds = frame_bounds['evaluation'][window_idx]
            
            # Step 1: set the frame bounds for the training environment
            for ticker, monitor in train_and_val_env.environments.items():
                env = monitor.env
                env.frame_bound = (training_bounds[0], training_bounds[1])
                env._process_data()
                env.reset()
                
            # Step 2: train 
            self.logger.info("Training the model...")
            agent = VisualLearner(train_and_val_env, self.config)
            
            # set up checkpoint directory for this fold/window
            checkpoint_freq = self.config.get('Checkpoint frequency')
            checkpoint_dir = os.path.join(
                self.config.get('Model save path'),
                f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints'
            )
            
            # train with checkpoint saving
            agent.train(
                checkpoint_save_path = checkpoint_dir,
                checkpoint_freq = checkpoint_freq
            )
            
            # Step 3: find all checkpoint files
            self.logger.info("Searching for model checkpoints")
            checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.zip')))
            if not checkpoint_files:
                self.logger.warning("No checkpoints found, using final model")
                checkpoint_files = [None]  # use final model as fallback
        
            # Step 4: set the validation environment frame bounds
            for ticker, monitor in train_and_val_env.environments.items():
                env = monitor.env
                env.frame_bound = (validation_bounds[0], validation_bounds[1])
                env._process_data()
                env.reset()
            
            # Step 5: evaluate checkpoints and select best model
            best_checkpoint, best_agent, best_validation_results, validation_results_by_checkpoint = \
                self.select_best_checkpoint(
                    checkpoint_files = checkpoint_files,
                    agent = agent,
                    train_and_val_env = train_and_val_env,
                )
            
            # Step 6: set the frame bounds for the test environment
            for ticker, monitor in test_env.environments.items():
                env = monitor.env
                env.frame_bound = (evaluation_bounds[0], evaluation_bounds[1])
                env._process_data()
                env.reset()
                
            # Step 7: Test the best checkpoint
            self.logger.info("Evaluating best model on test set...")
            test_results = self.evaluate(best_agent, test_env)

            # Step 8: run benchmark models
            MACD_results = self.evaluate(MACD(), test_env)
            SignR_results = self.evaluate(SignR(), test_env)
            BuyAndHold_results = self.evaluate(BuyAndHold(), test_env)
            Random_results = self.evaluate(Random(), test_env)

            # Step 8: combine results and return 
            fold_window_results = {
                'test results': test_results,
                'validation results': best_validation_results,
                'MACD results' : MACD_results,
                'SignR results' : SignR_results,
                'Buy and Hold results' : BuyAndHold_results,
                'Random results' : Random_results,
            }
            
            self.logger.info(f"""Completed training/validation/test for 
            fold {fold_idx+1}, window {window_idx+1}""")
            return fold_window_results
        
        except Exception as e:
            self.logger.error(f"Error during Train/Validate/Test: {e}")
            return {}
        
    @log_function_call
    def evaluate(
        self,
        strategy, 
        environment
    ) -> Dict: 
        """Evaluate a trained model or benchmark strategy in an environment.
        Update individual and aggregated performance metrics by environment"""

        try:
            stock_metrics = {}
            deterministic = self.config.get('Deterministic', True)
            
            # Stock-by-Stock Evaluation
            for stock, monitor in environment.environments.items():
                
                # initialize performance metrics
                portfolio_factors = []
                episode_reward = 0
                actions = []

                # reset the environment; return first observation and dataset
                env = monitor.env
                obs, info = env.reset() 
                obs = obs[0] if isinstance(obs, tuple) else obs
                data = environment.timeseries_data[stock]

                # store initial portfolio value
                initial_portfolio_factor = info.get('total_profit', 1.0)
                portfolio_factors.append(initial_portfolio_factor)

                done = False
                step_count = 0
                while not done:
                    
                    # ensure observation has correct shape for model prediction
                    if len(obs.shape) == 3:
                        obs_batch = np.expand_dims(obs, axis=0)
                    else:
                        obs_batch = obs
                    
                    # action prediction
                    if strategy.strategy_type == 'Agent':
                        action, _ = strategy.model.predict(obs_batch, deterministic=deterministic)
                    if strategy.strategy_type == 'Benchmark':
                        last_action = None if step_count == 0 else actions[step_count-1]
                        action = strategy.predict(data, last_action, step_count)

                    if isinstance(action, (list, np.ndarray)):
                        action = int(action[0] if len(action) > 0 else action)
                    else:
                        action = int(action)
                    actions.append(action)

                    # step; reward; next state
                    step_result = env.step(action)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                    # extract portfolio value from environment info
                    # gym_anytrading tracks normalized portfolio value in total_profit
                    portfolio_factor = info.get('total_profit')
                    portfolio_factors.append(portfolio_factor)
                    episode_reward += reward
                    step_count += 1

                # compute performance metrics from portfolio values
                metrics = compute_performance_metrics(
                    portfolio_factors = portfolio_factors,
                    start_date = self.config.get('Start date'), 
                    end_date = self.config.get('End date'),
                    sig_figs = 4,
                )

                # action distribution diagnostics
                unique_actions, action_counts = np.unique(actions, return_counts = True)
                action_distribution = dict(zip(unique_actions.tolist(), action_counts.tolist()))
                
                stock_metrics[stock] = {
                    'portfolio factors': portfolio_factors,
                    'episode reward': episode_reward,
                    'actions': actions,
                    'action distribution': action_distribution,
                    'num steps': step_count,
                    'cumulative return': metrics['cumulative_return'],
                    'annualized return': metrics['annualized_cumulative_return'],
                    'sharpe ratio': metrics['sharpe_ratio'],
                    'sortino ratio': metrics['sortino_ratio'],
                    'max drawdown': metrics['max_drawdown'],
                }
                
            self.logger.info("Evaluation complete with performance metrics")
            return stock_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return False
    
    

    
if __name__ == "__main__":
    
    from src.utils.configurations import load_config
    experiment_name = 'Development'
    config = load_config(experiment_name)
    
    dev = TemporalCrossValidation(experiment_name, config)
    dev.exe_experiment()
    
    