
from common.modules import np, random, os, json, Dict, List
import optuna
from src.utils.configurations import load_config
from src.utils.logging import get_logger
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.environment_pipeline import EnvironmentPipeline
from src.models.visual_agent import VisualAgent
from src.models.numeric_agent import NumericAgent


def get_single_fold(stocks: List[str], num_folds: int, stratification_type: str, random_seed: int) -> Dict[str, List[str]]:
    """Create a single fold split for hyperparameter tuning"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if stratification_type == 'random':
        shuffled = stocks.copy()
        random.shuffle(shuffled)
        stocks_per_fold = len(stocks) // num_folds
        remainder = len(stocks) % num_folds
        
        # Use first fold for training, rest for validation
        fold_size = stocks_per_fold + (1 if 0 < remainder else 0)
        train_stocks = shuffled[:fold_size]
        val_stocks = shuffled[fold_size:]
        
        return {'train': train_stocks, 'val': val_stocks}
    else:
        raise NotImplementedError(f"Stratification type '{stratification_type}' not implemented")


def get_single_window_bounds(timeseries: Dict, config: Dict) -> Dict[str, List[int]]:
    """Get frame bounds for single window (first window)"""
    df = next(iter(timeseries.values()))
    gaf_periods = config.get('GAF periods', 14)
    num_sequences = len(df) - gaf_periods + 1
    
    # Get and normalize ratios
    train_ratio = config.get('Training ratio', 0.75)
    val_ratio = config.get('Validation ratio', 0.05)
    eval_ratio = config.get('Evaluation ratio', 0.20)
    total_ratio = train_ratio + val_ratio + eval_ratio
    
    if abs(total_ratio - 1.0) > 0.01:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        eval_ratio /= total_ratio
    
    # Calculate window sizes for first window
    walk_throughs = config.get('Walk throughs', 5)
    train_window_size = int((num_sequences * train_ratio) / walk_throughs) - 1
    val_window_size = int(num_sequences * val_ratio)
    
    # First window bounds
    train_start = 0
    train_end = train_start + train_window_size
    val_start = train_end + 1
    val_end = val_start + val_window_size
    
    return {
        'training': [train_start, train_end],
        'validation': [val_start, val_end]
    }


def set_env_frame_bounds(env_pipeline, bounds: List[int], modalities: List[str] = ['image', 'numeric']) -> None:
    """Set frame bounds for all environments in a pipeline"""
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


def evaluate_agent(agent, environment, modality: str, deterministic: bool = True) -> float:
    """Evaluate agent and return average cumulative return across all stocks"""
    env_dict = getattr(environment, f'{modality}_environments', {})
    if not env_dict:
        return -float('inf')
    
    cumulative_returns = []
    
    for stock, monitor in env_dict.items():
        portfolio_factors = []
        env = monitor.env
        obs, info = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        
        portfolio_factors.append(info.get('total_profit', 1.0))
        done = False
        
        while not done:
            # Prepare observation
            obs_batch = np.expand_dims(obs, axis=0) if len(obs.shape) == 3 else obs
            
            # Get action
            action, _ = agent.model.predict(obs_batch, deterministic=deterministic)
            
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.ndim == 0 else int(action[0])
            elif isinstance(action, list):
                action = int(action[0]) if len(action) > 0 else 0
            else:
                action = int(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            portfolio_factors.append(info.get('total_profit', 1.0))
        
        # Compute cumulative return
        if len(portfolio_factors) > 1:
            cumulative_return = (portfolio_factors[-1] - portfolio_factors[0]) / portfolio_factors[0]
            cumulative_returns.append(cumulative_return)
    
    # Return average cumulative return
    if cumulative_returns:
        return float(np.mean(cumulative_returns))
    else:
        return -float('inf')


def objective(trial, config: Dict, train_env, val_env, agent_type: str = 'visual') -> float:
    """Optuna objective function for hyperparameter tuning"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('Learning rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('Batch size', [32, 64, 128, 256])
    n_steps = trial.suggest_int('Rollout steps', 512, 2048, step=256)
    gamma = trial.suggest_float('Gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_float('GAE lambda', 0.8, 0.99)
    clip_range = trial.suggest_float('Clip range', 0.1, 0.3)
    ent_coef = trial.suggest_float('Entropy coefficient', 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float('VF coefficient', 0.1, 1.0)
    max_grad_norm = trial.suggest_float('Max grad norm', 0.3, 1.0)
    n_epochs = trial.suggest_int('Epochs', 5, 20)
    
    # Visual agent specific
    if agent_type == 'visual':
        features_dim = trial.suggest_categorical('Feature dim', [128, 256, 512])
    else:
        features_dim = None
    
    # Create modified config with suggested hyperparameters
    trial_config = config.copy()
    
    # Initialize agent-specific hyperparameter dict if it doesn't exist
    agent_key = 'Visual agent hyperparameters' if agent_type == 'visual' else 'Numeric agent hyperparameters'
    if agent_key not in trial_config:
        trial_config[agent_key] = {}
    
    # Set hyperparameters in both general config and agent-specific config
    # (agents check agent-specific first, then fall back to general)
    hyperparams = {
        'Learning rate': learning_rate,
        'Batch size': batch_size,
        'Rollout steps': n_steps,
        'Gamma': gamma,
        'GAE lambda': gae_lambda,
        'Clip range': clip_range,
        'Entropy coefficient': ent_coef,
        'VF coefficient': vf_coef,
        'Max grad norm': max_grad_norm,
        'Epochs': n_epochs
    }
    
    if features_dim is not None:
        hyperparams['Feature dim'] = features_dim
    
    # Set in both places
    for key, value in hyperparams.items():
        trial_config[key] = value
        trial_config[agent_key][key] = value
    
    # Set Training epochs for the train() method (use a reasonable value for tuning)
    # This is the total timesteps for training - using rollout steps * a multiplier
    trial_config['Training epochs'] = n_steps * 10  # Train for 10 rollouts worth of steps
    
    # Create and train agent
    try:
        if agent_type == 'visual':
            agent = VisualAgent(train_env, trial_config)
        else:
            agent = NumericAgent(train_env, trial_config)
        
        # Train agent (no checkpoint saving for tuning)
        agent.train(checkpoint_save_path=None, checkpoint_freq=None)
        
        # Evaluate on validation set
        modality = 'image' if agent_type == 'visual' else 'numeric'
        deterministic = config.get('Deterministic', True)
        avg_cumulative_return = evaluate_agent(agent, val_env, modality, deterministic)
        
        return avg_cumulative_return
        
    except Exception as e:
        # Return very poor score if training fails
        logger = get_logger("hyperparameter_tuning")
        logger.warning(f"Trial failed: {e}")
        return -float('inf')


def run_hyperparameter_tuning(
    experiment_name: str,
    agent_type: str = 'visual',
    n_trials: int = 50,
    fold_idx: int = 0,
    window_idx: int = 0
) -> Dict:
    """Run hyperparameter tuning for specified experiment and agent type"""
    
    logger = get_logger("hyperparameter_tuning")
    logger.info(f"Starting hyperparameter tuning for {experiment_name} ({agent_type} agent)")
    
    # Load config
    config = load_config(experiment_name)
    random_seed = config.get('Random seed', 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load data
    logger.info("Loading timeseries data...")
    data_pipeline = DataPipeline(experiment_name)
    timeseries_data = data_pipeline.exe_data_pipeline(config)
    if not timeseries_data:
        raise RuntimeError("Failed to load timeseries data")
    
    # Create single fold split
    logger.info("Creating fold split...")
    stocks = list(timeseries_data.keys())
    num_folds = config.get('K folds', 5)
    stratification_type = config.get('Stratification type', 'random')
    fold_split = get_single_fold(stocks, num_folds, stratification_type, random_seed)
    
    train_stocks = fold_split['train']
    val_stocks = fold_split['val']
    
    train_set = {s: timeseries_data[s] for s in train_stocks}
    val_set = {s: timeseries_data[s] for s in val_stocks}
    
    # Create environments
    logger.info("Creating environments...")
    features = config.get('Features', [])
    target = config.get('Target', 'Close')
    gaf_periods = config.get('GAF periods', 14)
    lookback_window = config.get('Lookback window', 7)
    
    train_env = EnvironmentPipeline(
        experiment_name=experiment_name,
        timeseries=train_set,
        features=features,
        target=target,
        gaf_periods=gaf_periods,
        lookback_window=lookback_window
    )
    
    val_env = EnvironmentPipeline(
        experiment_name=experiment_name,
        timeseries=val_set,
        features=features,
        target=target,
        gaf_periods=gaf_periods,
        lookback_window=lookback_window
    )
    
    if not train_env.exe_env_pipeline('image') or not train_env.exe_env_pipeline('numeric'):
        raise RuntimeError("Failed to build training environments")
    
    if not val_env.exe_env_pipeline('image') or not val_env.exe_env_pipeline('numeric'):
        raise RuntimeError("Failed to build validation environments")
    
    # Get frame bounds for single window
    logger.info("Setting up temporal window...")
    frame_bounds = get_single_window_bounds(timeseries_data, config)
    
    # Set training bounds
    set_env_frame_bounds(train_env, frame_bounds['training'])
    set_env_frame_bounds(val_env, frame_bounds['validation'])
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    logger.info(f"Starting optimization with {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, config, train_env, val_env, agent_type),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best trial value: {best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Save to JSON
    output_dir = os.path.join("experiments", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"best_hyperparams_{agent_type}.json")
    
    output_data = {
        'experiment_name': experiment_name,
        'agent_type': agent_type,
        'best_value': float(best_value),
        'best_trial_number': study.best_trial.number,
        'best_hyperparameters': best_params,
        'n_trials': n_trials
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    logger.info(f"Best hyperparameters saved to {output_file}")
    
    return output_data

