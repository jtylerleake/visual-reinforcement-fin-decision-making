
from common.modules import os
current_directory = os.path.dirname(os.path.abspath(__file__))
experiment_name = os.path.basename(current_directory)


EXPERIMENT_CONFIG = {
    
    'Experiment name': experiment_name,
    'Random seed': 42,
    
    # Model and results save paths
    'Visual agent save path': f"./experiments/{experiment_name}/visual_models",
    'Numeric agent save path': f"./experiments/{experiment_name}/numeric_models", 
    'Results save path': f"./experiments/{experiment_name}/aggregated_statistics", 
    'Portfolio factors save path': f"./experiments/{experiment_name}/portfolio_factors", 
    
    # -------------------------------------------------------------------------
    # Hyperparameters (hardcoded from hyperparam tuning)
    # -------------------------------------------------------------------------
    
    'Lookback window': 21, 
    'Deterministic' : True,
    'Checkpoint frequency': 100,
    
    'Visual agent hyperparameters': {
        'Learning rate': 0.00175,
        'Batch size': 32,
        'Rollout steps': 2816,
        'Gamma': 0.90674,
        'GAE lambda': 0.82528,
        'Clip range': 0.13220,
        'Entropy coefficient': 0.00000,
        'VF coefficient': 0.10887,
        'Max grad norm': 0.72251,
        'Epochs': 9,
        'Feature dim': 256,
    },
    
    'Numeric agent hyperparameters': {
        'Learning rate': 0.00001,
        'Batch size': 128,
        'Rollout steps': 3072,
        'Gamma': 0.99363,
        'GAE lambda': 0.80246,
        'Clip range': 0.25003,
        'Entropy coefficient': 0.00000,
        'VF coefficient': 0.39161,
        'Max grad norm': 0.99929,
        'Epochs': 16,
    },
    
    # -------------------------------------------------------------------------
    # Experiment Design
    # -------------------------------------------------------------------------
    
    'K folds': 5,
    'Walk throughs': 5,
    'Stratification type': 'random',
    
    # note: the modulus of training ratio and walk throughs must equal 
    # zero in order to get uniform training window sizes
    
    'Evaluation ratio' : 0.20,
    'Validation ratio' : 0.05,
    'Training ratio' : 0.75,
    
    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    
    # Time and frequency attributes
    'Start date': '2004-12-30',
    'End date': '2024-12-30',
    'Update frequency': '1d',
    
    # Features / target
    'Features': ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA', 'RSI', 'OBV'],
    'Target': 'Close',
    'GAF periods': 14, 
    'SMA periods': 20,
    'RSI periods': 14,
    
    'Tickers' : ['AZO', 'CI', 'EMR', 'RCL', 'AEP', 'TRV', 'WMB', 'PWR', 'AMT', 'CDNS', 'GD', 'MMC', 'NEM', 'ORLY', 'SO', 'TT', 'DHR', 'GILD', 'HON', 'PLD', 'SYK', 'TXN', 'ABT', 'AMAT', 'AMGN', 'CAT', 'DIS', 'GS', 'T', 'TMO', 'AAPL', 'BRK.B', 'MU', 'ORCL', 'WFC', 'WMT', 'MAR', 'REGN', 'MCK', 'PH', 'KLAC', 'ETN', 'SPGI', 'UNP', 'MS', 'RTX', 'HD', 'PG', 'MSFT', 'JNJ']

}

