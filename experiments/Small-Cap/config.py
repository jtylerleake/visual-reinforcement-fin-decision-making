
from common.modules import os
current_directory = os.path.dirname(os.path.abspath(__file__))
experiment_name = os.path.basename(current_directory)


EXPERIMENT_CONFIG = {
    
    'Experiment name': experiment_name,
    'Random seed': 42,
    
    # ----------------------------------------------------------------
    # MODEL CONFIGURATION
    # ----------------------------------------------------------------
    
    # RL Models
    'Lookback window': 7,   # observation space lookback period
    'Deterministic' : True,
    'Training epochs' : 10,
    
    # Checkpoint variables
    'Checkpoint frequency': 100,
    'Visual agent save path': f".\experiments\{experiment_name}\\visual_models",
    'Numeric agent save path': f".\experiments\{experiment_name}\\numeric_models", 
    
    # ----------------------------------------------------------------
    # AGENT-SPECIFIC HYPERPARAMETERS (from hyperparameter tuning)
    # ----------------------------------------------------------------
    
    # Visual Agent Hyperparameters
    'Visual agent hyperparameters': {
        'Learning rate': 0.006411989123196969,
        'Batch size': 256,
        'Rollout steps': 4096,
        'Gamma': 0.9302533598825136,
        'GAE lambda': 0.9117136519309937,
        'Clip range': 0.1989187331282529,
        'Entropy coefficient': 1.0849278368344497e-08,
        'VF coefficient': 0.8916664120437997,
        'Max grad norm': 0.4868062316899426,
        'Epochs': 10,
        'Feature dim': 512
    },
    
    # Numeric Agent Hyperparameters
    'Numeric agent hyperparameters': {
        'Learning rate': 0.0002392009799441135,
        'Batch size': 32,
        'Rollout steps': 1792,
        'Gamma': 0.9934069286858497,
        'GAE lambda': 0.9074808566528025,
        'Clip range': 0.20402037672851311,
        'Entropy coefficient': 0.00012910415451389588,
        'VF coefficient': 0.8662065176611409,
        'Max grad norm': 0.6424830487310397,
        'Epochs': 10
    },
    
    # ----------------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # ----------------------------------------------------------------
    
    # K-Fold Cross Validation Parameters
    'K folds': 5,
    'Stratification type': 'random',
    
    # Temporal Walk-Forward Parameters
    # note: the modulus of training ratio and walk throughs must equal 
    # zero in order to get uniform training window sizes
    
    'Walk throughs': 5,         # # of walk-forwards per K-fold
    'Evaluation ratio' : 0.20,  # % data allocated to evaluation
    'Validation ratio' : 0.05,  # % data allocated to validation
    'Training ratio' : 0.75,    # % data allocated to training
    
    # ----------------------------------------------------------------
    # DATASET CONFIGURATION
    # ----------------------------------------------------------------
    
    # Time Series Date and Frequency Attributes
    'Start date': '2004-01-01',
    'End date': '2024-12-30',
    'Update frequency': '1d',
    
    # OHLC Features
    'Features': ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA', 'RSI', 'OBV'],
    'Target': 'Close',
    'GAF periods': 14, # historical periods in a single gaf image

    # Technical Indicator Features
    'SMA periods': 20,
    'RSI periods': 14,
    
    # Stocks
    'Tickers' : ['ARE', 'APA', 'AVY', 'BALL', 'BAX', 'BBY', 'CHRW', 'CPT', 'DRI', 'DECK', 'ESS', 'FFIV', 'FRT', 'HAS', 'HPQ', 'INCY', 'IFF', 'IP', 'JBHT', 'LH', 'MAA', 'NTAP', 'NDSN', 'NTRS', 'NVR', 'PNW', 'PPG', 'PTC', 'PHM', 'RVTY', 'SWK', 'TROW', 'TXT', 'COO', 'TRMB', 'TYL', 'UHS', 'WAT', 'WST', 'CPB', 'IPG', 'TAP', 'MGM', 'BXP', 'SNA', 'LII', 'KIM', 'GPC', 'STLD', 'VRSN']
       
}
