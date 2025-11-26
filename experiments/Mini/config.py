
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
    'Rollout steps' : 2048,
    'Batch size': 64,
    'Training epochs': 1000,  # total timesteps for training
    'Lookback window': 7,   # observation space lookback period
    'Deterministic' : True,
    'Learning rate': 0.001,
    'Gamma': 0.99, 
    'GAE lambda': 0.95,     
    'Clip range': 0.2,      
    'Entropy coefficient': 0.0,       
    'VF coefficient': 0.5,      
    'Max grad norm': 0.5,     
    'Epochs': 10,         
    'Feature dim': 256, 
    
    # Checkpoint variables
    'Checkpoint frequency': 100,
    'Visual agent save path': f".\experiments\{experiment_name}\\visual_models",
    'Numeric agent save path': f".\experiments\{experiment_name}\\numeric_models", 
    
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
    'Tickers' : ['BIIB', 'BRO', 'FE', 'PPL', 'STE', 'ADM', 'CSGP', 'EL', 'NUE', 'RJF']
}
