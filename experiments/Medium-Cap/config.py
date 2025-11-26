
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
        'Learning rate': 0.0073613462973597015,
        'Batch size': 128,
        'Rollout steps': 2304,
        'Gamma': 0.9497942586147119,
        'GAE lambda': 0.8658769234738106,
        'Clip range': 0.1114813972649783,
        'Entropy coefficient': 3.901094178742899e-05,
        'VF coefficient': 0.5624017982876938,
        'Max grad norm': 0.7039004968856596,
        'Epochs': 8,
        'Feature dim': 128
    },
    
    # Numeric Agent Hyperparameters
    'Numeric agent hyperparameters': {
        'Learning rate': 0.0072252971345909064,
        'Batch size': 128,
        'Rollout steps': 2816,
        'Gamma': 0.9808761602671783,
        'GAE lambda': 0.9569102268142909,
        'Clip range': 0.24330636188599397,
        'Entropy coefficient': 1.1895982634836095e-07,
        'VF coefficient': 0.35513761478796224,
        'Max grad norm': 0.8303937212647076,
        'Epochs': 8
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
    'Tickers' : ['BIIB', 'BRO', 'FE', 'PPL', 'STE', 'ADM', 'CSGP', 'EL', 'NUE', 'RJF', 'STT', 'A', 'DHI', 'EBAY', 'EQT', 'HSY', 'RMD', 'ROK', 'VMC', 'VTR', 'D', 'GWW', 'OKE', 'PSA', 'URI', 'AFL', 'FDX', 'SPG', 'SRE', 'TFC', 'TSCO', 'FITB', 'MCHP', 'ROL', 'MTB', 'CTSH', 'WAB', 'CCL', 'ACGL', 'YUM', 'ROST', 'ROP', 'EA', 'PCAR', 'XEL', 'CMI', 'MSI', 'ADSK', 'ALL', 'NSC']
       
}
