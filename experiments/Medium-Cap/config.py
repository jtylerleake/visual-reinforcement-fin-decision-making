
from common.modules import os
current_directory = os.path.dirname(os.path.abspath(__file__))
experiment_name = os.path.basename(current_directory)


EXPERIMENT_CONFIG = {
    
    'Experiment name': experiment_name,
    'Random seed': 42,
    
    # ----------------------------------------------------------------
    # MODEL CONFIGURATION
    # ----------------------------------------------------------------
    
    # Visual Reinforcement Learning Model
    'RL model': 'PPO', 
    'RL policy': 'CnnPolicy', 
    'Lookback window': 7,   # observation space lookback period
    'Rollout steps' : 2048,
    'Batch size': 64,
    'Deterministic' : True,
    'Training epochs': 1000, # total timesteps for training
    'Learning rate': 0.001,
    
    'Checkpoint frequency': 100,
    'Model save path': f".\experiments\{experiment_name}\models", # model storage
    
    # ----------------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # ----------------------------------------------------------------
    
    # Time Series Date and Frequency Attributes
    'Start date': '1994-01-01',
    'End date': '2025-09-30',
    'Update frequency': '1d',
    
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
    
    # OHLC Features
    'GAF features': ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA', 'RSI', 'OBV'],
    'GAF target': 'Close',
    'GAF timeseries periods': 14, # historical periods in a single gaf image

    # Technical Indicator Features
    'SMA periods': 20,
    'RSI periods': 14,

    # Stocks
    'Tickers' : ['FOX', 'EXE', 'PPL', 'BR', 'FE', 'FSLR', 'BRO', 'BIIB', 'CBOE', 'STE', 'NRG', 'VRSK', 'NUE', 'RJF', 'GEHC', 'CSGP', 'EL', 'ADM', 'VICI', 'STT', 'EQT', 'AMP', 'RMD', 'A', 'HSY', 'ROK', 'DHI', 'EBAY', 'VTR', 'VMC', 'GWW', 'LVS', 'OKE', 'D', 'PSA', 'URI', 'CBRE', 'MPWR', 'MSCI', 'NXPI', 'KMI', 'FDX', 'AFL', 'SRE', 'MPC', 'SPG', 'TFC', 'VST', 'PYPL', 'HLT']
       
}
