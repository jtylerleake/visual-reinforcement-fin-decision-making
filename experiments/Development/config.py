
from common.modules import os
current_directory = os.path.dirname(os.path.abspath(__file__))
experiment_name = os.path.basename(current_directory)


EXPERIMENT_CONFIG = {
    
    'Experiment name': experiment_name,
    
    # ----------------------------------------------------------------
    # MODEL CONFIGURATION
    # ----------------------------------------------------------------
    
    # Reinforcement Learning Model
    'RL model': 'PPO', 
    'RL policy': 'CnnPolicy', 
    'Lookback window': 5, # observation space lookback period
    'Rollout steps' : 100, # trajectory collection size
    'Batch size': 50,
    'Deterministic' : False,
    'Training epochs': 10,
    'Learning rate': 0.01,
    'Learning rate': 0.01,
    
    # ----------------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # ----------------------------------------------------------------
    
    # Time Series Date and Frequency Attributes
    'Start date': '2015-01-01',
    'End date': '2020-12-30',
    'Update frequency': '1d',
    
    # K-Fold Cross Validation Parameters
    'Random seed': 42,
    'K folds': 5,
    'Stratification type': 'random',
    'Checkpoint frequency': 100,
    
    # Temporal Walk-Forward Parameters
    
    # Note on settings:
    # (1) the sum of evaluation/validation/training ratios must sum to 1
    # (2) the modulus of training ratio and walk throughs must equal 0 in
    # order to get uniform training window sizes
    
    'Evaluation ratio' : 0.20, # % data allocated to evaluation
    'Validation ratio' : 0.05, # % data allocated to validation
    'Training ratio' : 0.75, # % data allocated to training
    'Walk throughs': 5, # number of walk-forwards per K-fold
    
    # Model Storage
    'Model save path': f"./experiments/./{experiment_name}./models",
    
    # ----------------------------------------------------------------
    # DATASET CONFIGURATION
    # ----------------------------------------------------------------
    
    # Stocks
    'Tickers': [
        "NVDA",
        "AAPL",
        "MSFT",
        "AMZN",
        "AVGO",
        # "TSLA",
        # "AXP",
        # "LLY",
        # "JPM",
        # "PLTR",
        # 'TRV',
        # 'UNH',
        # 'VZ',
        # 'WMT',
        # 'V',
        # 'MRK',
        # 'NKE',
        # 'PG',
        # 'CVX',
        # 'DIS',
    ],
    
    # OHLC Features
    'GAF features': ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA', 'RSI', 'OBV'],
    'GAF target': 'Close',
    'GAF timeseries periods': 14, # historical periods in a single gaf image

    # Technical Indicator Features
    'SMA periods': 30,
    'RSI periods': 30,
       
}
