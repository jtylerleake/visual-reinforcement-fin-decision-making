
EXPERIMENT_CONFIG = {
    
    # Name and Description
    'Experiment name': 'Dow 30 Experiment',
    'Experiment description': 'Trading Experiment for Dow Jones Industrial Average',  

    # Reinforcement Learning Parameters
    'RL model': 'PPO', 
    'RL policy': 'MlpPolicy', 
    
    # Stocks --> Dow Jones Industrial Average
    'Tickers': [

    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "AVGO",
    "TSLA",
    "BRK.b",
    "LLY",
    "JPM",
    "PLTR",
    "V",
    "MA",
    "JNJ",
    "ORCL",
    "AMD",
    "BAC",
    "ABBV",
    "HD",
    "GE",
    "UNH",
    "IBM",
    "MU",
    "MS",
    "CSCO",
    "CAT",
    "AXP",
    "C",
    "APP",
    "WFC",
    "RTX"
    
    ]




    
    # Time Series Date and Frequency Attributes
    'Start date': '2004-01-01',
    'End date': '2024-12-31',
    'Update frequency': '1d',

    # Observation Space Parameters
    'Lookback window': 30,

    # OHLC Features
    'GAF features': ['Close', 'High', 'Low', 'Open', 'SMA', 'RSI' 'OBV'],
    'GAF target': 'Close',
    'GAF timeseries periods': 14,

    # Technical Indicator Features
    'SMA periods': 30,
    'RSI periods': 30,

    # Training Parameters
    'Training epochs': 1000,
    'Learning rate': 0.001,
    'Learning rate': 0.001,
    
    # Experiment Design Parameters
    'K folds': 5,
    'Time splits': 5,
    'Stratification type': 'random',
    'Random seed': 42,

    # Model Storage
    'Model save path': './models',
    'Checkpoint frequency': 100,
    
}
