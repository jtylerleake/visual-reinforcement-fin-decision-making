
from common.modules import os, imut, dt, Dict, List, Any
from src.utils.logging import get_logger

PARENT_DIR = "experiments"


def load_config(experiment_name: str) -> Dict[str, Any]:
    """Load the configuration file for a specificied experiment"""
    
    logger = get_logger(experiment_name)

    try: 
        config_path = os.path.join(PARENT_DIR, experiment_name, "config.py")
        spec = imut.spec_from_file_location("config", config_path)
        config_module = imut.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_var_name = "EXPERIMENT_CONFIG"
        config = getattr(config_module, config_var_name)
        logger.info(f"Loaded configuration for experiment: {experiment_name}")
        return config
        
    except ImportError as e:
        logger.error(f"Could not import '{experiment_name}' config: {e}")
        raise
        
    except AttributeError as e:
        logger.error(f"Could not find config variable '{config_var_name}' \
                     in {experiment_name}_config.py: {e}")
        raise


def validate_config(experiment_name: str, config: Dict[str, Any]) -> bool:
    """Validate parameters in configuration dictionary"""
    
    logger = get_logger(experiment_name)
    
    try:
        
        required_params = [
            'Experiment name',
            'Tickers',
            'Start date',
            'End date',
            'GAF features',
            'GAF target',
            'GAF timeseries periods',
            'Lookback window',
            'RL model'
        ]
        
        # check for required parameters
        missing_params = [p for p in required_params if p not in config]
        if missing_params:
            logger.error(f"Missing required parameters: {missing_params}")
            return False
        
        # validate parameter types and values
        if not isinstance(config['Tickers'], list) or \
            len(config['Tickers']) == 0:
            logger.error("'tickers' must be a non-empty list")
            return False
        
        if not isinstance(config['GAF features'], list) or \
            len(config['GAF features']) == 0:
            logger.error("'GAF features' must be a non-empty list")
            return False
        
        if config['GAF timeseries periods'] <= 0:
            logger.error("'GAF timeseries periods' must be positive")
            return False
        
        if config['Lookback window'] <= 0:
            logger.error("'Lookback window' must be positive")
            return False
        
        # validate date format
        try:
            dt.strptime(config['Start date'], '%Y-%m-%d')
            dt.strptime(config['End date'], '%Y-%m-%d')
        except ValueError:
            logger.error("Date format must be 'YYYY-MM-DD'")
            return False
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False


def get_available_experiments() -> List[str]:
    """Retrieve all experiment config names from the experiment-configs dir"""
    
    try:
        available_experiments = []
        
        # check if config directory exists
        if not os.path.exists(PARENT_DIR):
            return available_experiments
        
        # iterate through all subdirectories
        for item in os.listdir(PARENT_DIR):
            
            item_path = os.path.join(PARENT_DIR, item)
            
            # skip if not a directory
            if not os.path.isdir(item_path):
                continue
            
            # check if config.py exists in this directory
            config_file = os.path.join(item_path, "config.py")
            if os.path.isfile(config_file):
                available_experiments.append(item)
        
        return available_experiments
        
    except Exception as e:
        return []