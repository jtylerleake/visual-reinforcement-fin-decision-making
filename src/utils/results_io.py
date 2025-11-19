
from common.modules import os, json, np, pd, glob, Dict, Any
import pickle
import gzip
from src.utils.logging import get_logger, log_function_call
from datetime import datetime


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


@log_function_call
def save_results(
    results: Dict,
    experiment_name: str,
    filepath: str = None,
    format: str = 'pickle',
    compress: bool = True,
    run_id: int = None
) -> str:
    """
    Save experiment results dictionary to file.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from temporal cross-validation experiment
    experiment_name : str
        Name of the experiment
    filepath : str, optional
        Full path to save file. If None, generates automatic filename.
    format : str
        File format: 'pickle' (default, preserves all types) or 'json' (portable, converts numpy)
    compress : bool
        Whether to compress the file (only for pickle format)
    run_id : int, optional
        Run ID to include in filename
        
    Returns:
    --------
    str
        Path to saved file
    """
    logger = get_logger(experiment_name, run_id=run_id)
    
    try:
        # Generate filepath if not provided
        if filepath is None:
            results_dir = os.path.join("experiments", experiment_name, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if run_id is not None:
                base_filename = f"run_{run_id}__{timestamp}"
            else:
                base_filename = timestamp
            
            if format == 'pickle':
                if compress:
                    filepath = os.path.join(results_dir, f"{base_filename}-results.pkl.gz")
                else:
                    filepath = os.path.join(results_dir, f"{base_filename}-results.pkl")
            elif format == 'json':
                filepath = os.path.join(results_dir, f"{base_filename}-results.json")
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")
        
        # Save based on format
        if format == 'pickle':
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Results saved to {filepath} (pickle format, {'compressed' if compress else 'uncompressed'})")
            
        elif format == 'json':
            # Convert numpy types to native Python types
            json_results = _convert_numpy_types(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filepath} (JSON format)")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


@log_function_call
def load_results(
    filepath: str,
    experiment_name: str = None
) -> Dict:
    """
    Load experiment results from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the results file
    experiment_name : str, optional
        Name of the experiment (for logging)
        
    Returns:
    --------
    Dict
        Loaded results dictionary
    """
    logger = get_logger(experiment_name) if experiment_name else get_logger("load_results")
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        # Determine format from file extension
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {filepath} (JSON format)")
            
        elif filepath.endswith('.pkl.gz') or filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
            logger.info(f"Results loaded from {filepath} (compressed pickle format)")
            
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            logger.info(f"Results loaded from {filepath} (pickle format)")
            
        else:
            # Try pickle first, then JSON
            try:
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"Results loaded from {filepath} (pickle format, auto-detected)")
            except:
                with open(filepath, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Results loaded from {filepath} (JSON format, auto-detected)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise


def get_latest_results(
    experiment_name: str,
    format: str = 'pickle'
) -> str:
    """
    Get the path to the most recently saved results file for an experiment.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    format : str
        Format to look for: 'pickle' or 'json'
        
    Returns:
    --------
    str
        Path to the latest results file, or None if not found
    """
    results_dir = os.path.join("experiments", experiment_name, "results")
    
    if not os.path.exists(results_dir):
        return None
    
    # Get all result files matching the format
    if format == 'pickle':
        pattern = "*-results.pkl*"
    else:
        pattern = "*-results.json"
    
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        return None
    
    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

