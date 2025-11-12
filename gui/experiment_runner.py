
from common.modules import os, sys, time, json, path, Dict, List, Any, Optional, dt
from src.utils.logging import get_logger
from src.utils.configurations import load_config, validate_config
from src.utils.configurations import get_available_experiments
from src.experiments.temporal_cross_validation import TemporalCrossValidation

config_dir = "experiment-configs"


class ExperimentRunner:
    
    """
    Utility class for ExperimentLauncher. Handles fetching and execution 
    of different experiments and configurations. 
    """
    
    def __init__(self):
        self.available_experiments = get_available_experiments()

    def run_experiment(
            self, 
            experiment_name: str, 
            run_id: Optional[str] = None
        ) -> Dict[str, Any]:
        """Run a specific experiment with a specific configuration"""
        
        try:
            # generate run_id if not provided
            if run_id is None:
                run_id = dt.now().strftime("%Y%m%d_%H%M%S")
            
            logger = get_logger(experiment_name, run_id=run_id)
            logger.info(f"Starting experiment: {experiment_name} (Run ID: {run_id})")
            
            # validate experiment exists
            if experiment_name not in self.available_experiments:
                raise ValueError(f"Unknown experiment: {experiment_name}")
            
            # load and validate configuration
            config = load_config(experiment_name)
            if not validate_config(experiment_name, config):
                raise ValueError(f"Invalid configuration: {experiment_name}")
            
            # run the experiment; save results
            run = TemporalCrossValidation(experiment_name, config, run_id=run_id)
            results = run.exe_experiment()
            self.save_results(experiment_name, results, run_id=run_id)
            
            logger.info(f"Experiment completed: {experiment_name} (Run ID: {run_id})")
            return results
            
        except Exception as e:
            try:
                run_id_for_error = run_id if 'run_id' in locals() else None
                logger = get_logger(experiment_name, run_id=run_id_for_error)
                logger.error(f"Error running experiment {experiment_name}: {e}")
            except:
                pass
            return {
                'experiment_name': experiment_name,
                'success': False,
                'error': str(e),
            }
    
    def save_results(
            self, 
            experiment_name: str, 
            results: Dict[str, Any], 
            run_id: Optional[str] = None
        ) -> None:
        """Save experiment results to file"""
        
        logger = get_logger(experiment_name, run_id=run_id)
        
        try:
            # create results directory
            results_dir = os.path.join(config_dir, experiment_name)
            os.makedirs(results_dir, exist_ok=True)
            
            # generate filename with run_id if provided
            timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            if run_id:
                filename = f"run_{run_id}__{timestamp}-results.json"
            else:
                filename = f"{timestamp}-results.json"
            filepath = os.path.join(results_dir, filename)
            
            # save
            save_results = results.copy()
            if 'config' in save_results:
                del save_results['config']
            
            with open(filepath, 'w') as f:
                json.dump(save_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_experiment_summary(
            self, 
            results: Dict[str, Any]
        ) -> str:
        """Generate a summary of experiment results"""
        
        try:
            summary = f"""
            Experiment Summary
            ==================
            Experiment: {results['experiment_name']}
            Configuration: {results['config_name']}
            Status: {'SUCCESS' if results['success'] else 'FAILED'}
            Duration: {results['duration_seconds']:.2f} seconds
            Start Time: {results['start_time']}
            End Time: {results['end_time']}
            Data Directory: {results['data_dir']}
            """
            
            if not results['success'] and 'error' in results:
                summary += f"Error: {results['error']}\n"
            
            return summary
            
        except Exception as e:
            # Use a basic logger if we can't get the experiment-specific one
            try:
                logger = get_logger("ExperimentSummary")
                logger.error(f"Error generating summary: {e}")
            except:
                pass
            return f"Error generating summary: {e}"
