
experiments=("Large-Cap" "Medium-Cap" "Small-Cap")
agents=("visual" "numeric")

for experiment in "${experiments[@]}"; do

    for agent in "${agents[@]}"; do

        # --- 1. execute hyperparameter tuning for each experiment subset --- #

        echo "--- Executing hyperparameter tuning for **$experiment** **$agent** ---"
        
        python3 -c "
# import the required modules
from src.utils.configurations import load_config
from src.experiments.hyperparameter_tuning import *

# load the experiment config file
experiment_name = '$experiment'
config = load_config(experiment_name)

# execute hyperparameter tuning; results will save to experiment directory
results = run_hyperparameter_tuning(
    experiment_name=experiment_name,
    agent_type='$agent',
    n_trials=50
)
"
        
        echo "Hyperparameter tuning completed for **$experiment** **$agent** "

    done
    
done

echo "Hyperparameter tuning completed and saved for all experiments"
echo "Exiting..."