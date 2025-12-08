
experiments=("Large-Cap" "Medium-Cap" "Small-Cap")

for experiment in "${experiments[@]}"; do

    # --- execute the experiment --- # 
    echo "--- Executing training for **$experiment** ---"
    
    python3 -c "
# import the required modules
from src.utils.configurations import load_config
from src.experiments.temporal_cross_validation import TemporalCrossValidation

# load the experiment config file
experiment_name = '$experiment'
config = load_config(experiment_name)

# execute the experiment in training mode; models will save to experiment directory
experiment = TemporalCrossValidation(experiment_name, config)
experiment.exe_experiment('training')
"
    
    echo " Completed **$experiment** experiment"

    # --- save the visual and numeric model checkpoints to the mounted storage --- #

    cd ~   
    rsync -av --info=progress2 visual-reinforcement-fin-decision-making/experiments/"${experiment}"/visual_models visual-reinforcement-fin-decision-making-storage/experiments/"${experiment}"
    rsync -av --info=progress2 visual-reinforcement-fin-decision-making/experiments/"${experiment}"/numeric_models visual-reinforcement-fin-decision-making-storage/experiments/"${experiment}"
    echo "Saved the model checkpoints to storage"
    cd visual-reinforcement-fin-decision-making
    
done

echo "All training experiments completed and saved"
echo "Exiting..."