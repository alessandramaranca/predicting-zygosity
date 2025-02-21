Current slurm script I am using for Della:

#!/bin/bash
#SBATCH --job-name=twin100test         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=64G         # memory 
#SBATCH --time=15:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ar0241@princeton.edu
#SBATCH --gres=gpu:1

module purge

module load anaconda3/2024.6

conda activate wandtest

# preprocessing
~/.conda/envs/tttestc/bin/python /home/ar0241/scratch/twins/preprocessing_instruct.py

# fine tuning
tune run lora_finetune_single_device --config /home/ar0241/scratch/twins/finetune_feb21.yaml  epochs=1

# evaluation
~/.conda/envs/wandtest/bin/python evaluate_model.py
