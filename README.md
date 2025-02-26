Slurm script I'm using for Della: 

#!/bin/bash
#SBATCH --job-name=abs_diff_test   # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=25G                      # Memory allocation
#SBATCH --time=23:00:00                # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin              # Email when job starts
#SBATCH --mail-type=end                # Email when job ends
#SBATCH --mail-user=ar0241@princeton.edu
#SBATCH --gres=gpu:1                    # Request 1 GPU

module purge
module load anaconda3/2024.6
conda activate envi
~/.conda/envs/envi/bin/python scratch/twins/preproc_feb25.py
tune run lora_finetune_single_device --config /home/ar0241/scratch/twins/finetune_feb21.yaml  epochs=3
~/.conda/envs/envi/bin/python scratch/twins/eval_feb26.py
~                                                                 
