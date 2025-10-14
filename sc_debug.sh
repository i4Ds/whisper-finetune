#!/bin/bash
#SBATCH --job-name=debug_whisper_finetune     # create a short name for your job
#SBATCH --cpus-per-task=8                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G              #This is the memory reserved per core.
#SBATCH --time=00:30:00         # total run time limit (HH:MM:SS)
#SBATCH --partition=a100     # or titanx
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --qos=gpu30min      # qos level

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate whisper_finetune

# Get env variables
export $(cat .env | xargs)

python src/whisper_finetune/scripts/finetune.py --config configs/config_1.yaml