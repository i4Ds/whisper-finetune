#!/bin/bash
#SBATCH --job-name=whisper_finetune     # create a short name for your job
#SBATCH --cpus-per-task=8                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G              #This is the memory reserved per core.
#SBATCH --time=24:00:00         # total run time limit (HH:MM:SS)
#SBATCH --partition=a100       # or titanx
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --qos=1week         # qos level
#SBATCH --exclude sgi61

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate whisper_finetune

# Get env variables
export $(cat .env | xargs)

python src/whisper_finetune/scripts/finetune.py --config configs/large-sg-corpus-mc.yaml