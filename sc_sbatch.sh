#!/bin/bash
#SBATCH --job-name=whisper_finetune     # create a short name for your job
#SBATCH --cpus-per-task=4                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=16G              #This is the memory reserved per core.
#SBATCH --time=48:00:00         # total run time limit (HH:MM:SS)
#SBATCH --partition=a100       # or titanx
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --qos=gpu1week         # qos level
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vincenzo293@gmail.com

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate whisper_finetune

# Get env variables
export $(cat .env | xargs)

python src/whisper_finetune/scripts/finetune.py --config configs/large-sg-corpus-mc.yaml