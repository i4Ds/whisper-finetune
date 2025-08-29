#!/bin/bash
#SBATCH --job-name=whisper_finetune_${CONFIG_NAME:-job}   # job name (we'll override CONFIG_NAME below)
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=80:00:00
#SBATCH --partition=a100-80g # or titanx
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu1week
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vincenzo293@gmail.com

# If you want each job to have a unique SLURM job name based on the config file name,
# uncomment the following two lines.  They’ll set CONFIG_NAME to the basename of $1 (no extension).
# CONFIG_NAME=$(basename "$1" .yaml)
# sbatch --job-name=whisper_finetune_${CONFIG_NAME} "$0" "$@"

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate whisper_finetune

# Load any environment variables from .env
export $(cat .env | xargs)

# Run the Python finetuning script, passing in the config file as "$1"
python src/whisper_finetune/scripts/finetune.py --config "$1"
