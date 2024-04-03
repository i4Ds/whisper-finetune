#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=whisper_ft
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --exclude=gpu22a,gpu22b,node15,sdas2
#SBATCH --output=logs/whisper_ft_%x_%j.out
#SBATCH --error=logs/whisper_ft_%x_%j.err

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate whisper-finetune

# Run the training script with the provided configuration file
python src/whisper_finetune/scripts/finetune.py --config configs/large-lora-sg-corpus.yaml