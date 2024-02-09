#!/bin/bash
#SBATCH --job-name=whisper_finetune     # create a short name for your job
#SBATCH --cpus-per-task=8                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=6G              #This is the memory reserved per core.
#SBATCH --time=84:00:00         # total run time limit (HH:MM:SS)
#SBATCH --partition=a100       # or titanx
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --qos=1week	           # qos level

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$((1 * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate whisper_finetune

# Get env variables
export $(cat .env | xargs)

python src/whisper_finetune/scripts/finetune.py --config configs/large.yaml